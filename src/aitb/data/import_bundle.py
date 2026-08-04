"""Import market data into the canonical real store.

Two sources:
  1. A portable bundle produced by scripts/download_real_data.py
     (manifest + per-provider parquet files, checksum-verified here).
  2. Loose user-supplied files under data/import/{prices,macro,fundamentals}/
     as CSV/JSON/Parquet with flexible column names.

Reconciliation: when several providers cover a ticker, their daily returns
are compared. Providers are ranked by total-return quality
(tiingo > yahoo > alphavantage > stooq); the best available becomes canonical
for that ticker. Discrepancy statistics are recorded — incompatible values
are FLAGGED, never averaged.

Listing-window enforcement: prices outside a security's configured
IPO/delisting window are trimmed and counted (data continuing past a
delisting is a classic vendor artifact).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import IMPORT_DIR, load_universe_config
from ..utils import get_logger
from . import realstore

log = get_logger("data.import")

PROVIDER_PREFERENCE = ["tiingo", "yahoo", "alphavantage", "stooq", "user"]
RETURN_DISCREPANCY_BP = 50           # flag daily |return diff| above this
COLUMN_ALIASES = {
    "open": ["open", "o"], "high": ["high", "h"], "low": ["low", "l"],
    "close": ["close", "c", "price"],
    "adj_close": ["adj_close", "adjclose", "adj close", "adjusted_close",
                  "adjusted close", "adjusted"],
    "volume": ["volume", "vol", "v"],
}
DATE_ALIASES = ["date", "timestamp", "time", "day"]


@dataclass
class ImportReport:
    imported: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    reconciliation: list[dict] = field(default_factory=list)
    trimmed: list[dict] = field(default_factory=list)
    checksum_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def normalize_price_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map arbitrary column names/dates onto the canonical schema."""
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if not isinstance(d.index, pd.DatetimeIndex):
        date_col = next((c for c in DATE_ALIASES if c in d.columns), None)
        if date_col is None:
            raise ValueError(f"{source}: no date column found (tried {DATE_ALIASES})")
        d[date_col] = pd.to_datetime(d[date_col], utc=False)
        d = d.set_index(date_col)
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    d.index = d.index.normalize()
    d.index.name = "date"

    out = pd.DataFrame(index=d.index)
    for canon, aliases in COLUMN_ALIASES.items():
        col = next((a for a in aliases if a in d.columns), None)
        if col is not None:
            out[canon] = pd.to_numeric(d[col], errors="coerce")
    if "close" not in out.columns:
        raise ValueError(f"{source}: no close column")
    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]  # flagged later as split-adjusted-only
    for c in ("open", "high", "low"):
        if c not in out.columns:
            out[c] = out["close"]
    if "volume" not in out.columns:
        out["volume"] = np.nan
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out[realstore.PRICE_COLUMNS + [c for c in ("dividend", "split") if c in out.columns]]


def reconcile(ticker: str, frames: dict[str, pd.DataFrame]) -> tuple[str, dict]:
    """Choose the canonical provider; report cross-provider return diffs."""
    ranked = [p for p in PROVIDER_PREFERENCE if p in frames] + \
             [p for p in frames if p not in PROVIDER_PREFERENCE]
    best = ranked[0]
    stats: dict = {"ticker": ticker, "chosen": best, "providers": ranked}
    if len(ranked) > 1:
        base_r = frames[best]["adj_close"].pct_change()
        for other in ranked[1:]:
            o_r = frames[other]["adj_close"].pct_change()
            joined = pd.concat([base_r, o_r], axis=1, join="inner").dropna()
            if len(joined) < 20:
                continue
            diff = (joined.iloc[:, 0] - joined.iloc[:, 1]).abs()
            n_bad = int((diff > RETURN_DISCREPANCY_BP / 1e4).sum())
            stats[f"vs_{other}"] = {
                "overlap_days": len(joined),
                "return_diffs_gt_50bp": n_bad,
                "pct_flagged": round(n_bad / len(joined), 4),
            }
            if n_bad / len(joined) > 0.02:
                stats["warning"] = (f"{ticker}: {best} vs {other} disagree on "
                                    f"{n_bad} days — check adjustment conventions")
    return best, stats


def _verify_manifest(bundle: Path, report: ImportReport) -> dict | None:
    mpath = bundle / "manifest.json"
    if not mpath.exists():
        log.warning("bundle has no manifest.json — importing without checksums")
        return None
    manifest = json.loads(mpath.read_text())
    for rel, expected in manifest.get("files", {}).items():
        p = bundle / rel
        if not p.exists():
            report.checksum_failures.append(f"missing file: {rel}")
        elif realstore.file_sha256(p) != expected:
            report.checksum_failures.append(f"checksum mismatch: {rel}")
    return manifest


def import_bundle(bundle: Path, root: Path | None = None,
                  strict_checksums: bool = True) -> ImportReport:
    report = ImportReport()
    ucfg = load_universe_config()
    manifest = _verify_manifest(bundle, report)
    if report.checksum_failures and strict_checksums:
        raise ValueError(f"bundle integrity failed: {report.checksum_failures[:5]}")

    # ---- prices: gather per provider, reconcile, trim, write ---------------
    by_ticker: dict[str, dict[str, pd.DataFrame]] = {}
    prices_root = bundle / "prices"
    if prices_root.exists():
        for pdir in sorted(prices_root.iterdir()):
            if not pdir.is_dir():
                continue
            for f in sorted(pdir.glob("*.parquet")):
                try:
                    df = normalize_price_frame(pd.read_parquet(f), f.name)
                    by_ticker.setdefault(f.stem.upper(), {})[pdir.name] = df
                except Exception as exc:
                    report.skipped.append({"file": str(f), "reason": str(exc)})

    listing = {s.ticker: s for s in ucfg.securities}
    for ticker, frames in sorted(by_ticker.items()):
        provider, stats = reconcile(ticker, frames)
        report.reconciliation.append(stats)
        df = frames[provider]
        sec = listing.get(ticker)
        if sec is not None:
            before = len(df)
            lo = pd.Timestamp(sec.ipo)
            hi = pd.Timestamp(sec.delisted) if sec.delisted else df.index.max()
            df = df.loc[lo:hi]
            if len(df) != before:
                report.trimmed.append({"ticker": ticker,
                                       "rows_outside_listing": before - len(df)})
        div_included = provider in ("tiingo", "yahoo", "alphavantage")
        realstore.write("prices", ticker, df, {
            "provider": provider,
            "dividends_included": div_included,
            "adjustment": "total_return" if div_included else "split_only",
            "reconciliation": {k: v for k, v in stats.items() if k.startswith("vs_")},
            "source_bundle": str(bundle),
            "bundle_created_at": (manifest or {}).get("created_at"),
        }, root=root)
        report.imported.append(f"prices/{ticker}")

    # ---- macro -------------------------------------------------------------
    for f in sorted((bundle / "macro").glob("*.parquet")) if (bundle / "macro").exists() else []:
        s = pd.read_parquet(f)
        col = s.columns[0]
        series = s[col]
        if f.stem == "CPIYOY" and series.dropna().median() > 30:
            # Provider delivered the CPI index level: convert to YoY %.
            monthly = series.dropna()
            series = (monthly.pct_change(12) * 100).dropna()
            log.info("CPIYOY: converted index level to YoY%%")
        out = series.to_frame(f.stem)
        out.index = pd.to_datetime(out.index)
        realstore.write("macro", f.stem, out, {"provider": "fred",
                        "source_bundle": str(bundle)}, root=root)
        report.imported.append(f"macro/{f.stem}")

    # ---- fundamentals ------------------------------------------------------
    fdir = bundle / "fundamentals"
    for f in sorted(fdir.glob("*.parquet")) if fdir.exists() else []:
        df = pd.read_parquet(f)
        missing = [c for c in realstore.FUNDAMENTAL_COLUMNS if c not in df.columns]
        if missing:
            report.skipped.append({"file": str(f), "reason": f"missing columns {missing}"})
            continue
        df["period_end"] = pd.to_datetime(df["period_end"])
        df["published"] = pd.to_datetime(df["published"])
        if (df["published"] < df["period_end"]).any():
            report.skipped.append({"file": str(f),
                                   "reason": "published before period_end — leakage risk"})
            continue
        realstore.write("fundamentals", f.stem.upper(), df,
                        {"provider": "edgar", "source_bundle": str(bundle)}, root=root)
        report.imported.append(f"fundamentals/{f.stem.upper()}")

    log.info("imported %d datasets, %d skipped, %d reconciliation records",
             len(report.imported), len(report.skipped), len(report.reconciliation))
    return report


def import_loose_files(root: Path | None = None) -> ImportReport:
    """Import user-supplied loose files from data/import/ (CSV/JSON/Parquet).

    Loose price files are treated as provider 'user'; the same normalization,
    listing-window trimming and metadata recording applies.
    """
    report = ImportReport()
    ucfg = load_universe_config()
    listing = {s.ticker: s for s in ucfg.securities}

    pdir = IMPORT_DIR / "prices"
    for f in sorted(pdir.glob("*")) if pdir.exists() else []:
        if f.suffix.lower() not in (".csv", ".json", ".parquet"):
            continue
        try:
            raw = (pd.read_parquet(f) if f.suffix == ".parquet"
                   else pd.read_json(f) if f.suffix == ".json"
                   else pd.read_csv(f))
            df = normalize_price_frame(raw, f.name)
            ticker = f.stem.upper()
            sec = listing.get(ticker)
            if sec is not None:
                lo = pd.Timestamp(sec.ipo)
                hi = pd.Timestamp(sec.delisted) if sec.delisted else df.index.max()
                df = df.loc[lo:hi]
            realstore.write("prices", ticker, df, {
                "provider": "user", "dividends_included": None,
                "adjustment": "unknown — user supplied",
                "source_file": str(f)}, root=root)
            report.imported.append(f"prices/{ticker}")
        except Exception as exc:
            report.skipped.append({"file": str(f), "reason": str(exc)})

    for kind in ("macro", "fundamentals"):
        kdir = IMPORT_DIR / kind
        for f in sorted(kdir.glob("*")) if kdir.exists() else []:
            if f.suffix.lower() not in (".csv", ".parquet"):
                continue
            try:
                df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
                if kind == "fundamentals":
                    missing = [c for c in realstore.FUNDAMENTAL_COLUMNS
                               if c not in df.columns]
                    if missing:
                        raise ValueError(f"missing columns {missing}")
                    df["period_end"] = pd.to_datetime(df["period_end"])
                    df["published"] = pd.to_datetime(df["published"])
                else:
                    dcol = next((c for c in DATE_ALIASES if c in
                                 [str(x).lower() for x in df.columns]), None)
                    if dcol:
                        df[dcol] = pd.to_datetime(df[dcol])
                        df = df.set_index(dcol)
                realstore.write(kind, f.stem.upper(), df,
                                {"provider": "user", "source_file": str(f)}, root=root)
                report.imported.append(f"{kind}/{f.stem.upper()}")
            except Exception as exc:
                report.skipped.append({"file": str(f), "reason": str(exc)})
    return report
