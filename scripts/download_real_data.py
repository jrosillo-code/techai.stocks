#!/usr/bin/env python3
"""Standalone real-data downloader — run this in a NETWORK-ENABLED environment.

Produces a portable bundle importable by scripts/import_data_bundle.py without
internet access:

    python scripts/download_real_data.py \
        --providers yahoo stooq fred sec \
        --start 1998-01-01 \
        --output data/export_bundle

Bundle layout:
    manifest.json                     checksums, coverage, failures, metadata
    prices/<PROVIDER>/<TICKER>.parquet   raw per-provider series (kept apart
                                         so the importer can reconcile — they
                                         are NEVER averaged together)
    macro/<SERIES>.parquet
    fundamentals/<TICKER>.parquet     SEC EDGAR point-in-time quarterlies

Behavior:
  * resumable — a symbol whose file already covers [start, now-5d] is skipped;
    a stale file is extended incrementally, not re-downloaded;
  * per-provider rate limiting and retry-with-backoff (see providers.py);
  * every failure is recorded per (provider, symbol) in the manifest —
    failures are explicit, never silently substituted;
  * checksums (sha256) for every file for transport integrity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aitb.config import load_universe_config
from aitb.utils import get_logger

log = get_logger("download_real_data")

MACRO_SERIES = {
    "FEDFUNDS": "FEDFUNDS", "DGS10": "DGS10", "DGS2": "DGS2",
    "T10Y2Y": "T10Y2Y", "CPIYOY": "CPIAUCSL",  # importer converts CPI level -> YoY
    "UNRATE": "UNRATE", "VIXCLS": "VIXCLS", "BAA10Y": "BAA10Y",
}
RATE_LIMIT_S = {"yahoo": 1.0, "stooq": 1.0, "fred": 0.5, "tiingo": 0.8,
                "alphavantage": 13.0, "sec": 0.15}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_provider(name: str):
    if name in ("yahoo", "stooq", "fred"):
        from aitb.data.providers import get_provider
        return get_provider(name)
    if name in ("tiingo", "alphavantage"):
        from aitb.data.providers_ext import get_extended_provider
        return get_extended_provider(name)
    if name == "sec":
        from aitb.data.providers_ext import EdgarFundamentals
        return EdgarFundamentals()
    raise KeyError(name)


def needs_refresh(path: Path, start: date, staleness_days: int = 5) -> str:
    """'skip' | 'extend' | 'full'"""
    if not path.exists():
        return "full"
    try:
        df = pd.read_parquet(path)
        first, last = df.index.min().date(), df.index.max().date()
    except Exception:
        return "full"
    if first > start + pd.Timedelta(days=40).to_pytimedelta():
        return "full"
    if (date.today() - last).days > staleness_days:
        return "extend"
    return "skip"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+",
                    default=["yahoo", "stooq", "fred", "sec"])
    ap.add_argument("--start", default="1998-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--output", default="data/export_bundle")
    ap.add_argument("--tickers", nargs="*", help="override universe tickers")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out = Path(args.output)
    (out / "macro").mkdir(parents=True, exist_ok=True)
    (out / "fundamentals").mkdir(parents=True, exist_ok=True)

    ucfg = load_universe_config()
    tickers = args.tickers or (ucfg.tickers + ucfg.benchmark_tickers)

    manifest: dict = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start": str(start), "end": str(end),
        "providers": args.providers,
        "adjustment_note": ("yahoo adj_close is dividend+split adjusted; "
                            "stooq is split-adjusted ONLY (no dividends); "
                            "tiingo adjClose is total-return quality"),
        "files": {}, "failures": [], "coverage": {},
    }

    price_providers = [p for p in args.providers
                       if p in ("yahoo", "stooq", "tiingo", "alphavantage")]
    for pname in price_providers:
        try:
            provider = make_provider(pname)
        except Exception as exc:
            log.error("provider %s unavailable: %s", pname, exc)
            manifest["failures"].append({"provider": pname, "symbol": "*",
                                         "error": str(exc)})
            continue
        pdir = out / "prices" / pname
        pdir.mkdir(parents=True, exist_ok=True)
        n_ok = 0
        for t in tickers:
            path = pdir / f"{t}.parquet"
            action = needs_refresh(path, start)
            if action == "skip":
                n_ok += 1
                continue
            try:
                if action == "extend":
                    old = pd.read_parquet(path)
                    tail = provider.fetch_daily(t, old.index.max().date(), end)
                    df = pd.concat([old, tail[tail.index > old.index.max()]])
                else:
                    df = provider.fetch_daily(t, start, end)
                df.to_parquet(path)
                n_ok += 1
                log.info("%s/%s: %d rows (%s)", pname, t, len(df), action)
            except Exception as exc:
                log.warning("%s/%s FAILED: %s", pname, t, exc)
                manifest["failures"].append({"provider": pname, "symbol": t,
                                             "error": str(exc)})
            time.sleep(RATE_LIMIT_S.get(pname, 1.0))
        manifest["coverage"][pname] = {"requested": len(tickers), "ok": n_ok}

    if "fred" in args.providers:
        provider = make_provider("fred")
        n_ok = 0
        for sid, fred_id in MACRO_SERIES.items():
            try:
                s = provider.fetch_series(fred_id, start, end).rename(sid)
                s.to_frame().to_parquet(out / "macro" / f"{sid}.parquet")
                n_ok += 1
            except Exception as exc:
                manifest["failures"].append({"provider": "fred", "symbol": sid,
                                             "error": str(exc)})
            time.sleep(RATE_LIMIT_S["fred"])
        manifest["coverage"]["fred"] = {"requested": len(MACRO_SERIES), "ok": n_ok}

    if "sec" in args.providers:
        edgar = make_provider("sec")
        n_ok = 0
        univ_only = [t for t in tickers if t in ucfg.tickers]
        for t in univ_only:
            path = out / "fundamentals" / f"{t}.parquet"
            if path.exists():
                n_ok += 1
                continue
            try:
                df = edgar.fetch_fundamentals(t)
                df.to_parquet(path)
                n_ok += 1
                log.info("sec/%s: %d quarters", t, len(df))
            except Exception as exc:
                manifest["failures"].append({"provider": "sec", "symbol": t,
                                             "error": str(exc)})
            time.sleep(RATE_LIMIT_S["sec"])
        manifest["coverage"]["sec"] = {"requested": len(univ_only), "ok": n_ok}

    for p in sorted(out.rglob("*.parquet")):
        manifest["files"][str(p.relative_to(out))] = sha256(p)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_fail = len(manifest["failures"])
    log.info("bundle complete: %d files, %d failures -> %s",
             len(manifest["files"]), n_fail, out / "manifest.json")
    print(f"\nBundle ready at {out}. Import with:\n"
          f"  python scripts/import_data_bundle.py --input {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
