"""Real-data quality gate.

Runs a battery of checks over the canonical real store and emits a verdict:

    PASS FOR RESEARCH
    PASS WITH LIMITATIONS
    FAIL — DO NOT BACKTEST

The verdict plus the full check log is written to
results/real/data_quality.json; the run scripts REFUSE to backtest in real
mode unless that file exists with a passing status and a store fingerprint
matching the current data (re-validate after any import).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import load_universe_config, results_dir
from ..utils import get_logger, stable_hash
from . import realstore
from .validation import validate_prices

log = get_logger("data.quality")

PASS = "PASS FOR RESEARCH"
PASS_LIMITED = "PASS WITH LIMITATIONS"
FAIL = "FAIL — DO NOT BACKTEST"

GATE_FILENAME = "data_quality.json"


@dataclass
class GateResult:
    status: str
    checks: list[dict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    store_fingerprint: str = ""

    def add(self, name: str, ok: bool, detail: str, fatal: bool = False):
        self.checks.append({"check": name, "ok": ok, "detail": detail,
                            "fatal": fatal})


def store_fingerprint(root: Path | None = None) -> str:
    """Deterministic hash over dataset keys + row counts + spans."""
    cov = realstore.coverage_summary(root)
    return stable_hash(cov, 16)


def run_gate(root: Path | None = None,
             min_universe_names: int = 10,
             max_missing_session_pct: float = 0.03) -> GateResult:
    ucfg = load_universe_config()
    res = GateResult(status=PASS)

    have_prices = realstore.available("prices", root)
    univ_have = [t for t in ucfg.tickers if t in have_prices]
    bench_have = [t for t in ucfg.benchmark_tickers if t in have_prices]

    # --- fatal checks -------------------------------------------------------
    ok = len(univ_have) >= min_universe_names
    res.add("universe_coverage", ok,
            f"{len(univ_have)}/{len(ucfg.tickers)} universe names present "
            f"(min {min_universe_names})", fatal=True)

    for b in ("SPY", "QQQ"):
        res.add(f"benchmark_{b}", b in have_prices,
                f"{b} {'present' if b in have_prices else 'MISSING'}", fatal=True)

    # --- per-ticker validation ---------------------------------------------
    try:
        from ..calendar import trading_sessions
        have_calendar = True
    except Exception:
        have_calendar = False

    n_err_tickers = 0
    delisted = {s.ticker: s for s in ucfg.securities if s.delisted}
    for t in have_prices:
        df = realstore.read("prices", t, root)
        issues = validate_prices(t, df)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            n_err_tickers += 1
            res.add(f"prices_{t}", False,
                    "; ".join(i.detail for i in errors), fatal=False)
        if have_calendar and len(df) > 252:
            sessions = trading_sessions(df.index.min().date(), df.index.max().date())
            missing = sessions.difference(df.index)
            pct = len(missing) / max(len(sessions), 1)
            if pct > max_missing_session_pct:
                res.add(f"gaps_{t}", False,
                        f"{pct:.1%} of exchange sessions missing", fatal=False)
        sec = delisted.get(t)
        if sec is not None and df.index.max() > pd.Timestamp(sec.delisted) + pd.Timedelta(days=5):
            res.add(f"postdelist_{t}", False,
                    f"data continues past delisting {sec.delisted}", fatal=False)

    if n_err_tickers > len(have_prices) * 0.2:
        res.add("error_ticker_share", False,
                f"{n_err_tickers}/{len(have_prices)} tickers have fatal price errors",
                fatal=True)

    # --- dividend / total-return coverage ----------------------------------
    split_only = []
    for t in have_prices:
        meta = realstore.read_meta("prices", t, root) or {}
        if meta.get("adjustment") == "split_only":
            split_only.append(t)
    if split_only:
        res.limitations.append(
            f"{len(split_only)} tickers have split-adjusted-only prices (no "
            f"dividends): {split_only[:10]}{'...' if len(split_only) > 10 else ''} — "
            "total returns are understated for dividend payers")

    # --- macro --------------------------------------------------------------
    have_macro = realstore.available("macro", root)
    if "FEDFUNDS" not in have_macro:
        res.limitations.append("no FEDFUNDS series — cash sleeve earns 0%")
    if "VIXCLS" not in have_macro:
        res.limitations.append("no VIX series — VIX-based regime strategies must be skipped")
    res.add("macro_coverage", len(have_macro) > 0,
            f"{len(have_macro)} macro series: {have_macro}")
    res.limitations.append(
        "macro series are revised data, not real-time vintages — regime "
        "strategies using CPI/UNRATE carry look-ahead risk and are excluded "
        "from the first real run")

    # --- fundamentals -------------------------------------------------------
    have_fund = realstore.available("fundamentals", root)
    if len(have_fund) < min(10, len(univ_have)):
        res.limitations.append(
            f"fundamentals cover only {len(have_fund)} names — fundamental "
            "strategies restricted to that subset or skipped")
    for t in have_fund:
        f = realstore.read("fundamentals", t, root)
        if (pd.to_datetime(f["published"]) < pd.to_datetime(f["period_end"])).any():
            res.add(f"fund_leak_{t}", False,
                    "published date precedes period_end", fatal=True)

    # --- delisting coverage (survivorship honesty) --------------------------
    delisted_have = [t for t in delisted if t in have_prices]
    if len(delisted_have) < len(delisted):
        res.limitations.append(
            f"only {len(delisted_have)}/{len(delisted)} configured delisted names "
            "have real history — free providers rarely carry delisted series; "
            "the universe is PARTIALLY current-constituent biased")

    # --- verdict ------------------------------------------------------------
    if any(c["fatal"] and not c["ok"] for c in res.checks):
        res.status = FAIL
    elif res.limitations or any(not c["ok"] for c in res.checks):
        res.status = PASS_LIMITED
    else:
        res.status = PASS
    res.store_fingerprint = store_fingerprint(root)
    return res


def write_gate(res: GateResult, mode_results: Path | None = None) -> Path:
    out_dir = mode_results or results_dir("real")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / GATE_FILENAME
    path.write_text(json.dumps({
        "status": res.status,
        "store_fingerprint": res.store_fingerprint,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "checks": res.checks,
        "limitations": res.limitations,
    }, indent=2))
    return path


def require_gate(root: Path | None = None,
                 mode_results: Path | None = None) -> dict:
    """Called by run scripts in real mode. Raises unless a current, passing
    gate file exists for the present store contents."""
    path = (mode_results or results_dir("real")) / GATE_FILENAME
    if not path.exists():
        raise RuntimeError(
            "real mode blocked: no data-quality gate found. "
            "Run scripts/validate_real_data.py first.")
    gate = json.loads(path.read_text())
    if gate["status"] == FAIL:
        raise RuntimeError(f"real mode blocked: data quality gate is '{FAIL}'")
    current = store_fingerprint(root)
    if gate.get("store_fingerprint") != current:
        raise RuntimeError(
            "real mode blocked: the real store changed since validation "
            "(fingerprint mismatch). Re-run scripts/validate_real_data.py.")
    return gate
