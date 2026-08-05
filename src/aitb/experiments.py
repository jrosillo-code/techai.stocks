"""Experiment registry and runner.

Every backtest — successful, mediocre, or failed — is recorded append-only in
``results/experiments.jsonl`` with its full spec, data version, split dates,
per-scenario metrics (development and holdout separately), and a stable ID.
Equity curves land in ``results/curves/<id>.parquet``. Re-running an identical
spec is skipped (resumable pipeline); nothing is ever deleted.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest.engine import BacktestResult, run_backtest
from .config import CostScenario, RESULTS_DIR, load_backtest_config
from .data.loader import MarketData
from .metrics import summary
from .strategies.base import Strategy
from .utils import get_logger, stable_hash
from .validation import (Split, concentration_diagnostics, contribution_shares,
                         leave_one_year_out, make_split, period_split_metrics,
                         probabilistic_sharpe, slice_dev, slice_holdout,
                         subperiod_table)

log = get_logger("experiments")


def _lineage(md: MarketData) -> dict:
    """Data-store fingerprint + freeze hash for real runs (empty-safe)."""
    out: dict = {}
    if md.data_mode == "real":
        try:
            from .data.quality import store_fingerprint
            out["store_fingerprint"] = store_fingerprint()
        except Exception:
            out["store_fingerprint"] = None
        try:
            from .freeze import load_freeze
            out["freeze_hash"] = load_freeze()["hash"]
        except Exception:
            out["freeze_hash"] = None
    return out


def _freeze_version() -> int:
    from .freeze import FREEZE_VERSION
    return FREEZE_VERSION


def universe_hash(md: MarketData) -> str:
    """Fingerprint of the exact investable roster a run saw.

    Ticker list AND listing windows: adding a delisted name, or correcting an
    IPO date, changes what a strategy could have held and therefore what its
    results mean. Two runs with different universe hashes are not comparable.
    """
    return stable_hash([[s.ticker, str(s.ipo), str(s.delisted)]
                        for s in md.universe.securities])


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return "unknown"


class ExperimentRegistry:
    """Append-only registry. Real and synthetic runs use DIFFERENT roots
    (results/real vs results/synthetic) and must never share one — construct
    via ``ExperimentRegistry.for_mode(mode)``."""

    def __init__(self, root: Path | None = None):
        self.root = root or (RESULTS_DIR / "synthetic")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "curves").mkdir(exist_ok=True)
        self.path = self.root / "experiments.jsonl"

    @classmethod
    def for_mode(cls, mode: str) -> "ExperimentRegistry":
        from .config import results_dir
        return cls(results_dir(mode))

    def existing_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        return {json.loads(line)["id"] for line in self.path.read_text().splitlines() if line.strip()}

    def append(self, record: dict) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        return pd.DataFrame(rows)

    def load_curve(self, exp_id: str) -> pd.DataFrame | None:
        p = self.root / "curves" / f"{exp_id}.parquet"
        return pd.read_parquet(p) if p.exists() else None


def experiment_id(strategy: Strategy, md: MarketData, scenario: str) -> str:
    """Stable ID for one (strategy, data, scenario) run.

    The universe fingerprint and freeze version are part of the identity
    (fixed 2026-08-05, finding AUD-016). They were not, and that was a real
    caching defect: the runner skips an experiment whose ID already exists and
    reloads its stored curve, so running the SAME strategy against an EXPANDED
    universe silently returned the old universe's results. It could not fire
    while a single freeze was in force — the universe is part of the freeze —
    but it fires the moment a new freeze widens the roster, which is exactly
    when the temptation to compare old and new numbers is highest.
    """
    from .freeze import FREEZE_VERSION
    return stable_hash({"spec": strategy.spec(), "provider": md.provider_name,
                        "data_mode": md.data_mode, "scenario": scenario,
                        "data_span": [str(md.calendar[0]), str(md.calendar[-1])],
                        "universe": universe_hash(md),
                        "freeze_version": FREEZE_VERSION})


def run_experiment(md: MarketData,
                   strategy: Strategy,
                   scenarios: dict[str, CostScenario],
                   registry: ExperimentRegistry,
                   split: Split | None = None,
                   notes: str = "",
                   force: bool = False) -> dict[str, BacktestResult]:
    """Run one strategy under every cost scenario; register every outcome."""
    bt_cfg = load_backtest_config()
    if split is None:
        split = make_split(md.calendar, bt_cfg.holdout_months)

    results: dict[str, BacktestResult] = {}
    existing = registry.existing_ids()
    weights = None

    bench = md.adj_close["QQQ"].pct_change() if "QQQ" in md.adj_close.columns else None
    rf = (md.macro["FEDFUNDS"].reindex(md.calendar).ffill() / 100 / 252
          if "FEDFUNDS" in md.macro.columns else 0.0)

    for scen_name, scen in scenarios.items():
        exp_id = experiment_id(strategy, md, scen_name)
        if exp_id in existing and not force:
            curve = registry.load_curve(exp_id)
            if curve is not None:
                r = curve["returns"]
                results[scen_name] = BacktestResult(
                    name=strategy.name, equity=curve["equity"], returns=r,
                    weights=pd.DataFrame(), turnover=curve.get("turnover", pd.Series(dtype=float)),
                    costs_paid=curve.get("costs", pd.Series(dtype=float)),
                    n_trades=-1, cost_scenario=scen_name, meta={"cached": True})
                continue

        try:
            if weights is None:
                weights = strategy.build(md)   # weights are cost-independent
            res = run_backtest(md, weights, scen, name=strategy.name,
                               initial_capital=bt_cfg.initial_capital)
        except Exception as exc:
            log.warning("experiment failed: %s [%s]: %s", strategy.name, scen_name, exc)
            registry.append({
                "id": exp_id, "status": "failed", "error": str(exc),
                "spec": strategy.spec(), "scenario": scen_name,
                "provider": md.provider_name, "data_mode": md.data_mode,
                "git": _git_commit(),
                "timestamp": datetime.now(timezone.utc).isoformat(), "notes": notes,
            })
            continue
        results[scen_name] = res

        r_dev = slice_dev(res.returns, split)
        r_hold = slice_holdout(res.returns, split)
        bench_dev = slice_dev(bench, split) if bench is not None else None
        bench_hold = slice_holdout(bench, split) if bench is not None else None

        record = {
            "id": exp_id,
            "status": "ok",
            "spec": strategy.spec(),
            "strategy": strategy.name,
            "family": strategy.family,
            "hypothesis": strategy.hypothesis,
            "scenario": scen_name,
            "provider": md.provider_name,
            "data_mode": md.data_mode,
            "universe_hash": universe_hash(md),
            "universe_size": len(md.universe.securities),
            "freeze_version": _freeze_version(),
            # Lineage (audit finding AUD-007): bind the record to the exact
            # validated data-store contents and governing freeze, so any
            # upstream change invalidates downstream artifacts detectably.
            "lineage": _lineage(md),
            "split": {"dev_end": str(split.dev_end.date()),
                      "holdout_start": str(split.holdout_start.date()),
                      "holdout_end": str(split.holdout_end.date())},
            "metrics_dev": summary(r_dev, bench_dev, rf if isinstance(rf, float) else rf.loc[r_dev.index]),
            "metrics_holdout": summary(r_hold, bench_hold, rf if isinstance(rf, float) else rf.reindex(r_hold.index).fillna(0.0)),
            "psr_dev": probabilistic_sharpe(r_dev),
            "annual_turnover": res.annual_turnover,
            "n_trades": res.n_trades,
            "total_costs": res.total_costs,
            "concentration": concentration_diagnostics(
                res.weights, md.adj_close.pct_change()) if not res.weights.empty else {},
            "contributions": contribution_shares(
                res.weights, md.adj_close.pct_change()) if not res.weights.empty else {},
            "period_split_2023": period_split_metrics(res.returns, "2023-01-01", bench),
            "leave_one_year_out": leave_one_year_out(res.returns),
            "subperiods": subperiod_table(res.returns, bt_cfg.subperiods).to_dict("records"),
            "git": _git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
        registry.append(record)
        curve = pd.DataFrame({"equity": res.equity, "returns": res.returns,
                              "turnover": res.turnover, "costs": res.costs_paid})
        curve.to_parquet(registry.root / "curves" / f"{exp_id}.parquet")
        log.info("ran %s [%s]: dev Sharpe %.2f, holdout Sharpe %.2f",
                 strategy.name, scen_name,
                 record["metrics_dev"].get("sharpe", float("nan")),
                 record["metrics_holdout"].get("sharpe", float("nan")))
    return results
