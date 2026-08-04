#!/usr/bin/env python3
"""Capacity analysis: re-run top strategies at multiple portfolio sizes.

The √participation impact model and 5%-of-ADV participation cap make costs a
function of portfolio size, so re-running at $100k / $1M / $10M / $100M
reveals where an edge stops scaling.

    python scripts/run_capacity.py [--data-mode synthetic] [--top 5]

Output: results/<mode>/capacity.csv with per-size CAGR/Sharpe, cost drag vs
the zero-cost run, and a capacity verdict (largest size keeping >= 80% of the
$100k Sharpe).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aitb.backtest.engine import run_backtest
from aitb.config import load_cost_scenarios
from aitb.data.loader import load_market_data
from aitb.experiments import ExperimentRegistry
from aitb.metrics import cagr, sharpe
from aitb.strategies import from_spec
from aitb.utils import get_logger

log = get_logger("run_capacity")

SIZES = [1e5, 1e6, 1e7, 1e8]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-mode", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--provider", default="synthetic")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    if args.data_mode == "real":
        from aitb.data.quality import require_gate
        require_gate()
        from aitb.freeze import verify_freeze
        verify_freeze()   # audit AUD-002: derived analytics must run frozen code

        md = load_market_data(mode="real")
    else:
        md = load_market_data(args.provider, mode="synthetic")

    registry = ExperimentRegistry.for_mode(args.data_mode)
    df = registry.load()
    ok = df[(df["status"] == "ok") & (df["scenario"] == "base")].copy()
    ok = ok[ok["family"] != "benchmark"]
    ok["dev_sharpe"] = ok["metrics_dev"].map(lambda m: (m or {}).get("sharpe") or float("nan"))
    top = ok.sort_values("dev_sharpe", ascending=False).drop_duplicates("strategy").head(args.top)

    scens = load_cost_scenarios()
    rows = []
    for rec in top.to_dict("records"):
        strat = from_spec(rec["spec"])
        weights = strat.build(md)
        zero = run_backtest(md, weights, scens["zero"], initial_capital=1e6)
        zero_cagr = cagr(zero.returns)
        per_size = {}
        for size in SIZES:
            res = run_backtest(md, weights, scens["base"], initial_capital=size)
            per_size[size] = {"cagr": cagr(res.returns), "sharpe": sharpe(res.returns)}
        base_sharpe = per_size[SIZES[0]]["sharpe"]
        capacity = max((s for s in SIZES
                        if per_size[s]["sharpe"] >= 0.8 * base_sharpe), default=SIZES[0])
        row = {"strategy": rec["strategy"], "family": rec["family"],
               "zero_cost_cagr": round(zero_cagr, 4),
               "capacity_est_usd": capacity}
        for s in SIZES:
            label = f"{int(s / 1e6)}M" if s >= 1e6 else f"{int(s / 1e3)}k"
            row[f"cagr_{label}"] = round(per_size[s]["cagr"], 4)
            row[f"sharpe_{label}"] = round(per_size[s]["sharpe"], 3)
            row[f"cost_drag_{label}"] = round(zero_cagr - per_size[s]["cagr"], 4)
        rows.append(row)
        log.info("%s: capacity ~$%s", rec["strategy"][:50], f"{capacity:,.0f}")

    out = pd.DataFrame(rows)
    out.to_csv(registry.root / "capacity.csv", index=False)
    log.info("capacity table -> %s", registry.root / "capacity.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
