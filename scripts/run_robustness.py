#!/usr/bin/env python3
"""Phase 4: robustness analysis over the experiment registry.

Produces, per family (base-cost scenario):
  * walk-forward parameter selection -> genuine OOS composite series,
  * moving-block bootstrap CIs on Sharpe for the best variant,
  * deflated Sharpe ratio using the family's FULL trial battery,
  * Monte Carlo trade-sequence drawdown analysis,
  * parameter-sensitivity tables.

Outputs land in results/robustness/ as JSON/CSV for the report builder.

Usage: python scripts/run_robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aitb.config import RESULTS_DIR, load_backtest_config
from aitb.experiments import ExperimentRegistry
from aitb.metrics import sharpe
from aitb.utils import get_logger
from aitb.validation import (block_bootstrap_ci, deflated_sharpe,
                             parameter_sensitivity, trade_sequence_monte_carlo,
                             walk_forward_select)

log = get_logger("run_robustness")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-mode", default="synthetic", choices=["synthetic", "real"])
    args = ap.parse_args()
    if args.data_mode == "real":
        from aitb.freeze import verify_freeze
        verify_freeze()   # audit AUD-002: derived analytics must run frozen code
    registry = ExperimentRegistry.for_mode(args.data_mode)
    df = registry.load()
    if df.empty:
        log.error("no experiments found — run scripts/run_experiments.py first")
        return 1
    ok = df[(df["status"] == "ok") & (df["scenario"] == "base")].copy()
    ok["dev_sharpe"] = ok["metrics_dev"].map(lambda m: (m or {}).get("sharpe"))

    out_dir = registry.root / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    bt_cfg = load_backtest_config()
    dev_end = None

    summary_rows = []
    for family, grp in ok.groupby("family"):
        if family == "benchmark":
            continue
        curves = {}
        for rec in grp.to_dict("records"):
            c = registry.load_curve(rec["id"])
            if c is not None:
                curves[rec["strategy"]] = c["returns"]
                if dev_end is None:
                    dev_end = pd.Timestamp(rec["split"]["dev_end"])
        if len(curves) < 2:
            continue

        # Walk-forward selection runs on DEVELOPMENT data only.
        dev_curves = {k: v.loc[:dev_end] for k, v in curves.items()}
        wf_oos, wf_log = walk_forward_select(
            dev_curves, min_train_months=bt_cfg.min_train_months,
            step_months=bt_cfg.step_months)
        wf_sharpe = sharpe(wf_oos) if len(wf_oos) else float("nan")

        # Audit finding AUD-008: the deflated-Sharpe trial battery must count
        # EVERY variant attempted in the family — errored runs enter as
        # zero-Sharpe trials so the multiple-testing correction is not
        # understated by dropping failures.
        n_failed = int((df[(df["status"] == "failed")
                           & (df.get("family", pd.Series(dtype=object)) == family)]
                        ).shape[0]) if "family" in df.columns else 0
        trial_sharpes = ([sharpe(v.dropna()) for v in dev_curves.values()]
                         + [0.0] * n_failed)
        best_name = max(dev_curves, key=lambda k: sharpe(dev_curves[k].dropna()))
        best = dev_curves[best_name].dropna()
        point, lo, hi = block_bootstrap_ci(best, n_boot=bt_cfg.n_bootstrap,
                                           seed=bt_cfg.seed)
        dsr = deflated_sharpe(best, trial_sharpes)
        mc = trade_sequence_monte_carlo(best, seed=bt_cfg.seed)

        sens = parameter_sensitivity(dev_curves)
        sens.to_csv(out_dir / f"sensitivity_{family}.csv", index=False)
        wf_log.to_csv(out_dir / f"walkforward_{family}.csv", index=False)
        if len(wf_oos):
            wf_oos.to_frame("returns").to_parquet(out_dir / f"walkforward_{family}.parquet")

        row = {
            "family": family,
            "n_variants": len(curves),
            "best_variant": best_name,
            "best_dev_sharpe": round(point, 3) if point == point else None,
            "sharpe_ci90_lo": round(lo, 3) if lo == lo else None,
            "sharpe_ci90_hi": round(hi, 3) if hi == hi else None,
            "deflated_sharpe_prob": round(dsr, 3) if dsr == dsr else None,
            "walkforward_oos_sharpe": round(wf_sharpe, 3) if wf_sharpe == wf_sharpe else None,
            "mc_mdd_median": round(mc["mdd_median"], 3),
            "mc_mdd_p95": round(mc["mdd_p95"], 3),
        }
        summary_rows.append(row)
        log.info("family %-12s best=%s DSR=%.2f wf_oos_sharpe=%s",
                 family, best_name[:40], dsr, row["walkforward_oos_sharpe"])

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "family_summary.csv", index=False)
    (out_dir / "family_summary.json").write_text(summary.to_json(orient="records", indent=2))
    log.info("robustness outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
