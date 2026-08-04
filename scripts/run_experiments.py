#!/usr/bin/env python3
"""Phase 2-3: run the full experiment grid from configs/experiments.yaml.

Every (strategy variant × cost scenario) is backtested and appended to the
registry with development/holdout metrics. Already-run experiment IDs are
skipped, so the pipeline is resumable and incremental.

Usage:
    python scripts/run_experiments.py [--provider synthetic] [--families xsmom,tsmom]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aitb.config import load_cost_scenarios, load_yaml
from aitb.data.loader import load_market_data
from aitb.experiments import ExperimentRegistry, run_experiment
from aitb.strategies import STRATEGY_CLASSES as CLASSES
from aitb.utils import get_logger

log = get_logger("run_experiments")


def expand_grid(entry: dict):
    cls = CLASSES[entry["class"]]
    grid = entry.get("grid", {})
    keys = sorted(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = {k: v for k, v in zip(keys, combo) if v is not None}
        yield cls(**params)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="synthetic",
                    help="synthetic-mode provider (ignored in real mode)")
    ap.add_argument("--data-mode", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--families", default="",
                    help="comma-separated subset of families to run")
    ap.add_argument("--scenarios", default="zero,base,stressed")
    args = ap.parse_args()

    if args.data_mode == "real":
        from aitb.data.quality import require_gate
        from aitb.freeze import registry_freeze_record, verify_freeze
        gate = require_gate()          # hard stop without a passing gate
        log.info("real-data gate: %s", gate["status"])
        # The first real run may only execute the FROZEN specification: any
        # drift in configs or strategy/engine/ranking code aborts here.
        freeze_doc = verify_freeze()
        md = load_market_data(mode="real")
    else:
        md = load_market_data(args.provider, mode="synthetic")

    scens = load_cost_scenarios()
    scens = {k: v for k, v in scens.items() if k in args.scenarios.split(",")}
    registry = ExperimentRegistry.for_mode(args.data_mode)
    spec = load_yaml("experiments.yaml")

    if args.data_mode == "real":
        # Record which freeze governs these runs, freeze the holdout
        # selection, and log the single sanctioned holdout access. Dev and
        # holdout metrics are computed in one pass — legitimate because the
        # specification was hashed before any real result existed.
        from aitb.holdout import freeze_selection, record_holdout_access, holdout_status
        from aitb.config import load_backtest_config
        existing_ids = ({r.get("id") for r in registry.load().to_dict("records")}
                        if registry.path.exists() else set())
        frecord = registry_freeze_record(freeze_doc, "real")
        if frecord["id"] not in existing_ids:
            registry.append(frecord)
        hs = holdout_status("real")
        if not hs["access_log"]:
            all_specs = [{"family": fam, "entries": spec.get(fam, [])}
                         for fam in sorted(spec)]
            freeze_selection(all_specs, "real",
                             holdout_start=str(load_backtest_config().holdout_months))
            record_holdout_access(
                "real", "first frozen real study — single-pass dev+holdout "
                        f"evaluation under freeze {freeze_doc['hash'][:12]}")
        elif hs.get("compromised"):
            log.warning("HOLDOUT IS MARKED COMPROMISED — reports will say so")

    # In real mode, skip families whose required data did not pass coverage.
    skip_families: set[str] = set()
    if args.data_mode == "real":
        if md.fundamentals.empty:
            skip_families.add("fundamental")
            log.warning("skipping 'fundamental' family: no validated real fundamentals")
        if "VIXCLS" not in md.macro.columns:
            skip_families.add("regime")
            log.warning("skipping 'regime' family: no VIX series in real store")

    families = args.families.split(",") if args.families else list(spec)
    n = 0
    deprecated_logged = {r.get("id") for r in
                         (registry.load().to_dict("records") if registry.path.exists() else [])}
    for family in families:
        if family in skip_families:
            continue
        for entry in spec.get(family, []):
            if entry.get("status") == "deprecated":
                # Deprecated variants are never run but stay visible in the
                # registry with their reason (research-integrity requirement).
                from datetime import datetime, timezone
                from aitb.utils import stable_hash
                dep_id = "dep_" + stable_hash({"class": entry["class"],
                                               "grid": entry.get("grid", {}),
                                               "mode": args.data_mode})
                if dep_id not in deprecated_logged:
                    registry.append({
                        "id": dep_id, "status": "deprecated",
                        "strategy": entry["class"], "family": family,
                        "spec": {"class": entry["class"], "family": family,
                                 "params": entry.get("grid", {})},
                        "reason": entry.get("reason", "unspecified"),
                        "data_mode": args.data_mode, "scenario": "n/a",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                continue
            for strat in expand_grid(entry):
                run_experiment(md, strat, scens, registry,
                               notes=(f"family_group={family};mode={args.data_mode};"
                                      f"status={entry.get('status', 'exploratory')};"
                                      f"hypothesis={entry.get('hypothesis', '')}"))
                n += 1
    log.info("completed %d strategy variants × %d scenarios [%s mode]",
             n, len(scens), args.data_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
