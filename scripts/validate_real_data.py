#!/usr/bin/env python3
"""Validate the canonical real store and write the data-quality gate.

    python scripts/validate_real_data.py

Exit codes: 0 = PASS (possibly with limitations), 1 = FAIL.
The gate file (results/real/data_quality.json) is required by every real-mode
run script and is invalidated automatically whenever the store changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aitb.data.quality import FAIL, run_gate, write_gate
from aitb.utils import get_logger

log = get_logger("validate_real_data")


def main() -> int:
    res = run_gate()
    path = write_gate(res)

    print(f"\n{'=' * 70}\nDATA QUALITY VERDICT: {res.status}\n{'=' * 70}")
    failed = [c for c in res.checks if not c["ok"]]
    for c in failed:
        print(f"  {'FATAL ' if c['fatal'] else ''}FAILED {c['check']}: {c['detail']}")
    for lim in res.limitations:
        print(f"  LIMITATION: {lim}")
    print(f"\nGate written to {path}")
    if res.status == FAIL:
        print("Backtesting in real mode is BLOCKED until these issues are fixed.")
        return 1
    print("Real-mode backtesting is permitted."
          + (" Note the limitations above." if res.limitations else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
