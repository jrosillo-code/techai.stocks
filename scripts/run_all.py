#!/usr/bin/env python3
"""Reproduce every major result end-to-end.

    python scripts/run_all.py --data-mode synthetic     # offline demonstration
    python scripts/run_all.py --data-mode real          # requires validated real data

Phases: tests -> data availability -> experiments -> robustness -> company
analysis -> report. Each phase is an ordinary script that can be run alone;
the experiment registry makes re-runs incremental.

Real mode HARD-FAILS unless scripts/validate_real_data.py has produced a
passing quality gate for the current store contents. It never falls back to
synthetic data.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        print(f"FAILED: {' '.join(cmd)} (exit {res.returncode})", file=sys.stderr)
        raise SystemExit(res.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-mode", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--provider", default="synthetic",
                    help="synthetic-mode provider (ignored in real mode)")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    mode = ["--data-mode", args.data_mode]

    if args.data_mode == "synthetic":
        print("=" * 70)
        print("SYNTHETIC MODE — results demonstrate the machinery, not markets.")
        print("=" * 70)
    else:
        # Fail fast: gate must exist and pass before anything else runs.
        run([py, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "from aitb.data.quality import require_gate; "
             "g = require_gate(); print('gate:', g['status'])"])

    if not args.skip_tests:
        run([py, "-m", "pytest", "tests/", "-q"])
    if args.data_mode == "synthetic":
        run([py, "scripts/download_data.py", "--provider", args.provider])
    run([py, "scripts/run_experiments.py", "--provider", args.provider, *mode])
    run([py, "scripts/run_robustness.py", *mode])
    run([py, "scripts/run_company_analysis.py", "--provider", args.provider, *mode])
    run([py, "scripts/make_report.py", "--provider", args.provider, *mode])
    print(f"\nAll phases complete. Report: reports/{args.data_mode}/research_report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
