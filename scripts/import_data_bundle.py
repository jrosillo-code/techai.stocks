#!/usr/bin/env python3
"""Import a portable data bundle (or loose files) into the canonical real store.

    python scripts/import_data_bundle.py --input data/export_bundle
    python scripts/import_data_bundle.py --loose      # data/import/** files

Writes an import report to results/real/import_report.json and reminds you to
run the validation gate before backtesting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aitb.config import results_dir
from aitb.data.import_bundle import import_bundle, import_loose_files
from aitb.utils import get_logger

log = get_logger("import_data_bundle")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="bundle directory (with manifest.json)")
    ap.add_argument("--loose", action="store_true",
                    help="import loose files from data/import/ instead")
    ap.add_argument("--no-strict-checksums", action="store_true")
    args = ap.parse_args()

    if not args.input and not args.loose:
        ap.error("provide --input BUNDLE_DIR or --loose")

    reports = []
    if args.input:
        reports.append(import_bundle(Path(args.input),
                                     strict_checksums=not args.no_strict_checksums))
    if args.loose:
        reports.append(import_loose_files())

    out_dir = results_dir("real")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = {
        "imported": sum((r.imported for r in reports), []),
        "skipped": sum((r.skipped for r in reports), []),
        "reconciliation": sum((r.reconciliation for r in reports), []),
        "trimmed": sum((r.trimmed for r in reports), []),
        "checksum_failures": sum((r.checksum_failures for r in reports), []),
    }
    (out_dir / "import_report.json").write_text(json.dumps(merged, indent=2, default=str))
    log.info("imported %d datasets (%d skipped) — report at %s",
             len(merged["imported"]), len(merged["skipped"]),
             out_dir / "import_report.json")
    warn = [r.get("warning") for r in merged["reconciliation"] if r.get("warning")]
    for w in warn:
        log.warning("RECONCILIATION: %s", w)
    print("\nNext step (required before any real backtest):\n"
          "  python scripts/validate_real_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
