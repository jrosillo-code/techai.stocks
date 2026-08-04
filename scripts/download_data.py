#!/usr/bin/env python3
"""Phase 1: download (or generate), validate and cache all datasets.

Usage:
    python scripts/download_data.py [--provider synthetic|stooq|yahoo] [--refresh]

With a network-enabled environment, use --provider stooq (no key needed) or
yahoo. In this repository's execution environment all market-data hosts are
blocked, so the default is the deterministic synthetic provider.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aitb.data.loader import load_market_data
from aitb.utils import get_logger

log = get_logger("download_data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="synthetic")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    md = load_market_data(args.provider, refresh=args.refresh)
    n_names = md.adj_close.shape[1]
    span = f"{md.calendar[0].date()} .. {md.calendar[-1].date()}"
    log.info("dataset ready: %d securities, %d days (%s)", n_names, len(md.calendar), span)
    warns = [i for i in md.issues if i.severity == "warn"]
    errs = [i for i in md.issues if i.severity == "error"]
    for i in errs:
        log.error("%s: %s — %s", i.ticker, i.kind, i.detail)
    log.info("validation: %d warnings, %d errors", len(warns), len(errs))
    for i in warns[:20]:
        log.info("  warn %s: %s — %s", i.ticker, i.kind, i.detail)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
