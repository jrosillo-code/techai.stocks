#!/usr/bin/env python3
"""Rebuild the published site, and REFUSE to look like it worked when it didn't.

    python scripts/publish_site.py            # real mode (the published site)
    python scripts/publish_site.py --mode synthetic --out /tmp/check

Why this exists rather than a one-line `build_site('real')`:

A rebuild on 2026-08-05 reported success and changed 45 files by exactly one
line each — every one a timestamp. No new pages, no new chart exports, nothing
from four freezes of work. The build had run an OLDER copy of the `aitb`
package than the checkout it was launched from, so it faithfully rebuilt the
old site. Nothing errored. `git add site && git commit && git push` then
published that non-change, and the only symptom was a diff nobody had reason to
read closely.

The failure is easy to hit because `python -c "sys.path.insert(0,'src'); ..."`
only wins if the current directory is the checkout AND no editable install of
an older copy shadows it. This project has already been bitten once by a
vendored snapshot on sys.path at freeze v2.

So this script states, before doing anything, WHERE the code came from, then
verifies afterwards that the output actually contains what the current code
should produce — and exits non-zero if it does not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
# Absolute, and ahead of everything: a relative 'src' depends on the working
# directory, which is exactly how the wrong package got imported before.
sys.path.insert(0, str(HERE / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="real", choices=["real", "synthetic"])
    ap.add_argument("--out", default=None,
                    help="build elsewhere (leaves the published site alone)")
    ap.add_argument("--allow-downgrade", action="store_true",
                    help="permit overwriting a real site with a synthetic one")
    args = ap.parse_args()

    import aitb
    from aitb.freeze import FREEZE_VERSION
    from aitb.platform.site import build_site
    from aitb.platform.tradingview import _GENERATORS

    resolved = Path(aitb.__file__).resolve().parent
    expected = (HERE / "src" / "aitb").resolve()
    print(f"aitb package : {resolved}")
    print(f"this checkout: {expected}")
    if resolved != expected:
        print("\nREFUSING: the imported aitb is NOT this checkout's.\n"
              "Something on sys.path shadows it — usually `pip install -e` "
              "pointing at another copy, or a vendored snapshot.\n"
              "Fix with:  python -m pip uninstall aitb   (then re-run)\n"
              "Building anyway would rebuild the OLD site and report success.",
              file=sys.stderr)
        return 2
    print(f"freeze       : v{FREEZE_VERSION}")
    print(f"chart exports: {len(_GENERATORS)} generators\n")

    out = Path(args.out) if args.out else None
    try:
        built = build_site(args.mode, out=out,
                           allow_downgrade=args.allow_downgrade)
    except RuntimeError as exc:
        print(f"\nBUILD REFUSED: {exc}", file=sys.stderr)
        return 3

    # Verify the OUTPUT, not the return code. A build that silently rebuilt an
    # older site would return a path just as happily as a correct one.
    problems = []
    charts = built / "charts.html"
    if not charts.exists():
        problems.append("charts.html was not written at all")
    else:
        cards = charts.read_text().count("class='chart'")
        if cards < len(_GENERATORS):
            problems.append(f"charts.html has {cards} cards but the code has "
                            f"{len(_GENERATORS)} chart exports")
    pine = list((built / "tradingview").glob("*.pine"))
    if len(pine) < len(_GENERATORS):
        problems.append(f"{len(pine)} .pine files written, expected "
                        f"{len(_GENERATORS)}")

    if problems:
        print("\nBUILD PRODUCED THE WRONG OUTPUT:", file=sys.stderr)
        for p in problems:
            print(f"  * {p}", file=sys.stderr)
        print("\nDo NOT commit this. The site does not match the code.",
              file=sys.stderr)
        return 4

    print(f"site built at {built}")
    print(f"  {len(pine)} TradingView exports")
    print(f"  charts.html: {charts.read_text().count(chr(39) + 'chart' + chr(39))} cards")
    print("\nNow:  git add site && git commit -m 'Rebuild site' && git push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
