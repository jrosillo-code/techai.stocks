#!/usr/bin/env python3
"""The `research` CLI — one coherent entry point for the platform.

    python scripts/research.py dashboard  [--data-mode synthetic]  build the site
    python scripts/research.py update     [--data-mode ...]        incremental experiments + site
    python scripts/research.py validate                            real-data quality gate
    python scripts/research.py compare A B [...]                   side-by-side metric table
    python scripts/research.py roadmap                             print ranked priorities
    python scripts/research.py ideas                               list the idea backlog
    python scripts/research.py assistant                           heuristic evidence review
    python scripts/research.py export     [--fmt md|csv]           research summary export
    python scripts/research.py tradingview                         write Pine scripts

Every subcommand is read-only over frozen artifacts except `update`, which
delegates to the governed pipeline scripts (freeze/gate rules apply there).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

ROOT = Path(__file__).parents[1]


def _run(cmd: list[str]) -> int:
    return subprocess.run([sys.executable] + cmd, cwd=ROOT).returncode


def cmd_dashboard(args) -> int:
    from aitb.platform.site import build_site
    out = build_site(args.data_mode)
    print(f"Research site built: {out / 'index.html'}")
    return 0


def cmd_update(args) -> int:
    rc = _run(["scripts/run_experiments.py", "--data-mode", args.data_mode])
    if rc:
        return rc
    _run(["scripts/run_robustness.py", "--data-mode", args.data_mode])
    _run(["scripts/make_report.py", "--data-mode", args.data_mode])
    return cmd_dashboard(args)


def cmd_validate(args) -> int:
    return _run(["scripts/validate_real_data.py"])


def cmd_compare(args) -> int:
    import pandas as pd
    from aitb.experiments import ExperimentRegistry
    registry = ExperimentRegistry.for_mode(args.data_mode)
    df = registry.load()
    ok = df[(df["status"] == "ok") & (df["scenario"] == "base")]
    rows = []
    for pat in args.strategies:
        hits = ok[ok["strategy"].str.contains(pat, regex=False)]
        for r in hits.to_dict("records"):
            dev, hold = r.get("metrics_dev") or {}, r.get("metrics_holdout") or {}
            rows.append({"strategy": r["strategy"][:60], "family": r["family"],
                         "dev_sharpe": dev.get("sharpe"), "holdout_sharpe": hold.get("sharpe"),
                         "cagr": dev.get("cagr"), "max_dd": dev.get("max_drawdown"),
                         "calmar": dev.get("calmar"), "turnover": r.get("annual_turnover"),
                         "psr": r.get("psr_dev")})
    if not rows:
        print("no matches"); return 1
    out = pd.DataFrame(rows).drop_duplicates("strategy")
    print(out.to_string(index=False))
    return 0


def cmd_roadmap(args) -> int:
    from aitb.platform.research_mgmt import build_roadmap
    for i, r in enumerate(build_roadmap(args.data_mode), 1):
        print(f"{i:2d}. [{r['priority_score']:.1f}] {r['title']}  "
              f"({r['category']}, {r['status']})")
    return 0


def cmd_ideas(args) -> int:
    from aitb.platform.research_mgmt import load_ideas
    for i in load_ideas():
        print(f"{i['id']:9s} [{i['status']:>12s}] {i['title']}")
    return 0


def cmd_assistant(args) -> int:
    from aitb.platform.research_mgmt import assistant_review
    sugg = assistant_review(args.data_mode)
    if not sugg:
        print("no suggestions (or no experiments recorded)")
    for s in sugg:
        print(f"[{s.kind}] {s.target}\n    {s.detail}")
    return 0


def cmd_export(args) -> int:
    import pandas as pd
    from aitb.config import results_dir
    from aitb.platform.catalog import platform_stats
    stats = platform_stats(args.data_mode)
    rank = results_dir(args.data_mode) / "strategy_ranking.csv"
    out_dir = ROOT / "reports" / args.data_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fmt == "csv" and rank.exists():
        print(f"ranking CSV already at {rank}")
        return 0
    lines = ["# Research summary", "",
             f"Data mode: {args.data_mode} "
             + ("(SYNTHETIC — demonstration only)" if args.data_mode == "synthetic" else ""),
             "", "## Platform state", ""]
    lines += [f"- {k}: {v}" for k, v in stats.items() if not isinstance(v, dict)]
    if rank.exists():
        r = pd.read_csv(rank).head(15)
        lines += ["", "## Top of the ranking", "", r.to_markdown(index=False)]
    path = out_dir / "research_summary.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")
    return 0


def cmd_tradingview(args) -> int:
    from aitb.platform.tradingview import export_all
    written = export_all(ROOT / "site" / "tradingview")
    for w in written:
        print(w)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="research")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dashboard", "update", "validate", "roadmap", "ideas",
                 "assistant", "tradingview"):
        p = sub.add_parser(name)
        p.add_argument("--data-mode", default="synthetic",
                       choices=["synthetic", "real"])
    p = sub.add_parser("compare")
    p.add_argument("strategies", nargs="+")
    p.add_argument("--data-mode", default="synthetic", choices=["synthetic", "real"])
    p = sub.add_parser("export")
    p.add_argument("--fmt", default="md", choices=["md", "csv"])
    p.add_argument("--data-mode", default="synthetic", choices=["synthetic", "real"])
    args = ap.parse_args()
    return globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    raise SystemExit(main())
