"""Ideas, genealogy, roadmap and the heuristic research assistant.

The "assistant" is deliberately rule-based and read-only: it inspects
recorded metrics and artifacts (never raw holdout data, never frozen code)
and emits suggestions. It cannot modify studies, registries, or freezes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import load_yaml, results_dir
from ..experiments import ExperimentRegistry
from ..holdout import holdout_status
from ..utils import get_logger

log = get_logger("platform.research")

VALID_IDEA_STATUSES = {"idea", "planned", "implementing", "testing",
                       "auditing", "accepted", "rejected", "archived"}


# ------------------------------------------------------------------- ideas --
def load_ideas() -> list[dict]:
    ideas = load_yaml("research_ideas.yaml")["ideas"]
    for i in ideas:
        if i.get("status") not in VALID_IDEA_STATUSES:
            raise ValueError(f"{i.get('id')}: invalid status '{i.get('status')}'")
    return ideas


# --------------------------------------------------------------- genealogy --
def load_genealogy() -> dict:
    return load_yaml("strategy_genealogy.yaml")["lines"]


def genealogy_for(class_name: str) -> list[dict]:
    """All version lines touching a strategy class."""
    out = []
    for line_id, line in load_genealogy().items():
        if any(v.get("strategy") == class_name for v in line["versions"]):
            out.append({"line": line_id, "title": line["title"],
                        "versions": line["versions"]})
    return out


# ----------------------------------------------------------------- roadmap --

def _result_driven_items(mode: str) -> list[dict]:
    """Priorities read off the study's OWN results.

    A backlog of ideas written before any result is a plan, not a response.
    These entries come from what the completed study actually shows, so the
    roadmap says what would change the answer rather than what was interesting
    to think about beforehand.
    """
    import pandas as pd
    out: list[dict] = []
    rank_path = results_dir(mode) / "strategy_ranking.csv"
    if not rank_path.exists():
        return out
    try:
        r = pd.read_csv(rank_path)
    except Exception:
        return out
    if r.empty or "verdict" not in r.columns:
        return out

    vc = r["verdict"].value_counts().to_dict()
    robust = int(vc.get("robust_candidate", 0))
    inconclusive = int(vc.get("inconclusive", 0))
    active = r[r["verdict"] != "benchmark"]

    if robust == 0 and len(active):
        best = active.sort_values("score", ascending=False).iloc[0]
        bench = r[r["verdict"] == "benchmark"]["score"].max()
        gap = float(bench) + 0.25 - float(best["score"])
        if gap > 0:
            out.append({
                "id": "RES-1",
                "title": "Add evidence, not strategies — the best result is inside the noise",
                "category": "evidence", "status": "planned",
                "priority_score": 5.0,
                "rationale": (
                    f"Nothing cleared the bar. The best strategy scored "
                    f"{float(best['score']):.2f} against a benchmark hurdle of "
                    f"{float(bench):.2f} plus a 0.25 margin — short by {gap:.2f}. "
                    "A gap that small cannot be resolved by trying more "
                    "strategies: each additional trial raises the deflated-Sharpe "
                    "correction and makes the bar higher for everything. It is "
                    "resolved by more independent data — other markets, longer "
                    "history, or the missing delisted names."),
                "dependencies": ["additional market history"]})

    if inconclusive >= 10:
        out.append({
            "id": "RES-2",
            "title": f"Retire or defend the {inconclusive} inconclusive variants",
            "category": "governance", "status": "planned",
            "priority_score": 3.5,
            "rationale": (
                f"{inconclusive} variants are neither accepted nor rejected. Each "
                "one still counts as a trial in the multiple-testing correction, "
                "so carrying them costs statistical power for every other result. "
                "Either state the economic case for keeping each, or deprecate "
                "them with a reason on the record."),
            "dependencies": []})

    # Long-only concentration: the structural finding of the v3 study.
    if "family" in r.columns:
        fams = set(r["family"].dropna().unique())
        if "longshort" not in fams:
            out.append({
                "id": "RES-3",
                "title": "Test whether anything survives once the market is hedged out",
                "category": "strategy", "status": "planned",
                "priority_score": 4.0,
                "rationale": (
                    "Every variant tested so far is long-only, which mechanically "
                    "forces a high correlation to the index — that was never a "
                    "finding about technology stocks, it was a property of the "
                    "study's own constraint. The engine has supported shorts and "
                    "charged borrow since v1 and nothing had used them."),
                "dependencies": []})
        else:
            out.append({
                "id": "RES-3",
                "title": "Price the short side properly before believing the hedged results",
                "category": "data", "status": "planned",
                "priority_score": 4.0,
                "rationale": (
                    "The hedged family is the only thing here that decorrelates "
                    "from the index, so its costs are the load-bearing assumption. "
                    "Borrow is currently a flat rate (50bp base, 150bp stressed). "
                    "Real borrow on hard-to-locate shares runs 5-50% and some "
                    "names cannot be shorted at all — and the universe contains "
                    "39 companies that died, exactly the names a lender would have "
                    "bought you in on. Until borrow is priced per name and per "
                    "date, every long-short result is an upper bound."),
                "dependencies": ["per-name borrow rates (IBKR shortable-shares file)"]})

    try:
        hs = holdout_status(mode)
        if len(hs.get("access_log", [])) >= 1:
            out.append({
                "id": "RES-4",
                "title": "The holdout is spent — the next honest test is forward, not backward",
                "category": "governance", "status": "planned",
                "priority_score": 4.5,
                "rationale": (
                    "The locked-away slice of history has been opened. Every "
                    "further backtest on it is development evidence, however it "
                    "is labelled. Genuinely out-of-sample evidence now has to "
                    "come from data nobody has seen: paper-trade forward under "
                    "docs/prospective_testing_protocol.md."),
                "dependencies": []})
    except Exception:
        pass
    return out


def build_roadmap(mode: str = "synthetic") -> list[dict]:
    """Ranked next research priorities: expected value / difficulty, with
    dependency and evidence context pulled from the backlog, the quality
    gate (when a real one exists), and open audit findings."""
    items = []
    for idea in load_ideas():
        if idea["status"] in ("accepted", "rejected", "archived"):
            continue
        ev = float(idea.get("expected_edge", 1))
        diff = float(idea.get("difficulty", 3))
        # data-blocking ideas get a boost: they gate entire families/tiers
        blocker = 1.5 if idea.get("category") == "data" else 1.0
        items.append({
            "id": idea["id"], "title": idea["title"],
            "category": idea.get("category", ""),
            "status": idea["status"],
            "priority_score": round(blocker * ev / max(diff, 0.5), 2),
            "rationale": idea.get("notes", "") or idea.get("hypothesis", "")[:140],
            "dependencies": idea.get("required_providers", []),
        })

    # Gate limitations (real mode) become roadmap entries automatically.
    gate_path = results_dir("real") / "data_quality.json"
    if gate_path.exists():
        import json
        gate = json.loads(gate_path.read_text())
        for i, lim in enumerate(gate.get("limitations", [])):
            # Title is the first clause; the full text is the rationale. Cutting
            # mid-word produced entries nobody could act on.
            head = lim.split(" — ")[0].split(";")[0].strip()
            items.append({"id": f"GATE-{i+1}", "title": f"Resolve: {head}",
                          "category": "data", "status": "planned",
                          "priority_score": 3.0, "rationale": lim,
                          "dependencies": []})

    items.extend(_result_driven_items(mode))

    # Open audit findings stay on the roadmap until closed.
    import json
    from pathlib import Path
    findings = Path("audit/findings/findings.jsonl")
    if findings.exists():
        for line in findings.read_text().splitlines():
            f = json.loads(line)
            if f.get("status", "").startswith("open"):
                items.append({"id": f["id"], "title": f"Audit follow-up: {f['title']}",
                              "category": "governance", "status": "planned",
                              "priority_score": 2.0, "rationale": f["description"][:140],
                              "dependencies": []})
    return sorted(items, key=lambda x: -x["priority_score"])


# --------------------------------------------------- heuristic assistant ----
@dataclass
class Suggestion:
    kind: str
    target: str
    detail: str


def assistant_review(mode: str = "synthetic",
                     corr_threshold: float = 0.95) -> list[Suggestion]:
    """Rule-based review of the recorded evidence. Read-only by construction:
    consumes registry records and derived CSVs only."""
    registry = ExperimentRegistry.for_mode(mode)
    df = registry.load()
    out: list[Suggestion] = []
    if df.empty:
        return out
    ok = df[(df["status"] == "ok") & (df["scenario"] == "base")].copy()

    # 1. Weak evidence: decent dev Sharpe but low PSR or heavy degradation.
    for r in ok.to_dict("records"):
        dev = (r.get("metrics_dev") or {})
        hold = (r.get("metrics_holdout") or {})
        ds, hs = dev.get("sharpe"), hold.get("sharpe")
        psr = r.get("psr_dev")
        if ds and ds > 0.5 and psr is not None and psr < 0.90:
            out.append(Suggestion("weak_evidence", r["strategy"],
                                  f"dev Sharpe {ds:.2f} but PSR only {psr:.2f} — "
                                  "treat as noise until more data"))
        if ds and hs is not None and ds - hs > 1.0:
            out.append(Suggestion("degradation", r["strategy"],
                                  f"dev→holdout Sharpe fell {ds - hs:.2f} — "
                                  "possible overfit to the development window"))

    # 2. Duplicated strategies: near-identical monthly return streams.
    #
    # This was a pairwise loop doing a concat + corr per pair. At 999 records
    # in the current cohort that is ~500,000 Python-level pandas calls and it
    # took 106 of the site build's 118 seconds — and it is QUADRATIC, so each
    # study made it worse. A build slow enough to skip is a build that gets
    # skipped, which is how a stale site gets published.
    #
    # One aligned frame and one .corr() computes the same numbers in C. The
    # min_periods threshold reproduces the old `len(joined) > 24` guard: pairs
    # overlapping by 24 months or fewer come back NaN and are dropped.
    curves = {}
    for r in ok.to_dict("records"):
        c = registry.load_curve(r["id"])
        if c is not None:
            curves[r["strategy"]] = (1 + c["returns"]).resample("ME").prod() - 1
    if len(curves) > 1:
        monthly = pd.DataFrame(curves).sort_index()
        corr = monthly.corr(min_periods=25).to_numpy()
        names = list(monthly.columns)
        classes = np.array([n.split("(")[0] for n in names])
        iu, ju = np.triu_indices(len(names), k=1)
        # Same class with nearby parameters is expected to correlate and is
        # not evidence of duplication, so those pairs never enter the result.
        keep = (classes[iu] != classes[ju]) & np.isfinite(corr[iu, ju]) \
            & (corr[iu, ju] > corr_threshold)
        for i, j in zip(iu[keep], ju[keep]):
            out.append(Suggestion("duplicate", f"{names[i]} ~ {names[j]}",
                                  f"monthly-return correlation "
                                  f"{corr[i, j]:.3f} — consider consolidating; "
                                  "they are not independent evidence"))

    # 3. Missing robustness artifacts per family.
    rob = registry.root / "robustness"
    fams = set(ok["family"]) - {"benchmark"}
    for fam in sorted(fams):
        if not (rob / f"sensitivity_{fam}.csv").exists():
            out.append(Suggestion("missing_test", fam,
                                  "no parameter-sensitivity table — run "
                                  "scripts/run_robustness.py"))

    # 4. Parameter instability from sensitivity tables.
    for fam in sorted(fams):
        p = rob / f"sensitivity_{fam}.csv"
        if p.exists():
            s = pd.read_csv(p)
            if "neighbor_stability" in s.columns and len(s) > 3:
                top = s.iloc[0]
                if (top.get("neighbor_stability") or 0) > 0.5 * abs(top.get("stat") or 1):
                    out.append(Suggestion(
                        "parameter_instability", fam,
                        f"best variant's neighbor stability "
                        f"{top['neighbor_stability']:.2f} vs stat {top['stat']:.2f} — "
                        "the peak may be isolated; prefer the smooth region"))
    return out
