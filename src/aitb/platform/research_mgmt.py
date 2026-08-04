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
            items.append({"id": f"GATE-{i+1}", "title": f"Resolve: {lim[:90]}",
                          "category": "data", "status": "planned",
                          "priority_score": 3.0, "rationale": lim,
                          "dependencies": []})

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
    curves = {}
    for r in ok.to_dict("records"):
        c = registry.load_curve(r["id"])
        if c is not None:
            curves[r["strategy"]] = (1 + c["returns"]).resample("ME").prod() - 1
    names = sorted(curves)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a.split("(")[0] == b.split("(")[0]:
                continue      # same class, nearby params: correlation expected
            joined = pd.concat([curves[a], curves[b]], axis=1, join="inner").dropna()
            if len(joined) > 24:
                rho = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
                if rho > corr_threshold:
                    out.append(Suggestion("duplicate", f"{a} ~ {b}",
                                          f"monthly-return correlation {rho:.3f} — "
                                          "consider consolidating; they are not "
                                          "independent evidence"))

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
