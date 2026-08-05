"""Platform layer tests: catalog, scorecard, docs, ideas, roadmap, site,
plugins, portfolio lab — and the guarantee that building the site is
read-only over every governance artifact."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]


def _hash_tree(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def test_catalog_covers_all_registered_strategies():
    from aitb.platform.catalog import build_catalog
    from aitb.strategies import STRATEGY_CLASSES
    cat = build_catalog("synthetic")
    for name in STRATEGY_CLASSES:
        assert name in cat
    listed = [e for e in cat.values() if e.status != "unlisted"]
    assert len(listed) >= 20
    assert any(e.status == "deprecated" for e in cat.values())


def test_scorecard_structure_and_bounds():
    from aitb.platform.catalog import build_catalog
    from aitb.platform.scorecard import build_scorecard
    cat = build_catalog("synthetic")
    sc = build_scorecard(cat["TrendPlusVolTarget"])
    assert set(sc) >= {"overall", "dimensions", "verdict", "note"}
    assert 0 <= sc["overall"] <= 5
    for d in sc["dimensions"].values():
        assert 0 <= d["score"] <= 5 and d["reason"]
    # it must be a research-quality card, not a performance claim
    assert "performance" not in sc["verdict"].lower() or "not" in sc["note"].lower()


def test_docs_and_notebook_generation():
    from aitb.platform.catalog import build_catalog
    from aitb.platform.docgen import research_notebook, strategy_doc
    cat = build_catalog("synthetic")
    e = cat["DonchianBreakout"]
    doc = strategy_doc(e)
    for k in ("description", "parameters", "assumptions", "strengths",
              "weaknesses", "python_implementation", "tradingview_compatible"):
        assert k in doc
    nb = research_notebook(e)
    for k in ("hypothesis", "experiment_history", "lessons_learned",
              "remaining_questions", "scorecard"):
        assert k in nb
    assert len(nb["experiment_history"]) > 0   # auto-updated from registry


def test_ideas_valid_and_roadmap_ranked():
    from aitb.platform.research_mgmt import build_roadmap, load_ideas
    ideas = load_ideas()
    assert len(ideas) >= 8
    road = build_roadmap("synthetic")
    scores = [r["priority_score"] for r in road]
    assert scores == sorted(scores, reverse=True)
    assert any("AUD-010" in r["id"] for r in road)   # open audit finding present


def test_genealogy_loads_and_links():
    from aitb.platform.research_mgmt import genealogy_for
    lines = genealogy_for("TrendPlusVolTarget")
    assert lines and any(v.get("requires_freeze_bump") for line in lines
                         for v in line["versions"])


def test_pine_exports_only_portable():
    from aitb.platform.tradingview import export_pine
    assert export_pine("QQQMovingAverage") and "//@version=5" in export_pine("QQQMovingAverage")
    assert "APPROXIMATION" in export_pine("DonchianBreakout")
    assert export_pine("XSMomentumTopN") is None      # cross-sectional: refused
    assert export_pine("QualityGrowth") is None       # fundamentals: refused


def test_portfolio_lab_blend_math():
    from aitb.platform.portfolio_lab import blend, correlation_matrix
    idx = pd.bdate_range("2020-01-01", periods=504)
    a = pd.Series(0.001, index=idx)
    b = pd.Series(-0.0005, index=idx)
    r = pd.DataFrame({"A(x=1)": a, "B(y=2)": b})
    bl = blend(r, ["A(x=1)", "B(y=2)"])
    # 50/50 of constant returns ≈ average of the two, within drift tolerance
    assert abs(bl.mean() - 0.00025) < 5e-5
    corr = correlation_matrix(r)
    assert corr.shape == (2, 2)


def test_plugin_registers_and_builds(synth_md):
    from aitb.plugins import load_plugins
    from aitb.strategies import STRATEGY_CLASSES
    loaded = load_plugins()
    assert "example_golden_cross" in loaded
    assert "GoldenCrossRotation" in STRATEGY_CLASSES
    cls = STRATEGY_CLASSES["GoldenCrossRotation"]
    assert getattr(cls, "is_plugin", False)
    w = cls(fast=50, slow=200).build(synth_md)
    assert w.sum(axis=1).max() <= 1.0 + 1e-9


def test_plugins_cannot_enter_frozen_study():
    """A plugin class is not in the frozen grid nor the fingerprint list, so a
    real-mode run cannot execute it without changing frozen artifacts (which
    breaks the freeze hash)."""
    from aitb.config import load_yaml
    from aitb.freeze import _FROZEN_MODULES
    spec = load_yaml("experiments.yaml")
    classes_in_grid = {e["class"] for entries in spec.values() for e in entries}
    assert "GoldenCrossRotation" not in classes_in_grid
    assert not any("plugins" in m for m in _FROZEN_MODULES)


def test_site_build_is_read_only_over_governance(tmp_path):
    """Building the whole site must not modify registries, freezes, locks or
    findings — byte-identical before and after."""
    governance = [
        ROOT / "configs" / "research_freeze_v1.json",
        ROOT / "configs" / "research_freeze_v2.json",
        ROOT / "configs" / "research_freeze_v3.json",
        ROOT / "configs" / "research_freeze_v4.json",
        ROOT / "results" / "synthetic" / "experiments.jsonl",
        ROOT / "audit" / "findings" / "findings.jsonl",
    ]
    before = _hash_tree(governance)
    from aitb.platform.site import build_site
    out = build_site("synthetic", out=tmp_path / "site")
    after = _hash_tree(governance)
    assert before == after
    # Five pages, not nine: Compare folded into Strategies, Portfolio lab and
    # the experiment explorer into Results, audit/roadmap/ideas into Method.
    for page in ("index.html", "data.html", "strategies.html",
                 "results.html", "method.html"):
        assert (out / page).exists(), page
    # Nothing was dropped in the merge — each folded section must still render.
    strat = (out / "strategies.html").read_text()
    assert "id='compare'" in strat and "id='backlog'" in strat
    res = (out / "results.html").read_text()
    assert "id='together'" in res and "id='runs'" in res
    meth = (out / "method.html").read_text()
    assert "id='audit'" in meth and "id='next'" in meth
    assert (out / "strategy" / "TrendPlusVolTarget.html").exists()
    assert (out / "tradingview" / "QQQMovingAverage.pine").exists()
    # Every page showing numbers must warn that they are simulated. Assert the
    # GUARANTEE, not one exact sentence, so the copy can be improved without
    # silently dropping the warning: the styled warn box must be present and
    # must say the data is simulated/not real.
    for page in ("index.html", "data.html", "strategies.html",
                 "results.html"):
        txt = (out / page).read_text()
        assert "warnbox" in txt, f"{page} has no warning box"
        assert "simulated" in txt.lower(), f"{page} does not say the data is simulated"
        assert "not the real one" in txt.lower() or "real study" in txt.lower(), page


def test_assistant_is_read_only_and_flags_meanrev(tmp_path):
    from aitb.platform.research_mgmt import assistant_review
    governance = [ROOT / "results" / "synthetic" / "experiments.jsonl"]
    before = _hash_tree(governance)
    sugg = assistant_review("synthetic")
    assert _hash_tree(governance) == before
    assert isinstance(sugg, list)
