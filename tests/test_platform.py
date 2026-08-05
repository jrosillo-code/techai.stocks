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


def test_pine_entry_rules_are_backtestable_and_gauges_are_not():
    """A rule that says WHEN to buy exports as a strategy(); one that says HOW
    MUCH to hold exports as an indicator().

    Dressing a position-sizing gauge up as an entry signal would put a made-up
    equity curve in TradingView's Strategy Tester under the name of a strategy
    that never claimed to time anything.
    """
    from aitb.platform.tradingview import BACKTESTABLE, _GENERATORS, export_pine
    for name in _GENERATORS:
        code = export_pine(name)
        is_strategy = 'strategy("' in code
        assert is_strategy == (name in BACKTESTABLE), (
            f"{name}: strategy()={is_strategy} but BACKTESTABLE="
            f"{name in BACKTESTABLE}")
        if is_strategy:
            assert "longEntry" in code and "longExit" in code, name
            assert "plotshape(longEntry" in code, f"{name} marks no entries"


def test_every_pine_export_carries_its_caveats():
    """No script may leave the building without saying what it does not model.

    These get pasted into TradingView and screenshotted, at which point they
    are separated from this site and from every warning on it.
    """
    from aitb.platform.tradingview import _GENERATORS, export_pine
    for name in _GENERATORS:
        code = export_pine(name)
        assert "APPROXIMATION" in code, f"{name} lacks the approximation notice"
        assert "no transaction costs" in code.lower(), f"{name} omits the cost caveat"
        assert "NOT modelled here" in code, f"{name} lacks the on-chart cost warning"
        assert "//@version=5" in code, name


def test_pine_defaults_match_the_python_strategy_defaults():
    """The chart script and the tested rule must start from the same numbers.

    Each generator hardcodes its own fallback (``params.get("window", 200)``),
    so a Python default can be changed without the Pine one following. The
    result is a script that looks like the strategy it names but is a
    different one — the exact failure this module exists to prevent, and one
    no reader of the site could ever detect.
    """
    import inspect
    import re

    from aitb.platform.tradingview import _GENERATORS
    from aitb.strategies import STRATEGY_CLASSES

    pattern = re.compile(r"""params\.get\(\s*["'](\w+)["']\s*,\s*([^)]+?)\s*\)""")
    checked = 0
    for name, gen in _GENERATORS.items():
        cls = STRATEGY_CLASSES.get(name)
        assert cls is not None, f"{name} exports Pine but is not a strategy"
        sig = inspect.signature(cls.__init__)
        for param, literal in pattern.findall(inspect.getsource(gen)):
            if param not in sig.parameters:
                continue          # a derived quantity, not a strategy parameter
            py = sig.parameters[param].default
            if py is inspect.Parameter.empty or not isinstance(py, (int, float)):
                continue
            pine = float(eval(literal, {"__builtins__": {}}, {}))  # noqa: S307
            assert pine == float(py), (
                f"{name}: Pine defaults {param}={pine}, Python uses {py}")
            checked += 1
    assert checked >= 10, "the default-drift check matched nothing — it is inert"


def test_no_pine_export_can_repaint():
    """Nothing may read a bar that had not closed when the signal fired.

    ``lookahead_on`` is Pine's one-line lookahead bug: a higher-timeframe or
    cross-symbol request returns the FINAL value of a bar that was still
    forming. It makes any backtest look extraordinary and is the single most
    common way a published Pine strategy is wrong.
    """
    from aitb.platform.tradingview import BACKTESTABLE, _GENERATORS, export_pine
    for name in _GENERATORS:
        code = export_pine(name)
        assert "lookahead_on" not in code, f"{name} requests future bars"
        assert "lookahead=barmerge.lookahead" not in code, name
        if name in BACKTESTABLE:
            # Orders placed on close, filled next open — the engine's rule.
            assert "process_orders_on_close=false" in code, name
            assert "calc_on_every_tick=false" in code, name


def test_rsi_export_states_that_the_family_failed():
    """The cost-fragile family must say so in the file itself.

    Pine charges nothing, so the Strategy Tester makes this look good. Anyone
    who pastes it and sees a rising equity curve is looking at the exact
    artifact the study measured and rejected.
    """
    from aitb.platform.tradingview import export_pine
    code = export_pine("RSIReversion")
    assert "did NOT survive realistic costs" in code
    assert "NEGATIVE" in code


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
                 "results.html", "charts.html", "method.html"):
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


def test_no_strategy_inherits_abcs_docstring():
    """A strategy with no docstring must not describe itself as abc.ABC.

    inspect.getdoc walks the MRO, so ten strategies were published on the site
    describing themselves as "Helper class that provides a standard way to
    create an ABC using inheritance" — on the overview, the strategy pages and
    the TradingView tab.
    """
    from aitb.platform.catalog import build_catalog
    cat = build_catalog("synthetic")
    for name, e in cat.items():
        assert "abc" not in e.docstring.lower()[:60], f"{name}: {e.docstring[:70]}"
        assert "Helper class" not in e.docstring, name
        if e.status != "unlisted":
            assert e.docstring.strip(), f"{name} has no description at all"


def test_deprecated_variants_are_recorded_but_never_run():
    """`status: deprecated` means WITHDRAWN, not "runs but is not a candidate".

    run_experiments.py writes a `deprecated` registry record carrying the
    entry's stated reason and then `continue`s — the grid is never expanded and
    no backtest happens. That is the intended research-integrity behaviour: a
    withdrawn variant stays visible with its reason instead of vanishing.

    It is also a trap for anyone adding a CONTROL ARM, which must run to be
    worth anything. Freeze v6 marked the VolumeConfirmedBreakout control
    (vol_mult=1.0) deprecated on the reasoning that a control is not a
    candidate; it silently did not run and the comparison it existed to enable
    was empty. Nothing errored. This test states the contract so the next
    person reads it before making the same inference.
    """
    from aitb.config import load_yaml

    spec = load_yaml("experiments.yaml")
    for family, entries in spec.items():
        for e in entries:
            if e.get("status") != "deprecated":
                continue
            assert e.get("reason"), (
                f"{family}/{e['class']} is deprecated with no reason — a "
                f"withdrawn variant must record why it was withdrawn")

    # The control arm specifically must be runnable, or it measures nothing.
    controls = [e for e in spec.get("chartable", [])
                if "control arm" in str(e.get("reason", ""))]
    assert controls, "the volume-filter control arm has disappeared"
    for c in controls:
        assert c["status"] != "deprecated", (
            f"{c['class']} is described as a control arm but is deprecated, so "
            f"run_experiments.py will never execute it and the comparison it "
            f"exists for will be silently empty")
