"""Adversarial audit regression tests (findings AUD-001..AUD-011).

Each test reproduces an audit finding's failure mode and pins the fix.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))
from conftest import make_toy_md

from aitb.backtest.engine import run_backtest
from aitb.config import CostScenario

ZERO = CostScenario("zero", "z", 0, 0, 0, 0, 0)
BASE = CostScenario("base", "b", 0.5, 2.5, 2.5, 10.0, 50.0)


# ---------------------------------------------------- AUD-001: freeze scope --
def test_freeze_binds_analytics_and_data_layer():
    """Metrics, validation, tiers, loader, importer, gate, calendar, registry
    logic and tax must be fingerprinted — an edit to any of them can change
    reported results."""
    from aitb.freeze import _FROZEN_MODULES
    required = [
        "src/aitb/metrics.py", "src/aitb/validation.py", "src/aitb/tiers.py",
        "src/aitb/calendar.py", "src/aitb/experiments.py", "src/aitb/tax.py",
        "src/aitb/data/loader.py", "src/aitb/data/import_bundle.py",
        "src/aitb/data/quality.py", "src/aitb/data/security_master.py",
    ]
    missing = [m for m in required if m not in _FROZEN_MODULES]
    assert not missing, f"freeze does not bind: {missing}"


# ------------------------------------------- AUD-002: freeze at report time --
def test_all_real_mode_scripts_verify_freeze():
    for script in ("run_experiments.py", "run_robustness.py", "run_capacity.py",
                   "run_company_analysis.py", "make_report.py"):
        src = (Path(__file__).parents[1] / "scripts" / script).read_text()
        assert "verify_freeze" in src, f"{script} runs real mode without freeze check"


# --------------------------------------- AUD-003: holdout tamper-evidence ----
def test_holdout_lock_deletion_is_detected(tmp_path, monkeypatch):
    import aitb.holdout as ho
    monkeypatch.setattr(ho, "results_dir", lambda mode: tmp_path)
    ho.freeze_selection([{"class": "X"}], "real", "2024-01-01")
    ho.record_holdout_access("real", "sanctioned access")
    assert not ho.holdout_status("real")["compromised"]
    # Attacker deletes the lock file; the registry mirror survives.
    (tmp_path / "holdout_lock.json").unlink()
    st = ho.holdout_status("real")
    assert st["compromised"], "deleting holdout_lock.json must not restore clean status"
    assert any(v["kind"] == "lock_registry_mismatch" for v in st["violations"])


def test_holdout_lock_rewrite_is_detected(tmp_path, monkeypatch):
    import aitb.holdout as ho
    monkeypatch.setattr(ho, "results_dir", lambda mode: tmp_path)
    ho.freeze_selection([{"class": "X"}], "real", "2024-01-01")
    ho.record_holdout_access("real", "first")
    ho.record_holdout_access("real", "second (compromising)")
    # Attacker rewrites the lock to show a single clean access with a fake chain.
    lock = json.loads((tmp_path / "holdout_lock.json").read_text())
    lock["access_log"] = [{"kind": "access", "purpose": "first", "chain": "fake"}]
    lock["compromised"] = False
    (tmp_path / "holdout_lock.json").write_text(json.dumps(lock))
    assert ho.holdout_status("real")["compromised"]


def test_holdout_events_bind_freeze_hash(tmp_path, monkeypatch):
    import aitb.holdout as ho
    monkeypatch.setattr(ho, "results_dir", lambda mode: tmp_path)
    ho.freeze_selection([{"class": "X"}], "real", "2024-01-01")
    st = ho.record_holdout_access("real", "check bindings")
    ev = st["access_log"][0]
    assert "freeze_hash" in ev and "store_fingerprint" in ev and "chain" in ev


# ----------------------------------- AUD-004/005: decision brief fail-closed --
def test_brief_fail_closed_source_contract():
    src = (Path(__file__).parents[1] / "scripts" / "make_decision_brief.py").read_text()
    # missing capacity row can no longer pass
    assert "(not len(crow)) or" not in src
    assert "cap_est is not None and cap_est >= 1e6" in src
    # compromised holdout / bad gate / stale gate force DO NOTHING
    assert "blocked_reason" in src and "COMPROMISED" in src
    assert 'gate.get("status") not in ("PASS FOR RESEARCH", "PASS WITH LIMITATIONS")' in src
    assert "store_fingerprint" in src
    # non-finite values blocked
    assert "_finite" in src


# ------------------------------------------------ AUD-006: CAGR annualization --
def test_cagr_uses_elapsed_calendar_time():
    from aitb.metrics import cagr
    idx = pd.bdate_range("2020-01-02", "2023-12-29")
    years_true = (idx[-1] - idx[0]).days / 365.25
    daily = (1.10) ** (years_true / len(idx)) - 1
    r = pd.Series(daily, index=idx)
    assert abs(cagr(r) - 0.10) < 1e-4, f"CAGR {cagr(r):.4%} != 10% on b-day calendar"


# ------------------------------------------------------- AUD-007: lineage ----
def test_real_records_carry_store_fingerprint():
    from aitb.experiments import _lineage
    md = make_toy_md()
    md.data_mode = "real"
    lin = _lineage(md)
    assert "store_fingerprint" in lin and "freeze_hash" in lin
    md.data_mode = "synthetic"
    assert _lineage(md) == {}          # synthetic runs don't claim real lineage


# ------------------------------------------------- AUD-008: DSR trial count --
def test_failed_variants_count_as_trials():
    src = (Path(__file__).parents[1] / "scripts" / "run_robustness.py").read_text()
    assert "n_failed" in src and '[0.0] * n_failed' in src


# ------------------------------------------------- AUD-009: tier fail-closed --
def test_tier_metadata_omission_cannot_promote():
    from aitb.tiers import assign_tier
    clean_gate = {"limitations": []}
    t, r = assign_tier({"family": "", "status": "ok"}, clean_gate)
    assert t != "A" and "metadata" in r
    t, _ = assign_tier({"family": "made_up_family", "status": "ok"}, clean_gate)
    assert t != "A"
    t, _ = assign_tier({}, clean_gate)          # no status at all
    assert t == "C"


# ------------------------------------------- AUD-011: accounting invariant ----
def test_engine_invariant_holds_on_stress_paths():
    md = make_toy_md(n_days=300)
    flip = (np.arange(300) // 3) % 2
    w = pd.DataFrame(0.0, index=md.calendar, columns=["AAA", "BBB", "CCC"])
    w["AAA"] = np.where(flip == 0, 0.6, 0.1)
    w["BBB"] = np.where(flip == 0, 0.1, 0.6)
    w["CCC"] = -0.2                                     # short leg + borrow
    res = run_backtest(md, w, BASE, check_invariants=True)   # raises on violation
    assert res.equity.iloc[-1] > 0


def test_engine_invariant_with_delisting():
    md = make_toy_md(n_days=40, daily_ret=0.001)
    for panel in (md.open, md.high, md.low, md.close, md.adj_close, md.dollar_volume):
        panel.loc[panel.index[21]:, "AAA"] = np.nan
    w = pd.DataFrame(0.0, index=md.calendar, columns=["AAA", "BBB"])
    w.iloc[2:20, 0] = 0.5
    w.iloc[2:, 1] = 0.5
    run_backtest(md, w, ZERO, check_invariants=True)    # must not raise


# ----------------------------------------- timing / corporate-action audits ---
def test_split_between_signal_and_execution_no_fake_pnl():
    """1:10 raw split overnight between signal close and execution open."""
    md = make_toy_md(n_days=20, daily_ret=0.0)
    raw = md.close.copy()
    raw.loc[raw.index[6]:, "AAA"] /= 10.0
    md.close = raw
    w = pd.DataFrame(0.0, index=md.calendar, columns=["AAA"])
    w.iloc[5:] = 1.0
    res = run_backtest(md, w, ZERO)
    assert abs(res.returns.iloc[6]) < 1e-9


def test_weekend_filing_visible_next_session_only():
    from aitb.config import load_universe_config
    from aitb.data.loader import MarketData
    from aitb.features import pit_fundamental_panel
    cal = pd.bdate_range("2020-01-01", periods=60, name="date")
    f = pd.DataFrame({"ticker": ["AAA"], "period_end": [pd.Timestamp("2019-12-31")],
                      "published": [pd.Timestamp("2020-02-01")],   # a Saturday
                      "revenue": [10.0], "eps": [1.0], "fcf": [1.0], "shares": [1.0]})
    px = pd.DataFrame(100.0, index=cal, columns=["AAA"])
    md = MarketData(open=px, high=px, low=px, close=px, adj_close=px,
                    dollar_volume=px * 1e6, macro=pd.DataFrame(index=cal),
                    fundamentals=f, universe=load_universe_config())
    panel = pit_fundamental_panel(md, "revenue", "level")
    assert panel.loc["2020-01-31", "AAA"] != panel.loc["2020-01-31", "AAA"] or \
        pd.isna(panel.loc["2020-01-31", "AAA"])        # Friday before: invisible
    assert panel.loc["2020-02-03", "AAA"] == 10.0      # Monday after: visible


def test_close_predicting_next_open_gives_no_edge():
    """Adversarial: all price movement happens overnight and the signal
    'knows' tomorrow's jump. Fills at next open mean the predicted jump is
    already in the fill price — averaged over seeds there must be NO edge,
    while perfect capture would compound enormously."""
    realized, perfect = [], []
    for seed in range(20):
        md = make_toy_md(n_days=100, daily_ret=0.0)
        rng = np.random.default_rng(seed)
        jumps = rng.choice([-0.02, 0.02], size=100)
        close = pd.Series(100.0 * np.cumprod(1 + jumps), index=md.calendar)
        open_ = close.shift(1).fillna(100.0) * (1 + jumps)
        for panel, vals in ((md.close, close), (md.adj_close, close), (md.open, open_)):
            panel["AAA"] = vals
        w = pd.DataFrame(0.0, index=md.calendar, columns=["AAA"])
        w["AAA"] = (np.roll(jumps, -1) > 0).astype(float)
        res = run_backtest(md, w, ZERO)
        realized.append(res.equity.iloc[-1] / 1e6)
        perfect.append(float(np.prod(1 + jumps[jumps > 0])))
    mean_realized = float(np.mean(realized))
    mean_perfect = float(np.mean(perfect))
    assert mean_perfect > 2.0                      # the leak, if any, is huge
    assert 0.85 < mean_realized < 1.15, (          # engine captures none of it
        f"mean realized growth {mean_realized:.3f} suggests same-bar leakage")


# ------------------------------------------------ statistical safeguards -----
def test_null_strategies_rarely_survive_verdict_layer():
    """30 random-return strategies: the verdict layer must reject the vast
    majority; the layered brief criteria (bootstrap CI, tier, capacity)
    handle the residual."""
    from aitb.metrics import summary
    from aitb.ranking import rank_experiments
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2010-01-01", periods=3500)
    hold = idx[-500]

    def rec(name, r, family):
        dev, ho = r[r.index < hold], r[r.index >= hold]
        return {"status": "ok", "scenario": "base", "data_mode": "synthetic",
                "strategy": name, "family": family,
                "metrics_dev": summary(dev), "metrics_holdout": summary(ho),
                "psr_dev": 0.5, "annual_turnover": 2.0,
                "concentration": {"top_name_share": 0.2},
                "subperiods": [{"period": p, "sharpe": float(rng.normal(0, .5))}
                               for p in ("a", "b", "c", "d")],
                "spec": {"class": name, "params": {"a": 1}}}

    bench = pd.Series(rng.normal(0.0004, 0.011, len(idx)), index=idx)
    records = [rec("BuyAndHold(ticker=QQQ)", bench, "benchmark")]
    for i in range(30):
        records.append(rec(f"Null{i}",
                           pd.Series(rng.normal(0, 0.011, len(idx)), index=idx), "null"))
    out = rank_experiments(pd.DataFrame(records))
    n_robust = int((out[out["family"] == "null"]["verdict"] == "robust_candidate").sum())
    assert n_robust <= 2, f"{n_robust}/30 null strategies passed the verdict layer"


def test_zero_turnover_zero_trading_cost():
    md = make_toy_md(n_days=100)
    w = pd.DataFrame(0.0, index=md.calendar, columns=["AAA"])  # never trades
    res = run_backtest(md, w, BASE)
    assert res.total_costs == 0.0 and res.n_trades == 0


# ------------------------------- AUD-016: experiment ID must bind the universe --
def test_experiment_id_binds_universe(synth_md):
    """Two runs over different investable rosters are different experiments.

    Without this the runner's cache is actively harmful: it skips any ID it has
    already seen and reloads the stored curve, so expanding the universe would
    return the OLD universe's results while labelling them as the new study.
    """
    import dataclasses
    from aitb.experiments import experiment_id, universe_hash

    full = synth_md
    narrow_secs = full.universe.securities[:12]
    narrow_u = dataclasses.replace(full.universe, securities=narrow_secs)
    narrow = dataclasses.replace(full, universe=narrow_u)

    assert universe_hash(full) != universe_hash(narrow)

    from aitb.strategies import STRATEGY_CLASSES
    strat = STRATEGY_CLASSES["XSMomentumTopN"](lookback_days=126, top_n=5)
    assert experiment_id(strat, full, "base") != experiment_id(strat, narrow, "base")
    # ...and identical inputs must still be stable, or nothing is ever cached.
    assert experiment_id(strat, full, "base") == experiment_id(strat, full, "base")


def test_delisting_date_change_changes_universe_hash(synth_md):
    """The hash covers listing windows, not just the ticker list.

    Moving a delisting date changes what a strategy could have held and for how
    long. That is a different experiment even though the roster is identical.
    """
    import dataclasses
    import datetime as dt
    from aitb.experiments import universe_hash

    secs = list(synth_md.universe.securities)
    i = next(i for i, s in enumerate(secs) if s.delisted is not None)
    secs[i] = dataclasses.replace(secs[i], delisted=dt.date(2018, 1, 2))
    moved = dataclasses.replace(synth_md,
                                universe=dataclasses.replace(synth_md.universe,
                                                             securities=secs))
    assert universe_hash(synth_md) != universe_hash(moved)


def test_ranking_uses_one_universe_cohort():
    """A leaderboard must never mix results from two different rosters."""
    from aitb.ranking import current_cohort, rank_experiments

    def rec(uh, size, sharpe, ts):
        return {"status": "ok", "scenario": "base", "data_mode": "synthetic",
                "strategy": f"S{sharpe}(x=1)", "family": "xsmom",
                "universe_hash": uh, "universe_size": size, "timestamp": ts,
                "metrics_dev": {"sharpe": sharpe}, "metrics_holdout": {"sharpe": sharpe},
                "subperiods": [], "spec": {"params": {}}}

    df = pd.DataFrame([rec("OLD", 38, 3.0, "2026-08-01T00:00:00Z"),
                       rec("NEW", 120, 0.5, "2026-08-05T00:00:00Z")])
    assert set(current_cohort(df)["universe_hash"]) == {"NEW"}
    out = rank_experiments(df)
    assert len(out) == 1 and out.iloc[0]["dev_sharpe"] == 0.5


# ------------------ AUD-017: registry must not accumulate duplicate IDs -------
def test_registry_has_no_duplicate_ids():
    """The committed registry must contain each experiment ID exactly once.

    ExperimentRegistry.append() has no uniqueness guard (AUD-017, open), so two
    concurrent runners will both write the same IDs. That has happened. Until
    the guard lands with the next freeze bump, this test is the backstop that
    keeps a duplicated registry out of the repository.
    """
    from aitb.experiments import ExperimentRegistry
    reg = ExperimentRegistry.for_mode("synthetic")
    if not reg.path.exists():
        pytest.skip("no synthetic registry present")
    ids = [json.loads(l)["id"] for l in reg.path.read_text().splitlines() if l.strip()]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, (
        f"{len(dupes)} duplicated experiment IDs in {reg.path} — this is what a "
        "second concurrent runner looks like. Re-run as a single process.")


def test_site_baseline_set_matches_ranking_hurdle():
    """The page saying "the baselines they had to beat" must name the same
    funds the ranking actually used as the hurdle.

    Two hard-coded lists in two modules is how a site ends up describing a bar
    that was never applied.
    """
    import inspect
    from aitb import ranking
    from aitb.platform import site
    src = inspect.getsource(ranking.rank_experiments)
    for ticker in site._ETF_BENCH:
        assert f'"{ticker}"' in src, (
            f"{ticker} is shown as a diversified baseline on the site but is "
            "not in ranking.py's hurdle set")


def test_strategy_family_describes_what_it_is():
    """`family` must describe the construction, not the source file.

    It is part of every experiment's identity, it groups the deflated-Sharpe
    trial batteries, and it drives the correlation-by-family chart. Five
    long-only screens were once filed under `longshort` purely because they
    shared a module, which made the site report that hedged strategies
    correlate 0.83 with the index while the actual hedged ones correlate ~0.

    A strategy is long-short only if it can hold a negative weight.
    """
    from aitb.strategies import STRATEGY_CLASSES
    import inspect

    claimed = {n for n, c in STRATEGY_CLASSES.items()
               if getattr(c, "family", "") == "longshort"}
    assert claimed, "no long-short strategies registered"
    for name in claimed:
        src = inspect.getsource(STRATEGY_CLASSES[name].build)
        shorts = ("-" in src and ("short" in src.lower() or "hedge" in src.lower()))
        assert shorts, (
            f"{name} is labelled longshort but its build() never takes a "
            "negative position")


def test_market_neutral_family_really_is_uncorrelated(synth_md):
    """The claim the site makes about the longshort family must be true.

    If a long-only strategy is ever mislabelled into this family again, the
    family median correlation jumps and the site's headline conclusion becomes
    false. This checks the property directly rather than trusting the label.
    """
    import numpy as np
    from aitb.backtest.engine import run_backtest
    from aitb.strategies import STRATEGY_CLASSES

    w = STRATEGY_CLASSES["MarketNeutralMomentum"](top_n=10).build(synth_md)
    res = run_backtest(synth_md, w, ZERO, name="mn", initial_capital=1_000_000)
    q = synth_md.adj_close["QQQ"].pct_change().reindex(res.returns.index)
    m = pd.DataFrame({"s": res.returns, "q": q})
    m = (1 + m).resample("ME").prod() - 1
    corr = float(m["s"].corr(m["q"]))
    assert abs(corr) < 0.35, (
        f"the market-neutral book correlates {corr:+.2f} with QQQ — it is not "
        "market-neutral, and the site's central conclusion depends on it being so")


def test_registry_families_match_the_current_classes():
    """No committed record may claim a family its class does not have.

    A record carrying a stale family is the fingerprint of a corrected
    mislabelling that was never cleaned up. It ranks the same strategy twice
    under two labels and splits its deflated-Sharpe trial battery in half —
    which makes both halves look more significant than the whole.
    """
    from aitb.experiments import ExperimentRegistry
    from aitb.strategies import STRATEGY_CLASSES

    for mode in ("synthetic", "real"):
        reg = ExperimentRegistry.for_mode(mode)
        if not reg.path.exists():
            continue
        bad = []
        for line in reg.path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cls = (r.get("spec") or {}).get("class")
            c = STRATEGY_CLASSES.get(cls)
            fam = r.get("family")
            if c is None or fam is None:
                continue
            if fam != getattr(c, "family", fam):
                bad.append((cls, fam, c.family))
        assert not bad, (
            f"{mode}: records whose family contradicts the class definition — "
            f"{sorted(set(bad))[:5]}")


# ------- AUD-020: failed trials must still enter the correction battery -------
def test_failed_trials_are_recovered_for_the_multiple_testing_correction():
    """A failed run is still a trial and must count against significance.

    AUD-008 established this. The fix then silently died: failure records carry
    only `spec` — no family, no universe_hash, no freeze_version — so the
    cohort filter dropped them and the battery matched on a `family` column
    that is never present on a failure. n_failed was permanently zero, and
    every family's deflated Sharpe was overstated as a result.

    Observed live: the first freeze-v4 real study had 48 failed runs, 33 of
    them in the current cohort, every one invisible to the correction.
    """
    import importlib.util
    from pathlib import Path

    spec_ = importlib.util.spec_from_file_location(
        "run_robustness", Path(__file__).parents[1] / "scripts" / "run_robustness.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)

    cohort = pd.DataFrame([
        {"status": "ok", "family": "longshort", "timestamp": "2026-08-05T10:00:00Z"},
        {"status": "ok", "family": "quality", "timestamp": "2026-08-05T10:05:00Z"},
    ])
    all_df = pd.concat([cohort, pd.DataFrame([
        # current cohort, failed — must be recovered, family read from spec
        {"status": "failed", "timestamp": "2026-08-05T10:02:00Z",
         "spec": {"class": "ResearchIntensity", "family": "quality"}},
        {"status": "failed", "timestamp": "2026-08-05T10:03:00Z",
         "spec": {"class": "AccrualQuality", "family": "quality"}},
        # an OLDER cohort's failure — must NOT be counted against this study
        {"status": "failed", "timestamp": "2026-08-01T09:00:00Z",
         "spec": {"class": "OldThing", "family": "quality"}},
    ])], ignore_index=True)

    rec = mod.recover_failed_trials(all_df, cohort)
    assert len(rec) == 2, f"expected 2 in-cohort failures, got {len(rec)}"
    assert set(rec["family"]) == {"quality"}
    assert int((rec["family"] == "quality").sum()) == 2
    assert int((rec["family"] == "longshort").sum()) == 0

    # No failures at all must not explode.
    empty = mod.recover_failed_trials(cohort, cohort)
    assert len(empty) == 0
