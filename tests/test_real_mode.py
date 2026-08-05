"""Real/synthetic separation, canonical store, quality gate, holdout lock."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aitb.data import realstore
from aitb.data.realstore import RealDataMissing


def _price_frame(start="2015-01-02", periods=2600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, periods)))
    df = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.003, periods)),
        "high": close * 1.01, "low": close * 0.99, "close": close,
        "adj_close": close * 1.02, "volume": 1e7,
    }, index=idx)
    df.index.name = "date"
    return df


def test_real_mode_fails_without_data(tmp_path, monkeypatch):
    """Real mode must hard-fail when the store is empty — never fall back."""
    import aitb.config as cfg
    import aitb.data.realstore as rs
    monkeypatch.setattr(cfg, "REAL_DATA_DIR", tmp_path / "real")
    monkeypatch.setattr(rs, "REAL_DATA_DIR", tmp_path / "real")
    from aitb.data.loader import load_market_data
    with pytest.raises(RealDataMissing):
        load_market_data(mode="real")


def test_real_store_roundtrip_and_meta(tmp_path):
    df = _price_frame()
    realstore.write("prices", "TEST", df, {"provider": "yahoo",
                    "adjustment": "total_return"}, root=tmp_path)
    back = realstore.read("prices", "TEST", root=tmp_path)
    pd.testing.assert_frame_equal(df, back, check_freq=False)
    meta = realstore.read_meta("prices", "TEST", root=tmp_path)
    assert meta["provider"] == "yahoo"
    assert meta["sha256"] == realstore.file_sha256(tmp_path / "prices" / "TEST.parquet")
    assert realstore.available("prices", root=tmp_path) == ["TEST"]


def test_registry_separation_by_mode():
    from aitb.experiments import ExperimentRegistry
    real = ExperimentRegistry.for_mode("real")
    synth = ExperimentRegistry.for_mode("synthetic")
    assert real.root != synth.root
    assert real.root.name == "real" and synth.root.name == "synthetic"


def test_ranking_refuses_mixed_modes():
    from aitb.ranking import rank_experiments
    df = pd.DataFrame([
        {"status": "ok", "scenario": "base", "data_mode": "real",
         "strategy": "a", "family": "x", "metrics_dev": {}, "metrics_holdout": {}},
        {"status": "ok", "scenario": "base", "data_mode": "synthetic",
         "strategy": "b", "family": "x", "metrics_dev": {}, "metrics_holdout": {}},
    ])
    with pytest.raises(ValueError, match="data modes"):
        rank_experiments(df)


def test_quality_gate_fail_on_empty(tmp_path):
    from aitb.data.quality import run_gate, FAIL
    res = run_gate(root=tmp_path)
    assert res.status == FAIL
    assert any(c["fatal"] and not c["ok"] for c in res.checks)


def test_quality_gate_pass_with_limitations(tmp_path):
    from aitb.config import load_universe_config
    from aitb.data.quality import run_gate, FAIL
    ucfg = load_universe_config()
    for i, t in enumerate((ucfg.tickers[:12] + ["SPY", "QQQ"])):
        realstore.write("prices", t, _price_frame(seed=i),
                        {"provider": "yahoo", "adjustment": "total_return"},
                        root=tmp_path)
    res = run_gate(root=tmp_path)
    assert res.status != FAIL
    assert res.limitations  # delisted coverage etc. must be disclosed
    assert res.store_fingerprint


def test_gate_fingerprint_invalidates_on_change(tmp_path):
    from aitb.config import load_universe_config
    from aitb.data.quality import run_gate, write_gate, require_gate
    ucfg = load_universe_config()
    for i, t in enumerate((ucfg.tickers[:12] + ["SPY", "QQQ"])):
        realstore.write("prices", t, _price_frame(seed=i),
                        {"provider": "yahoo"}, root=tmp_path)
    res = run_gate(root=tmp_path)
    out = tmp_path / "results"
    write_gate(res, mode_results=out)
    assert require_gate(root=tmp_path, mode_results=out)["status"] == res.status
    # Store changes -> fingerprint mismatch -> gate rejected.
    realstore.write("prices", "ZZZZ", _price_frame(seed=99),
                    {"provider": "yahoo"}, root=tmp_path)
    with pytest.raises(RuntimeError, match="fingerprint"):
        require_gate(root=tmp_path, mode_results=out)


def test_fundamental_leak_check_is_fatal(tmp_path):
    from aitb.config import load_universe_config
    from aitb.data.quality import run_gate, FAIL
    ucfg = load_universe_config()
    for i, t in enumerate((ucfg.tickers[:12] + ["SPY", "QQQ"])):
        realstore.write("prices", t, _price_frame(seed=i),
                        {"provider": "yahoo"}, root=tmp_path)
    bad = pd.DataFrame({
        "period_end": [pd.Timestamp("2020-03-31")],
        "published": [pd.Timestamp("2020-03-01")],   # BEFORE period end
        "revenue": [1.0], "eps": [0.1], "fcf": [0.5], "shares": [100.0]})
    realstore.write("fundamentals", "NVDA", bad, {"provider": "test"}, root=tmp_path)
    assert run_gate(root=tmp_path).status == FAIL


def test_holdout_lock_discipline(tmp_path, monkeypatch):
    import aitb.holdout as ho
    monkeypatch.setattr(ho, "results_dir", lambda mode: tmp_path)
    h = ho.freeze_selection([{"class": "X", "params": {}}], "real", "2024-08-01")
    assert h
    st = ho.record_holdout_access("real", "final evaluation")
    assert not st["compromised"]
    st = ho.record_holdout_access("real", "peeking again")
    assert st["compromised"]           # second access flips the flag


def test_holdout_access_before_freeze_is_violation(tmp_path, monkeypatch):
    import aitb.holdout as ho
    monkeypatch.setattr(ho, "results_dir", lambda mode: tmp_path)
    st = ho.record_holdout_access("real", "premature peek")
    assert st["compromised"]
    assert st["violations"][0]["kind"] == "access_before_freeze"


def test_real_mode_site_cannot_show_synthetic_results(tmp_path, monkeypatch):
    """Building the site in real mode must read the REAL registry only.

    The two modes share every line of rendering code, so the only thing keeping
    simulated numbers off a real-data site is that the mode is threaded all the
    way down. If it is ever dropped on one path, the page silently publishes
    synthetic results with no warning banner — the single worst failure this
    project could have.
    """
    from aitb.config import results_dir
    from aitb.platform.catalog import current_registry, load_registry

    synth_root = results_dir("synthetic")
    real_root = results_dir("real")
    assert synth_root != real_root, "modes must not share a results root"

    # The real registry is empty (no real study has run). Anything the real
    # mode reports must therefore be empty too — if it is not, it is reading
    # the synthetic side.
    real = load_registry("real")
    if not real.empty:
        pytest.skip("a real study exists in this checkout")
    assert current_registry("real").empty
    from aitb.platform.catalog import platform_stats
    stats = platform_stats("real")
    assert stats["experiments_ok"] == 0, (
        "real mode reported experiments while the real registry is empty — "
        "a synthetic result has leaked into the real-data view")


def test_real_mode_site_omits_names_with_no_price_history(monkeypatch):
    """A company the providers could not serve must not appear on the site.

    It was never in any backtest, so listing it would claim coverage that does
    not exist. The page must also say how many were dropped and how many of
    those were dead companies, because dropping THOSE reintroduces exactly the
    survivorship bias the roster exists to prevent.
    """
    from aitb.data import realstore
    from aitb.platform import site

    kept = ["NVDA", "MSFT", "AAPL", "AMZN", "SPY", "QQQ"]
    monkeypatch.setattr(realstore, "available",
                        lambda kind, root=None: kept if kind == "prices" else [])
    html_out = site._data_page("real")

    assert "NVDA" in html_out
    # A configured name with no history must be absent from the roster tables.
    assert ">WCOM<" not in html_out and ">NT<" not in html_out
    assert "had no usable price history" in html_out
    assert "survivorship bias" in html_out


def test_real_research_record_is_version_controlled():
    """The real study's governance artifacts must be committable.

    results/* is gitignored so the 340 MB of equity curves stay out. That
    default silently swallowed the real registry, ranking, quality gate and
    holdout lock too — so a finished real study would have published rendered
    HTML while dropping the evidence behind it. A study whose record is not
    committed is not reproducible, which defeats the freeze entirely.

    Equity curves must STILL be ignored: they are large and regenerable.
    """
    import subprocess
    from pathlib import Path
    root = Path(__file__).parents[1]

    must_commit = [
        "results/real/experiments.jsonl",
        "results/real/strategy_ranking.csv",
        "results/real/company_analysis.csv",
        "results/real/data_quality.json",
        "results/real/holdout_lock.json",
        "results/real/run_fingerprint.json",
        "results/real/robustness/family_summary.csv",
    ]
    must_ignore = ["results/real/curves/abc123.parquet"]

    def ignored(rel: str) -> bool:
        return subprocess.run(["git", "check-ignore", "-q", rel],
                              cwd=root).returncode == 0

    wrongly_ignored = [p for p in must_commit if ignored(p)]
    assert not wrongly_ignored, (
        "these would be silently dropped from a real study's commit: "
        f"{wrongly_ignored}")

    not_ignored = [p for p in must_ignore if not ignored(p)]
    assert not not_ignored, f"equity curves must stay out of git: {not_ignored}"


def test_dashboard_reports_actual_coverage_not_configured(monkeypatch):
    """The headline "companies in the test" must be what the study had.

    The dashboard is the page most people read and the least likely to be
    checked against the gate. Reporting the configured roster there while the
    run only had a subset overstates coverage on the most visible number on the
    site — and does it in the direction that flatters.
    """
    from aitb.data import realstore
    from aitb.platform import site
    from aitb.platform.catalog import build_catalog, platform_stats
    from aitb.experiments import ExperimentRegistry
    from aitb.config import load_universe_config

    kept = ["NVDA", "MSFT", "AAPL", "SPY", "QQQ"]
    monkeypatch.setattr(realstore, "available",
                        lambda kind, root=None: kept if kind == "prices" else [])

    html_out = site._dashboard("real", platform_stats("real"),
                               build_catalog("real"),
                               ExperimentRegistry.for_mode("real"))
    configured = len(load_universe_config().securities)
    assert f">{configured}</div><div class='k'>companies in the test" not in html_out, (
        "dashboard claims the full configured roster was tested")
    assert "had no data and were excluded" in html_out
