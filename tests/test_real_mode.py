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
