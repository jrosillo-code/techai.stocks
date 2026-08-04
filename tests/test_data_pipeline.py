"""Integration tests for the synthetic data pipeline and universe."""
import numpy as np
import pandas as pd

from aitb.data.validation import validate_prices
from aitb.portfolio import cap_weights, equal_weight, rebalance_schedule


def test_synthetic_determinism():
    from aitb.data.synthetic import SyntheticProvider
    from datetime import date
    a = SyntheticProvider().fetch_daily("NVDA", date(2020, 1, 1), date(2021, 1, 1))
    b = SyntheticProvider().fetch_daily("NVDA", date(2020, 1, 1), date(2021, 1, 1))
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_respects_listing_dates(synth_md):
    px = synth_md.adj_close
    assert px["ARM"].loc[:"2023-09-13"].isna().all()      # pre-IPO: no prices
    assert px["ARM"].loc["2023-09-15":].notna().any()
    assert px["XLNX"].loc["2022-03-01":].isna().all()     # post-delisting: none
    assert px["SUNW"].loc["2010-02-01":].isna().all()


def test_loaded_panels_validate_clean(synth_md):
    errors = [i for i in synth_md.issues if i.severity == "error"]
    assert not errors, errors


def test_adjusted_vs_unadjusted_dividend_gap(synth_md):
    """Dividend payers: total-return series must outgrow the price series."""
    tr = synth_md.adj_close["MSFT"].dropna()
    px = synth_md.close["MSFT"].dropna()
    growth_gap = (tr.iloc[-1] / tr.iloc[0]) / (px.iloc[-1] / px.iloc[0])
    assert growth_gap > 1.2  # ~1% yield over ~36y


def test_validation_catches_bad_frames():
    idx = pd.bdate_range("2020-01-01", periods=50)
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                       "close": 100.0, "adj_close": 100.0, "volume": 1e6}, index=idx)
    bad = df.copy()
    bad.iloc[10, bad.columns.get_loc("close")] = -5
    kinds = {i.kind for i in validate_prices("T", bad)}
    assert "nonpositive" in kinds
    stale = {i.kind for i in validate_prices("T", df)}
    assert "stale" in stale  # constant closes flagged


def test_macro_series_present(synth_md):
    for col in ("FEDFUNDS", "DGS10", "VIXCLS", "CPIYOY"):
        assert col in synth_md.macro.columns
        assert synth_md.macro[col].notna().mean() > 0.9


def test_cap_weights_respects_cap():
    w = pd.DataFrame({"A": [0.6, 0.8], "B": [0.3, 0.1], "C": [0.1, 0.1]})
    capped = cap_weights(w, 0.4)
    assert (capped.max(axis=1) <= 0.4 + 1e-9).all()
    np.testing.assert_allclose(capped.sum(axis=1), w.sum(axis=1), rtol=1e-6)


def test_equal_weight_and_schedule():
    idx = pd.bdate_range("2020-01-01", periods=60)
    sel = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
    w = equal_weight(sel)
    assert np.allclose(w.to_numpy(), 0.5)
    sched = rebalance_schedule(w, "ME")
    # Before the first month-end decision there is deliberately no position;
    # from then on the schedule holds the decided weights.
    assert np.allclose(sched.iloc[-20:].to_numpy(), 0.5)
    assert np.allclose(sched.iloc[0].to_numpy(), 0.0)
