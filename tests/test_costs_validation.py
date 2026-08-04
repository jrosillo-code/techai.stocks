import numpy as np
import pandas as pd
import pytest

from aitb.config import CostScenario, load_cost_scenarios
from aitb.costs import daily_borrow_cost, one_way_cost
from aitb.validation import (block_bootstrap_ci, deflated_sharpe,
                             expected_max_sharpe, probabilistic_sharpe,
                             walk_forward_select)


BASE = CostScenario("base", "base", 0.5, 2.5, 2.5, 10.0, 50.0)


def test_one_way_cost_components():
    # 100k trade, huge ADV -> fixed cost only: 5.5 bp of 100k = $55.
    assert abs(one_way_cost(100_000, 1e12, BASE) - 55.0) < 0.5
    # participation impact: trade = ADV -> +10bp * sqrt(1) = $100 extra.
    c = one_way_cost(100_000, 100_000, BASE)
    assert abs(c - (55.0 + 100.0)) < 0.5
    assert one_way_cost(0, 1e9, BASE) == 0.0


def test_borrow_cost_annualizes():
    daily = daily_borrow_cost(1_000_000, BASE)
    assert abs(daily * 252 - 1_000_000 * 0.005) < 1e-6


def test_cost_scenarios_load_and_order():
    scens = load_cost_scenarios()
    assert set(scens) >= {"zero", "low", "base", "stressed"}
    assert (scens["zero"].fixed_one_way_bps < scens["low"].fixed_one_way_bps
            < scens["base"].fixed_one_way_bps < scens["stressed"].fixed_one_way_bps)


def test_psr_positive_series_high():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    good = pd.Series(rng.normal(0.001, 0.01, 1500), index=idx)
    noise = pd.Series(rng.normal(0.0, 0.01, 1500), index=idx)
    assert probabilistic_sharpe(good) > 0.95
    assert probabilistic_sharpe(noise) < 0.95  # zero-mean noise: no confidence


def test_deflated_sharpe_punishes_many_trials():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=1250)
    r = pd.Series(rng.normal(0.0004, 0.01, 1250), index=idx)
    few = deflated_sharpe(r, [0.5, 0.6])
    many = deflated_sharpe(r, list(rng.normal(0.3, 0.5, 200)))
    assert many < few  # same series, more trials -> less confidence


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0


def test_walk_forward_selects_on_prior_data_only():
    idx = pd.bdate_range("2010-01-01", periods=2520)
    # Variant A great in the first half, variant B great in the second half.
    a = pd.Series(np.where(np.arange(2520) < 1260, 0.002, -0.001), index=idx)
    b = pd.Series(np.where(np.arange(2520) < 1260, -0.001, 0.002), index=idx)
    oos, log = walk_forward_select({"A": a, "B": b},
                                   min_train_months=24, step_months=12)
    # Early steps must pick A (B's later glory is invisible at selection time).
    assert log.iloc[0]["selected"] == "A"
    # And the composite must lag the regime change (keeps picking A into B's era).
    assert (log["selected"] == "A").sum() >= 2


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=1000)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000), index=idx)
    point, lo, hi = block_bootstrap_ci(r, n_boot=200, seed=3)
    assert lo <= point <= hi
