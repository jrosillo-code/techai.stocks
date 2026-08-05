import numpy as np
import pandas as pd
import pytest

from aitb.backtest.engine import run_backtest
from aitb.config import CostScenario

from conftest import make_toy_md

ZERO = CostScenario("zero", "zero", 0, 0, 0, 0, 0)
BASE = CostScenario("base", "base", 0.5, 2.5, 2.5, 10.0, 50.0)


def test_single_asset_full_weight_tracks_asset():
    md = make_toy_md()
    w = pd.DataFrame(1.0, index=md.calendar, columns=["AAA"])
    res = run_backtest(md, w, ZERO)
    # After the initial fill, portfolio return must equal the asset's return.
    asset_r = md.adj_close["AAA"].pct_change()
    diff = (res.returns - asset_r).iloc[5:].abs().max()
    assert diff < 1e-10
    # And total equity ≈ initial * price growth from the entry fill.
    entry_px = md.open["AAA"].iloc[1]
    expected = 1_000_000 * md.adj_close["AAA"].iloc[-1] / entry_px
    assert abs(res.equity.iloc[-1] / expected - 1) < 1e-6


def test_costs_strictly_reduce_equity():
    md = make_toy_md()
    # Alternate 100% AAA / 100% BBB weekly to force turnover.
    w = pd.DataFrame(0.0, index=md.calendar, columns=["AAA", "BBB"])
    flip = (np.arange(len(md.calendar)) // 5) % 2
    w["AAA"] = (flip == 0).astype(float)
    w["BBB"] = (flip == 1).astype(float)
    res_zero = run_backtest(md, w, ZERO)
    res_base = run_backtest(md, w, BASE)
    assert res_base.equity.iloc[-1] < res_zero.equity.iloc[-1]
    assert res_base.total_costs > 0
    assert res_base.annual_turnover > 5  # heavy churn by construction


def test_cash_position_earns_nothing_without_macro():
    md = make_toy_md()
    w = pd.DataFrame(0.0, index=md.calendar, columns=["AAA"])
    w.iloc[0] = 0.0
    res = run_backtest(md, w, ZERO)
    assert abs(res.equity.iloc[-1] - 1_000_000) < 1e-6


def test_half_and_half_weights_sum():
    md = make_toy_md()
    w = pd.DataFrame(0.5, index=md.calendar, columns=["AAA", "BBB"])
    res = run_backtest(md, w, ZERO)
    # Realized weights should hover near 50/50 (rebalanced daily within tol).
    late = res.weights.iloc[-1]
    assert abs(late["AAA"] - 0.5) < 0.02 and abs(late["BBB"] - 0.5) < 0.02


def test_short_position_pays_borrow():
    md = make_toy_md(daily_ret=0.0)  # flat prices: isolate borrow cost
    w = pd.DataFrame(0.0, index=md.calendar, columns=["AAA"])
    w["AAA"] = -0.5
    res = run_backtest(md, w, BASE)
    assert res.equity.iloc[-1] < 1_000_000  # borrow fees bled the account


def test_short_into_a_delisting_is_not_a_windfall():
    """A short in a company that stops trading must not print free money.

    Shorting a stock to zero is a 100% gain, and the universe deliberately
    contains 39 companies that died. If the engine closed a short at zero the
    moment prices ended, the long-short family added in v4 would post
    spectacular returns for a reason that has nothing to do with its signal —
    and it would be nearly impossible to spot in aggregate results.

    The engine marks a name with no fresh quote at its LAST price instead. That
    is conservative for shorts (reality often does hand you the full 100%) and
    optimistic for longs (reality usually gives you less than the last print).
    Both directions are disclosed; neither is silently favourable.
    """
    import numpy as np
    import pandas as pd
    from aitb.config import load_universe_config
    from aitb.data.loader import MarketData

    cal = pd.bdate_range("2020-01-01", periods=60, name="date")
    px = pd.DataFrame({"AAA": 100.0, "BBB": 100.0}, index=cal)
    px.loc[cal[30]:, "AAA"] = np.nan          # AAA stops trading
    md = MarketData(open=px, high=px, low=px, close=px, adj_close=px,
                    dollar_volume=pd.DataFrame(1e12, index=cal,
                                               columns=["AAA", "BBB"]),
                    macro=pd.DataFrame(index=cal), fundamentals=pd.DataFrame(),
                    universe=load_universe_config(), provider_name="toy")

    w = pd.DataFrame(0.0, index=cal, columns=["AAA", "BBB"])
    w.iloc[5:, 0] = -0.5                       # short the doomed name
    w.iloc[5:, 1] = 0.5
    res = run_backtest(md, w, ZERO, name="short-into-delisting",
                       initial_capital=1_000_000)

    before = res.equity.iloc[29]
    after = res.equity.iloc[-1]
    # Prices were flat throughout, so a correct engine changes nothing. A
    # write-off of the short would add ~50% of equity.
    assert after == pytest.approx(before, rel=1e-6), (
        f"equity moved {after / before - 1:+.1%} when a short's quotes ended — "
        "the position was written off rather than marked at its last price")
