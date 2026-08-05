"""Bias-prevention regression tests.

1. The engine can never fill on the signal bar (structural next-open lag).
2. Signals computed on truncated data match signals computed on full data —
   i.e. features do not read the future.
3. Point-in-time fundamentals are invisible before their publication date.
4. Delisted names drop out of the investable mask after their delisting date;
   names are not investable before IPO + seasoning.
"""
import numpy as np
import pytest
import pandas as pd

from aitb.backtest.engine import run_backtest
from aitb.config import CostScenario
from aitb.features import momentum, rsi, donchian_breakout, pit_fundamental_panel

from conftest import make_toy_md

ZERO = CostScenario("zero", "zero", 0, 0, 0, 0, 0)


def test_engine_fills_next_open_not_signal_close():
    md = make_toy_md()
    cal = md.calendar
    # Signal fires exactly once, on day 10.
    w = pd.DataFrame(0.0, index=cal, columns=["AAA"])
    w.iloc[10] = 1.0
    res = run_backtest(md, w, ZERO)
    # Day 10 close: still all cash (trade cannot have happened yet).
    assert res.weights.iloc[10]["AAA"] == 0.0
    # Day 11: position is on.
    assert res.weights.iloc[11]["AAA"] > 0.9


def test_engine_pnl_uses_fill_after_signal():
    """A perfect-foresight signal must NOT capture the signal day's return."""
    md = make_toy_md()
    cal = md.calendar
    r = md.adj_close["AAA"].pct_change()
    # 'Cheating' weights: long only on days whose OWN return was positive.
    w = pd.DataFrame(0.0, index=cal, columns=["AAA"])
    w.loc[r > 0, "AAA"] = 1.0
    res = run_backtest(md, w, ZERO)
    # If the engine leaked, this strategy would earn every positive day and
    # skip every negative one — vastly beating buy & hold. It must not.
    bh = (1 + r.fillna(0)).prod()
    strat = res.equity.iloc[-1] / 1_000_000
    assert strat <= bh * 1.001


def test_features_are_causal():
    md = make_toy_md(n_days=250)
    px_full = md.adj_close
    px_trunc = px_full.iloc[:200]
    for fn in (lambda p: momentum(p, 63, 21), lambda p: rsi(p, 2),
               lambda p: donchian_breakout(p, 55)):
        full = fn(px_full).iloc[:200]
        trunc = fn(px_trunc)
        pd.testing.assert_frame_equal(full, trunc)


def test_fundamentals_respect_publication_date():
    from aitb.data.loader import MarketData
    from aitb.config import load_universe_config
    cal = pd.bdate_range("2020-01-01", periods=300, name="date")
    fundamentals = pd.DataFrame({
        "ticker": ["AAA"], "period_end": [pd.Timestamp("2020-03-31")],
        "published": [pd.Timestamp("2020-05-15")],
        "revenue": [100.0], "eps": [1.0], "fcf": [10.0], "shares": [1000.0],
    })
    px = pd.DataFrame(100.0, index=cal, columns=["AAA"])
    md = MarketData(open=px, high=px, low=px, close=px, adj_close=px,
                    dollar_volume=px * 1e6, macro=pd.DataFrame(index=cal),
                    fundamentals=fundamentals, universe=load_universe_config())
    panel = pit_fundamental_panel(md, "revenue", "level")
    # Invisible through period end and up to the publication date...
    assert panel.loc["2020-03-31":"2020-05-14", "AAA"].isna().all()
    # ...visible from publication onward.
    assert (panel.loc["2020-05-15":, "AAA"] == 100.0).all()
    # as-of accessor agrees
    assert md.fundamentals_asof("AAA", pd.Timestamp("2020-05-01")).empty
    assert len(md.fundamentals_asof("AAA", pd.Timestamp("2020-06-01"))) == 1


def test_universe_is_point_in_time(synth_md):
    from aitb.universe import investable_mask
    mask = investable_mask(synth_md)
    # XLNX delisted 2022-02-14: never investable afterwards.
    assert not mask.loc["2022-03-01":, "XLNX"].any()
    # But it WAS investable before (no survivorship deletion).
    assert mask.loc["2015-01-01":"2021-06-01", "XLNX"].any()
    # ARM IPO'd 2023-09-14: not investable before IPO + seasoning.
    assert not mask.loc[:"2023-12-31", "ARM"].any()


def _truncate(md, upto: str):
    """A MarketData view ending at `upto` — as if today were that date."""
    import dataclasses
    cut = pd.Timestamp(upto)
    frames = {f: getattr(md, f).loc[:cut] for f in
              ("open", "high", "low", "close", "adj_close", "dollar_volume",
               "macro")}
    fund = md.fundamentals
    if not fund.empty:
        fund = fund[fund["published"] <= cut]
    return dataclasses.replace(md, fundamentals=fund, **frames)


# Every strategy added under freeze v3, with settings cheap enough to build
# twice. These are new signal paths — rolling betas, cross-name correlation
# matrices, breadth over the whole universe, publication-gated growth
# acceleration — and each is a fresh opportunity to read the future by
# accident.
V3_STRATEGIES = [
    ("ResidualMomentum", dict(lookback_days=126, top_n=8)),
    ("MultiHorizonMomentum", dict(top_n=8)),
    ("LowVolatilityTech", dict(top_n=12)),
    ("BreadthGatedBasket", dict(basket="megacap_ai")),
    ("ThemeRotation", dict(top_themes=3)),
    ("FundamentalAcceleration", dict(top_n=8)),
    ("EqualRiskContribution", dict(basket="megacap_ai")),
    ("MinCorrelationSleeve", dict(top_n=10)),
]


# Freeze v4: long-short and hedged constructions. These are the first
# strategies in the study that can hold a negative position, so they exercise
# engine paths (borrow accrual, short marks) nothing else has.
V4_STRATEGIES = [
    ("MarketNeutralMomentum", dict(top_n=10)),
    ("BetaHedgedBasket", dict(basket="megacap_ai")),
    ("GrossProfitability", dict(top_n=12)),
    ("PostEarningsDrift", dict(top_n=10)),
    ("DispersionTimedSelection", dict(top_n=8)),
]


@pytest.mark.parametrize("cls_name,params", V4_STRATEGIES,
                         ids=[c for c, _ in V4_STRATEGIES])
def test_v4_strategies_are_causal(synth_md, cls_name, params):
    """Same guarantee as the v3 set: history ending at D must produce the same
    weights for dates before D as a run that could see everything after."""
    from aitb.strategies import STRATEGY_CLASSES
    cut, compare_to = "2021-06-30", "2021-04-30"
    strat = STRATEGY_CLASSES[cls_name](**params)
    full = strat.build(synth_md).loc[:compare_to]
    trunc = strat.build(_truncate(synth_md, cut)).loc[:compare_to]
    cols = [c for c in full.columns if c in trunc.columns]
    assert cols and full[cols].abs().to_numpy().sum() > 0
    pd.testing.assert_frame_equal(full[cols], trunc[cols], atol=1e-12)


# Freeze v5: designed single-symbol so the Python rule and the chart rule are
# the same rule. ATRTrailingStop carries a hand-rolled state loop, which is
# exactly where a peek at today's price hides.
V5_STRATEGIES = [
    ("ATRTrailingStop", dict(entry_window=100)),
    ("FiftyTwoWeekHighProximity", dict(top_n=10)),
    ("QuietTrend", dict()),
    # Freeze v6. VolumeConfirmedBreakout is the study's only volume signal, so
    # it is the only strategy whose causality depends on a panel other than
    # price; TurnOfMonth reads no market data at all, which makes truncation a
    # weak test of it — see the dedicated test below.
    ("VolumeConfirmedBreakout", dict()),
    ("TurnOfMonth", dict()),
    # Freeze v8. GaussianTrendBands/Hold carry the filter whose CENTRED form
    # would read the future; Supertrend carries a hand-rolled ratchet loop,
    # which is where a peek at today's band hides.
    ("GaussianTrendBands", dict()),
    ("GaussianTrendHold", dict()),
    ("Supertrend", dict()),
    ("ADXTrendStrength", dict()),
    ("RelativeStrengthNewHigh", dict()),
]


@pytest.mark.parametrize("cls_name,params", V5_STRATEGIES,
                         ids=[c for c, _ in V5_STRATEGIES])
def test_v5_strategies_are_causal(synth_md, cls_name, params):
    from aitb.strategies import STRATEGY_CLASSES
    cut, compare_to = "2021-06-30", "2021-04-30"
    strat = STRATEGY_CLASSES[cls_name](**params)
    full = strat.build(synth_md).loc[:compare_to]
    trunc = strat.build(_truncate(synth_md, cut)).loc[:compare_to]
    cols = [c for c in full.columns if c in trunc.columns]
    assert cols and full[cols].abs().to_numpy().sum() > 0
    pd.testing.assert_frame_equal(full[cols], trunc[cols], atol=1e-12)


def test_atr_stop_exits_on_yesterdays_stop_not_todays_low():
    """The trailing stop must use the peak known BEFORE today's bar.

    A stop that updates its peak with today's close and then tests today's
    close against it can never trigger — the position rides every decline to
    the exact bottom and exits nowhere. That is a silent, extremely flattering
    bug, and it shows up as a suspiciously good drawdown rather than an error.
    """
    import numpy as np
    import pandas as pd
    from aitb.config import load_universe_config
    from aitb.data.loader import MarketData
    from aitb.strategies import STRATEGY_CLASSES

    # Real tickers: the investable mask only admits names in the universe.
    cal = pd.bdate_range("2015-01-01", periods=400, name="date")
    path = np.concatenate([np.linspace(100, 300, 300), np.linspace(300, 90, 100)])
    px = pd.DataFrame({"NVDA": path, "MSFT": path}, index=cal)
    md = MarketData(open=px, high=px * 1.01, low=px * 0.99, close=px,
                    adj_close=px,
                    dollar_volume=pd.DataFrame(1e12, index=cal, columns=["NVDA", "MSFT"]),
                    macro=pd.DataFrame(index=cal), fundamentals=pd.DataFrame(),
                    universe=load_universe_config(), provider_name="toy")

    w = STRATEGY_CLASSES["ATRTrailingStop"](entry_window=50, atr_window=22,
                                            atr_mult=3.0).build(md)
    held = w["NVDA"] > 0
    assert held.any(), "never entered a 200-bar uptrend"
    # It must be OUT well before the bottom — a stop that never fires is the bug.
    assert not held.iloc[-1], "still holding at the very bottom: the stop never fired"
    exited = held[held].index.max()
    assert exited < cal[380], f"exited only at {exited.date()} — far too late for a 3-ATR stop"


def _toy_md(px, dollar_volume=None):
    """A two-name MarketData over real tickers, for hand-checkable rules."""
    import pandas as pd
    from aitb.config import load_universe_config
    from aitb.data.loader import MarketData
    cal = px.index
    dv = (dollar_volume if dollar_volume is not None
          else pd.DataFrame(1e12, index=cal, columns=px.columns))
    return MarketData(open=px, high=px * 1.01, low=px * 0.99, close=px,
                      adj_close=px, dollar_volume=dv,
                      macro=pd.DataFrame(index=cal), fundamentals=pd.DataFrame(),
                      universe=load_universe_config(), provider_name="toy")


def test_latch_matches_the_loop_it_replaces():
    """The vectorised hold-until-exit must equal the breakout family's loop.

    The loop is the reference implementation and has been running since v1.
    Replacing it with three chained pandas calls is only safe if the two are
    identical on every session, including the ties the fast version resolves by
    ordering rather than by an explicit branch.
    """
    import numpy as np
    import pandas as pd
    from aitb.strategies.chartable import _latch

    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=400)
    cols = ["NVDA", "MSFT", "AAPL"]
    entry = pd.DataFrame(rng.random((400, 3)) < 0.04, index=idx, columns=cols)
    exit_ = pd.DataFrame(rng.random((400, 3)) < 0.06, index=idx, columns=cols)
    assert (entry & exit_).to_numpy().any(), "no ties — the test proves nothing"

    pos = pd.DataFrame(0.0, index=idx, columns=cols)
    state = entry.iloc[0].astype(float)
    for i in range(len(idx)):
        state = ((state.astype(bool) & ~exit_.iloc[i]) | entry.iloc[i]).astype(float)
        pos.iloc[i] = state

    pd.testing.assert_frame_equal(pos, _latch(entry, exit_))


def test_volume_filter_actually_refuses_thin_breakouts():
    """The turnover filter must change the answer, not decorate it.

    An inert filter is the failure mode that matters here: the strategy would
    silently become a plain breakout, keep its name, and be reported as
    evidence about volume when it tested nothing of the kind. So: the same
    price path twice, differing only in turnover on the breakout day.
    """
    import numpy as np
    import pandas as pd
    from aitb.strategies import STRATEGY_CLASSES

    cal = pd.bdate_range("2015-01-01", periods=200, name="date")
    path = np.concatenate([np.full(150, 100.0), np.linspace(101, 160, 50)])
    px = pd.DataFrame({"NVDA": path, "MSFT": path}, index=cal)

    quiet = pd.DataFrame(1e9, index=cal, columns=px.columns)
    loud = quiet.copy()
    loud.iloc[150:] = 1e11          # 100x its own baseline once the move starts

    strat = STRATEGY_CLASSES["VolumeConfirmedBreakout"]
    thin = strat(entry_window=55, vol_window=50, vol_mult=1.5).build(_toy_md(px, quiet))
    heavy = strat(entry_window=55, vol_window=50, vol_mult=1.5).build(_toy_md(px, loud))

    assert heavy["NVDA"].sum() > 0, "refused a breakout on 100x turnover"
    assert thin["NVDA"].sum() == 0, (
        "took a breakout on flat turnover — the volume filter is inert and the "
        "strategy is a plain breakout wearing a volume label")


def test_turn_of_month_holds_only_inside_its_calendar_window():
    """Exposure must be decided by the calendar and by nothing else.

    This is the one rule in the study that reads no market data, which is its
    entire justification — it cannot be the same bet as everything else. A bug
    that let price leak into the window would destroy that property while
    leaving the results looking perfectly reasonable.
    """
    import numpy as np
    import pandas as pd
    from aitb.strategies import STRATEGY_CLASSES

    cal = pd.bdate_range("2015-01-01", periods=500, name="date")
    rng = np.random.default_rng(3)
    path = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, 500)))
    px = pd.DataFrame({"NVDA": path, "MSFT": path * 1.3}, index=cal)

    w = STRATEGY_CLASSES["TurnOfMonth"](start_day=26, end_day=5).build(_toy_md(px))
    held = w["NVDA"] > 0
    in_window = pd.Series((cal.day >= 26) | (cal.day <= 5), index=cal)
    # Held on every in-window session the mask admits, and on no other.
    assert not (held & ~in_window).any(), "held outside the calendar window"
    assert held.sum() > 100, "held on almost nothing — the window never opened"

    # And the answer must not move when the prices do.
    shuffled = px.sample(frac=1.0, random_state=1).set_axis(cal)
    w2 = STRATEGY_CLASSES["TurnOfMonth"](start_day=26, end_day=5).build(_toy_md(shuffled))
    pd.testing.assert_frame_equal(w, w2)


def test_market_neutral_is_actually_neutral(synth_md):
    """A dollar-neutral book must net to ~zero and carry real short exposure.

    If the short leg silently evaporated — a masking bug, a basket too narrow
    to fill both books — the strategy would quietly become long-only and would
    correlate with the market exactly like everything else, while still being
    labelled market-neutral.
    """
    from aitb.strategies import STRATEGY_CLASSES
    w = STRATEGY_CLASSES["MarketNeutralMomentum"](top_n=10).build(synth_md)
    active = w[w.abs().sum(axis=1) > 1e-9]
    assert len(active) > 500, "market-neutral book almost never traded"
    net = active.sum(axis=1)
    assert net.abs().max() < 1e-9, f"book is not dollar-neutral (max net {net.abs().max()})"
    assert (active < -1e-9).any().any(), "no short positions were ever taken"
    gross = active.abs().sum(axis=1)
    assert gross.max() <= 1.0 + 1e-9, f"gross exposure exceeded 1.0 ({gross.max()})"


def test_beta_hedge_shorts_the_index(synth_md):
    """The hedge leg must be negative and sized by a trailing beta."""
    from aitb.strategies import STRATEGY_CLASSES
    w = STRATEGY_CLASSES["BetaHedgedBasket"](basket="megacap_ai").build(synth_md)
    hedge = w["QQQ"]
    assert (hedge < -1e-9).any(), "hedge leg is never short"
    assert hedge.min() >= -1.5 - 1e-9, "hedge exceeded max_hedge"
    # Before enough history exists to estimate beta there must be no hedge —
    # a default of 1.0 would be a fabricated position.
    assert abs(hedge.iloc[0]) < 1e-12


@pytest.mark.parametrize("cls_name,params", V3_STRATEGIES,
                         ids=[c for c, _ in V3_STRATEGIES])
def test_v3_strategies_are_causal(synth_md, cls_name, params):
    """Weights built with history ending at date D must equal the weights the
    same strategy produced for those dates when it could see everything after.

    The last rebalance period is excluded from the comparison: with data cut
    mid-month the "last trading day of the month" is genuinely a different day,
    so the two runs legitimately rebalance on different dates there. Everything
    before that must match exactly.
    """
    from aitb.strategies import STRATEGY_CLASSES
    cut, compare_to = "2021-06-30", "2021-04-30"
    strat = STRATEGY_CLASSES[cls_name](**params)

    full = strat.build(synth_md).loc[:compare_to]
    trunc = strat.build(_truncate(synth_md, cut)).loc[:compare_to]

    cols = [c for c in full.columns if c in trunc.columns]
    assert cols, f"{cls_name} produced no overlapping columns"
    assert full[cols].abs().to_numpy().sum() > 0, f"{cls_name} never held anything"
    pd.testing.assert_frame_equal(full[cols], trunc[cols], atol=1e-12)


def test_ml_purge_gap():
    """The ML strategy's training window must end `horizon` days before the
    prediction window starts (no overlapping labels across the boundary)."""
    import inspect
    from aitb.strategies import ml
    src = inspect.getsource(ml.MLRankStrategy.build)
    assert "- 1 - horizon" in src  # purge gap present in split arithmetic


def test_gaussian_filter_is_causal_not_a_centred_kernel():
    """The centreline must lag a move, never anticipate it.

    A Gaussian filter in signal processing is a CENTRED kernel: it weights
    bars on both sides of the point it smooths. Applied to prices that reads
    the future — the centreline bends into a reversal before the reversal
    happens, every band entry looks prescient, and the equity curve is
    fiction. It is the most common way a good-looking band script is silently
    wrong, and nothing about the output looks wrong.

    On a unit step, a causal filter is still near zero AT the step and crosses
    halfway some bars later. A centred kernel is already at ~0.5 on the step
    bar, because half its weight sits on bars that have not happened.
    """
    import numpy as np
    import pandas as pd
    from aitb.strategies.chartable import _gaussian

    step = pd.DataFrame({"a": np.r_[np.zeros(300), np.ones(300)]})
    f = _gaussian(step, 40, 4)["a"]

    assert f.iloc[300] < 0.05, (
        f"filter reads {f.iloc[300]:.3f} on the step bar — a causal filter "
        f"cannot know the step happened yet; this kernel is centred")
    lag = int((f >= 0.5).idxmax()) - 300
    assert lag > 0, "filter reached half the step at or before the step bar"
    assert f.iloc[:300].abs().max() < 1e-9, "filter moved BEFORE the step"


def test_supertrend_flips_out_of_a_collapse():
    """The ratchet must fire. A band that never triggers is the flattering bug.

    Supertrend's whole claim is that it tightens into an advance and so exits
    early on the turn. If the flip compares today's close to today's band
    instead of yesterday's, it can ride a collapse to the bottom without ever
    flipping — and that shows up as an excellent drawdown, not as an error.
    """
    import numpy as np
    import pandas as pd
    from aitb.strategies import STRATEGY_CLASSES

    cal = pd.bdate_range("2015-01-01", periods=400, name="date")
    path = np.concatenate([np.linspace(100, 300, 300), np.linspace(300, 90, 100)])
    px = pd.DataFrame({"NVDA": path, "MSFT": path}, index=cal)

    w = STRATEGY_CLASSES["Supertrend"](atr_window=10, atr_mult=3.0).build(
        _toy_md(px))
    held = w["NVDA"] > 0
    assert held.any(), "never went long a 300-bar uptrend"
    assert not held.iloc[-1], "still long at the bottom — the band never flipped"


def test_relative_strength_reads_the_ratio_not_the_price():
    """The signal must come from price/benchmark, not from price alone.

    If the benchmark leg were dropped — a rename, a column that silently
    became NaN — this degrades into a plain breakout while keeping the name
    and the claim. So: hold the stock fixed and move only the benchmark. The
    answer must change.
    """
    import numpy as np
    import pandas as pd
    from aitb.config import load_universe_config
    from aitb.data.loader import MarketData
    from aitb.strategies import STRATEGY_CLASSES

    cal = pd.bdate_range("2015-01-01", periods=600, name="date")
    rng = np.random.default_rng(11)
    stock = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.015, 600)))

    def md_with(bench_path):
        px = pd.DataFrame({"NVDA": stock, "MSFT": stock * 1.1,
                           "QQQ": bench_path}, index=cal)
        return MarketData(open=px, high=px * 1.01, low=px * 0.99, close=px,
                          adj_close=px,
                          dollar_volume=pd.DataFrame(1e12, index=cal,
                                                     columns=px.columns),
                          macro=pd.DataFrame(index=cal),
                          fundamentals=pd.DataFrame(),
                          universe=load_universe_config(), provider_name="toy")

    flat = np.full(600, 100.0)
    strong = 100 * np.exp(np.cumsum(np.full(600, 0.0012)))   # benchmark beats it
    cls = STRATEGY_CLASSES["RelativeStrengthNewHigh"]
    a = cls(bench="QQQ").build(md_with(flat))["NVDA"]
    b = cls(bench="QQQ").build(md_with(strong))["NVDA"]

    assert a.sum() > 0, "never bought a rising stock against a flat benchmark"
    assert b.sum() < a.sum(), (
        "the benchmark got stronger and the rule did not notice — it is "
        "reading price, not relative strength")
