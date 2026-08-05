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
