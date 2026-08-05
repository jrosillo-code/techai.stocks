"""Factor strategies added under research freeze v3.

Every idea here exists to attack a SPECIFIC finding of the v2 study, not to
add variety for its own sake. The v2 result that mattered was negative: across
24 strategy families the pairwise correlation of monthly returns was mostly
above 0.85, because almost every variant ended up holding the same handful of
megacap names. They were variations on one bet.

So the question these ask is not "can we score higher" but "can anything in
this sector behave DIFFERENTLY":

  ResidualMomentum       strip out the shared market/tech factor first, then
                         rank on what is left (Blitz, Huij & Martens 2011)
  MultiHorizonMomentum   require agreement across 3/6/12 months rather than
                         fitting one lookback window
  LowVolatilityTech      the defensive end of a famously high-beta sector
  BreadthGatedBasket     participation, not price, as the regime signal
  ThemeRotation          rotate across sub-industries instead of names
  FundamentalAcceleration  second derivative of growth, not its level

All are long-only, use trailing information only, and inherit the engine's
uniform next-open execution. None of them has been run at the time of writing
— the hypotheses above are recorded before results exist, which is the point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import MarketData
from ..features import (momentum, pit_fundamental_panel, realized_vol, sma,
                        xs_zscore)
from ..portfolio import (equal_weight, inverse_vol_weight, rebalance_schedule,
                         top_n_selection)
from ..universe import investable_mask
from .base import Strategy


def _rolling_beta(r: pd.DataFrame, b: pd.Series, window: int) -> pd.DataFrame:
    """Trailing beta of each column to `b` over `window` days.

    cov/var with a common denominator; NaNs propagate rather than being filled,
    so a name with insufficient history simply has no beta yet.
    """
    bm = b.rolling(window).mean()
    var = b.rolling(window).var()
    cov = r.rolling(window).cov(b)
    beta = cov.div(var.replace(0, np.nan), axis=0)
    del bm
    return beta


class ResidualMomentum(Strategy):
    """Momentum measured on returns with the benchmark's influence removed.

    Each name's daily return is regressed on the benchmark over a trailing
    window (rolling beta, no intercept refit — the mean is removed inside the
    residual), and momentum is accumulated on the residual instead of the raw
    return. Two names that both rose 40% because the whole sector rose 40% get
    a residual momentum of roughly zero; a name that rose 40% while the sector
    rose 10% does not.
    """
    family = "factor"
    hypothesis = ("Ranking on benchmark-neutral residual return, rather than "
                  "total return, selects genuinely idiosyncratic winners and "
                  "produces a portfolio less correlated with simply owning the "
                  "sector.")

    def __init__(self, lookback_days: int = 126, skip_days: int = 21,
                 beta_window: int = 252, top_n: int = 8, bench: str = "QQQ",
                 rebalance: str = "ME", basket: str | None = None):
        super().__init__(lookback_days=lookback_days, skip_days=skip_days,
                         beta_window=beta_window, top_n=top_n, bench=bench,
                         rebalance=rebalance, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        px = md.adj_close[mask.columns]
        r = px.pct_change()
        b = md.adj_close[p["bench"]].pct_change()

        beta = _rolling_beta(r, b, p["beta_window"])
        resid = r.sub(beta.mul(b, axis=0))

        # Cumulative residual return over [T-lookback, T-skip]: sum of daily
        # residuals is the right accumulation here (they are excess-of-factor,
        # not a price path), and it keeps the statistic linear in beta.
        cum = resid.rolling(p["lookback_days"] - p["skip_days"]).sum().shift(p["skip_days"])
        # Scale by residual volatility so a high-idio-vol name does not rank
        # first purely for being noisy.
        rvol = resid.rolling(p["lookback_days"]).std().replace(0, np.nan)
        score = (cum / rvol).where(mask)

        sel = top_n_selection(score, p["top_n"]).where(score > 0, 0.0)
        return rebalance_schedule(equal_weight(sel), p["rebalance"])


class MultiHorizonMomentum(Strategy):
    """Hold only names trending on EVERY horizon tested.

    Single-lookback momentum invites exactly the overfitting the study is built
    to resist: 63, 126 and 252 days give different answers and whichever
    backtests best gets reported. Requiring agreement removes the choice.
    """
    family = "factor"
    hypothesis = ("A name trending on 3-, 6- and 12-month horizons "
                  "simultaneously is in a genuine product cycle rather than a "
                  "one-window artifact; demanding consensus should cut turnover "
                  "and whipsaw at the cost of missing early entries.")

    def __init__(self, horizons: str = "63,126,252", top_n: int = 8,
                 min_agree: int = 3, weighting: str = "equal",
                 rebalance: str = "ME", basket: str | None = None):
        super().__init__(horizons=horizons, top_n=top_n, min_agree=min_agree,
                         weighting=weighting, rebalance=rebalance, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        px = md.adj_close[mask.columns]

        horizons = [int(h) for h in str(p["horizons"]).split(",")]
        agree = None
        rank_sum = None
        for h in horizons:
            m = momentum(px, h, skip_days=21)
            positive = (m > 0).astype(float).where(m.notna())
            agree = positive if agree is None else agree.add(positive)
            z = xs_zscore(m, mask)
            rank_sum = z if rank_sum is None else rank_sum.add(z)

        eligible = (agree >= p["min_agree"]).where(mask, False)
        score = rank_sum.where(eligible)
        sel = top_n_selection(score, p["top_n"]).where(eligible, 0.0)
        if p["weighting"] == "inverse_vol":
            w = inverse_vol_weight(px, sel)
        else:
            w = equal_weight(sel)
        return rebalance_schedule(w, p["rebalance"])


class LowVolatilityTech(Strategy):
    """Own the calmest names in a violent sector.

    A deliberate falsification target. The low-volatility anomaly is one of the
    most replicated results in equities, but technology is where high beta has
    historically been rewarded — if the anomaly does not survive here, that is
    a finding worth recording, not a failure to hide.
    """
    family = "factor"
    hypothesis = ("Within technology, the lowest-volatility names deliver "
                  "better risk-adjusted returns than the cap-weighted sector, "
                  "even though they give up raw return. Expected to LOSE on "
                  "CAGR and win on Sharpe and drawdown — or to fail outright, "
                  "which is equally informative.")

    def __init__(self, vol_window: int = 126, top_n: int = 12,
                 trend_filter: bool = False, rebalance: str = "ME",
                 basket: str | None = None):
        super().__init__(vol_window=vol_window, top_n=top_n,
                         trend_filter=trend_filter, rebalance=rebalance,
                         basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        px = md.adj_close[mask.columns]

        vol = realized_vol(px, p["vol_window"])
        # Rank ascending: lowest vol scores highest.
        score = (-vol).where(mask)
        if p["trend_filter"]:
            score = score.where(px > sma(px, 200))
        sel = top_n_selection(score, p["top_n"])
        return rebalance_schedule(equal_weight(sel), p["rebalance"])


class BreadthGatedBasket(Strategy):
    """Hold the basket while the sector's PARTICIPATION is healthy.

    Every existing regime filter in the study keys off one index's price. That
    fails the same way twice: a handful of megacaps can hold an index above its
    moving average while the other ninety names are already in a bear market.
    Breadth — the share of the universe above its own 200-day average — sees
    that; the index price does not.
    """
    family = "factor"
    hypothesis = ("Deteriorating breadth precedes index-level drawdowns, so a "
                  "breadth gate should exit earlier and with fewer whipsaws "
                  "than a price-trend gate on the index itself.")

    def __init__(self, basket: str = "megacap_ai", breadth_window: int = 200,
                 threshold: float = 0.45, smooth_days: int = 21,
                 min_exposure: float = 0.0, fallback: str = "IEF",
                 rebalance: str = "W-FRI"):
        super().__init__(basket=basket, breadth_window=breadth_window,
                         threshold=threshold, smooth_days=smooth_days,
                         min_exposure=min_exposure, fallback=fallback,
                         rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        # Breadth is measured across the WHOLE universe, not the held basket —
        # the point is to use information the basket itself does not contain.
        full = investable_mask(md)
        px_all = md.adj_close[full.columns]
        above = (px_all > sma(px_all, p["breadth_window"])).astype(float)
        breadth = above.where(px_all.notna()).mean(axis=1)
        breadth = breadth.rolling(p["smooth_days"], min_periods=1).mean()

        mask = investable_mask(md, md.universe.baskets[p["basket"]])
        w = equal_weight(mask)
        # Before enough history exists to measure breadth, stay invested rather
        # than sitting in cash on a NaN — the alternative silently converts
        # "unknown" into a market call.
        gate = (breadth >= p["threshold"]).astype(float)
        gate = gate.where(breadth.notna(), 1.0)
        gate = gate.clip(lower=p["min_exposure"])

        w = w.mul(gate, axis=0)
        fb = p["fallback"]
        if fb in md.adj_close.columns:
            w[fb] = ((1 - w.sum(axis=1)).clip(lower=0.0)
                     * md.adj_close[fb].notna()).fillna(0.0)
        return rebalance_schedule(w, p["rebalance"])


class ThemeRotation(Strategy):
    """Rotate across sub-industries, holding every name in the winners.

    Selection happens one level up from the stock: rank the theme baskets
    (semis, semi-cap equipment, EDA, cloud, cyber, networking, robotics, AI
    power) by their own equal-weighted momentum, then hold the top themes
    equally weighted internally. Stock-level idiosyncratic risk is diversified
    away on purpose — the bet is entirely on which part of the sector leads.
    """
    family = "factor"
    hypothesis = ("Capital-spending cycles rotate between layers of the "
                  "technology stack with enough persistence to trade at the "
                  "theme level, and doing so avoids the single-name "
                  "concentration that penalises stock-level momentum.")

    _THEMES = ["ai_compute", "semi_equipment", "eda_tools", "cloud_platforms",
               "enterprise_ai", "cybersecurity", "networking",
               "dc_infrastructure", "robotics", "ai_power",
               "internet_platforms"]

    def __init__(self, lookback_days: int = 126, top_themes: int = 3,
                 rebalance: str = "ME"):
        super().__init__(lookback_days=lookback_days, top_themes=top_themes,
                         rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        themes = [t for t in self._THEMES if t in md.universe.baskets]
        masks = {t: investable_mask(md, md.universe.baskets[t]) for t in themes}

        # Each theme's own equal-weighted total-return index, built only from
        # names investable on each date.
        theme_px = {}
        for t, m in masks.items():
            if m.empty or not m.any().any():
                continue
            r = md.adj_close[m.columns].pct_change().where(m)
            theme_px[t] = (1 + r.mean(axis=1).fillna(0.0)).cumprod()
        if not theme_px:
            raise ValueError("ThemeRotation: no themes have investable members")

        tp = pd.DataFrame(theme_px)
        tmom = momentum(tp, p["lookback_days"], skip_days=21)
        sel_themes = top_n_selection(tmom, p["top_themes"]).where(tmom > 0, 0.0)

        w = pd.DataFrame(0.0, index=md.calendar, columns=md.adj_close.columns)
        share = sel_themes.div(sel_themes.sum(axis=1).replace(0, np.nan), axis=0)
        for t in tp.columns:
            m = masks[t]
            inner = equal_weight(m)                      # equal within theme
            w[m.columns] = w[m.columns].add(
                inner.mul(share[t].reindex(inner.index).fillna(0.0), axis=0),
                fill_value=0.0)
        return rebalance_schedule(w.fillna(0.0), p["rebalance"])


class FundamentalAcceleration(Strategy):
    """Rank on the CHANGE in revenue growth, not its level.

    Uses the point-in-time panel, so a quarter enters only on its publication
    date. Growth levels are widely known and largely in the price; the second
    derivative — growth that is itself speeding up — is the part that tends to
    surprise. This is the fundamental analogue of momentum, and it inherits
    momentum's weakness: it turns hardest exactly when cycles peak.
    """
    family = "factor"
    hypothesis = ("Acceleration in published revenue growth precedes price "
                  "revaluation more reliably than the level of growth, which is "
                  "already discounted.")

    def __init__(self, top_n: int = 8, rebalance: str = "QE",
                 require_positive_growth: bool = True,
                 basket: str | None = None):
        super().__init__(top_n=top_n, rebalance=rebalance,
                         require_positive_growth=require_positive_growth,
                         basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        if md.fundamentals.empty:
            raise ValueError("FundamentalAcceleration requires point-in-time "
                             "fundamentals; none are present in this data mode")

        accel = pit_fundamental_panel(md, "revenue", "accel")
        growth = pit_fundamental_panel(md, "revenue", "yoy_growth")
        cols = [c for c in accel.columns if c in mask.columns]
        accel, growth = accel[cols], growth[cols]

        score = accel.where(mask[cols])
        if p["require_positive_growth"]:
            score = score.where(growth > 0)
        sel = top_n_selection(score, p["top_n"]).where(score > 0, 0.0)
        w = equal_weight(sel).reindex(columns=mask.columns, fill_value=0.0)
        return rebalance_schedule(w, p["rebalance"])
