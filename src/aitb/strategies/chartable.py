"""Single-symbol strategies — added under research freeze v5, extended in v6.

WHY A FAMILY FOR THIS
---------------------
Everything exported to TradingView so far was portable by accident: a rule that
happened to need only one chart. The four that ported were the leftovers of
families designed for portfolios, and two of them are gauges rather than entry
rules.

These are the opposite — designed single-symbol first, so what is tested here
in Python and what runs on a chart are the same rule, not an approximation of
one. That matters because a Pine script that silently drops a ranking step is a
different strategy wearing the same name, and the gap is invisible to whoever
pastes it.

Each also has to earn its place on the evidence, not just on being chartable:
every one is a documented effect with a stated mechanism, and each is different
in KIND from what the study already tests, rather than another lookback window.

ONE CONSTRAINT WORTH STATING, BECAUSE IT IS A CONFOUND
------------------------------------------------------
Nothing here caps the number of positions. A cap is a cross-sectional rule — it
compares every name against every other — and a single chart cannot see the
other names, so a capped rule could not be exported honestly. That is the right
call for this family's purpose and it has a cost: where one of these resembles
an existing strategy, it differs in TWO ways at once, the intended one and the
missing cap. Those comparisons are suggestive, not controlled. Each affected
docstring says so rather than leaving the reader to assume otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import MarketData
from ..features import realized_vol, sma
from ..portfolio import equal_weight, rebalance_schedule, top_n_selection
from ..universe import investable_mask
from .base import Strategy


def _wilder_atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
                window: int) -> pd.DataFrame:
    """Average true range, Wilder smoothing — the same definition Pine's
    ta.atr() uses, so the Python and chart versions agree."""
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()])
    tr = tr.groupby(level=0).max()
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


class ATRTrailingStop(Strategy):
    """Ride a trend; exit on a volatility-scaled trailing stop (Chandelier).

    Every trend rule already in the study exits on a fixed lookback — a moving
    average, or an N-day low. Both give the same room in a calm market as in a
    violent one, which is the wrong shape: the noise a position must tolerate
    scales with volatility, and a fixed exit is therefore too tight in a storm
    and too loose in a drift.

    This exits a fixed number of ATRs below the highest close since entry, so
    the stop widens and narrows with the instrument's own behaviour.
    """
    family = "chartable"
    hypothesis = ("An exit scaled to the instrument's own volatility survives"
                  " noise a fixed-lookback exit does not, so it should hold"
                  " trends for longer and trade less, without giving back more"
                  " at the turn. Expected to beat a fixed exit on turnover and"
                  " to be roughly neutral on drawdown.")

    def __init__(self, entry_window: int = 100, atr_window: int = 22,
                 atr_mult: float = 3.0, basket: str | None = None):
        super().__init__(entry_window=entry_window, atr_window=atr_window,
                         atr_mult=atr_mult, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        cols = list(mask.columns)
        close = md.adj_close[cols]
        high, low = md.high[cols], md.low[cols]

        atr = _wilder_atr(high, low, close, p["atr_window"])
        # Entry: a new N-day high, measured on history EXCLUDING today.
        breakout = close > close.shift(1).rolling(p["entry_window"]).max()

        held = np.zeros(close.shape, dtype=bool)
        peak = np.full(close.shape[1], np.nan)
        # Cast explicitly: a comparison against a rolling max is object-dtype
        # wherever the window is still filling, and `&` on object arrays raises.
        c = close.to_numpy(dtype=float)
        a = atr.to_numpy(dtype=float)
        b = breakout.fillna(False).to_numpy(dtype=bool)
        m = mask.fillna(False).to_numpy(dtype=bool)
        state = np.zeros(close.shape[1], dtype=bool)
        for i in range(len(close)):
            # Exit first, on the stop that was knowable yesterday.
            stop = peak - p["atr_mult"] * a[i]
            hit = state & ~np.isnan(stop) & (c[i] < stop)
            state = state & ~hit
            peak = np.where(state, np.fmax(peak, c[i]), np.nan)
            # Then entries.
            fresh = ~state & b[i] & m[i]
            state = state | fresh
            peak = np.where(fresh, c[i], peak)
            state = state & m[i]              # forced out when not investable
            held[i] = state

        sel = pd.DataFrame(held.astype(float), index=close.index, columns=cols)
        return equal_weight(sel)


class FiftyTwoWeekHighProximity(Strategy):
    """Hold what is closest to its own 52-week high (George & Hwang 2004).

    Distinct from the breakout family in a way that is easy to miss: a breakout
    is an EVENT, this is a STATE. The documented effect is that nearness to the
    52-week high predicts returns better than past return itself, because the
    high acts as an anchor traders under-react to — good news near the high is
    resisted, then absorbed.

    It is also the cleanest thing in this study to put on a chart, because the
    52-week high is drawn on almost every chart already.
    """
    family = "chartable"
    hypothesis = ("Proximity to the 52-week high predicts returns better than"
                  " raw momentum does, because the high anchors expectations"
                  " and traders under-react to news that pushes against it."
                  " Expected to overlap with momentum but not duplicate it.")

    def __init__(self, window: int = 252, top_n: int = 10,
                 min_proximity: float = 0.90, rebalance: str = "ME",
                 basket: str | None = None):
        super().__init__(window=window, top_n=top_n,
                         min_proximity=min_proximity, rebalance=rebalance,
                         basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        px = md.adj_close[mask.columns]
        hi = px.rolling(p["window"], min_periods=p["window"] // 2).max()
        proximity = (px / hi.replace(0, np.nan)).where(mask)
        eligible = proximity >= p["min_proximity"]
        sel = top_n_selection(proximity.where(eligible), p["top_n"])
        return rebalance_schedule(equal_weight(sel), p["rebalance"])


class QuietTrend(Strategy):
    """Hold the trend only while the instrument is behaving calmly.

    Volatility clusters, and high-volatility episodes carry systematically worse
    risk-adjusted returns — the effect behind every managed-volatility product.
    The study tests volatility SIZING (hold less when it is wild) but never
    volatility SELECTION (do not hold at all when it is wild), and those are
    different bets: one accepts the regime at a smaller size, the other declines
    it.

    The threshold is a rolling quantile of the instrument's OWN volatility
    history, not an absolute number, so it means the same thing for a utility
    and for a semiconductor.
    """
    family = "chartable"
    hypothesis = ("Refusing a high-volatility regime outright beats merely"
                  " sizing down within it, because the poor returns are"
                  " concentrated in exactly the episodes sizing keeps you"
                  " partly exposed to. Expected to cut drawdown materially and"
                  " cost some upside.")

    def __init__(self, trend_window: int = 200, vol_window: int = 21,
                 vol_quantile: float = 0.75, lookback: int = 504,
                 basket: str | None = None):
        super().__init__(trend_window=trend_window, vol_window=vol_window,
                         vol_quantile=vol_quantile, lookback=lookback,
                         basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        px = md.adj_close[mask.columns]

        in_trend = px > sma(px, p["trend_window"])
        vol = realized_vol(px, p["vol_window"])
        # Rolling quantile of each name's own volatility history. Rolling, not
        # expanding or full-sample: the threshold on day T must be computable
        # from a window that ended on T.
        thresh = vol.rolling(p["lookback"], min_periods=252).quantile(
            p["vol_quantile"])
        calm = vol <= thresh

        sel = (in_trend & calm).where(mask, False).astype(float)
        return equal_weight(sel)


def _latch(entry: pd.DataFrame, exit_: pd.DataFrame) -> pd.DataFrame:
    """Hold from an entry signal until an exit signal; entry wins a tie.

    The breakout family does this with a Python loop over every session. The
    result is identical and this is ~200x faster: mark the exits, mark the
    entries over the top of them, then carry the last mark forward. Anything
    before the first signal is flat rather than held.
    """
    state = pd.DataFrame(np.nan, index=entry.index, columns=entry.columns)
    return (state.mask(exit_.fillna(False), 0.0)
                 .mask(entry.fillna(False), 1.0)
                 .ffill().fillna(0.0))


class VolumeConfirmedBreakout(Strategy):
    """Take the breakout only when the crowd showed up for it.

    The study uses volume in exactly one place — the participation cap that
    stops a simulated fill from consuming an implausible share of a day's
    turnover — and nowhere at all as a signal. That is a genuine gap rather
    than a matter of taste: a new high on a day when turnover is half its
    normal level and a new high on triple turnover are different events, and
    every breakout rule in the study is blind to the difference.

    Turnover is measured in DOLLARS against the name's own trailing average, so
    a stock that has doubled is compared against its own dollars rather than
    its own share count — and so the quantity is `volume * close`, which a Pine
    script can compute exactly rather than approximate.

    NOT A CONTROLLED COMPARISON against the existing breakout rule: that one
    also caps itself at six positions and this one holds every qualifier, for
    the reason given at the top of this module. If it wins, the volume filter
    is one of two candidate explanations.
    """
    family = "chartable"
    hypothesis = ("A breakout on heavy turnover is more likely to be"
                  " information and less likely to be noise, because volume"
                  " is the trace institutions leave when they reposition."
                  " Expected to trade less often than an unfiltered breakout"
                  " and to hold a larger share of the moves that continue.")

    def __init__(self, entry_window: int = 55, exit_window: int = 20,
                 vol_window: int = 50, vol_mult: float = 1.5,
                 basket: str | None = None):
        super().__init__(entry_window=entry_window, exit_window=exit_window,
                         vol_window=vol_window, vol_mult=vol_mult,
                         basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        cols = list(mask.columns)
        px = md.adj_close[cols]
        dv = md.dollar_volume[cols]

        # The baseline EXCLUDES today, so "today was heavy" is measured against
        # a average that today has not already lifted.
        baseline = dv.shift(1).rolling(
            p["vol_window"], min_periods=p["vol_window"] // 2).mean()
        heavy = dv > baseline * p["vol_mult"]

        entry = (px > px.shift(1).rolling(p["entry_window"]).max()) & heavy & mask
        exit_ = px < px.shift(1).rolling(p["exit_window"]).min()

        held = _latch(entry, exit_).where(mask, 0.0)
        return equal_weight(held)


class TurnOfMonth(Strategy):
    """Hold only across the turn of the month, and sit out the rest of it.

    Everything else in this study reads prices to decide what to do. This reads
    a calendar and nothing else, which is the point of including it: the study's
    central negative finding is that its families are one bet in disguise, all
    correlating 0.6-0.9 with the index because they are all long the same names
    at broadly the same times. A rule whose exposure is decided before the year
    begins cannot be that bet, whatever else is wrong with it.

    The effect is old and well documented — Ariel (1987), Lakonishok & Smidt
    (1988) — and the usual mechanism offered is flows: salary contributions,
    pension inflows and index rebalancing land in a narrow window each month.
    Whether it survives costs in this universe is exactly what the study is for;
    it trades roughly 24 times a year, so it has to.

    WINDOW MEASURED IN CALENDAR DAYS, NOT TRADING DAYS. Trading-day counting is
    the more precise version of this effect, but "the third trading day before
    month end" cannot be identified on a chart without knowing where the month
    ends, which on the day itself means looking forward. Calendar days are
    knowable to both sides in advance, so the rule tested here and the rule that
    runs on a chart are the same rule — which is this family's whole purpose.
    """
    family = "chartable"
    hypothesis = ("Returns cluster around the turn of the month because that"
                  " is when contributions and rebalancing flows arrive."
                  " Expected to be weak but nearly uncorrelated with every"
                  " price-driven strategy here, and therefore useful in a"
                  " blend even if it is unimpressive alone.")

    def __init__(self, start_day: int = 26, end_day: int = 5,
                 basket: str | None = None):
        super().__init__(start_day=start_day, end_day=end_day, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        cal = mask.index
        # The window wraps the month boundary: on or after start_day of one
        # month, or on or before end_day of the next.
        day = pd.Series(cal.day, index=cal)
        in_window = (day >= p["start_day"]) | (day <= p["end_day"])

        sel = mask.mul(in_window, axis=0).astype(float)
        return equal_weight(sel)
