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


def _gaussian(px: pd.DataFrame, period: int, poles: int = 4) -> pd.DataFrame:
    """Ehlers' N-pole Gaussian filter — smoother than an EMA at equal lag.

    CAUSALITY, WHICH IS THE WHOLE POINT HERE. A "Gaussian filter" in signal
    processing is a CENTRED kernel: it weights bars on both sides of the point
    it is smoothing. Applied to a price series that reads the future, and it is
    the most common way a good-looking band script is silently wrong — the
    centreline bends into a reversal before the reversal happens, so entries
    look prescient and the equity curve is fiction.

    This is Ehlers' formulation instead: a single-pole recursion applied
    `poles` times in series. Cascading N one-pole filters approximates a
    Gaussian response by the central limit theorem while only ever reading
    bars that have closed. `alpha` is solved so the cascade's cutoff lands at
    `period`, which is why it is not simply 2/(period+1).

        beta  = (1 - cos(2*pi/period)) / (2^(1/poles) - 1)
        alpha = -beta + sqrt(beta^2 + 2*beta)

    ewm(adjust=False) IS that recursion, and it seeds from the first
    observation — the Pine version seeds the same way rather than from zero, so
    the two agree bar for bar instead of only after the transient decays.
    """
    beta = (1.0 - np.cos(2.0 * np.pi / period)) / (2.0 ** (1.0 / poles) - 1.0)
    alpha = -beta + np.sqrt(beta * beta + 2.0 * beta)
    out = px
    for _ in range(poles):
        out = out.ewm(alpha=alpha, adjust=False).mean()
    return out


class GaussianTrendBands(Strategy):
    """Buy the volatility-band break, sell back to the centreline.

    Built for names that move violently — the AI and semiconductor complex —
    where the cost of being slow out is larger than the cost of being early in.
    Entry demands a close ABOVE the upper band, which is a genuine volatility
    expansion rather than a drift over an average. The exit is deliberately
    tight: back to the centreline, not to the lower band, because in a name
    that can lose a third of its value in a month the round trip from the top
    band to the bottom band is most of the move.

    Paired with GaussianTrendHold, which is the same machinery parameterised
    the opposite way. Neither is claimed to suit any particular universe — both
    are run over the AI baskets AND the broad tech roster, and the study says
    which fits where. That comparison is the reason they exist as a pair.
    """
    family = "chartable"
    hypothesis = ("In high-volatility names a band break is information and a"
                  " drift above an average is not, so demanding the break and"
                  " exiting fast at the centreline should keep more of each"
                  " advance than a trend filter does. Expected to trade more"
                  " often than a trend rule and to cut drawdown materially.")

    def __init__(self, period: int = 40, poles: int = 4, atr_window: int = 22,
                 band_mult: float = 1.5, exit_mult: float = 0.0,
                 min_hold: int = 0, basket: str | None = None):
        super().__init__(period=period, poles=poles, atr_window=atr_window,
                         band_mult=band_mult, exit_mult=exit_mult,
                         min_hold=min_hold, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        cols = list(mask.columns)
        close = md.adj_close[cols]

        centre = _gaussian(close, p["period"], p["poles"])
        atr = _wilder_atr(md.high[cols], md.low[cols], close, p["atr_window"])
        upper = centre + p["band_mult"] * atr

        # v9: the exit sits exit_mult ATRs BELOW the centreline, opening a dead
        # band between entry and exit. exit_mult=0 is the v8 rule exactly and is
        # kept in the grid as the control arm.
        exit_level = centre - p["exit_mult"] * atr
        entry = (close > upper) & mask
        exit_ = close < exit_level
        held = _latch(entry, exit_, int(p["min_hold"]))
        return equal_weight(held.where(mask, 0.0))


class GaussianTrendHold(Strategy):
    """Hold the trend while it holds; sell only when the band breaks down.

    The mirror image of GaussianTrendBands, and built for steadier compounding
    names — broad large-cap technology — where the expensive mistake is being
    shaken out of a trend that was intact. Entry is permissive: a close above a
    RISING centreline, no breakout required. The exit is wide: a close below
    the lower band, which in a calm name is a long way down.

    The centreline must be rising, measured over `confirm_days`, not merely
    above where it was yesterday. A filter that reacts to one day of slope is a
    filter that whipsaws, which is the failure this parameterisation exists to
    avoid.
    """
    family = "chartable"
    hypothesis = ("In steadier names the cost of a false exit exceeds the cost"
                  " of a late one, so a permissive entry with a wide"
                  " volatility stop should compound better than a rule that"
                  " demands confirmation to get in and gives it up quickly."
                  " Expected to trade far less and to hold larger drawdowns.")

    def __init__(self, period: int = 60, poles: int = 4, atr_window: int = 22,
                 band_mult: float = 2.5, confirm_days: int = 5,
                 min_hold: int = 0, basket: str | None = None):
        super().__init__(period=period, poles=poles, atr_window=atr_window,
                         band_mult=band_mult, confirm_days=confirm_days,
                         min_hold=min_hold, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        cols = list(mask.columns)
        close = md.adj_close[cols]

        centre = _gaussian(close, p["period"], p["poles"])
        atr = _wilder_atr(md.high[cols], md.low[cols], close, p["atr_window"])
        lower = centre - p["band_mult"] * atr

        rising = centre > centre.shift(p["confirm_days"])
        entry = (close > centre) & rising & mask
        exit_ = close < lower
        held = _latch(entry, exit_, int(p["min_hold"]))
        return equal_weight(held.where(mask, 0.0))


def _wilder_smooth(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Wilder's smoothing — the recursion behind ATR, DI and ADX alike."""
    return x.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


class Supertrend(Strategy):
    """The band that ratchets: flip long on a close above it, out on a break.

    The most widely used trend flip on TradingView, and absent from this study
    until now. It differs from the Chandelier stop already here in two ways
    that matter: the basis is the bar's midpoint rather than the close, and the
    band RATCHETS — once it has moved up it never moves back down while the
    trend holds, so it tightens into an advance instead of breathing with every
    volatility spike.

    Included partly because it is ubiquitous. A rule that thousands of people
    trade is worth measuring honestly rather than dismissing, and the study is
    the only way to find out whether the ubiquity is earned.
    """
    family = "chartable"
    hypothesis = ("A ratcheting mid-price band gives back less at the turn"
                  " than a stop that re-widens on every volatility spike,"
                  " because it never loosens while the trend is intact."
                  " Expected to beat the Chandelier stop on drawdown and to"
                  " trade somewhat more.")

    def __init__(self, atr_window: int = 10, atr_mult: float = 3.0,
                 min_hold: int = 0, basket: str | None = None):
        super().__init__(atr_window=atr_window, atr_mult=atr_mult,
                         min_hold=min_hold, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        cols = list(mask.columns)
        close = md.adj_close[cols]
        high, low = md.high[cols], md.low[cols]

        atr = _wilder_atr(high, low, close, p["atr_window"])
        hl2 = (high + low) / 2.0
        raw_up = (hl2 - p["atr_mult"] * atr).to_numpy(dtype=float)
        raw_dn = (hl2 + p["atr_mult"] * atr).to_numpy(dtype=float)
        c = close.to_numpy(dtype=float)
        m = mask.fillna(False).to_numpy(dtype=bool)

        n, k = c.shape
        up = np.full(k, np.nan)
        dn = np.full(k, np.nan)
        trend = np.zeros(k, dtype=bool)          # True = long
        age = np.zeros(k, dtype=int)             # sessions since the flip up
        min_hold = int(p["min_hold"])            # v9; 0 == the v8 rule exactly
        held = np.zeros((n, k), dtype=bool)
        for i in range(n):
            pu, pd_ = up.copy(), dn.copy()
            up = raw_up[i]
            dn = raw_dn[i]
            if i:
                # Ratchet: the band only tightens while the side holds.
                keep_u = ~np.isnan(pu) & (c[i - 1] > pu)
                up = np.where(keep_u, np.fmax(up, pu), up)
                keep_d = ~np.isnan(pd_) & (c[i - 1] < pd_)
                dn = np.where(keep_d, np.fmin(dn, pd_), dn)
                # Flip on yesterday's band, which is the only one that was known.
                age = np.where(trend, age + 1, age)
                up_flip = ~np.isnan(pd_) & (c[i] > pd_)
                dn_flip = (~np.isnan(pu) & (c[i] < pu) & trend
                           & (age >= min_hold))
                was = trend
                trend = np.where(up_flip, True, np.where(dn_flip, False, trend))
                age = np.where(trend & ~was, 0, age)
            held[i] = trend & m[i]

        sel = pd.DataFrame(held.astype(float), index=close.index, columns=cols)
        return equal_weight(sel)


class ADXTrendStrength(Strategy):
    """Trade direction only when the trend is strong enough to be worth it.

    Every trend rule in this study measures DIRECTION — is price above an
    average, is it making new highs. None measures STRENGTH: whether there is a
    trend at all, as opposed to a drift that happens to be pointing up. Those
    are different questions, and Wilder's ADX is the standard answer to the
    second one.

    The distinction has teeth in a chop: a moving-average rule is fully
    invested in a sideways market that keeps crossing its own average, which is
    where trend following does its worst damage. This one declines to
    participate until ADX says the move has structure.
    """
    family = "chartable"
    hypothesis = ("Requiring trend STRENGTH as well as direction should cut"
                  " the whipsaw losses that dominate trend following in"
                  " sideways markets, at the cost of entering late in the"
                  " moves that do work. Expected to trade much less than a"
                  " moving-average rule with a better win rate.")

    def __init__(self, di_window: int = 14, adx_window: int = 14,
                 adx_min: float = 20.0, adx_exit: float | None = None,
                 min_hold: int = 0, basket: str | None = None):
        super().__init__(di_window=di_window, adx_window=adx_window,
                         adx_min=adx_min, adx_exit=adx_exit,
                         min_hold=min_hold, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        cols = list(mask.columns)
        close, high, low = md.adj_close[cols], md.high[cols], md.low[cols]

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        atr = _wilder_atr(high, low, close, p["di_window"]).replace(0, np.nan)
        plus_di = 100.0 * _wilder_smooth(plus_dm, p["di_window"]) / atr
        minus_di = 100.0 * _wilder_smooth(minus_dm, p["di_window"]) / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = _wilder_smooth(dx, p["adx_window"])

        # v9: enter above adx_min, leave only below adx_exit. A single
        # threshold makes the rule flip every time ADX jitters across it, and
        # ADX jitters constantly — this measured 50x annual turnover in v8,
        # the worst in the study bar the deprecated unfiltered breakout.
        # adx_exit=None means "same as adx_min", i.e. the v8 rule exactly, and
        # is kept in the grid as the control arm.
        exit_floor = p["adx_min"] if p["adx_exit"] is None else p["adx_exit"]
        directional = plus_di > minus_di
        entry = directional & (adx >= p["adx_min"]) & mask
        exit_ = ~directional | (adx < exit_floor)
        held = _latch(entry, exit_, int(p["min_hold"]))
        return equal_weight(held.where(mask, 0.0))


class RelativeStrengthNewHigh(Strategy):
    """Own what is beating the index, and sell it when it stops.

    Cross-sectional strength is the thing this study most wants to test and
    least able to put on a chart: ranking 81 names against each other needs 81
    charts. Strength against a single BENCHMARK is the exception — it is one
    ratio line, and TradingView draws it from one extra `request.security`
    call. So this is the one genuinely relative rule in the chartable family.

    The signal is the ratio price/benchmark making a new N-day high, not the
    price doing so. A stock at a new high in a market that is also at a new
    high has told you nothing; a stock whose RATIO is at a new high is being
    accumulated relative to everything else. This is the O'Neil / Minervini
    relative-strength line, and it is the closest a single chart gets to the
    cross-sectional momentum the rest of the study relies on.
    """
    family = "chartable"
    hypothesis = ("Strength measured against the index carries information"
                  " that strength measured against the stock's own past does"
                  " not, because it separates the name from the market it sits"
                  " in. Expected to overlap with momentum but to hold up"
                  " better when the whole index is rising.")

    def __init__(self, bench: str = "QQQ", entry_window: int = 63,
                 exit_window: int = 21, trend_window: int = 200,
                 basket: str | None = None):
        super().__init__(bench=bench, entry_window=entry_window,
                         exit_window=exit_window, trend_window=trend_window,
                         basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        if p["bench"] not in md.adj_close.columns:
            raise ValueError(
                f"RelativeStrengthNewHigh needs the benchmark {p['bench']!r}, "
                f"which is not in the loaded panel")
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        cols = list(mask.columns)
        px = md.adj_close[cols]
        bench = md.adj_close[p["bench"]]

        rs = px.div(bench.replace(0, np.nan), axis=0)
        # New highs measured on history EXCLUDING today, so today's close is
        # compared against a level that was knowable before it printed.
        rs_hi = rs.shift(1).rolling(p["entry_window"]).max()
        rs_lo = rs.shift(1).rolling(p["exit_window"]).min()
        # A rising ratio in a falling stock is still a falling stock.
        in_trend = px > sma(px, p["trend_window"])

        entry = (rs > rs_hi) & in_trend & mask
        exit_ = rs < rs_lo
        return equal_weight(_latch(entry, exit_).where(mask, 0.0))


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


def _latch(entry: pd.DataFrame, exit_: pd.DataFrame,
           min_hold: int = 0) -> pd.DataFrame:
    """Hold from an entry signal until an exit signal; entry wins a tie.

    The breakout family does this with a Python loop over every session. The
    vectorised form below is identical and ~200x faster: mark the exits, mark
    the entries over the top of them, then carry the last mark forward.
    Anything before the first signal is flat rather than held.

    ``min_hold`` forbids exiting for N sessions after entry. Added under freeze
    v9 against a measured failure, not a hunch: the v8 chart rules turned over
    35-50x a year against 7.3x for the plain trend filter, because a rule with
    a tight exit that behaves sensibly on ONE chart becomes a churn engine when
    81 names each flip independently. A floor on holding length is the blunter
    of the two standard cures for that (the other is a dead band on the exit).

    min_hold=0 reproduces the pre-v9 behaviour EXACTLY and is kept in the grid
    as the control arm, so the cure's effect is measured rather than assumed.
    """
    if min_hold <= 0:
        state = pd.DataFrame(np.nan, index=entry.index, columns=entry.columns)
        return (state.mask(exit_.fillna(False), 0.0)
                     .mask(entry.fillna(False), 1.0)
                     .ffill().fillna(0.0))

    en = entry.fillna(False).to_numpy(dtype=bool)
    ex = exit_.fillna(False).to_numpy(dtype=bool)
    n, k = en.shape
    out = np.zeros((n, k), dtype=float)
    state = np.zeros(k, dtype=bool)
    age = np.zeros(k, dtype=int)
    for i in range(n):
        age = np.where(state, age + 1, age)
        # Entry still wins a tie, exactly as the vectorised form does.
        fresh = en[i] & ~state
        # Exit only where the position has been open long enough.
        gone = state & ex[i] & ~en[i] & (age >= min_hold)
        state = (state | fresh) & ~gone
        age = np.where(fresh, 0, age)
        out[i] = state
    return pd.DataFrame(out, index=entry.index, columns=entry.columns)


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


class ShortSqueezeCandidate(Strategy):
    """Buy strength that is being fought; sell when the fight is over.

    The study's first signal that reads neither price, nor volume, nor a
    filing. FINRA publishes the share of each session's tape that was sold
    short — the closest free read on POSITIONING rather than on what happened.
    A stock making new highs on 20% short volume and one making new highs on
    55% are the same bar on a chart and different events underneath: the
    second is climbing against people betting the other way, and their covering
    is fuel the first does not have.

    THE MEASURE IS RELATIVE, NECESSARILY. Bona-fide market making is a large
    and variable share of reported short volume — a market maker who shorts to
    fill your buy and covers seconds later is in this series and in nobody's
    short interest. So the LEVEL is close to meaningless and only movement
    against the name's own history is readable. This uses a rolling percentile
    of each name's own short share, never an absolute threshold.

    NOT SHORT INTEREST, which is the number most people mean by "squeeze": that
    is a stock of open positions published twice a month with a settlement lag.
    This is a daily flow. They are different data and the difference is the
    single easiest way to misread this strategy.

    NOT CHARTABLE, and it is the only rule in this family that is not. FINRA
    short volume is not a TradingView data feed, so there is no honest Pine
    export — an approximation using price and volume alone would be a different
    strategy wearing this one's name. It is here because the family is where
    the single-symbol rules live, and it is single-symbol; the Chart-it page
    lists it under what deliberately does not port.

    BLIND BEFORE 2009-07-31, when FINRA began publishing. Any result is silent
    across the dot-com collapse and 2008 — the two most informative regimes in
    the study's window — and the leave-one-year-out check will show that as
    missing years rather than as stability.
    """
    family = "chartable"
    hypothesis = ("Strength against heavy short participation continues"
                  " further than unopposed strength, because covering adds"
                  " demand that is mechanical rather than discretionary."
                  " Expected to be rare, concentrated, and to fail badly if"
                  " the level rather than the relative measure is what"
                  " actually carries the signal.")

    def __init__(self, lookback: int = 252, pctile: float = 0.80,
                 trend_window: int = 100, exit_pctile: float = 0.50,
                 min_hold: int = 0, basket: str | None = None):
        super().__init__(lookback=lookback, pctile=pctile,
                         trend_window=trend_window, exit_pctile=exit_pctile,
                         min_hold=min_hold, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        ss = md.short_share
        if ss is None or ss.empty:
            raise ValueError(
                "ShortSqueezeCandidate requires FINRA short-sale volume, which "
                "is not in this store. Download it with "
                "`scripts/download_real_data.py --providers finra` — it is free "
                "and needs no key. Refusing rather than approximating it from "
                "price and volume, which would be a different strategy.")

        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        cols = [c for c in mask.columns if c in ss.columns]
        if not cols:
            raise ValueError(
                "no overlap between the investable universe and the FINRA "
                "short-volume panel")
        mask = mask[cols]
        px = md.adj_close[cols]
        share = ss[cols].reindex(px.index).ffill(limit=5)

        # Rolling percentile of the name's OWN short share. Rolling, not
        # expanding: the threshold on day T must come from a window ending T.
        hi = share.rolling(p["lookback"], min_periods=126).quantile(p["pctile"])
        lo = share.rolling(p["lookback"], min_periods=126).quantile(p["exit_pctile"])
        in_trend = px > sma(px, p["trend_window"])

        entry = (share >= hi) & in_trend & mask
        exit_ = (share <= lo) | ~in_trend
        held = _latch(entry, exit_, int(p["min_hold"]))
        return equal_weight(held.where(mask, 0.0))


class HedgedChartSignal(Strategy):
    """A chart signal's long book, with the market subtracted from it.

    THIS IS THE STUDY'S OWN LESSON, APPLIED. Three separate times the same
    finding has arrived from a different direction:

      * freeze v3 — 208 long-only variants all correlated 0.7-0.9 with the
        index, and that was a property of the study's own constraint, not a
        fact about technology stocks;
      * freeze v4 — the ONLY construction ever to clear the bar on real prices
        was BetaHedgedBasket, which was also the only one whose correlation to
        the index was near zero;
      * freezes v6-v9 — TurnOfMonth was added specifically to be uncorrelated
        and correlated +0.53 anyway, because its exposure was still long the
        market whenever it was on; and 92 long-only chart variants were
        rejected while losing to equal-weighting eight AI megacaps.

    A long-only chart rule on technology is the index bet minus the fees. This
    keeps the signal's SELECTION and subtracts the market it is riding, using
    the construction that has actually survived: short the hedge instrument at
    the long book's trailing beta.

    hedge_ratio=0.0 is the CONTROL ARM — the identical signal, unhedged. It is
    in the grid so the hedge's contribution is measured against the same rule
    rather than inferred from an earlier cohort under a different freeze.

    NOT A SINGLE-CHART STRATEGY, and the family docstring's promise does not
    extend to it: Pine cannot hold two instruments in one script. The signal
    leg exports as a normal chart strategy; the hedge is a second position the
    reader has to put on themselves, which the export says in as many words.
    """
    family = "chartable"
    hypothesis = ("A chart signal's value, if it has any, is in WHICH names it"
                  " picks and WHEN it is on — not in the market exposure that"
                  " comes attached. Hedging the index away should leave either"
                  " genuine selection value or nothing, and this measures"
                  " which. Expected to cut return sharply and correlation to"
                  " near zero; the question is whether Sharpe survives.")

    _SIGNALS = {
        "gaussian_bands": ("GaussianTrendBands", {}),
        "gaussian_hold": ("GaussianTrendHold", {}),
        "supertrend": ("Supertrend", {}),
        "rs_new_high": ("RelativeStrengthNewHigh", {}),
    }

    def __init__(self, signal: str = "gaussian_bands", hedge: str = "QQQ",
                 beta_window: int = 126, max_hedge: float = 1.5,
                 hedge_ratio: float = 1.0, period: int | None = None,
                 basket: str | None = "megacap_ai"):
        super().__init__(signal=signal, hedge=hedge, beta_window=beta_window,
                         max_hedge=max_hedge, hedge_ratio=hedge_ratio,
                         period=period, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        from . import STRATEGY_CLASSES

        p = self.params
        if p["signal"] not in self._SIGNALS:
            raise ValueError(f"unknown signal {p['signal']!r}; "
                             f"expected one of {sorted(self._SIGNALS)}")
        if p["hedge"] not in md.adj_close.columns:
            raise ValueError(f"hedge instrument {p['hedge']} not in the data")

        cls_name, extra = self._SIGNALS[p["signal"]]
        kw = dict(extra, basket=p["basket"])
        # `period` means different things to different signals and does not
        # exist on Supertrend at all, so it is passed only where it applies
        # rather than silently ignored.
        if p["period"] is not None:
            import inspect
            if "period" in inspect.signature(
                    STRATEGY_CLASSES[cls_name].__init__).parameters:
                kw["period"] = p["period"]
        long_book = STRATEGY_CLASSES[cls_name](**kw).build(md)

        if p["hedge"] in long_book.columns:      # never hedge with itself
            long_book = long_book.drop(columns=[p["hedge"]])

        cols = list(long_book.columns)
        bench = md.adj_close[p["hedge"]].pct_change()
        port = (long_book.shift(1) * md.adj_close[cols].pct_change()).sum(axis=1)
        exposure = long_book.abs().sum(axis=1).clip(0.0, 1.0)

        # Beta must be measured on the book AS IF FULLY INVESTED, then scaled
        # by today's exposure — not measured on the timed return and scaled
        # again. A signal that is on 60% of the time has a timed beta of about
        # 0.6x its invested beta; multiplying that by exposure a second time
        # under-hedges by the same factor, which is a hedge that looks right
        # and removes two thirds of what it claims to. Measured on the
        # synthetic panel across three signals, the double-damped version left
        # correlation to QQQ at +0.14 to +0.22; this leaves +0.05 to +0.12.
        # Not the -0.005 an always-invested hedge achieves, and it should not
        # be: a timed book turns on and off faster than a 126-day trailing
        # beta can follow, so some market exposure survives by construction.
        invested = port / exposure.shift(1).replace(0.0, np.nan)
        cov = invested.rolling(p["beta_window"], min_periods=63).cov(bench)
        var = bench.rolling(p["beta_window"], min_periods=63).var()
        beta = (cov / var.replace(0, np.nan)).clip(0.0, p["max_hedge"])

        out = long_book.copy()
        # No hedge until a beta exists. An unhedged stub is honest; a
        # fabricated beta of 1.0 would not be.
        out[p["hedge"]] = (-beta * exposure * p["hedge_ratio"]).fillna(0.0)
        return out.fillna(0.0)
