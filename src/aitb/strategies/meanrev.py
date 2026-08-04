"""Short-horizon mean reversion within the tech universe.

Economic rationale: liquidity provision — sharp short-term selloffs in liquid
names overshoot and partially retrace. These are exactly the strategies most
sensitive to transaction costs and most likely to be illusory after realistic
spreads, so they must be judged under the stressed cost scenario.
"""
from __future__ import annotations

import pandas as pd

from ..data.loader import MarketData
from ..features import above_sma, bollinger_z, rsi, short_term_reversal
from ..portfolio import equal_weight
from ..universe import investable_mask
from .base import Strategy


class RSIReversion(Strategy):
    """Buy quality-of-trend names on RSI(2) oversold, exit on recovery.

    Long-term uptrend filter (200-day SMA) restricts entries to names in
    uptrends — reversion trades against the short move, not the major trend.
    """
    family = "meanrev"
    hypothesis = "Oversold snaps back within days in uptrending liquid names."

    def __init__(self, rsi_window: int = 2, entry: float = 10.0, exit: float = 60.0,
                 trend_window: int = 200, max_positions: int = 5):
        super().__init__(rsi_window=rsi_window, entry=entry, exit=exit,
                         trend_window=trend_window, max_positions=max_positions)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md)
        px = md.adj_close[mask.columns]
        r = rsi(px, p["rsi_window"])
        uptrend = above_sma(px, p["trend_window"]).astype(bool)

        entry = (r < p["entry"]) & uptrend & mask
        exit_ = (r > p["exit"]) | ~uptrend
        # Stateful position: in after entry until exit fires.
        pos = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        state = entry.iloc[0].astype(float)
        for i in range(len(px.index)):
            state = ((state.astype(bool) & ~exit_.iloc[i]) | entry.iloc[i]).astype(float)
            pos.iloc[i] = state
        # Cap concurrent positions at max_positions (highest conviction =
        # lowest RSI first).
        ranks = r.where(pos.astype(bool)).rank(axis=1, method="first")
        pos = pos.where(ranks <= p["max_positions"], 0.0)
        return equal_weight(pos)


class ShortTermReversal(Strategy):
    """Weekly reversal: buy the biggest 5-day losers among liquid names."""
    family = "meanrev"
    hypothesis = "5-day losers outperform over the following week (pre-cost)."

    def __init__(self, lookback_days: int = 5, top_n: int = 5,
                 trend_window: int = 200, rebalance: str = "W-FRI"):
        super().__init__(lookback_days=lookback_days, top_n=top_n,
                         trend_window=trend_window, rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        from ..portfolio import rebalance_schedule, top_n_selection
        p = self.params
        mask = investable_mask(md)
        px = md.adj_close[mask.columns]
        rev = short_term_reversal(px, p["lookback_days"]).where(mask)
        uptrend = above_sma(px, p["trend_window"]).astype(bool)
        sel = top_n_selection(rev.where(uptrend), p["top_n"])
        return rebalance_schedule(equal_weight(sel), p["rebalance"])


class BollingerReversion(Strategy):
    """Buy names stretched below their lower band, in uptrends only."""
    family = "meanrev"
    hypothesis = "-2σ stretches revert toward the mean in liquid uptrends."

    def __init__(self, window: int = 20, entry_z: float = -2.0, exit_z: float = 0.0,
                 trend_window: int = 200, max_positions: int = 5):
        super().__init__(window=window, entry_z=entry_z, exit_z=exit_z,
                         trend_window=trend_window, max_positions=max_positions)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md)
        px = md.adj_close[mask.columns]
        z = bollinger_z(px, p["window"])
        uptrend = above_sma(px, p["trend_window"]).astype(bool)
        entry = (z < p["entry_z"]) & uptrend & mask
        exit_ = (z > p["exit_z"]) | ~uptrend
        pos = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        state = entry.iloc[0].astype(float)
        for i in range(len(px.index)):
            state = ((state.astype(bool) & ~exit_.iloc[i]) | entry.iloc[i]).astype(float)
            pos.iloc[i] = state
        ranks = z.where(pos.astype(bool)).rank(axis=1, method="first")
        pos = pos.where(ranks <= p["max_positions"], 0.0)
        return equal_weight(pos)
