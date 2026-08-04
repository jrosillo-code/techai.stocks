"""Breakout / volatility-expansion strategies.

Economic rationale: multi-month highs attract attention and force
underweighted institutions in; volatility compression often precedes
information arrival. Both are classic trend-initiation entries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import MarketData
from ..features import donchian_high, realized_vol
from ..portfolio import equal_weight
from ..universe import investable_mask
from .base import Strategy


class DonchianBreakout(Strategy):
    """Enter on an N-day-high breakout, exit on an M-day-low break."""
    family = "breakout"
    hypothesis = "New multi-month highs precede continued advances."

    def __init__(self, entry_window: int = 55, exit_window: int = 20,
                 max_positions: int = 6, regime_filter: bool = True):
        super().__init__(entry_window=entry_window, exit_window=exit_window,
                         max_positions=max_positions, regime_filter=regime_filter)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md)
        px = md.adj_close[mask.columns]
        hi = donchian_high(px, p["entry_window"])
        lo = px.shift(1).rolling(p["exit_window"]).min()

        entry = (px > hi) & mask
        exit_ = px < lo
        if p["regime_filter"] and "QQQ" in md.adj_close.columns:
            q = md.adj_close["QQQ"]
            risk_on = q > q.rolling(200).mean()
            entry = entry.mul(risk_on, axis=0).astype(bool)

        pos = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        state = entry.iloc[0].astype(float)
        for i in range(len(px.index)):
            state = ((state.astype(bool) & ~exit_.iloc[i]) | entry.iloc[i]).astype(float)
            pos.iloc[i] = state
        # Recency-of-breakout conviction: strongest = closest above prior high.
        strength = (px / hi - 1).where(pos.astype(bool))
        ranks = strength.rank(axis=1, ascending=False, method="first")
        pos = pos.where(ranks <= p["max_positions"], 0.0)
        return equal_weight(pos)


class VolCompressionBreakout(Strategy):
    """Only take breakouts that emerge from a volatility squeeze."""
    family = "breakout"
    hypothesis = "Breakouts from compressed vol carry more information."

    def __init__(self, entry_window: int = 40, squeeze_pct: float = 0.25,
                 exit_window: int = 20, max_positions: int = 6):
        super().__init__(entry_window=entry_window, squeeze_pct=squeeze_pct,
                         exit_window=exit_window, max_positions=max_positions)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md)
        px = md.adj_close[mask.columns]
        vol = realized_vol(px, 21)
        vol_rank = vol.rolling(252).rank(pct=True)  # trailing percentile of own vol
        squeezed = vol_rank < p["squeeze_pct"]

        hi = donchian_high(px, p["entry_window"])
        lo = px.shift(1).rolling(p["exit_window"]).min()
        entry = (px > hi) & squeezed.shift(1).fillna(False) & mask
        exit_ = px < lo

        pos = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        state = entry.iloc[0].astype(float)
        for i in range(len(px.index)):
            state = ((state.astype(bool) & ~exit_.iloc[i]) | entry.iloc[i]).astype(float)
            pos.iloc[i] = state
        ranks = (px / hi - 1).where(pos.astype(bool)).rank(axis=1, ascending=False, method="first")
        pos = pos.where(ranks <= p["max_positions"], 0.0)
        return equal_weight(pos)
