"""Risk-managed buy-and-hold: keep the exposure, manage the path.

These strategies never pick stocks — they hold a fixed basket and modulate
gross exposure (volatility targeting, drawdown de-risking) or weighting
(inverse-vol). They answer the study's central question: can risk management
beat raw buy-and-hold of the same assets after costs?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import MarketData
from ..portfolio import equal_weight, inverse_vol_weight, rebalance_schedule, vol_target
from ..universe import investable_mask
from .base import Strategy


class VolTargetedBasket(Strategy):
    family = "riskmanaged"
    hypothesis = "Constant-risk sizing improves compounding vs constant-dollar."

    def __init__(self, basket: str = "megacap_ai", target_vol: float = 0.25,
                 vol_window: int = 63, fallback: str = "IEF",
                 rebalance: str = "W-FRI"):
        super().__init__(basket=basket, target_vol=target_vol,
                         vol_window=vol_window, fallback=fallback,
                         rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md, md.universe.baskets[p["basket"]])
        w = equal_weight(mask)
        w = vol_target(w, md.adj_close[mask.columns], p["target_vol"],
                       window=p["vol_window"])
        fb = p["fallback"]
        if fb in md.adj_close.columns:
            w[fb] = ((1 - w.sum(axis=1)).clip(lower=0.0)
                     * md.adj_close[fb].notna()).fillna(0.0)
        return rebalance_schedule(w, p["rebalance"])


class TrendPlusVolTarget(Strategy):
    """Combined overlay: QQQ 200-day trend gates GROSS exposure (with
    hysteresis to cut whipsaw), volatility targeting scales it, remainder in
    Treasuries. Exposure changes are gradual, not binary."""
    family = "riskmanaged"
    hypothesis = ("Trend removes the deep left tail; vol targeting smooths the "
                  "path — the two address different failure modes.")

    def __init__(self, basket: str = "megacap_ai", target_vol: float = 0.20,
                 vol_window: int = 63, trend_window: int = 200,
                 hysteresis: float = 0.02, min_exposure: float = 0.2,
                 fallback: str = "IEF", rebalance: str = "W-FRI"):
        super().__init__(basket=basket, target_vol=target_vol,
                         vol_window=vol_window, trend_window=trend_window,
                         hysteresis=hysteresis, min_exposure=min_exposure,
                         fallback=fallback, rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md, md.universe.baskets[p["basket"]])
        w = equal_weight(mask)
        w = vol_target(w, md.adj_close[mask.columns], p["target_vol"],
                       window=p["vol_window"])

        # Hysteresis trend gate on QQQ: off only below (1-h)*SMA, back on
        # only above (1+h)*SMA — the band suppresses whipsaw at the line.
        q = md.adj_close["QQQ"]
        sma = q.rolling(p["trend_window"]).mean()
        upper, lower = sma * (1 + p["hysteresis"]), sma * (1 - p["hysteresis"])
        state = np.ones(len(q))
        on = True
        qv, uv, lv = q.to_numpy(), upper.to_numpy(), lower.to_numpy()
        for i in range(len(q)):
            if np.isnan(uv[i]):
                state[i] = 1.0
                continue
            if on and qv[i] < lv[i]:
                on = False
            elif not on and qv[i] > uv[i]:
                on = True
            state[i] = 1.0 if on else p["min_exposure"]
        gate = pd.Series(state, index=q.index)

        w = w.mul(gate, axis=0)
        fb = p["fallback"]
        if fb in md.adj_close.columns:
            w[fb] = ((1 - w.sum(axis=1)).clip(lower=0.0)
                     * md.adj_close[fb].notna()).fillna(0.0)
        return rebalance_schedule(w, p["rebalance"])


class DrawdownDeRisk(Strategy):
    """Cut basket exposure as its trailing drawdown deepens; restore on recovery."""
    family = "riskmanaged"
    hypothesis = "Deep drawdowns cluster; de-risking into them cuts the left tail."

    def __init__(self, basket: str = "megacap_ai", dd_start: float = -0.10,
                 dd_full: float = -0.30, fallback: str = "IEF",
                 rebalance: str = "W-FRI"):
        super().__init__(basket=basket, dd_start=dd_start, dd_full=dd_full,
                         fallback=fallback, rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md, md.universe.baskets[p["basket"]])
        w = equal_weight(mask)
        # Trailing drawdown of the basket's own (pre-cost) equity curve.
        basket_ret = (w.shift(1) * md.adj_close[mask.columns].pct_change()).sum(axis=1)
        eq = (1 + basket_ret).cumprod()
        dd = eq / eq.cummax() - 1
        span = p["dd_start"] - p["dd_full"]
        exposure = ((dd - p["dd_full"]) / span).clip(0.0, 1.0)
        w = w.mul(exposure, axis=0)
        fb = p["fallback"]
        if fb in md.adj_close.columns:
            w[fb] = ((1 - exposure).clip(0.0, 1.0)
                     * md.adj_close[fb].notna()).fillna(0.0)
        return rebalance_schedule(w, p["rebalance"])


class InverseVolBasket(Strategy):
    family = "riskmanaged"
    hypothesis = "Risk-balanced weights avoid concentration in the wildest name."

    def __init__(self, basket: str = "megacap_ai", rebalance: str = "ME"):
        super().__init__(basket=basket, rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md, md.universe.baskets[p["basket"]])
        w = inverse_vol_weight(md.adj_close[mask.columns], mask)
        return rebalance_schedule(w, p["rebalance"])
