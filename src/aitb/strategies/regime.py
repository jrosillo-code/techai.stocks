"""Regime-based allocation: scale tech exposure by trailing macro/market state.

All regime inputs are trailing (trend, 63-day rate changes, VIX level,
breadth); definitions contain no future information by construction.
"""
from __future__ import annotations

import pandas as pd

from ..data.loader import MarketData
from ..features import macro_regime_flags
from ..portfolio import equal_weight, rebalance_schedule
from ..universe import investable_mask
from .base import Strategy


class RegimeSwitchedTech(Strategy):
    """Equal-weight tech universe, scaled down as risk regimes stack up.

    Each active risk flag (QQQ below 200dma, VIX high, rates rising, weak
    breadth, credit stress) cuts exposure by `cut`; the remainder sits in the
    fallback ETF.
    """
    family = "regime"
    hypothesis = "Tech drawdowns cluster in identifiable macro risk regimes."

    def __init__(self, cut: float = 0.25, fallback: str = "IEF",
                 rebalance: str = "W-FRI",
                 flags: tuple = ("qqq_below_200", "vix_high", "rates_rising", "breadth_weak")):
        super().__init__(cut=cut, fallback=fallback, rebalance=rebalance, flags=flags)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        rf = macro_regime_flags(md)
        risk = pd.DataFrame(index=md.calendar)
        if "qqq_below_200" in p["flags"] and "qqq_above_200" in rf.columns:
            risk["qqq_below_200"] = 1 - rf["qqq_above_200"]
        if "vix_high" in p["flags"] and "vix_high" in rf.columns:
            risk["vix_high"] = rf["vix_high"]
        if "rates_rising" in p["flags"] and "rates_rising" in rf.columns:
            risk["rates_rising"] = rf["rates_rising"]
        if "breadth_weak" in p["flags"] and "breadth_200" in rf.columns:
            risk["breadth_weak"] = (rf["breadth_200"] < 0.4).astype(float)
        if "credit_stress" in p["flags"] and "credit_stress" in rf.columns:
            risk["credit_stress"] = rf["credit_stress"]

        exposure = (1 - p["cut"] * risk.sum(axis=1)).clip(0.0, 1.0)
        mask = investable_mask(md)
        w = equal_weight(mask).mul(exposure, axis=0)
        fb = p["fallback"]
        if fb in md.adj_close.columns:
            w[fb] = ((1 - exposure) * md.adj_close[fb].notna()).fillna(0.0)
        return rebalance_schedule(w, p["rebalance"])


class SemisLeadership(Strategy):
    """Overweight semis when SOXX leads QQQ, else broad tech; cash if both weak.

    Semiconductors are the cycle's canary: their relative strength has
    historically led broader tech risk appetite.
    """
    family = "regime"
    hypothesis = "Semiconductor relative strength leads tech risk appetite."

    def __init__(self, lookback_days: int = 126, fallback: str = "IEF"):
        super().__init__(lookback_days=lookback_days, fallback=fallback)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        lb = p["lookback_days"]
        cols = [c for c in ("SOXX", "QQQ", p["fallback"]) if c in md.adj_close.columns]
        px = md.adj_close[cols]
        soxx_rel = px["SOXX"].pct_change(lb) - px["QQQ"].pct_change(lb)
        qqq_trend = px["QQQ"] > px["QQQ"].rolling(200).mean()

        w = pd.DataFrame(0.0, index=md.calendar, columns=cols)
        semis_lead = (soxx_rel > 0) & qqq_trend
        broad = (soxx_rel <= 0) & qqq_trend
        w.loc[semis_lead, "SOXX"] = 1.0
        w.loc[broad, "QQQ"] = 1.0
        fb = p["fallback"]
        w.loc[~qqq_trend & px[fb].notna(), fb] = 1.0
        return rebalance_schedule(w, "ME")
