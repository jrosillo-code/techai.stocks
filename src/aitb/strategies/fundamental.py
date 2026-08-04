"""Point-in-time fundamental strategies: quality/growth and valuation-aware
growth.

All fundamental inputs come from ``pit_fundamental_panel``, which only makes a
quarter visible after its publication date. Cross-sectional z-scores are
computed within the investable set on each date.
"""
from __future__ import annotations

import pandas as pd

from ..data.loader import MarketData
from ..features import pit_fundamental_panel, price_to_sales, xs_zscore
from ..portfolio import equal_weight, rebalance_schedule, top_n_selection
from ..universe import investable_mask
from .base import Strategy


class QualityGrowth(Strategy):
    """Composite: revenue growth + growth acceleration + FCF margin."""
    family = "fundamental"
    hypothesis = "Compounding fundamentals precede compounding prices."

    def __init__(self, top_n: int = 8, rebalance: str = "QE",
                 w_growth: float = 0.4, w_accel: float = 0.3, w_margin: float = 0.3):
        super().__init__(top_n=top_n, rebalance=rebalance, w_growth=w_growth,
                         w_accel=w_accel, w_margin=w_margin)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md)
        growth = pit_fundamental_panel(md, "revenue", "yoy_growth")
        accel = pit_fundamental_panel(md, "revenue", "accel")
        rev = pit_fundamental_panel(md, "revenue", "ttm")
        fcf = pit_fundamental_panel(md, "fcf", "ttm")
        cols = [c for c in mask.columns if c in growth.columns]
        margin = fcf[cols] / rev[cols].replace(0.0, float("nan"))

        m = mask[cols]
        z = (p["w_growth"] * xs_zscore(growth[cols], m)
             + p["w_accel"] * xs_zscore(accel[cols], m)
             + p["w_margin"] * xs_zscore(margin.astype(float), m))
        sel = top_n_selection(z, p["top_n"], m)
        return rebalance_schedule(equal_weight(sel), p["rebalance"])


class ValuationAwareGrowth(Strategy):
    """Quality growth, but avoid the most expensive names vs their own history.

    Valuation is judged against the stock's OWN trailing P/S distribution
    (structural margin differences make cross-company P/S comparisons unfair).
    """
    family = "fundamental"
    hypothesis = "Growth wins, but entry valuation caps forward returns."

    def __init__(self, top_n: int = 8, rebalance: str = "QE",
                 max_ps_own_pct: float = 0.85, history_days: int = 756):
        super().__init__(top_n=top_n, rebalance=rebalance,
                         max_ps_own_pct=max_ps_own_pct, history_days=history_days)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md)
        growth = pit_fundamental_panel(md, "revenue", "yoy_growth")
        rev = pit_fundamental_panel(md, "revenue", "ttm")
        fcf = pit_fundamental_panel(md, "fcf", "ttm")
        ps = price_to_sales(md)
        cols = [c for c in mask.columns if c in growth.columns and c in ps.columns]
        m = mask[cols]

        margin = (fcf[cols] / rev[cols].replace(0.0, float("nan"))).astype(float)
        z = xs_zscore(growth[cols], m) + 0.5 * xs_zscore(margin, m)
        # Own-history valuation percentile (trailing 3y), NaN-safe.
        ps_pct = ps[cols].rolling(p["history_days"], min_periods=252).rank(pct=True)
        not_stretched = ps_pct < p["max_ps_own_pct"]
        sel = top_n_selection(z.where(not_stretched), p["top_n"], m)
        return rebalance_schedule(equal_weight(sel), p["rebalance"])
