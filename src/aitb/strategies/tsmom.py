"""Time-series momentum / trend-following on individual names or baskets.

Economic rationale: slow diffusion of information plus investor herding makes
absolute uptrends persist; exiting downtrends avoids the deep tech drawdowns
(-80% dot-com, -55% GFC, -35% 2022) at the cost of whipsaws in choppy tape.
"""
from __future__ import annotations

import pandas as pd

from ..data.loader import MarketData
from ..features import above_sma, momentum
from ..portfolio import equal_weight, rebalance_schedule
from ..universe import investable_mask
from .base import Strategy


class TrendFollowCash(Strategy):
    """Hold each investable name only while its own trend filter is on; the
    freed capital goes to a Treasury/cash ETF."""
    family = "tsmom"
    hypothesis = "Per-name absolute trend avoids left-tail drawdowns."

    def __init__(self, sma_window: int = 200, fallback: str = "IEF",
                 rebalance: str = "W-FRI", basket: str | None = None):
        super().__init__(sma_window=sma_window, fallback=fallback,
                         rebalance=rebalance, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        trend_on = above_sma(md.adj_close[mask.columns], p["sma_window"]).where(mask, 0.0)
        n_universe = mask.sum(axis=1).astype(float).replace(0.0, float("nan"))
        w = trend_on.div(n_universe, axis=0).fillna(0.0)
        fb = p["fallback"]
        if fb and fb in md.adj_close.columns:
            w[fb] = ((1 - w.sum(axis=1)).clip(lower=0.0)
                     * md.adj_close[fb].notna()).fillna(0.0)
        return rebalance_schedule(w, p["rebalance"])


class AbsoluteMomentum(Strategy):
    """Hold names whose own trailing return beats T-bills; else cash ETF."""
    family = "tsmom"
    hypothesis = "Dual-momentum absolute leg: positive excess return persists."

    def __init__(self, lookback_days: int = 252, fallback: str = "BIL",
                 rebalance: str = "ME", basket: str | None = None):
        super().__init__(lookback_days=lookback_days, fallback=fallback,
                         rebalance=rebalance, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        lb = p["lookback_days"]
        mom = momentum(md.adj_close[mask.columns], lb)
        cash_hurdle = (1 + md.macro.get("FEDFUNDS", pd.Series(0, index=md.calendar))
                       .reindex(md.calendar).ffill() / 100) ** (lb / 252) - 1
        on = mom.gt(cash_hurdle, axis=0).where(mask, False)
        w = equal_weight(on)
        fb = p["fallback"]
        if fb and fb in md.adj_close.columns:
            w[fb] = ((1 - w.sum(axis=1)).clip(lower=0.0)
                     * md.adj_close[fb].notna()).fillna(0.0)
        return rebalance_schedule(w, p["rebalance"])


class DualMomentum(Strategy):
    """Relative momentum across tech baskets + absolute filter vs cash."""
    family = "tsmom"
    hypothesis = "Rotate into the strongest theme; stand aside in downtrends."

    def __init__(self, lookback_days: int = 126, fallback: str = "IEF"):
        super().__init__(lookback_days=lookback_days, fallback=fallback)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        etfs = [t for t in ("QQQ", "XLK", "SOXX", "IGV") if t in md.adj_close.columns]
        mom = momentum(md.adj_close[etfs], p["lookback_days"])
        valid = mom.notna().any(axis=1)
        best = pd.Series(pd.NA, index=mom.index, dtype=object)
        best[valid] = mom[valid].idxmax(axis=1)
        best_mom = mom.max(axis=1)
        w = pd.DataFrame(0.0, index=md.calendar, columns=etfs + [p["fallback"]])
        for t in etfs:
            w.loc[(best == t) & (best_mom > 0), t] = 1.0
        fb_ok = md.adj_close[p["fallback"]].notna()
        w.loc[(best_mom <= 0) & fb_ok, p["fallback"]] = 1.0
        return rebalance_schedule(w, "ME")
