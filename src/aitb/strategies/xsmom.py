"""Cross-sectional momentum within the technology universe.

Economic rationale: relative winners in innovation-driven sectors keep winning
while their product cycle runs; the danger in a narrow universe is that
"momentum" collapses into "always long the single hottest name", which the
ranking diagnostics and concentration penalties downstream are built to catch.
"""
from __future__ import annotations

import pandas as pd

from ..data.loader import MarketData
from ..features import momentum, risk_adjusted_momentum
from ..portfolio import equal_weight, inverse_vol_weight, rebalance_schedule, score_weight, top_n_selection
from ..universe import investable_mask
from .base import Strategy


class XSMomentumTopN(Strategy):
    family = "xsmom"
    hypothesis = "Relative strength persists among tech leaders."

    def __init__(self, lookback_days: int = 126, skip_days: int = 21,
                 top_n: int = 5, weighting: str = "equal",
                 rebalance: str = "ME", basket: str | None = None):
        super().__init__(lookback_days=lookback_days, skip_days=skip_days,
                         top_n=top_n, weighting=weighting,
                         rebalance=rebalance, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        px = md.adj_close[mask.columns]
        mom = momentum(px, p["lookback_days"], p["skip_days"]).where(mask)
        sel = top_n_selection(mom, p["top_n"])
        # Require positive momentum — never buy a falling knife just because
        # it ranks 5th of 5.
        sel = sel.where(mom > 0, 0.0)
        if p["weighting"] == "equal":
            w = equal_weight(sel)
        elif p["weighting"] == "inverse_vol":
            w = inverse_vol_weight(px, sel)
        elif p["weighting"] == "momentum":
            w = score_weight(mom, sel)
        else:
            raise ValueError(p["weighting"])
        return rebalance_schedule(w, p["rebalance"])


class XSRiskAdjMomentum(Strategy):
    family = "xsmom"
    hypothesis = "Sharpe-style momentum favors steady trends over spikes."

    def __init__(self, lookback_days: int = 126, top_n: int = 5,
                 rebalance: str = "ME"):
        super().__init__(lookback_days=lookback_days, top_n=top_n, rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md)
        ram = risk_adjusted_momentum(md.adj_close[mask.columns],
                                     p["lookback_days"]).where(mask)
        sel = top_n_selection(ram, p["top_n"]).where(ram > 0, 0.0)
        return rebalance_schedule(equal_weight(sel), p["rebalance"])


class RelativeStrengthVsBench(Strategy):
    """Long names outperforming QQQ over the lookback, equal weight."""
    family = "xsmom"
    hypothesis = "Beating the tech tape itself is a stronger signal than raw return."

    def __init__(self, lookback_days: int = 126, top_n: int = 8,
                 bench: str = "QQQ", rebalance: str = "ME"):
        super().__init__(lookback_days=lookback_days, top_n=top_n,
                         bench=bench, rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md)
        px = md.adj_close[mask.columns]
        rel = momentum(px, p["lookback_days"]).sub(
            momentum(md.adj_close[[p["bench"]]], p["lookback_days"])[p["bench"]], axis=0
        ).where(mask)
        sel = top_n_selection(rel, p["top_n"]).where(rel > 0, 0.0)
        return rebalance_schedule(equal_weight(sel), p["rebalance"])
