"""Example plugin: golden-cross rotation (template for new strategy families).

This is ALL the boilerplate a new strategy needs. Once registered it appears
automatically in the catalog, dashboard, documentation, comparison engine and
research site. To run it in a REAL study it must additionally be added to the
frozen grid (configs/experiments.yaml) and the freeze fingerprint list —
which forces a freeze version bump, by design.
"""
import pandas as pd

from aitb.data.loader import MarketData
from aitb.features import sma
from aitb.plugins import register
from aitb.portfolio import equal_weight, rebalance_schedule
from aitb.strategies.base import Strategy
from aitb.universe import investable_mask


@register
class GoldenCrossRotation(Strategy):
    """Hold names whose 50-day SMA is above their 200-day SMA (golden cross)."""

    family = "tsmom"
    hypothesis = "The 50/200 golden cross is a slow, robust trend proxy."

    def __init__(self, fast: int = 50, slow: int = 200, rebalance: str = "ME"):
        super().__init__(fast=fast, slow=slow, rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md)
        px = md.adj_close[mask.columns]
        crossed = (sma(px, p["fast"]) > sma(px, p["slow"])).where(mask, False)
        return rebalance_schedule(equal_weight(crossed), p["rebalance"])
