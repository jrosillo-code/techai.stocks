"""Benchmark strategies every candidate must beat.

Includes single-ticker buy & hold, equal-weight and cap-weight universe
portfolios, the 200-day moving-average timing baseline on QQQ, and a simple
12-1 time-series momentum baseline. Complexity that cannot beat these after
costs is rejected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import MarketData
from ..features import above_sma, momentum, pit_fundamental_panel
from ..portfolio import cap_weights, equal_weight, rebalance_schedule
from ..universe import investable_mask
from .base import Strategy


class BuyAndHold(Strategy):
    family = "benchmark"
    hypothesis = "Own the asset; the baseline all timing must beat."

    def __init__(self, ticker: str):
        super().__init__(ticker=ticker)

    def build(self, md: MarketData) -> pd.DataFrame:
        t = self.params["ticker"]
        px = md.adj_close[t]
        w = pd.DataFrame(0.0, index=md.calendar, columns=[t])
        w.loc[px.notna(), t] = 1.0
        return w


class EqualWeightUniverse(Strategy):
    family = "benchmark"
    hypothesis = "Naive diversification across the investable tech universe."

    def __init__(self, rebalance: str = "ME", basket: str | None = None):
        super().__init__(rebalance=rebalance, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        basket = self.params["basket"]
        tickers = md.universe.baskets[basket] if basket else None
        mask = investable_mask(md, tickers)
        w = equal_weight(mask)
        return rebalance_schedule(w, self.params["rebalance"])


class CapWeightUniverse(Strategy):
    family = "benchmark"
    hypothesis = "Market-cap weighting: let winners ride, PIT shares only."

    def __init__(self, rebalance: str = "QE", basket: str | None = None,
                 max_weight: float = 0.25):
        super().__init__(rebalance=rebalance, basket=basket, max_weight=max_weight)

    def build(self, md: MarketData) -> pd.DataFrame:
        basket = self.params["basket"]
        tickers = md.universe.baskets[basket] if basket else None
        mask = investable_mask(md, tickers)
        shares = pit_fundamental_panel(md, "shares", "level")
        cols = [c for c in mask.columns if c in shares.columns]
        cap = (md.close[cols] * shares[cols]).where(mask[cols])
        if cap.dropna(how="all").empty:
            # Fallback when the provider has no shares data: trailing dollar
            # volume as a size proxy (documented approximation).
            cap = md.dollar_volume[mask.columns].rolling(63).mean().where(mask)
        tot = cap.sum(axis=1).replace(0, np.nan)
        w = cap.div(tot, axis=0).fillna(0.0)
        w = cap_weights(w, self.params["max_weight"])
        return rebalance_schedule(w, self.params["rebalance"])


class QQQMovingAverage(Strategy):
    family = "benchmark"
    hypothesis = "Classic trend baseline: QQQ when above its 200-day SMA, else cash."

    def __init__(self, window: int = 200, fallback: str | None = "IEF"):
        super().__init__(window=window, fallback=fallback)

    def build(self, md: MarketData) -> pd.DataFrame:
        w0 = self.params["window"]
        fb = self.params["fallback"]
        sig = above_sma(md.adj_close[["QQQ"]], w0)["QQQ"]
        cols = ["QQQ"] + ([fb] if fb else [])
        w = pd.DataFrame(0.0, index=md.calendar, columns=cols)
        w["QQQ"] = sig
        if fb:
            fb_ok = md.adj_close[fb].notna()
            w[fb] = ((1 - sig) * fb_ok).fillna(0.0)
        return w[md.adj_close["QQQ"].notna()]


class SimpleMomentum12_1(Strategy):
    family = "benchmark"
    hypothesis = "Textbook 12-1 momentum, top-N equal weight, monthly."

    def __init__(self, top_n: int = 5):
        super().__init__(top_n=top_n)

    def build(self, md: MarketData) -> pd.DataFrame:
        mask = investable_mask(md)
        mom = momentum(md.adj_close[mask.columns], 252, 21).where(mask)
        ranks = mom.rank(axis=1, ascending=False, method="first")
        sel = (ranks <= self.params["top_n"]).astype(float)
        return rebalance_schedule(equal_weight(sel), "ME")
