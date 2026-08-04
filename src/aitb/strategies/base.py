"""Strategy interface.

A Strategy converts MarketData into a frame of target weights. Row T is the
portfolio the strategy WANTS after seeing the close of T; the engine fills it
at the open of T+1. Strategies must never call anything that peeks forward —
all provided feature helpers are trailing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..data.loader import MarketData


class Strategy(ABC):
    family: str = "unspecified"
    hypothesis: str = ""

    def __init__(self, **params):
        self.params = params

    @property
    def name(self) -> str:
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.__class__.__name__}({inner})"

    @abstractmethod
    def build(self, md: MarketData) -> pd.DataFrame:
        """Return date × ticker target weights (rows: information date T)."""

    def spec(self) -> dict:
        return {"class": self.__class__.__name__, "family": self.family,
                "params": self.params, "hypothesis": self.hypothesis}
