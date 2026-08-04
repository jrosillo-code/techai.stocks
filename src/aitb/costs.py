"""Transaction-cost models.

One-way cost of trading `trade_value` dollars of a security on a given day:

    cost = trade_value * (fixed_bps + impact_bps * sqrt(participation)) / 1e4

where participation = |trade_value| / trailing 21-day average dollar volume.
Borrow cost accrues daily on short market value. ETF expense ratios are
embedded in their return series by the data layer, not double-counted here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostScenario


def one_way_cost(trade_value: float, adv_dollars: float, scen: CostScenario) -> float:
    """Dollar cost of a single one-way trade."""
    tv = abs(float(trade_value))
    if tv == 0.0:
        return 0.0
    fixed = scen.fixed_one_way_bps
    if adv_dollars and adv_dollars > 0 and scen.impact_coeff_bps > 0:
        participation = min(tv / adv_dollars, 1.0)
        impact = scen.impact_coeff_bps * np.sqrt(participation)
    else:
        impact = 0.0
    return tv * (fixed + impact) / 1e4


def daily_borrow_cost(short_value: float, scen: CostScenario) -> float:
    """Daily financing cost on absolute short market value."""
    return abs(float(short_value)) * scen.borrow_bps / 1e4 / 252.0


def rolling_adv(dollar_volume: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Trailing average daily dollar volume, shifted so day T uses data
    through T-1 (the trade decided on T's signal executes at T+1's open,
    priced against ADV known at decision time)."""
    return dollar_volume.rolling(window, min_periods=5).mean().shift(1)
