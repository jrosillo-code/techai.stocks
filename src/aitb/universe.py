"""Point-in-time investable universe.

``investable_mask`` returns a boolean date × ticker frame that is True only
when, using information available through that date, the name was actually
tradeable:

  * listed (a price exists) and past the IPO seasoning window,
  * not yet delisted,
  * trailing price above the configured floor,
  * trailing 63-day median dollar volume above the liquidity floor.

Because every input is a trailing statistic, a mask row for day T contains no
information from after T; strategies consume it with the same T -> T+1-open
execution convention as any other signal. Delisted names remain in the panel
until their delisting date — removing them entirely is how survivorship bias
arises (tests cover this).
"""
from __future__ import annotations

import pandas as pd

from .data.loader import MarketData


def investable_mask(md: MarketData, tickers: list[str] | None = None) -> pd.DataFrame:
    u = md.universe
    tickers = [t for t in (tickers or u.tickers) if t in md.adj_close.columns]
    px = md.adj_close[tickers]
    raw_px = md.close[tickers]
    dv = md.dollar_volume[tickers]

    listed = px.notna()
    # Seasoning: require `seasoning_days` of history since first quote.
    seasoned = listed.cumsum() > u.seasoning_days
    liquid = dv.rolling(63, min_periods=21).median() > u.min_median_dollar_volume
    price_ok = raw_px.rolling(5, min_periods=1).min() > u.min_price

    mask = listed & seasoned & liquid.fillna(False) & price_ok.fillna(False)

    # Enforce hard delisting dates from config (belt and braces: the price
    # panel already ends there for delisted names).
    for sec in u.securities:
        if sec.delisted is not None and sec.ticker in mask.columns:
            mask.loc[mask.index > pd.Timestamp(sec.delisted), sec.ticker] = False
    return mask


def basket_mask(md: MarketData, basket: str) -> pd.DataFrame:
    """Investable mask restricted to a configured theme basket."""
    members = md.universe.baskets[basket]
    return investable_mask(md, tickers=members)
