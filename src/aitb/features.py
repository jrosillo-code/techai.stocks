"""Signal building blocks.

Every function maps price/fundamental history to a date × ticker frame where
the value at date T uses only data through the close of T. Nothing here shifts
for execution — the engine applies the uniform one-bar lag, so features stay
in "information time".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.loader import MarketData

ANN = 252


# ---------------------------------------------------------------- momentum --
def momentum(px: pd.DataFrame, lookback_days: int, skip_days: int = 0) -> pd.DataFrame:
    """Total return over [T-lookback, T-skip]. skip=21 gives 12-1 momentum."""
    shifted = px.shift(skip_days)
    return shifted / shifted.shift(lookback_days - skip_days) - 1


def risk_adjusted_momentum(px: pd.DataFrame, lookback_days: int,
                           skip_days: int = 21) -> pd.DataFrame:
    mom = momentum(px, lookback_days, skip_days)
    vol = px.pct_change().rolling(lookback_days).std() * np.sqrt(ANN)
    return mom / vol.replace(0, np.nan)


def sma(px: pd.DataFrame, window: int) -> pd.DataFrame:
    return px.rolling(window).mean()


def above_sma(px: pd.DataFrame, window: int) -> pd.DataFrame:
    return (px > sma(px, window)).astype(float)


def distance_from_sma(px: pd.DataFrame, window: int) -> pd.DataFrame:
    m = sma(px, window)
    return px / m - 1


# ------------------------------------------------------------ mean reversion --
def rsi(px: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    delta = px.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    dn = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def short_term_reversal(px: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Negative of the trailing `window`-day return (high = oversold)."""
    return -(px / px.shift(window) - 1)


def bollinger_z(px: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    m = px.rolling(window).mean()
    sd = px.rolling(window).std()
    return (px - m) / sd.replace(0, np.nan)


# ------------------------------------------------------------------ breakout --
def donchian_high(px: pd.DataFrame, window: int) -> pd.DataFrame:
    """Prior `window`-day high EXCLUDING today (so a breakout is vs history)."""
    return px.shift(1).rolling(window).max()


def donchian_breakout(px: pd.DataFrame, window: int) -> pd.DataFrame:
    return (px > donchian_high(px, window)).astype(float)


# ---------------------------------------------------------------- volatility --
def realized_vol(px: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    return px.pct_change().rolling(window).std() * np.sqrt(ANN)


def inverse_vol_weights(px: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    iv = 1 / realized_vol(px, window).replace(0, np.nan)
    return iv.div(iv.sum(axis=1), axis=0)


# --------------------------------------------------------------------- ranks --
def xs_rank(f: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cross-sectional percentile rank in [0, 1] per date; masked names NaN."""
    x = f.where(mask) if mask is not None else f
    return x.rank(axis=1, pct=True)


def xs_zscore(f: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    x = f.where(mask) if mask is not None else f
    mu = x.mean(axis=1)
    sd = x.std(axis=1).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0)


# ------------------------------------------------------- fundamentals (PIT) --
def pit_fundamental_panel(md: MarketData, field: str,
                          transform: str = "level") -> pd.DataFrame:
    """Daily date × ticker panel of a fundamental field, point-in-time.

    Each quarterly value enters the panel on its PUBLICATION date (not the
    fiscal period end) and is forward-filled until the next publication.
    transform:
        level      raw value
        ttm        trailing 4-quarter sum
        yoy_growth TTM value vs TTM one year earlier
        accel      change in YoY growth vs previous quarter's YoY growth
    """
    f = md.fundamentals
    cal = md.calendar
    out = pd.DataFrame(index=cal, columns=sorted(f["ticker"].unique()) if not f.empty else [],
                       dtype=float)
    if f.empty:
        return out
    for t, grp in f.groupby("ticker"):
        grp = grp.sort_values("period_end").reset_index(drop=True)
        vals = grp[field].astype(float)
        if transform in ("ttm", "yoy_growth", "accel"):
            ttm = vals.rolling(4).sum()
            if transform == "ttm":
                series = ttm
            else:
                yoy = ttm / ttm.shift(4) - 1
                series = yoy if transform == "yoy_growth" else yoy - yoy.shift(1)
        else:
            series = vals
        pub = pd.DatetimeIndex(grp["published"])
        s = pd.Series(series.to_numpy(), index=pub).dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[t] = s.reindex(cal, method="ffill")
    return out


def price_to_sales(md: MarketData) -> pd.DataFrame:
    """Market cap / TTM revenue, using PIT shares and revenue."""
    rev_ttm = pit_fundamental_panel(md, "revenue", "ttm")
    shares = pit_fundamental_panel(md, "shares", "level")
    cols = [c for c in rev_ttm.columns if c in md.close.columns]
    mktcap = md.close[cols] * shares[cols]
    return mktcap / rev_ttm[cols].replace(0, np.nan)


# --------------------------------------------------------------------- macro --
def macro_regime_flags(md: MarketData) -> pd.DataFrame:
    """Binary/continuous regime descriptors from macro + index data.

    All are trailing quantities: trend uses SMA through T, rate direction uses
    a 63-day change, VIX is the level on T.
    """
    out = pd.DataFrame(index=md.calendar)
    if "QQQ" in md.adj_close.columns:
        q = md.adj_close["QQQ"]
        out["qqq_above_200"] = (q > q.rolling(200).mean()).astype(float)
        out["qqq_ret_63"] = q.pct_change(63)
    m = md.macro
    if "FEDFUNDS" in m.columns:
        out["rates_rising"] = (m["FEDFUNDS"].diff(63) > 0.05).astype(float)
    if "VIXCLS" in m.columns:
        out["vix"] = m["VIXCLS"]
        out["vix_high"] = (m["VIXCLS"] > 25).astype(float)
    if "T10Y2Y" in m.columns:
        out["curve_inverted"] = (m["T10Y2Y"] < 0).astype(float)
    if "BAA10Y" in m.columns:
        out["credit_stress"] = (m["BAA10Y"] > m["BAA10Y"].rolling(252).median() * 1.25).astype(float)
    # Breadth: share of universe names above their own 200-day SMA.
    univ = [s.ticker for s in md.universe.securities if s.ticker in md.adj_close.columns]
    above = above_sma(md.adj_close[univ], 200)
    listed = md.adj_close[univ].notna()
    out["breadth_200"] = above.where(listed).mean(axis=1)
    return out
