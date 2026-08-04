"""Performance and risk metrics for daily return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def _clean(r: pd.Series) -> pd.Series:
    return r.dropna()


def total_return(r: pd.Series) -> float:
    return float((1 + _clean(r)).prod() - 1)


def cagr(r: pd.Series) -> float:
    """Compound annual growth rate over ELAPSED CALENDAR TIME.

    Audit finding AUD-006: dividing the row count by 252 misstates years on
    calendars with a different session density (a plain business-day calendar
    has ~261 rows/year, understating CAGR by ~35bp/yr). Elapsed days / 365.25
    is calendar-correct regardless of session convention.
    """
    r = _clean(r)
    if len(r) < 2:
        return np.nan
    if isinstance(r.index, pd.DatetimeIndex):
        years = max((r.index[-1] - r.index[0]).days, 1) / 365.25
    else:
        years = len(r) / ANN
    growth = float((1 + r).prod())
    if growth <= 0:
        return -1.0
    return growth ** (1 / years) - 1


def ann_vol(r: pd.Series) -> float:
    return float(_clean(r).std() * np.sqrt(ANN))


def sharpe(r: pd.Series, rf_daily: pd.Series | float = 0.0) -> float:
    r = _clean(r)
    if isinstance(rf_daily, pd.Series):
        ex = r - rf_daily.reindex(r.index).fillna(0.0)
    else:
        ex = r - rf_daily
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan


def sortino(r: pd.Series, rf_daily: float = 0.0) -> float:
    r = _clean(r) - rf_daily
    downside = r[r < 0]
    dd = downside.std()
    return float(r.mean() / dd * np.sqrt(ANN)) if dd and dd > 0 else np.nan


def drawdown_series(r: pd.Series) -> pd.Series:
    eq = (1 + _clean(r)).cumprod()
    return eq / eq.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    dd = drawdown_series(r)
    return float(dd.min()) if len(dd) else np.nan


def avg_drawdown(r: pd.Series) -> float:
    dd = drawdown_series(r)
    in_dd = dd[dd < 0]
    return float(in_dd.mean()) if len(in_dd) else 0.0


def max_drawdown_duration_days(r: pd.Series) -> int:
    dd = drawdown_series(r)
    at_peak = dd >= -1e-12
    run = 0
    worst = 0
    for flag in at_peak:
        run = 0 if flag else run + 1
        worst = max(worst, run)
    return worst


def calmar(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    return float(cagr(r) / abs(mdd)) if mdd and mdd < 0 else np.nan


def var_cvar(r: pd.Series, level: float = 0.95) -> tuple[float, float]:
    r = _clean(r)
    if len(r) < 20:
        return np.nan, np.nan
    var = float(np.quantile(r, 1 - level))
    tail = r[r <= var]
    return var, float(tail.mean()) if len(tail) else var


def beta_alpha(r: pd.Series, bench: pd.Series) -> tuple[float, float]:
    df = pd.concat([_clean(r), _clean(bench)], axis=1, join="inner").dropna()
    if len(df) < 60:
        return np.nan, np.nan
    x, y = df.iloc[:, 1], df.iloc[:, 0]
    cov = np.cov(y, x)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan
    alpha = float((y.mean() - beta * x.mean()) * ANN)
    return beta, alpha


def tracking_error(r: pd.Series, bench: pd.Series) -> float:
    df = pd.concat([_clean(r), _clean(bench)], axis=1, join="inner").dropna()
    return float((df.iloc[:, 0] - df.iloc[:, 1]).std() * np.sqrt(ANN))


def information_ratio(r: pd.Series, bench: pd.Series) -> float:
    df = pd.concat([_clean(r), _clean(bench)], axis=1, join="inner").dropna()
    diff = df.iloc[:, 0] - df.iloc[:, 1]
    sd = diff.std()
    return float(diff.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan


def capture(r: pd.Series, bench: pd.Series) -> tuple[float, float]:
    df = pd.concat([_clean(r), _clean(bench)], axis=1, join="inner").dropna()
    up = df[df.iloc[:, 1] > 0]
    dn = df[df.iloc[:, 1] < 0]
    upc = float(up.iloc[:, 0].mean() / up.iloc[:, 1].mean()) if len(up) else np.nan
    dnc = float(dn.iloc[:, 0].mean() / dn.iloc[:, 1].mean()) if len(dn) else np.nan
    return upc, dnc


def win_rate(r: pd.Series) -> float:
    r = _clean(r)
    active = r[r != 0]
    return float((active > 0).mean()) if len(active) else np.nan


def profit_factor(r: pd.Series) -> float:
    r = _clean(r)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else np.nan


def best_worst(r: pd.Series) -> dict[str, float]:
    r = _clean(r)
    out: dict[str, float] = {"best_day": float(r.max()), "worst_day": float(r.min())}
    for label, freq in (("month", "ME"), ("quarter", "QE"), ("year", "YE")):
        agg = (1 + r).resample(freq).prod() - 1
        agg = agg[agg != 0]
        if len(agg):
            out[f"best_{label}"] = float(agg.max())
            out[f"worst_{label}"] = float(agg.min())
    return out


def rolling_sharpe(r: pd.Series, window_years: int = 1) -> pd.Series:
    w = window_years * ANN
    r = _clean(r)
    mean = r.rolling(w).mean()
    sd = r.rolling(w).std()
    return (mean / sd * np.sqrt(ANN)).rename(f"rolling_sharpe_{window_years}y")


def rolling_cagr(r: pd.Series, window_years: int = 1) -> pd.Series:
    w = window_years * ANN
    growth = (1 + _clean(r)).rolling(w).apply(np.prod, raw=True)
    return (growth ** (1 / window_years) - 1).rename(f"rolling_cagr_{window_years}y")


def summary(r: pd.Series,
            bench: pd.Series | None = None,
            rf_daily: pd.Series | float = 0.0,
            turnover_ann: float | None = None) -> dict[str, float]:
    """Full metric dictionary for one daily-return series."""
    r = _clean(r)
    var95, cvar95 = var_cvar(r)
    out = {
        "total_return": total_return(r),
        "cagr": cagr(r),
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r, rf_daily),
        "sortino": sortino(r),
        "calmar": calmar(r),
        "max_drawdown": max_drawdown(r),
        "avg_drawdown": avg_drawdown(r),
        "max_dd_duration_days": max_drawdown_duration_days(r),
        "var_95": var95,
        "cvar_95": cvar95,
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "win_rate": win_rate(r),
        "profit_factor": profit_factor(r),
        "n_days": int(len(r)),
    }
    out.update(best_worst(r))
    if bench is not None:
        b, a = beta_alpha(r, bench)
        upc, dnc = capture(r, bench)
        df = pd.concat([r, _clean(bench)], axis=1, join="inner").dropna()
        out.update({
            "beta": b, "alpha_ann": a,
            "correlation": float(df.iloc[:, 0].corr(df.iloc[:, 1])) if len(df) > 30 else np.nan,
            "tracking_error": tracking_error(r, bench),
            "information_ratio": information_ratio(r, bench),
            "upside_capture": upc, "downside_capture": dnc,
        })
    if turnover_ann is not None:
        out["annual_turnover"] = turnover_ann
    return out
