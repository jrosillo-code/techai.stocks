"""Validation and anti-overfitting machinery.

  * chronological development/holdout split (holdout untouched by selection),
  * walk-forward parameter selection (select on trailing window, trade forward),
  * moving-block bootstrap CIs for Sharpe and CAGR,
  * Probabilistic and Deflated Sharpe Ratio (Bailey & López de Prado),
  * parameter-sensitivity grids,
  * subperiod/regime metrics,
  * strategy correlation / redundancy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import cagr, sharpe, summary

EULER_GAMMA = 0.5772156649015329


# ------------------------------------------------------------------ splits --
@dataclass(frozen=True)
class Split:
    dev_start: pd.Timestamp
    dev_end: pd.Timestamp        # last day usable for any selection/tuning
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp


def make_split(cal: pd.DatetimeIndex, holdout_months: int) -> Split:
    holdout_start = cal[-1] - pd.DateOffset(months=holdout_months)
    dev = cal[cal < holdout_start]
    hold = cal[cal >= holdout_start]
    return Split(dev[0], dev[-1], hold[0], hold[-1])


def slice_dev(r: pd.Series, split: Split) -> pd.Series:
    return r.loc[:split.dev_end]


def slice_holdout(r: pd.Series, split: Split) -> pd.Series:
    return r.loc[split.holdout_start:]


# ------------------------------------------------- walk-forward selection ---
def walk_forward_select(returns_by_param: dict[str, pd.Series],
                        min_train_months: int = 60,
                        step_months: int = 12,
                        criterion=sharpe) -> tuple[pd.Series, pd.DataFrame]:
    """Genuine out-of-sample composite for a parameter family.

    At each step date, pick the parameter set with the best `criterion` over
    all PRIOR data, then hold that variant's returns for the next step window.
    Returns (composite_oos_returns, selection_log).
    """
    all_idx = None
    for r in returns_by_param.values():
        all_idx = r.index if all_idx is None else all_idx.union(r.index)
    panel = pd.DataFrame({k: v for k, v in returns_by_param.items()}).reindex(all_idx)

    start = all_idx[0] + pd.DateOffset(months=min_train_months)
    steps = pd.date_range(start, all_idx[-1], freq=f"{step_months}MS")
    out = pd.Series(0.0, index=all_idx, name="walk_forward")
    out[:] = np.nan
    log_rows = []
    for i, step in enumerate(steps):
        train = panel.loc[:step - pd.Timedelta(days=1)]
        scores = {k: criterion(train[k].dropna()) for k in panel.columns}
        scores = {k: v for k, v in scores.items() if np.isfinite(v)}
        if not scores:
            continue
        best = max(scores, key=scores.get)
        nxt = steps[i + 1] if i + 1 < len(steps) else all_idx[-1] + pd.Timedelta(days=1)
        window = panel.loc[step:nxt - pd.Timedelta(days=1), best]
        out.loc[window.index] = window
        log_rows.append({"step": step, "selected": best,
                         "train_score": scores[best]})
    return out.dropna(), pd.DataFrame(log_rows)


# ----------------------------------------------------------------- bootstrap --
def block_bootstrap_ci(r: pd.Series, stat=sharpe, n_boot: int = 2000,
                       block: int = 21, seed: int = 0,
                       level: float = 0.90) -> tuple[float, float, float]:
    """Moving-block bootstrap CI (point, lo, hi) preserving short-range
    autocorrelation."""
    x = r.dropna().to_numpy()
    n = len(x)
    if n < 2 * block:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(n_boot, n_blocks))
    stats_out = np.empty(n_boot)
    idx_template = np.arange(block)
    for b in range(n_boot):
        sample = x[(starts[b][:, None] + idx_template).ravel()[:n]]
        s = pd.Series(sample)
        stats_out[b] = stat(s)
    stats_out = stats_out[np.isfinite(stats_out)]
    if len(stats_out) == 0:
        return np.nan, np.nan, np.nan
    alpha = (1 - level) / 2
    return float(stat(pd.Series(x))), float(np.quantile(stats_out, alpha)), float(np.quantile(stats_out, 1 - alpha))


def trade_sequence_monte_carlo(r: pd.Series, n_paths: int = 2000,
                               seed: int = 0) -> dict[str, float]:
    """Shuffle daily returns to gauge path dependence of drawdown/CAGR."""
    x = r.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    mdds = np.empty(n_paths)
    for i in range(n_paths):
        perm = rng.permutation(x)
        eq = np.cumprod(1 + perm)
        mdds[i] = (eq / np.maximum.accumulate(eq) - 1).min()
    return {"mdd_median": float(np.median(mdds)),
            "mdd_p95": float(np.quantile(mdds, 0.05)),
            "observed_order_luck": float((mdds > (pd.Series(x).pipe(_mdd))).mean())}


def _mdd(s: pd.Series) -> float:
    eq = (1 + s).cumprod()
    return float((eq / eq.cummax() - 1).min())


# ---------------------------------------------- PSR / DSR (multiple testing) --
def probabilistic_sharpe(r: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > benchmark_sr) given the observed series (annualized
    inputs are converted internally to per-period units)."""
    x = r.dropna()
    T = len(x)
    if T < 60:
        return np.nan
    sr = x.mean() / x.std() if x.std() > 0 else np.nan  # per-period SR
    if not np.isfinite(sr):
        return np.nan
    sr_b = benchmark_sr / np.sqrt(252)
    g3 = float(stats.skew(x))
    g4 = float(stats.kurtosis(x, fisher=False))
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr_b) * np.sqrt(T - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_trial_sr: float) -> float:
    """E[max SR] across n_trials of zero-true-SR strategies (per-period units)."""
    if n_trials <= 1 or var_trial_sr <= 0:
        return 0.0
    sd = np.sqrt(var_trial_sr)
    return sd * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                 + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(r: pd.Series, trial_sharpes_ann: list[float]) -> float:
    """DSR: PSR against the expected-max Sharpe of the whole trial battery.

    `trial_sharpes_ann` must include EVERY variant tried in the family —
    failed ones too. Forgetting failures is exactly the bias this corrects.
    """
    if len(trial_sharpes_ann) < 2:
        return probabilistic_sharpe(r, 0.0)
    per_period = np.array(trial_sharpes_ann, dtype=float) / np.sqrt(252)
    per_period = per_period[np.isfinite(per_period)]
    sr_star = expected_max_sharpe(len(per_period), float(np.var(per_period)))
    return probabilistic_sharpe(r, sr_star * np.sqrt(252))


# ------------------------------------------------------------- sensitivity --
def parameter_sensitivity(returns_by_param: dict[str, pd.Series],
                          stat=sharpe) -> pd.DataFrame:
    rows = [{"variant": k, "stat": stat(v.dropna()), "n_days": len(v.dropna())}
            for k, v in returns_by_param.items()]
    df = pd.DataFrame(rows).sort_values("stat", ascending=False)
    df["neighbor_stability"] = df["stat"].rolling(3, center=True, min_periods=1).std()
    return df


def subperiod_table(r: pd.Series, subperiods: dict[str, tuple],
                    bench: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for name, (lo, hi) in subperiods.items():
        seg = r.loc[str(lo):str(hi)]
        if len(seg) < 60:
            continue
        row = {"period": name, "start": seg.index[0].date(), "end": seg.index[-1].date(),
               "cagr": cagr(seg), "sharpe": sharpe(seg), "max_dd": _mdd(seg)}
        if bench is not None:
            bseg = bench.loc[str(lo):str(hi)]
            row["bench_cagr"] = cagr(bseg)
            row["excess_cagr"] = row["cagr"] - row["bench_cagr"]
        rows.append(row)
    return pd.DataFrame(rows)


def strategy_correlation(returns: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(returns).corr()


def stationary_bootstrap_ci(r: pd.Series, stat=sharpe, n_boot: int = 2000,
                            mean_block: int = 21, seed: int = 0,
                            level: float = 0.90) -> tuple[float, float, float]:
    """Politis–Romano stationary bootstrap: geometric block lengths (mean
    `mean_block`), wrap-around resampling. Complements the fixed-block
    bootstrap; run both and compare (block-length sensitivity)."""
    x = r.dropna().to_numpy()
    n = len(x)
    if n < 2 * mean_block:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(n)
        restarts = rng.random(n) < p
        steps = rng.integers(n, size=n)
        for t in range(1, n):
            idx[t] = steps[t] if restarts[t] else (idx[t - 1] + 1) % n
        out[b] = stat(pd.Series(x[idx]))
    out = out[np.isfinite(out)]
    if not len(out):
        return np.nan, np.nan, np.nan
    a = (1 - level) / 2
    return float(stat(pd.Series(x))), float(np.quantile(out, a)), float(np.quantile(out, 1 - a))


def contribution_shares(weights: pd.DataFrame, asset_returns: pd.DataFrame,
                        top_n: int = 5) -> dict:
    """P&L share of the top 1/3/5 contributors and of NVDA specifically."""
    contrib = (weights.shift(1) * asset_returns.reindex(weights.index)).sum()
    contrib = contrib[np.isfinite(contrib)]
    total = contrib.sum()
    if total == 0 or contrib.empty:
        return {}
    ranked = contrib.sort_values(ascending=False)
    out = {
        "top1_share": float(ranked.iloc[:1].sum() / total),
        "top3_share": float(ranked.iloc[:3].sum() / total),
        "top5_share": float(ranked.iloc[:top_n].sum() / total),
        "top_contributors": list(ranked.index[:top_n]),
    }
    if "NVDA" in contrib.index:
        out["nvda_share"] = float(contrib["NVDA"] / total)
    return out


def period_split_metrics(r: pd.Series, split_date: str = "2023-01-01",
                         bench: pd.Series | None = None) -> dict:
    """Metrics before/after a date — the post-2022 AI-rally dependence test."""
    pre, post = r.loc[:split_date], r.loc[split_date:]
    out = {}
    for label, seg in (("pre", pre), ("post", post)):
        if len(seg) < 60:
            continue
        out[label] = {"cagr": cagr(seg), "sharpe": sharpe(seg), "max_dd": _mdd(seg)}
        if bench is not None:
            bseg = bench.reindex(seg.index).dropna()
            if len(bseg) > 60:
                out[label]["excess_cagr"] = cagr(seg) - cagr(bseg)
    if "pre" in out and "post" in out:
        post_only = out["post"]["sharpe"] > 0.5 and out["pre"]["sharpe"] <= 0.0
        out["ai_rally_dependent"] = bool(post_only)
    return out


def leave_one_year_out(r: pd.Series, stat=sharpe) -> dict:
    """Stat when each calendar year is removed — a strategy carried by one
    year is fragile."""
    years = sorted(set(r.dropna().index.year))
    if len(years) < 4:
        return {}
    vals = {}
    for y in years:
        vals[y] = stat(r[r.index.year != y])
    s = pd.Series(vals)
    full = stat(r)
    return {"full": float(full), "min": float(s.min()), "max": float(s.max()),
            "worst_year_removed": int(s.idxmax()),  # removing it helps most
            "most_supportive_year": int(s.idxmin()),
            "range": float(s.max() - s.min())}


def concentration_diagnostics(weights: pd.DataFrame,
                              asset_returns: pd.DataFrame) -> dict[str, float]:
    """How much of the P&L comes from a single name? (fragility indicator)"""
    contrib = (weights.shift(1) * asset_returns.reindex(weights.index)).sum()
    total = contrib.sum()
    if total == 0 or not np.isfinite(total) or contrib.abs().isna().all():
        return {"top_name_share": np.nan, "top_name": ""}
    top = contrib.abs().idxmax()
    return {"top_name_share": float(contrib[top] / total), "top_name": str(top)}
