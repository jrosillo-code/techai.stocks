"""Portfolio laboratory: combine STRATEGIES instead of securities.

Answers: how correlated are strategies, do they diversify, do they fail in
the same regimes, and can mediocre-but-uncorrelated strategies blend into
something better than any single one? Read-only over recorded curves; blend
returns are simple monthly-rebalanced equal-weight combinations of net
(base-cost) strategy returns — costs of switching BETWEEN strategies are not
modeled and this is flagged in every output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..experiments import ExperimentRegistry
from ..metrics import cagr, max_drawdown, sharpe
from ..utils import get_logger

log = get_logger("platform.portfolio_lab")


def load_strategy_returns(mode: str = "synthetic",
                          include_benchmarks: bool = True,
                          max_strategies: int = 40) -> pd.DataFrame:
    """Daily net (base-scenario) return panel, best-scoring variant per class."""
    registry = ExperimentRegistry.for_mode(mode)
    df = registry.load()
    if df.empty:
        return pd.DataFrame()
    ok = df[(df["status"] == "ok") & (df["scenario"] == "base")].copy()
    ok["dev_sharpe"] = ok["metrics_dev"].map(lambda m: (m or {}).get("sharpe") or np.nan)
    ok["cls"] = ok["strategy"].str.split("(").str[0]
    if not include_benchmarks:
        ok = ok[ok["family"] != "benchmark"]
    best = (ok.sort_values("dev_sharpe", ascending=False)
              .drop_duplicates("cls").head(max_strategies))
    cols = {}
    for rec in best.to_dict("records"):
        c = registry.load_curve(rec["id"])
        if c is not None:
            cols[rec["strategy"]] = c["returns"]
    return pd.DataFrame(cols)


def correlation_matrix(returns: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    monthly = (1 + returns).resample(freq).prod() - 1
    return monthly.corr()


def regime_cofailure(returns: pd.DataFrame, subperiods: dict) -> pd.DataFrame:
    """Sharpe per strategy per named regime window — co-failure at a glance."""
    rows = {}
    for name, (lo, hi) in subperiods.items():
        seg = returns.loc[str(lo):str(hi)]
        if len(seg) < 60:
            continue
        rows[name] = seg.apply(lambda s: sharpe(s.dropna()))
    return pd.DataFrame(rows)


def blend(returns: pd.DataFrame, names: list[str],
          weights: list[float] | None = None) -> pd.Series:
    """Monthly-rebalanced weighted blend of strategy return streams."""
    sub = returns[names].dropna(how="all")
    w = np.array(weights if weights is not None else [1 / len(names)] * len(names))
    w = w / w.sum()
    # monthly-rebalanced: within each month weights drift with performance
    out = []
    for _, month in sub.groupby(pd.Grouper(freq="ME")):
        month = month.fillna(0.0)
        eq = (1 + month).cumprod()
        port = (eq * w).sum(axis=1)
        prev = pd.concat([pd.Series([1.0]), port.iloc[:-1]])
        out.append(pd.Series(port.values / prev.values - 1, index=month.index))
    return pd.concat(out).rename("+".join(n.split("(")[0] for n in names))


def blend_report(returns: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Every pair among the top-N Sharpe strategies: does 50/50 beat both?"""
    stats = {c: sharpe(returns[c].dropna()) for c in returns.columns}
    top = sorted(stats, key=stats.get, reverse=True)[:top_n]
    rows = []
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            if a.split("(")[0] == b.split("(")[0]:
                continue
            pair = blend(returns, [a, b])
            joined = correlation_matrix(returns[[a, b]]).iloc[0, 1]
            rows.append({
                "a": a[:45], "b": b[:45],
                "corr_monthly": round(float(joined), 2),
                "sharpe_a": round(stats[a], 2), "sharpe_b": round(stats[b], 2),
                "sharpe_blend": round(sharpe(pair.dropna()), 2),
                "blend_beats_both": bool(sharpe(pair.dropna()) > max(stats[a], stats[b])),
                "max_dd_blend": round(max_drawdown(pair.dropna()), 2),
                "cagr_blend": round(cagr(pair.dropna()), 3),
            })
    out = pd.DataFrame(rows)
    return out.sort_values("sharpe_blend", ascending=False) if len(out) else out
