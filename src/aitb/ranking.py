"""Composite strategy ranking.

CAGR alone is never the criterion. The composite score rewards risk-adjusted
holdout performance, stability, cost robustness and simplicity, and penalizes
turnover, single-name concentration, regime concentration and complexity.
Scores are only comparable within one data version / universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _nz(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def score_record(rec: dict) -> dict:
    """Composite score for one registry record (base-cost scenario row)."""
    dev = rec.get("metrics_dev", {}) or {}
    hold = rec.get("metrics_holdout", {}) or {}
    subs = pd.DataFrame(rec.get("subperiods", []))

    dev_sharpe = _nz(dev.get("sharpe"))
    hold_sharpe = _nz(hold.get("sharpe"))
    calmar = _nz(dev.get("calmar"))
    max_dd = _nz(dev.get("max_drawdown"), -1.0)
    psr = _nz(rec.get("psr_dev"), 0.5)
    turnover = _nz(rec.get("annual_turnover"))
    n_params = len((rec.get("spec") or {}).get("params", {}))
    top_share = _nz((rec.get("concentration") or {}).get("top_name_share"), 1.0)

    # Stability: fraction of subperiods with positive Sharpe, and its spread.
    if len(subs) >= 3 and "sharpe" in subs:
        s = subs["sharpe"].astype(float)
        stability = float((s > 0).mean())
        regime_spread = float(s.std())
        pre2023 = subs[~subs["period"].isin(["ai_rally"])]["sharpe"].astype(float)
        fails_pre_ai = bool(len(pre2023) >= 2 and (pre2023 <= 0).mean() > 0.6)
    else:
        stability, regime_spread, fails_pre_ai = 0.5, 1.0, False

    # IS -> holdout degradation.
    degradation = max(dev_sharpe - hold_sharpe, 0.0)

    score = (
        2.0 * hold_sharpe
        + 1.0 * dev_sharpe
        + 0.5 * min(calmar, 3.0)
        + 1.0 * stability
        + 1.0 * (psr - 0.5)
        - 1.0 * degradation
        - 0.3 * min(turnover / 5.0, 2.0)          # >5x/yr turnover starts to hurt
        - 0.5 * max(top_share - 0.5, 0.0) * 2.0   # >50% of P&L from one name
        - 0.1 * max(n_params - 3, 0)              # complexity
        - 0.3 * min(regime_spread / 2.0, 1.0)
        + 0.5 * max_dd                            # max_dd is negative
        - (1.0 if fails_pre_ai else 0.0)
    )
    return {
        "strategy": rec.get("strategy"),
        "family": rec.get("family"),
        "score": round(float(score), 3),
        "dev_sharpe": round(dev_sharpe, 2),
        "holdout_sharpe": round(hold_sharpe, 2),
        "dev_cagr": round(_nz(dev.get("cagr")), 4),
        "holdout_cagr": round(_nz(hold.get("cagr")), 4),
        "max_drawdown": round(max_dd, 3),
        "calmar": round(calmar, 2),
        "psr": round(psr, 3),
        "annual_turnover": round(turnover, 2),
        "top_name_share": round(top_share, 2),
        "stability": round(stability, 2),
        "degradation": round(degradation, 2),
        "fails_pre_ai_rally": fails_pre_ai,
    }


def current_cohort(registry_df: pd.DataFrame) -> pd.DataFrame:
    """Rows belonging to the newest universe cohort in the registry.

    The registry is append-only across freezes, so it accumulates results run
    against DIFFERENT universes. A 33-name roster and a 120-name roster are not
    the same experiment, and a score computed on one says nothing about the
    other — ranking them together would be the universe-level version of the
    evidence-tier mixing the study already forbids. Older cohorts stay in the
    registry (nothing is ever deleted) but leave the leaderboard.
    """
    if registry_df.empty or "universe_hash" not in registry_df.columns:
        return registry_df
    known = registry_df[registry_df["universe_hash"].notna()]
    if known.empty:
        return registry_df
    # A cohort is (universe, freeze). The universe alone is not enough: freeze
    # v4 added a strategy family without touching the roster, so v3 and v4
    # results share a universe hash while belonging to different studies with
    # different trial counts. Ranking them together would understate the
    # multiple-testing correction for both.
    row = known.sort_values("timestamp").iloc[-1]
    latest_u = row["universe_hash"]
    sel = registry_df["universe_hash"] == latest_u
    if "freeze_version" in registry_df.columns and pd.notna(row.get("freeze_version")):
        sel &= registry_df["freeze_version"] == row["freeze_version"]
    return registry_df[sel]


def rank_experiments(registry_df: pd.DataFrame,
                     scenario: str = "base") -> pd.DataFrame:
    """Rank all OK experiments run under `scenario`, plus cost-robustness
    columns comparing against the stressed scenario when present.

    Only the newest universe cohort is ranked (see ``current_cohort``).
    """
    if "data_mode" in registry_df.columns:
        modes = set(registry_df["data_mode"].dropna().unique())
        if len(modes) > 1:
            raise ValueError(
                f"refusing to rank across data modes {sorted(modes)} — real and "
                "synthetic experiments live in separate registries by design")
    ok = registry_df[registry_df["status"] == "ok"]
    ok = current_cohort(ok)
    base = ok[ok["scenario"] == scenario]
    rows = [score_record(rec) for rec in base.to_dict("records")]
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    stressed = ok[ok["scenario"] == "stressed"]
    if not stressed.empty:
        s_map = {r["strategy"]: _nz((r.get("metrics_dev") or {}).get("sharpe"))
                 for r in stressed.to_dict("records")}
        out["stressed_sharpe"] = out["strategy"].map(s_map)
        out["cost_fragile"] = (out["dev_sharpe"] > 0.5) & (out["stressed_sharpe"] < 0.25 * out["dev_sharpe"])
        out.loc[out["cost_fragile"].fillna(False), "score"] -= 1.0

    out = out.sort_values("score", ascending=False).reset_index(drop=True)

    # The hurdle for active strategies is RELATIVE: the best score among the
    # simple, diversified benchmarks (index ETFs, equal/cap-weight universes,
    # the 200dma baseline, 12-1 momentum). Single-company buy & hold is
    # excluded from the hurdle — picking the one mega-winner ex post is not a
    # fair alternative anyone could have chosen ex ante.
    # Broad or sector index funds: diversified, buyable in advance by anyone,
    # and therefore fair members of the hurdle. Any other single-ticker
    # buy & hold is one company chosen with hindsight and stays excluded.
    etf_bh = {f"BuyAndHold(ticker={t})" for t in
              ("SPY", "QQQ", "XLK", "SOXX", "SMH", "IGV",
               "VGT", "IWM", "ARKK", "SKYY", "CIBR", "BOTZ")}
    is_bench = out["family"] == "benchmark"
    diversified = is_bench & (~out["strategy"].str.startswith("BuyAndHold(")
                              | out["strategy"].isin(etf_bh))
    hurdle = out.loc[diversified, "score"].max() if diversified.any() else 0.0
    margin = 0.25

    def verdict(row) -> str:
        if row["family"] == "benchmark":
            return "benchmark"
        if (row["score"] > hurdle + margin and row["holdout_sharpe"] > 0
                and not row.get("cost_fragile", False)):
            return "robust_candidate"
        if row["score"] > hurdle - 1.0 and row["holdout_sharpe"] > 0:
            return "inconclusive"
        return "rejected"
    out["verdict"] = out.apply(verdict, axis=1)
    out.attrs["benchmark_hurdle"] = float(hurdle)
    return out
