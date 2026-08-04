#!/usr/bin/env python3
"""Phase 5: per-company analysis — buy & hold vs timing overlays.

For every company in the universe:
  * buy & hold since IPO (and since the common start date),
  * standardized trailing 1/3/5/10-year returns,
  * volatility, max drawdown, correlation with QQQ,
  * a 200-day SMA timing overlay on THAT stock alone (cash -> IEF fallback),
    under zero and base costs, so "own it" vs "trade it" is answered per name.

Output: results/company_analysis.csv (+ parquet).

Usage: python scripts/run_company_analysis.py [--provider synthetic]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aitb.backtest.engine import run_backtest
from aitb.config import results_dir, load_cost_scenarios
from aitb.data.loader import load_market_data
from aitb.features import above_sma
from aitb.metrics import ann_vol, cagr, max_drawdown, sharpe, summary
from aitb.utils import get_logger

log = get_logger("company_analysis")


def trailing_return(r: pd.Series, years: int) -> float:
    n = years * 252
    if len(r) < n:
        return np.nan
    seg = r.iloc[-n:]
    return float((1 + seg).prod() ** (1 / years) - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="synthetic")
    ap.add_argument("--data-mode", default="synthetic", choices=["synthetic", "real"])
    args = ap.parse_args()

    if args.data_mode == "real":
        from aitb.data.quality import require_gate
        require_gate()
        from aitb.freeze import verify_freeze
        verify_freeze()   # audit AUD-002: derived analytics must run frozen code

        md = load_market_data(mode="real")
    else:
        md = load_market_data(args.provider, mode="synthetic")
    scens = load_cost_scenarios()
    qqq = md.adj_close["QQQ"].pct_change()

    rows = []
    for sec in md.universe.securities:
        t = sec.ticker
        if t not in md.adj_close.columns:
            continue
        px = md.adj_close[t].dropna()
        if len(px) < 504:
            continue
        r = px.pct_change().dropna()

        # --- timing overlay: hold the stock only above its 200d SMA --------
        sig = above_sma(md.adj_close[[t]], 200)[t]
        w = pd.DataFrame(0.0, index=md.calendar, columns=[t, "IEF"])
        w[t] = sig.reindex(md.calendar).fillna(0.0)
        ief_ok = md.adj_close["IEF"].notna() if "IEF" in md.adj_close.columns else False
        w["IEF"] = ((1 - w[t]) * ief_ok).fillna(0.0)
        w = w.loc[px.index[0]:]
        overlay = {}
        for scen_name in ("zero", "base"):
            try:
                res = run_backtest(md, w, scens[scen_name], name=f"sma200_{t}")
                overlay[scen_name] = res.returns
            except Exception as exc:
                log.warning("overlay failed for %s [%s]: %s", t, scen_name, exc)

        row = {
            "ticker": t,
            "name": sec.name,
            "sector": sec.sector,
            "ipo": sec.ipo,
            "delisted": sec.delisted,
            "first_data": px.index[0].date(),
            "last_data": px.index[-1].date(),
            "bh_cagr_since_ipo": cagr(r),
            "bh_vol": ann_vol(r),
            "bh_sharpe": sharpe(r),
            "bh_max_dd": max_drawdown(r),
            "corr_qqq": float(pd.concat([r, qqq], axis=1).corr().iloc[0, 1]),
            "ret_1y": trailing_return(r, 1),
            "ret_3y": trailing_return(r, 3),
            "ret_5y": trailing_return(r, 5),
            "ret_10y": trailing_return(r, 10),
        }
        for scen_name, orets in overlay.items():
            row[f"overlay_cagr_{scen_name}"] = cagr(orets)
            row[f"overlay_sharpe_{scen_name}"] = sharpe(orets)
            row[f"overlay_max_dd_{scen_name}"] = max_drawdown(orets)
        if "base" in overlay:
            row["timing_adds_value_after_costs"] = bool(
                sharpe(overlay["base"]) > sharpe(r) and cagr(overlay["base"]) > 0)
        rows.append(row)
        log.info("%-5s B&H cagr=%6.1f%% dd=%5.0f%% | overlay(base) cagr=%s",
                 t, 100 * row["bh_cagr_since_ipo"], 100 * row["bh_max_dd"],
                 f"{100 * row.get('overlay_cagr_base', float('nan')):.1f}%%")

    out = pd.DataFrame(rows)
    rdir = results_dir(args.data_mode)
    rdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(rdir / "company_analysis.csv", index=False)
    out.to_parquet(rdir / "company_analysis.parquet")
    log.info("wrote %d company rows to %s", len(out), rdir / "company_analysis.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
