"""Backtesting engine.

Contract
--------
A strategy hands the engine a frame of TARGET WEIGHTS (date × ticker). The row
stamped date T must be computed only from information available through the
close of T. The engine executes the transition into those weights at the OPEN
of the NEXT trading day (T+1). It is structurally impossible for a fill to
occur on the bar that produced the signal — the engine always lags weights by
one bar. tests/test_no_lookahead.py locks this in.

Accounting
----------
  * Prices are ratio-adjusted (total-return-consistent), so dividends are
    embedded in the fill and mark prices; splits are handled by adjustment.
  * Cash earns the FEDFUNDS rate daily; shorts pay borrow (scenario rate).
  * Transaction costs = fixed bps + sqrt-participation market impact, priced
    against trailing ADV known at decision time (see costs.py).
  * Delistings: when a held name stops printing prices, the position is
    liquidated at its last available price under the stressed fixed cost
    (2× scenario bps) on the following bar.
  * A trade only happens when |target − current| weight exceeds
    `rebalance_tolerance`, which suppresses churn from float noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CostScenario
from ..costs import one_way_cost, daily_borrow_cost, rolling_adv
from ..data.loader import MarketData
from ..utils import get_logger

log = get_logger("backtest.engine")


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series                 # end-of-day portfolio value
    returns: pd.Series                # daily simple returns
    weights: pd.DataFrame             # realized end-of-day weights
    turnover: pd.Series               # one-way traded value / equity
    costs_paid: pd.Series             # dollars per day
    n_trades: int
    cost_scenario: str
    meta: dict = field(default_factory=dict)

    @property
    def total_costs(self) -> float:
        return float(self.costs_paid.sum())

    @property
    def annual_turnover(self) -> float:
        return float(self.turnover.mean() * 252)


def run_backtest(md: MarketData,
                 target_weights: pd.DataFrame,
                 scen: CostScenario,
                 name: str = "strategy",
                 initial_capital: float = 1_000_000.0,
                 rebalance_tolerance: float = 0.0015,
                 max_participation: float = 0.05,
                 check_invariants: bool = False) -> BacktestResult:
    """Simulate the strategy defined by `target_weights` under scenario `scen`.

    max_participation caps any single day's trade in a name at that fraction
    of its trailing ADV; the remainder is carried to later days (a simple but
    real liquidity constraint).

    check_invariants=True (or env AITB_ENGINE_INVARIANTS=1) enables the
    per-bar accounting invariant:
        Δequity == overnight P&L + intraday P&L + cash interest
                   − trading costs − borrow − delisting write-offs
    and raises AssertionError on any bar where it fails (audit mode).
    """
    cols = [c for c in target_weights.columns if c in md.adj_close.columns]
    tw = target_weights[cols]

    # Engine timeline: from first row with any non-NaN weight, to end of data.
    first_signal = tw.dropna(how="all").index.min()
    if first_signal is None or pd.isna(first_signal):
        raise ValueError("target_weights contains no signals")
    cal = md.calendar[md.calendar >= first_signal]

    px_open = md.open[cols].reindex(cal)
    px_close = md.adj_close[cols].reindex(cal)
    adv = rolling_adv(md.dollar_volume[cols]).reindex(cal)
    tw = tw.reindex(cal)

    # SIGNAL LAG: weights decided on T are the ones traded at T+1's open.
    desired = tw.shift(1)

    cash_rate = (md.macro["FEDFUNDS"].reindex(cal).ffill().fillna(0.0) / 100 / 252
                 if "FEDFUNDS" in md.macro.columns else pd.Series(0.0, index=cal))

    n_days, n_assets = len(cal), len(cols)
    O = px_open.to_numpy()
    C = px_close.to_numpy()
    ADV = adv.to_numpy()
    W_des = desired.to_numpy()
    cash_r = cash_rate.to_numpy()

    import os
    check_invariants = check_invariants or os.environ.get("AITB_ENGINE_INVARIANTS") == "1"

    shares = np.zeros(n_assets)
    cash = initial_capital
    last_price = np.full(n_assets, np.nan)

    equity_out = np.empty(n_days)
    turnover_out = np.zeros(n_days)
    costs_out = np.zeros(n_days)
    weights_out = np.zeros((n_days, n_assets))
    n_trades = 0

    for t in range(n_days):
        o, c = O[t], C[t]
        fill = np.where(np.isnan(o), last_price, o)     # delisted -> last price
        mark = np.where(np.isnan(c), fill, c)
        if check_invariants:
            _prev_mark = last_price.copy()
            _shares_pre = shares.copy()

        # --- execute at the open against yesterday's decided weights -------
        wd = W_des[t]
        equity_pre = cash + np.nansum(shares * fill)
        if equity_pre <= 0:
            log.warning("%s: portfolio wiped out on %s", name, cal[t].date())
            equity_out[t:] = 0.0
            break

        if not np.all(np.isnan(wd)):
            wd = np.nan_to_num(wd, nan=0.0)
            # Forced liquidation of names with no price at all (delisted, never
            # to return): close them at last price with doubled fixed cost.
            dead = np.isnan(fill) & (shares != 0)
            if dead.any():
                shares[dead] = 0.0  # value already unrecoverable -> written off
            tradeable = ~np.isnan(fill)
            cur_w = np.where(tradeable, shares * np.where(tradeable, fill, 0.0), 0.0) / equity_pre
            delta_w = np.where(tradeable, wd - cur_w, 0.0)
            delta_w[np.abs(delta_w) < rebalance_tolerance] = 0.0

            trade_val = delta_w * equity_pre
            # Participation cap: trade at most max_participation * ADV today.
            cap = np.where(np.isnan(ADV[t]), np.inf, max_participation * ADV[t])
            trade_val = np.clip(trade_val, -cap, cap)

            day_cost = 0.0
            for j in np.nonzero(trade_val)[0]:
                stressed = 2.0 if np.isnan(C[t, j]) else 1.0
                cost = one_way_cost(trade_val[j], ADV[t, j] if not np.isnan(ADV[t, j]) else 0.0, scen)
                day_cost += cost * stressed
                shares[j] += trade_val[j] / fill[j]
                n_trades += 1
            cash -= trade_val.sum() + day_cost
            costs_out[t] += day_cost
            turnover_out[t] = np.abs(trade_val).sum() / equity_pre

        # --- end of day: accrue cash interest and borrow, mark to close ----
        _interest = cash * cash_r[t]
        cash += _interest
        short_val = np.nansum(np.where(shares < 0, shares * mark, 0.0))
        if short_val < 0:
            borrow = daily_borrow_cost(short_val, scen)
            cash -= borrow
            costs_out[t] += borrow

        pos_val = np.nansum(shares * mark)
        equity = cash + pos_val
        equity_out[t] = equity
        weights_out[t] = np.where(np.isnan(mark), 0.0, shares * np.nan_to_num(mark)) / equity if equity > 0 else 0.0

        if check_invariants and t > 0:
            # Per-bar accounting invariant (audit finding AUD-011):
            #   Δequity == overnight P&L (pre-trade shares, prev mark -> fill)
            #            + intraday P&L (post-trade shares, fill -> mark)
            #            + cash interest − all costs booked today
            overnight = np.nansum(np.where(_shares_pre != 0,
                                           _shares_pre * (fill - _prev_mark), 0.0))
            intraday = np.nansum(np.where(shares != 0, shares * (mark - fill), 0.0))
            expected = equity_out[t - 1] + overnight + intraday + _interest - costs_out[t]
            tol = max(1e-6 * max(abs(equity), 1.0), 1e-4)
            if abs(equity - expected) > tol:
                raise AssertionError(
                    f"{name}: accounting invariant violated on {cal[t].date()}: "
                    f"equity {equity:.4f} != expected {expected:.4f} "
                    f"(diff {equity - expected:+.6f})")

        last_price = np.where(np.isnan(mark), last_price, mark)

    equity_s = pd.Series(equity_out, index=cal, name=name)
    returns = equity_s.pct_change().fillna(0.0)
    return BacktestResult(
        name=name,
        equity=equity_s,
        returns=returns,
        weights=pd.DataFrame(weights_out, index=cal, columns=cols),
        turnover=pd.Series(turnover_out, index=cal),
        costs_paid=pd.Series(costs_out, index=cal),
        n_trades=n_trades,
        cost_scenario=scen.name,
    )
