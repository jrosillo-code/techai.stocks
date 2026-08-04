"""Approximate tax-aware analysis for a US taxable investor.

This is an APPROXIMATION, clearly separated from pre-tax results. It does not
do lot-level accounting; it models annual settlement from turnover:

  * average holding period ≈ 1 / annual turnover;
  * the realized fraction of each year's gain ≈ min(turnover, 1);
  * realized gains are short-term when the implied holding period < 1 year,
    long-term otherwise;
  * dividends (estimated from the total-return vs price divergence, or a
    configured yield) are taxed annually at the dividend rate;
  * net losses carry forward; no tax-loss harvesting unless enabled, and then
    only as a crude annual offset with a wash-sale haircut;
  * `deferred` mode taxes nothing until final liquidation (LT rate), the
    tax-advantaged-account comparison.

Good enough to answer "does the strategy's edge survive taxes?", not good
enough to file returns with.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TaxConfig:
    st_rate: float = 0.37
    lt_rate: float = 0.20
    dividend_rate: float = 0.20
    harvest_losses: bool = False
    wash_sale_haircut: float = 0.30   # fraction of harvested losses disallowed
    deferred: bool = False            # IRA/401k-style: tax only at liquidation


@dataclass
class TaxResult:
    pre_tax_equity: pd.Series
    after_tax_equity: pd.Series
    taxes_paid: pd.Series             # by year
    pre_tax_cagr: float
    after_tax_cagr: float
    tax_drag_annual: float


def apply_tax_overlay(returns: pd.Series, turnover: pd.Series,
                      dividend_yield_annual: float = 0.005,
                      cfg: TaxConfig = TaxConfig()) -> TaxResult:
    r = returns.dropna()
    if len(r) < 252:
        raise ValueError("need at least one year of returns for tax analysis")

    yearly_r = (1 + r).groupby(r.index.year).prod() - 1
    yearly_to = turnover.reindex(r.index).fillna(0.0).groupby(r.index.year).sum()

    equity = 1.0
    equity_path = []
    taxes = {}
    loss_carry = 0.0
    for year in yearly_r.index:
        gross = yearly_r.loc[year]
        to = float(yearly_to.get(year, 0.0))
        start_eq = equity
        gain = start_eq * gross

        div = start_eq * dividend_yield_annual
        tax = div * cfg.dividend_rate if not cfg.deferred else 0.0

        if not cfg.deferred:
            realized_frac = min(to, 1.0)
            realized = gain * realized_frac
            holding_years = 1.0 / to if to > 0 else np.inf
            rate = cfg.st_rate if holding_years < 1.0 else cfg.lt_rate
            if realized > 0:
                offset = min(loss_carry, realized)
                loss_carry -= offset
                tax += (realized - offset) * rate
            elif realized < 0:
                usable = -realized * ((1 - cfg.wash_sale_haircut)
                                      if cfg.harvest_losses else 1.0)
                loss_carry += usable
        equity = start_eq + gain - tax
        taxes[year] = tax
        equity_path.append((year, equity))

    if cfg.deferred:
        # Liquidation at the end: LT tax on all accumulated gain.
        final_gain = equity - 1.0
        if final_gain > 0:
            liq_tax = final_gain * cfg.lt_rate
            equity -= liq_tax
            taxes[yearly_r.index[-1]] = taxes.get(yearly_r.index[-1], 0.0) + liq_tax
            equity_path[-1] = (yearly_r.index[-1], equity)

    years = len(yearly_r)
    pre_eq = (1 + yearly_r).cumprod()
    after_eq = pd.Series(dict(equity_path))
    pre_cagr = float(pre_eq.iloc[-1] ** (1 / years) - 1)
    after_cagr = float(max(after_eq.iloc[-1], 1e-9) ** (1 / years) - 1)
    return TaxResult(
        pre_tax_equity=pre_eq,
        after_tax_equity=after_eq,
        taxes_paid=pd.Series(taxes),
        pre_tax_cagr=pre_cagr,
        after_tax_cagr=after_cagr,
        tax_drag_annual=pre_cagr - after_cagr,
    )
