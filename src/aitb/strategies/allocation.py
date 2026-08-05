"""Allocation strategies added under research freeze v3.

These do not pick stocks. They take a fixed basket and ask whether a smarter
WEIGHTING scheme earns its keep — the cheapest possible improvement, since it
requires no forecasting skill at all.

The v2 study already showed inverse-volatility weighting beating equal weight
on Sharpe for the megacap basket. That leaves two questions it could not
answer with 33 names:

  EqualRiskContribution  inverse-vol ignores correlation. Does accounting for
                         it help, or is the extra machinery unpaid complexity?
  MinCorrelationSleeve   if the sector is one bet, can you at least hold the
                         least-redundant corner of it?

Both are long-only, use trailing covariance only, and rebalance on a schedule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import MarketData
from ..portfolio import cap_weights, rebalance_schedule
from ..universe import investable_mask
from .base import Strategy


def _hold_between(w: pd.DataFrame, stamps: pd.Index) -> pd.DataFrame:
    """Carry rebalance-date weights forward; zero before the first one.

    Same convention as ``portfolio.rebalance_schedule``, but these strategies
    compute weights only ON the rebalance dates (covariance over ~100 names is
    far too expensive to evaluate daily), so the rows in between are blanked
    and forward-filled rather than sampled from a daily frame.
    """
    out = w.copy()
    out[~w.index.isin(stamps)] = np.nan
    return out.ffill().fillna(0.0)


def _erc_weights(cov: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Equal-risk-contribution weights by fixed-point iteration.

    Solves w_i ∝ w_i / (Σw)_i until each holding contributes the same share of
    portfolio variance. Cheaper and far more stable than a general optimizer,
    and it cannot return a short position — both properties matter more here
    than hitting the exact optimum.
    """
    n = cov.shape[0]
    if n == 0:
        return np.zeros(0)
    w = np.full(n, 1.0 / n)
    for _ in range(iterations):
        mrc = cov @ w                       # marginal risk contribution
        mrc = np.where(np.abs(mrc) < 1e-14, 1e-14, mrc)
        w_new = w / mrc
        s = w_new.sum()
        if not np.isfinite(s) or s <= 0:
            return np.full(n, 1.0 / n)
        w_new /= s
        if np.max(np.abs(w_new - w)) < 1e-8:
            return w_new
        w = w_new
    return w


class EqualRiskContribution(Strategy):
    """Weight so every holding contributes the same share of portfolio risk.

    Inverse-volatility weighting is equal-risk-contribution's special case
    under the assumption that everything is equally correlated. In this
    universe that assumption is badly wrong — semis move together far more
    tightly than semis and software — so the two should differ measurably.
    """
    family = "allocation"
    hypothesis = ("Accounting for correlation as well as volatility produces a "
                  "genuinely risk-balanced portfolio and a better Sharpe than "
                  "inverse-vol weighting, which double-counts the risk of "
                  "clustered names.")

    def __init__(self, basket: str = "megacap_ai", cov_window: int = 126,
                 max_weight: float = 0.25, rebalance: str = "ME"):
        super().__init__(basket=basket, cov_window=cov_window,
                         max_weight=max_weight, rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md, md.universe.baskets[p["basket"]])
        px = md.adj_close[mask.columns]
        rets = px.pct_change()

        # Covariance is only recomputed on rebalance dates — daily covariance
        # over ~100 names is both expensive and pointless, since the weights
        # are held constant between rebalances anyway.
        stamps = rets.groupby(pd.Grouper(freq=p["rebalance"])).tail(1).index
        w = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
        win = p["cov_window"]
        cols = list(rets.columns)

        for ts in stamps:
            live = mask.loc[ts]
            names = [c for c in cols if bool(live.get(c, False))]
            if len(names) < 2:
                if names:
                    w.loc[ts, names] = 1.0 / len(names)
                continue
            hist = rets.loc[:ts, names].tail(win)
            hist = hist.dropna(axis=1, thresh=max(21, win // 2))
            if hist.shape[1] < 2:
                w.loc[ts, names] = 1.0 / len(names)
                continue
            cov = hist.cov().to_numpy()
            # Ridge shrinkage toward a diagonal: a 126-day window over dozens
            # of names is near-singular, and an unshrunk inverse would put
            # enormous weight on whichever pair happened to look uncorrelated.
            cov = 0.9 * cov + 0.1 * np.diag(np.diag(cov))
            if not np.all(np.isfinite(cov)):
                w.loc[ts, names] = 1.0 / len(names)
                continue
            ws = _erc_weights(cov)
            w.loc[ts, list(hist.columns)] = ws

        return cap_weights(_hold_between(w, stamps), p["max_weight"])


class MinCorrelationSleeve(Strategy):
    """Hold the names least correlated with the rest of the sector.

    Each name is scored by its average trailing correlation to every other
    investable name; the lowest scores are held, equally weighted. This is a
    direct attack on the v2 finding that the strategy set was one bet: if a
    less-redundant corner of technology exists, this finds it. If the resulting
    portfolio still moves with QQQ, that is strong evidence the sector simply
    has no independent corner — which is worth knowing.
    """
    family = "allocation"
    hypothesis = ("A sleeve selected purely for low average correlation to the "
                  "rest of the sector diversifies a core technology holding "
                  "better than any return-driven selection, at the cost of "
                  "lower standalone return.")

    def __init__(self, corr_window: int = 252, top_n: int = 10,
                 rebalance: str = "QE", basket: str | None = None):
        super().__init__(corr_window=corr_window, top_n=top_n,
                         rebalance=rebalance, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        rets = md.adj_close[mask.columns].pct_change()

        stamps = rets.groupby(pd.Grouper(freq=p["rebalance"])).tail(1).index
        w = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
        win = p["corr_window"]

        for ts in stamps:
            live = mask.loc[ts]
            names = [c for c in rets.columns if bool(live.get(c, False))]
            if len(names) < 2:
                if names:
                    w.loc[ts, names] = 1.0 / len(names)
                continue
            hist = rets.loc[:ts, names].tail(win)
            hist = hist.dropna(axis=1, thresh=max(21, win // 2))
            if hist.shape[1] < 2:
                w.loc[ts, names] = 1.0 / len(names)
                continue
            corr = hist.corr()
            # Average correlation to everything else (exclude the diagonal 1.0).
            n = corr.shape[0]
            avg = (corr.sum(axis=1) - 1.0) / max(n - 1, 1)
            avg = avg.dropna()
            if avg.empty:
                continue
            picks = list(avg.nsmallest(min(p["top_n"], len(avg))).index)
            w.loc[ts, picks] = 1.0 / len(picks)

        return _hold_between(w, stamps)
