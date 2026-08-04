"""Machine-learning strategy: walk-forward cross-sectional ridge ranking.

Deliberately simple — a regularized linear model over transparent features —
so that any edge is attributable and comparable against the traditional
baselines. Anti-leakage measures:

  * expanding-window walk-forward refits (annual), predictions only on data
    strictly after the training window,
  * a purge gap of `horizon` days between train end and prediction start so
    overlapping forward-return labels never straddle the boundary,
  * feature standardization fitted on the training window only,
  * fixed seed; no hyperparameter search outside the training window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..data.loader import MarketData
from ..features import distance_from_sma, momentum, realized_vol, short_term_reversal
from ..portfolio import equal_weight, rebalance_schedule, top_n_selection
from ..universe import investable_mask
from .base import Strategy


def _feature_panel(md: MarketData, cols: list[str]) -> dict[str, pd.DataFrame]:
    px = md.adj_close[cols]
    return {
        "mom_21": momentum(px, 21),
        "mom_63": momentum(px, 63),
        "mom_126": momentum(px, 126),
        "mom_252_21": momentum(px, 252, 21),
        "rev_5": short_term_reversal(px, 5),
        "vol_63": realized_vol(px, 63),
        "dist_200": distance_from_sma(px, 200),
    }


class MLRankStrategy(Strategy):
    family = "ml"
    hypothesis = "A shrunk linear blend of momentum/vol features ranks forward returns."

    def __init__(self, horizon: int = 21, top_n: int = 6, alpha: float = 10.0,
                 refit_months: int = 12, min_train_years: int = 5,
                 rebalance: str = "ME"):
        super().__init__(horizon=horizon, top_n=top_n, alpha=alpha,
                         refit_months=refit_months, min_train_years=min_train_years,
                         rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md)
        cols = list(mask.columns)
        px = md.adj_close[cols]
        feats = _feature_panel(md, cols)
        fnames = sorted(feats)

        horizon = p["horizon"]
        fwd = px.shift(-horizon) / px - 1  # label: only ever used for TRAINING rows

        cal = px.index
        refit_dates = pd.date_range(cal[0], cal[-1], freq=f"{p['refit_months']}MS")
        min_train = p["min_train_years"] * 252

        scores = pd.DataFrame(np.nan, index=cal, columns=cols)
        for i, refit in enumerate(refit_dates):
            train_end_pos = cal.searchsorted(refit) - 1 - horizon  # purge gap
            if train_end_pos < min_train:
                continue
            pred_start = cal.searchsorted(refit)
            pred_end = (cal.searchsorted(refit_dates[i + 1])
                        if i + 1 < len(refit_dates) else len(cal))
            if pred_start >= len(cal):
                break

            tr_idx = cal[:train_end_pos]
            X_parts, y_parts = [], []
            for f in fnames:
                X_parts.append(feats[f].loc[tr_idx].where(mask.loc[tr_idx]).to_numpy().ravel())
            Xtr = np.column_stack(X_parts)
            ytr = fwd.loc[tr_idx].where(mask.loc[tr_idx]).to_numpy().ravel()
            ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
            if ok.sum() < 500:
                continue
            Xtr, ytr = Xtr[ok], ytr[ok]
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1.0
            model = Ridge(alpha=p["alpha"], random_state=0)
            model.fit((Xtr - mu) / sd, ytr)

            te_idx = cal[pred_start:pred_end]
            Xte = np.column_stack([
                feats[f].loc[te_idx].to_numpy().ravel() for f in fnames
            ])
            pred = np.full(len(Xte), np.nan)
            ok_te = np.isfinite(Xte).all(axis=1)
            if ok_te.any():
                pred[ok_te] = model.predict((Xte[ok_te] - mu) / sd)
            scores.loc[te_idx] = pred.reshape(len(te_idx), len(cols))

        sel = top_n_selection(scores.where(mask), p["top_n"]).where(scores > 0, 0.0)
        return rebalance_schedule(equal_weight(sel), p["rebalance"])
