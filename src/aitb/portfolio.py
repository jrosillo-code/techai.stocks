"""Portfolio construction: turn scores/selections into constrained weights.

All functions take date × ticker frames and return target-weight frames whose
rows sum to at most 1 (long-only) or respect gross limits (long-short). The
constraint helper enforces per-name caps by iterative redistribution — no
unconstrained optimizer output ever reaches the engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import realized_vol


def equal_weight(selection: pd.DataFrame) -> pd.DataFrame:
    """selection: boolean/0-1 frame of names to hold each date."""
    sel = selection.astype(float)
    n = sel.sum(axis=1).replace(0, np.nan)
    return sel.div(n, axis=0).fillna(0.0)


def score_weight(scores: pd.DataFrame, selection: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weights proportional to positive scores (rank- or momentum-weighting)."""
    s = scores.where(selection.astype(bool)) if selection is not None else scores
    s = s.clip(lower=0)
    tot = s.sum(axis=1).replace(0, np.nan)
    return s.div(tot, axis=0).fillna(0.0)


def inverse_vol_weight(px: pd.DataFrame, selection: pd.DataFrame,
                       window: int = 63) -> pd.DataFrame:
    iv = (1 / realized_vol(px, window).replace(0, np.nan)).where(selection.astype(bool))
    tot = iv.sum(axis=1).replace(0, np.nan)
    return iv.div(tot, axis=0).fillna(0.0)


def top_n_selection(scores: pd.DataFrame, n: int,
                    mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Boolean frame holding the top-`n` scores each date (within mask)."""
    s = scores.where(mask.astype(bool)) if mask is not None else scores
    ranks = s.rank(axis=1, ascending=False, method="first")
    return (ranks <= n).astype(float)


def cap_weights(w: pd.DataFrame, max_weight: float = 0.20,
                iterations: int = 25) -> pd.DataFrame:
    """Cap per-name weight, redistributing the excess pro-rata among names
    still below the cap. If every name hits the cap, the residual is left
    unallocated (implicitly cash). Weight totals are preserved whenever
    cap × n_names ≥ row total."""
    out = w.to_numpy(copy=True)
    eps = 1e-12
    for _ in range(iterations):
        over = out > max_weight + eps
        if not over.any():
            break
        excess = np.where(over, out - max_weight, 0.0).sum(axis=1, keepdims=True)
        out = np.where(over, max_weight, out)
        below = (out > eps) & (out < max_weight - eps)
        room = np.where(below, out, 0.0)
        room_tot = room.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = out + np.where(room_tot > eps, room / np.where(room_tot > eps, room_tot, 1.0) * excess, 0.0)
    return pd.DataFrame(np.minimum(out, max_weight), index=w.index, columns=w.columns)


def vol_target(weights: pd.DataFrame, px: pd.DataFrame,
               target_ann_vol: float = 0.20, window: int = 63,
               max_leverage: float = 1.0) -> pd.DataFrame:
    """Scale gross exposure so trailing portfolio vol ≈ target (no leverage
    above `max_leverage`; the remainder is implicitly cash)."""
    port_ret = (weights.shift(1) * px.pct_change()).sum(axis=1)
    trailing = port_ret.rolling(window).std() * np.sqrt(252)
    scale = (target_ann_vol / trailing.replace(0, np.nan)).clip(upper=max_leverage).fillna(1.0)
    return weights.mul(scale, axis=0)


def rebalance_schedule(weights: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Hold weights constant between scheduled rebalance dates.

    The weight decided at the last trading day of each period is carried
    forward (in decision space) until the next rebalance; the engine still
    applies its uniform next-open execution lag.
    """
    stamps = weights.groupby(pd.Grouper(freq=freq)).tail(1).index
    out = weights.copy()
    keep = weights.index.isin(stamps)
    out[~keep] = np.nan
    return out.ffill().fillna(0.0)
