"""Data-quality validation.

Checks each price frame for: missing calendar dates, duplicates, non-positive
prices, OHLC ordering violations, extreme jumps, and stale (flat) stretches.
Issues are returned as structured records so the pipeline can log them and a
report can disclose them; hard errors raise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Issue:
    ticker: str
    kind: str
    detail: str
    severity: str  # "warn" | "error"


def validate_prices(ticker: str, df: pd.DataFrame,
                    calendar: pd.DatetimeIndex | None = None,
                    max_daily_move: float = 0.65) -> list[Issue]:
    issues: list[Issue] = []
    if df.empty:
        return [Issue(ticker, "empty", "no rows", "error")]

    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique()
        issues.append(Issue(ticker, "duplicates", f"{len(dupes)} duplicated dates", "error"))
    if not df.index.is_monotonic_increasing:
        issues.append(Issue(ticker, "unsorted", "index not sorted", "error"))

    for col in ("open", "high", "low", "close", "adj_close"):
        bad = int((df[col] <= 0).sum())
        if bad:
            issues.append(Issue(ticker, "nonpositive", f"{bad} rows with {col} <= 0", "error"))

    ohlc_bad = int(((df["high"] < df["low"]) |
                    (df["high"] < df[["open", "close"]].max(axis=1) * 0.999) |
                    (df["low"] > df[["open", "close"]].min(axis=1) * 1.001)).sum())
    if ohlc_bad:
        issues.append(Issue(ticker, "ohlc_order", f"{ohlc_bad} rows violate OHLC ordering", "warn"))

    r = df["adj_close"].pct_change().abs()
    jumps = int((r > max_daily_move).sum())
    if jumps:
        issues.append(Issue(ticker, "jump", f"{jumps} daily moves > {max_daily_move:.0%}", "warn"))

    flat = (df["close"].diff() == 0).astype(int)
    max_flat = int(flat.groupby((flat == 0).cumsum()).cumsum().max() or 0)
    if max_flat >= 10:
        issues.append(Issue(ticker, "stale", f"{max_flat} consecutive unchanged closes", "warn"))

    if calendar is not None:
        expected = calendar[(calendar >= df.index.min()) & (calendar <= df.index.max())]
        missing = expected.difference(df.index)
        if len(missing) > 0.02 * len(expected):
            issues.append(Issue(ticker, "gaps",
                                f"{len(missing)} of {len(expected)} calendar dates missing", "warn"))
    return issues


def assert_clean(ticker: str, df: pd.DataFrame, **kw) -> list[Issue]:
    issues = validate_prices(ticker, df, **kw)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise ValueError(f"{ticker}: {'; '.join(i.detail for i in errors)}")
    return issues
