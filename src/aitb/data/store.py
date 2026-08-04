"""Local data store: raw parquet cache with metadata and incremental updates.

Layout:
    data/raw/<provider>/prices/<TICKER>.parquet
    data/raw/<provider>/macro/<SERIES>.parquet
    data/raw/<provider>/fundamentals/<TICKER>.parquet
    data/raw/<provider>/_meta/<key>.json      (source, timestamps, row counts)

Raw data is stored exactly as the provider returned it. Processed panels are
built by the loader and cached under data/processed/. Unchanged data is never
re-downloaded: refreshes fetch only from the last cached bar forward.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR
from ..utils import get_logger
from .providers import PriceProvider

log = get_logger("data.store")


class DataStore:
    def __init__(self, provider: PriceProvider, root: Path | None = None):
        self.provider = provider
        self.root = (root or DATA_DIR) / "raw" / provider.name
        for sub in ("prices", "macro", "fundamentals", "_meta"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ io --
    def _meta_path(self, kind: str, key: str) -> Path:
        return self.root / "_meta" / f"{kind}_{key}.json"

    def _write_meta(self, kind: str, key: str, df: pd.DataFrame) -> None:
        meta = {
            "provider": self.provider.name,
            "key": key,
            "kind": kind,
            "rows": int(len(df)),
            "first": str(df.index.min()) if len(df) else None,
            "last": str(df.index.max()) if len(df) else None,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._meta_path(kind, key).write_text(json.dumps(meta, indent=2))

    def read_meta(self, kind: str, key: str) -> dict | None:
        p = self._meta_path(kind, key)
        return json.loads(p.read_text()) if p.exists() else None

    # -------------------------------------------------------------- prices --
    def get_prices(self, ticker: str, start: date, end: date,
                   refresh: bool = False) -> pd.DataFrame:
        path = self.root / "prices" / f"{ticker}.parquet"
        cached = pd.read_parquet(path) if path.exists() else None

        if cached is not None and not refresh:
            have_start = cached.index.min().date()
            have_end = cached.index.max().date()
            if have_start <= start and have_end >= end:
                return cached.loc[str(start):str(end)]
            # Incremental extension: fetch only the missing tail.
            if have_start <= start and have_end < end:
                tail = self.provider.fetch_daily(ticker, have_end, end)
                cached = pd.concat([cached, tail[tail.index > cached.index.max()]])
                cached.to_parquet(path)
                self._write_meta("prices", ticker, cached)
                return cached.loc[str(start):str(end)]

        df = self.provider.fetch_daily(ticker, start, end)
        df.to_parquet(path)
        self._write_meta("prices", ticker, df)
        return df

    # --------------------------------------------------------------- macro --
    def get_series(self, series_id: str, start: date, end: date,
                   refresh: bool = False) -> pd.Series:
        path = self.root / "macro" / f"{series_id}.parquet"
        if path.exists() and not refresh:
            s = pd.read_parquet(path)[series_id]
            if s.index.min().date() <= start and s.index.max().date() >= end:
                return s.loc[str(start):str(end)]
        s = self.provider.fetch_series(series_id, start, end)
        s.to_frame().to_parquet(path)
        self._write_meta("macro", series_id, s.to_frame())
        return s

    # -------------------------------------------------------- fundamentals --
    def get_fundamentals(self, ticker: str, refresh: bool = False) -> pd.DataFrame:
        path = self.root / "fundamentals" / f"{ticker}.parquet"
        if path.exists() and not refresh:
            return pd.read_parquet(path)
        if not hasattr(self.provider, "fetch_fundamentals"):
            return pd.DataFrame()
        df = self.provider.fetch_fundamentals(ticker)  # type: ignore[attr-defined]
        df.to_parquet(path)
        return df
