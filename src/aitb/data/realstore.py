"""Canonical real-data store.

Layout (all Parquet, one file per security/series):

    data/real/prices/<TICKER>.parquet         open, high, low, close,
                                              adj_close, volume  (+ optional
                                              dividend, split columns)
    data/real/macro/<SERIES>.parquet          value
    data/real/fundamentals/<TICKER>.parquet   period_end, published, revenue,
                                              eps, fcf, shares (+ optional
                                              extended XBRL fields)
    data/real/earnings/<TICKER>.parquet       ann_date, timing, eps, eps_est,
                                              revenue, revenue_est
    data/real/_meta/<kind>_<key>.json         provider, downloaded_at,
                                              adjustment method, coverage,
                                              source URL, schema_version

This store is populated ONLY by scripts/download_real_data.py (networked
environment) or scripts/import_data_bundle.py (portable bundle). The loader in
real mode reads from here and from nowhere else — a missing dataset is a hard
error, never a silent synthetic substitution.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import REAL_DATA_DIR
from ..utils import get_logger

log = get_logger("data.realstore")

SCHEMA_VERSION = "1.0"
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
FUNDAMENTAL_COLUMNS = ["period_end", "published", "revenue", "eps", "fcf", "shares"]
EARNINGS_COLUMNS = ["ann_date", "timing"]  # timing: bmo/amc/during/unknown

KINDS = ("prices", "macro", "fundamentals", "earnings")


class RealDataMissing(RuntimeError):
    """Raised when real mode requires a dataset that is not in the store."""


def _root(root: Path | None) -> Path:
    return root or REAL_DATA_DIR


def _meta_path(kind: str, key: str, root: Path | None = None) -> Path:
    return _root(root) / "_meta" / f"{kind}_{key}.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(kind: str, key: str, df: pd.DataFrame, meta: dict,
          root: Path | None = None) -> Path:
    """Write one dataset + its metadata record (provider, timestamps, etc.)."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind '{kind}'")
    base = _root(root)
    (base / kind).mkdir(parents=True, exist_ok=True)
    (base / "_meta").mkdir(parents=True, exist_ok=True)
    path = base / kind / f"{key}.parquet"
    df.to_parquet(path)
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind, "key": key,
        "rows": int(len(df)),
        "first": str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) and len(df) else None,
        "last": str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) and len(df) else None,
        "sha256": file_sha256(path),
        "written_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    _meta_path(kind, key, root).write_text(json.dumps(record, indent=2, default=str))
    return path


def read(kind: str, key: str, root: Path | None = None) -> pd.DataFrame:
    path = _root(root) / kind / f"{key}.parquet"
    if not path.exists():
        raise RealDataMissing(f"real dataset missing: {kind}/{key} "
                              f"(populate via download_real_data.py or import_data_bundle.py)")
    return pd.read_parquet(path)


def read_meta(kind: str, key: str, root: Path | None = None) -> dict | None:
    p = _meta_path(kind, key, root)
    return json.loads(p.read_text()) if p.exists() else None


def available(kind: str, root: Path | None = None) -> list[str]:
    d = _root(root) / kind
    return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []


def coverage_summary(root: Path | None = None) -> dict:
    out: dict = {}
    for kind in KINDS:
        keys = available(kind, root)
        entry = {"count": len(keys), "keys": keys}
        spans = {}
        for k in keys:
            m = read_meta(kind, k, root) or {}
            spans[k] = {"first": m.get("first"), "last": m.get("last"),
                        "rows": m.get("rows"), "provider": m.get("provider")}
        entry["spans"] = spans
        out[kind] = entry
    return out
