"""Canonical security master: permanent IDs behind mutable tickers.

Guards against the classic real-data traps:
  * renames (FB→META, SUNW→JAVA) creating phantom new securities,
  * a recycled or changed symbol splicing two unrelated price histories,
  * acquisitions (XLNX→AMD, EMC→DELL) silently vanishing from the record.

``resolve(symbol, on_date)`` maps a ticker AS OF A DATE to its sid;
``stitch_history`` concatenates per-symbol price frames for one sid, refusing
to bridge across a gap with a fabricated return.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import pandas as pd
import yaml

from ..config import CONFIG_DIR


@dataclass(frozen=True)
class TickerSpan:
    symbol: str
    start: date
    end: date | None  # None = current


@dataclass
class SecurityRecord:
    sid: str
    name: str
    tickers: list[TickerSpan]
    exchange: str = ""
    sector: str = ""
    theme: str = ""
    security_type: str = "common"
    successor: dict | None = None
    notes: str = ""

    @property
    def current_symbol(self) -> str | None:
        for t in self.tickers:
            if t.end is None:
                return t.symbol
        return None

    @property
    def all_symbols(self) -> list[str]:
        return [t.symbol for t in self.tickers]


def _to_date(v) -> date | None:
    if v is None:
        return None
    return v if isinstance(v, date) else date.fromisoformat(str(v))


@lru_cache(maxsize=1)
def load_master() -> dict[str, SecurityRecord]:
    with open(CONFIG_DIR / "security_master.yaml") as fh:
        raw = yaml.safe_load(fh)
    out: dict[str, SecurityRecord] = {}
    for s in raw["securities"]:
        spans = [TickerSpan(t["symbol"], _to_date(t["from"]), _to_date(t["to"]))
                 for t in s["tickers"]]
        out[s["sid"]] = SecurityRecord(
            sid=s["sid"], name=s["name"], tickers=spans,
            exchange=s.get("exchange", ""), sector=s.get("sector", ""),
            theme=s.get("theme", ""), security_type=s.get("security_type", "common"),
            successor=s.get("successor"), notes=s.get("notes", ""))
    return out


def resolve(symbol: str, on_date: date | None = None) -> SecurityRecord | None:
    """Which security did `symbol` denote on `on_date` (default: any time)?"""
    for rec in load_master().values():
        for span in rec.tickers:
            if span.symbol != symbol:
                continue
            if on_date is None:
                return rec
            end = span.end or date(2100, 1, 1)
            if span.start <= on_date <= end:
                return rec
    return None


def symbols_for(sid: str) -> list[TickerSpan]:
    return load_master()[sid].tickers


def stitch_history(sid: str, frames_by_symbol: dict[str, pd.DataFrame],
                   max_gap_days: int = 7) -> pd.DataFrame:
    """Concatenate per-symbol frames into one continuous sid history.

    Frames are clipped to each symbol's validity window, sorted, and checked:
    overlapping dates or a gap longer than `max_gap_days` between consecutive
    spans raise, because bridging either would fabricate returns.
    """
    rec = load_master()[sid]
    pieces = []
    for span in rec.tickers:
        df = frames_by_symbol.get(span.symbol)
        if df is None or df.empty:
            continue
        lo = pd.Timestamp(span.start)
        hi = pd.Timestamp(span.end) if span.end else df.index.max()
        pieces.append(df.loc[lo:hi])
    if not pieces:
        return pd.DataFrame()
    pieces.sort(key=lambda d: d.index.min())
    for a, b in zip(pieces, pieces[1:]):
        if b.index.min() <= a.index.max():
            raise ValueError(f"{sid}: overlapping ticker spans at {b.index.min()}")
        gap = (b.index.min() - a.index.max()).days
        if gap > max_gap_days:
            raise ValueError(f"{sid}: {gap}-day gap between ticker spans — "
                             "refusing to stitch across it")
    return pd.concat(pieces)


def audit_duplicates() -> list[str]:
    """Symbols claimed by more than one sid with overlapping validity."""
    claims: dict[str, list[tuple[str, date, date]]] = {}
    problems = []
    for rec in load_master().values():
        for span in rec.tickers:
            end = span.end or date(2100, 1, 1)
            for other_sid, lo, hi in claims.get(span.symbol, []):
                if span.start <= hi and lo <= end:
                    problems.append(f"{span.symbol}: {rec.sid} overlaps {other_sid}")
            claims.setdefault(span.symbol, []).append((rec.sid, span.start, end))
    return problems
