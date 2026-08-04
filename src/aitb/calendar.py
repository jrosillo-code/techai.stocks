"""US exchange trading calendar.

Real-data runs use the maintained `exchange_calendars` XNYS calendar (NYSE):
official holidays (New Year, MLK, Washington, **Good Friday**, Memorial,
Juneteenth, July 4, Labor, Thanksgiving, Christmas), historical unscheduled
closures (9/11, Hurricane Sandy, presidential mourning days) and early-close
session metadata.

Synthetic mode may keep the plain business-day calendar (the generator was
built on it); mixing the two calendars inside one run is prevented by the
loader, which owns calendar selection per data mode.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=4)
def _xnys():
    import exchange_calendars as xc
    # Default XNYS only covers ~20 years back; request the full research span.
    return xc.get_calendar("XNYS", start="1990-01-01")


def trading_sessions(start: date, end: date) -> pd.DatetimeIndex:
    """NYSE trading sessions in [start, end] (timezone-naive dates)."""
    cal = _xnys()
    lo = max(pd.Timestamp(start), cal.first_session)
    hi = min(pd.Timestamp(end), cal.last_session)
    sessions = cal.sessions_in_range(lo, hi)
    return pd.DatetimeIndex(sessions.tz_localize(None) if sessions.tz is not None
                            else sessions, name="date")


def business_day_calendar(start: date, end: date) -> pd.DatetimeIndex:
    """Plain Mon-Fri calendar (synthetic mode only)."""
    return pd.bdate_range(start, end, name="date")


def is_session(day: pd.Timestamp) -> bool:
    return _xnys().is_session(pd.Timestamp(day).normalize())


def next_session(day: pd.Timestamp) -> pd.Timestamp:
    """First session strictly after `day` — used to roll rebalance dates that
    fall on holidays/weekends forward instead of silently dropping them."""
    cal = _xnys()
    ts = pd.Timestamp(day).normalize()
    if cal.is_session(ts):
        nxt = cal.next_session(ts)
    else:
        nxt = cal.date_to_session(ts, direction="next")
    return pd.Timestamp(nxt.tz_localize(None) if getattr(nxt, "tz", None) else nxt)


def early_closes(start: date, end: date) -> pd.DatetimeIndex:
    """Early-close sessions in range (e.g. day after Thanksgiving, Christmas
    Eve). Strategies do not trade differently on them, but the report
    discloses them and intraday extensions will need them."""
    cal = _xnys()
    ec = cal.early_closes.tz_localize(None) if getattr(cal.early_closes, "tz", None) \
        else cal.early_closes
    return pd.DatetimeIndex([d for d in ec if pd.Timestamp(start) <= d <= pd.Timestamp(end)])


def get_calendar(mode: str, start: date, end: date) -> pd.DatetimeIndex:
    """Calendar for a data mode: exchange sessions for real, b-days for synthetic."""
    if mode == "real":
        return trading_sessions(start, end)
    return business_day_calendar(start, end)
