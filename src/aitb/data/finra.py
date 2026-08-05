"""FINRA daily short-sale volume — the study's first non-price, non-fundamental input.

WHY THIS DATA
-------------
Every signal in this study reads price, or volume, or a filing. All three
describe what happened; none describes WHO did it. Short volume is the closest
free proxy for positioning: the share of a day's consolidated tape that was
sold short. A stock rising on 20% short volume and one rising on 55% are the
same bar on a chart and different events underneath — the second is climbing
against people betting the other way, and the covering is fuel the first does
not have.

FINRA publishes this daily, free, for every NMS security, with no key and no
licence. It is the only free read on positioning the study can get.

WHAT IT IS NOT, WHICH MATTERS MORE THAN WHAT IT IS
--------------------------------------------------
This is emphatically NOT short interest, and the two are routinely confused:

  * SHORT VOLUME is a flow — shares sold short during one session, as a
    fraction of that session's volume. It says nothing about how many shares
    remain short at the close.
  * SHORT INTEREST is a stock — open short positions, published twice a month
    with a settlement lag. That is the number most people mean.

A market maker who sells short to fill a buy order and covers seconds later
appears in short volume and never in short interest. Bona-fide market making
is a large and variable share of the total, which is why the level of this
series is nearly meaningless and only its movement RELATIVE TO ITS OWN HISTORY
is worth reading. Any strategy built on this must use a per-name relative
measure, never an absolute threshold.

COVERAGE AND LIMITS, ALL OF WHICH ARE DISCLOSED RATHER THAN PAPERED OVER
------------------------------------------------------------------------
  * Daily files begin 2009-07-31. There is no history before that, so any
    strategy reading this is silent across the dot-com collapse and the 2008
    crisis — the two most informative regimes in the study's window.
  * The consolidated (CNMS) file covers off-exchange plus exchange-reported
    volume. It is not the full consolidated tape and its ratio to true total
    volume drifts over the years as off-exchange share has grown. A level
    comparison across a decade is not valid; a comparison against the name's
    own trailing window is.
  * One file per session, roughly 4,300 of them for the full history. Each is
    small, but a first download is thousands of requests — hence the resume
    logic below, which keys on the SESSION rather than the symbol.

The parser here is pure and takes text, so it is fully tested offline; the
fetching lives behind the same rate-limited HTTP helper every other provider
uses.
"""
from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd

from ..utils import get_logger

log = get_logger("data.finra")

# One file per session. CNMS = consolidated NMS tape.
BASE_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{stamp}.txt"

# The first session FINRA published. Requesting earlier dates returns a 404,
# which the provider layer treats as permanent and does not retry (AUD-014).
FIRST_SESSION = date(2009, 7, 31)

COLUMNS = ["date", "ticker", "short_volume", "short_exempt_volume",
           "total_volume"]


def short_volume_url(day: date) -> str:
    return BASE_URL.format(stamp=day.strftime("%Y%m%d"))


def parse_daily(text: str) -> pd.DataFrame:
    """Parse one CNMS daily file into a tidy frame.

    The file is pipe-delimited with a header row and a trailing record count
    line that is NOT data — dropping it by column count rather than by
    position, because a truncated download loses the tail and a positional
    drop would then silently discard a real row instead.
    """
    raw = pd.read_csv(io.StringIO(text), sep="|", dtype=str,
                      keep_default_na=False)
    expected = {"Date", "Symbol", "ShortVolume", "TotalVolume"}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"FINRA file is missing columns {sorted(missing)} — "
                         f"got {list(raw.columns)}")

    # The trailer ("Records: 8123") parses as a row whose Symbol is blank.
    raw = raw[raw["Symbol"].str.strip() != ""]
    raw = raw[raw["Date"].str.fullmatch(r"\d{8}")]

    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], format="%Y%m%d"),
        "ticker": raw["Symbol"].str.strip().str.upper(),
        "short_volume": pd.to_numeric(raw["ShortVolume"], errors="coerce"),
        "short_exempt_volume": pd.to_numeric(
            raw.get("ShortExemptVolume", pd.Series(index=raw.index, dtype=str)),
            errors="coerce"),
        "total_volume": pd.to_numeric(raw["TotalVolume"], errors="coerce"),
    })
    # A row with no total volume carries no ratio and is dropped rather than
    # imputed — a zero denominator would become an infinite short share.
    out = out[out["total_volume"] > 0]
    return out[COLUMNS].reset_index(drop=True)


def short_share(df: pd.DataFrame) -> pd.DataFrame:
    """Add the only column any strategy should read: short / total.

    Deliberately NOT clipped to [0, 1]. The consolidated file's short volume
    can legitimately exceed its own total on thin names because the two are
    reported from different feeds, and a value above 1 is a data-quality signal
    worth seeing in the gate rather than a number worth silently flattening.
    """
    out = df.copy()
    out["short_share"] = out["short_volume"] / out["total_volume"]
    return out


def sessions_between(start: date, end: date) -> list[date]:
    """Every weekday in range, from FINRA's first published session onward.

    Weekdays rather than exchange sessions on purpose: a market holiday simply
    has no file, and a 404 is already handled as a permanent, unretried miss.
    Pulling the exchange calendar in here would couple the downloader to the
    calendar module for no gain.
    """
    day = max(start, FIRST_SESSION)
    out = []
    while day <= end:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def to_panel(frames: list[pd.DataFrame], tickers: list[str]) -> pd.DataFrame:
    """Stack parsed daily files into a date x ticker short-share panel."""
    if not frames:
        return pd.DataFrame()
    wanted = {t.upper() for t in tickers}
    keep = [f[f["ticker"].isin(wanted)] for f in frames]
    keep = [f for f in keep if not f.empty]
    if not keep:
        return pd.DataFrame()
    allrows = short_share(pd.concat(keep, ignore_index=True))
    panel = allrows.pivot_table(index="date", columns="ticker",
                                values="short_share", aggfunc="last")
    return panel.sort_index()
