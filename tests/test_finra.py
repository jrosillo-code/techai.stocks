"""FINRA short-sale volume: parsing, and the confusions it must not permit."""
from datetime import date

import pandas as pd
import pytest

SAMPLE = """Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
20260805|NVDA|12000000|1500|30000000|Q
20260805|MSFT|4000000|0|20000000|Q
20260805|THIN|900|0|1000|Q
20260805|BADR|500|0|0|Q
Records: 4
"""


def test_parses_and_drops_the_trailer_not_a_real_row():
    from aitb.data.finra import parse_daily
    df = parse_daily(SAMPLE)
    assert set(df["ticker"]) == {"NVDA", "MSFT", "THIN"}, (
        "BADR has zero total volume and must be dropped; the 'Records:' "
        "trailer must not survive as a row")
    assert df["date"].iloc[0] == pd.Timestamp("2026-08-05")
    assert df["short_volume"].iloc[0] == 12_000_000


def test_a_truncated_file_loses_rows_not_meaning():
    """Dropping the trailer by position would eat a real row on a short read.

    A download cut off mid-file has no 'Records:' line. If the parser removed
    the last row positionally it would silently discard a genuine symbol, and
    the loss would look exactly like a symbol that did not trade.
    """
    from aitb.data.finra import parse_daily
    truncated = "\n".join(SAMPLE.splitlines()[:-1])
    df = parse_daily(truncated)
    assert set(df["ticker"]) == {"NVDA", "MSFT", "THIN"}


def test_short_share_is_not_silently_clipped():
    """A ratio above 1 is a data-quality signal, not a number to flatten.

    Short and total volume come from different feeds, so a thin name can
    legitimately report short > total. Clipping it to 1.0 would hide a
    reporting problem the quality gate should see.
    """
    from aitb.data.finra import parse_daily, short_share
    odd = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
           "20260805|WEIRD|1500|0|1000|Q\n")
    df = short_share(parse_daily(odd))
    assert df["short_share"].iloc[0] == pytest.approx(1.5)


def test_missing_columns_fail_loudly():
    """A changed file format must raise, not yield an empty frame.

    An empty frame would flow through the pipeline as 'this name has no short
    data', which is indistinguishable from a symbol FINRA genuinely does not
    cover — and the study would record a coverage gap instead of a broken
    parser.
    """
    from aitb.data.finra import parse_daily
    with pytest.raises(ValueError, match="missing columns"):
        parse_daily("Date|Symbol|Something\n20260805|NVDA|1\n")


def test_history_cannot_start_before_finra_published_any():
    """The series begins 2009-07-31 and no strategy may pretend otherwise.

    Anything reading this data is blind across the dot-com collapse and 2008 —
    the two most informative regimes in the study's window. Silently returning
    dates FINRA never published would turn that gap into invisible NaNs.
    """
    from aitb.data.finra import FIRST_SESSION, sessions_between
    days = sessions_between(date(1998, 1, 1), date(2009, 8, 5))
    assert days[0] == FIRST_SESSION
    assert all(d >= FIRST_SESSION for d in days)
    assert all(d.weekday() < 5 for d in days), "weekend files never exist"


def test_panel_pivots_to_date_by_ticker():
    from aitb.data.finra import parse_daily, to_panel
    day2 = SAMPLE.replace("20260805", "20260806")
    panel = to_panel([parse_daily(SAMPLE), parse_daily(day2)], ["NVDA", "MSFT"])
    assert list(panel.columns) == ["MSFT", "NVDA"]
    assert len(panel) == 2
    assert panel["NVDA"].iloc[0] == pytest.approx(0.4)


def test_short_squeeze_refuses_without_the_data_it_names():
    """It must fail closed, not fall back to a price-and-volume lookalike.

    An approximation built from price and volume would be a different strategy
    wearing this one's name, and the substitution would be invisible in the
    results — the exact failure mode AUD-022 produced from the other direction,
    where absent data read as a data limitation instead of a bug.
    """
    import dataclasses

    from aitb.data.loader import load_market_data
    from aitb.strategies import STRATEGY_CLASSES

    # Construct the absence rather than relying on the synthetic provider
    # lacking the panel — it MIRRORS the real one now (freeze v11), precisely
    # so this strategy can be exercised before a real run. A real store that
    # was never given a FINRA download is the case under test.
    md = load_market_data("synthetic", mode="synthetic")
    empty = dataclasses.replace(md, short_share=pd.DataFrame())
    with pytest.raises(ValueError, match="FINRA short-sale volume"):
        STRATEGY_CLASSES["ShortSqueezeCandidate"]().build(empty)


def test_short_squeeze_holds_nothing_before_finra_published_anything():
    """The pre-2009 blind spot must be visible as zero exposure.

    FINRA began publishing 2009-07-31. A rule reading it is silent across the
    dot-com collapse and 2008 — the two most informative regimes in the study's
    window. If missing history quietly became a tradeable signal, the
    leave-one-year-out check would report stability across years the strategy
    could not have traded at all.
    """
    import dataclasses

    import numpy as np
    from aitb.data.loader import load_market_data
    from aitb.strategies import STRATEGY_CLASSES

    md = load_market_data("synthetic", mode="synthetic")
    rng = np.random.default_rng(5)
    cols = list(md.adj_close.columns[:40])
    panel = pd.DataFrame(rng.uniform(0.15, 0.6, (len(md.calendar), len(cols))),
                         index=md.calendar, columns=cols)
    panel.loc[:"2009-07-30"] = np.nan

    w = STRATEGY_CLASSES["ShortSqueezeCandidate"]().build(
        dataclasses.replace(md, short_share=panel))
    assert w.loc[:"2009-07-30"].abs().to_numpy().sum() == 0.0
    assert w.loc["2011-01-01":].abs().to_numpy().sum() > 0, "never traded at all"


def test_short_squeeze_is_not_offered_as_a_chart_export():
    """FINRA short volume is not a TradingView feed, so there is no honest Pine.

    Every other rule in the chartable family exports. This one must appear in
    the refused list with a reason, rather than shipping an approximation built
    from data a chart does have.
    """
    from aitb.platform.tradingview import _GENERATORS, export_pine
    assert "ShortSqueezeCandidate" not in _GENERATORS
    assert export_pine("ShortSqueezeCandidate") is None
