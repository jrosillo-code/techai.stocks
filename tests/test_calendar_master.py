"""Exchange calendar and security-master tests."""
import pandas as pd
import pytest

from aitb.calendar import early_closes, get_calendar, next_session, trading_sessions
from aitb.data.security_master import (audit_duplicates, load_master, resolve,
                                       stitch_history)


# --------------------------------------------------------------- calendar ---
def test_holidays_excluded():
    s = trading_sessions(pd.Timestamp("2023-01-01").date(), pd.Timestamp("2023-12-31").date())
    for holiday in ("2023-01-02",   # New Year (observed)
                    "2023-04-07",   # Good Friday
                    "2023-07-04",   # Independence Day
                    "2023-11-23",   # Thanksgiving
                    "2023-12-25"):  # Christmas
        assert pd.Timestamp(holiday) not in s, holiday
    assert pd.Timestamp("2023-07-03") in s      # regular (early-close) session
    assert len(s) == 250                        # 2023 NYSE session count


def test_early_close_sessions_flagged():
    ec = early_closes(pd.Timestamp("2023-01-01").date(), pd.Timestamp("2023-12-31").date())
    assert pd.Timestamp("2023-11-24") in ec     # day after Thanksgiving
    assert pd.Timestamp("2023-07-03") in ec


def test_rebalance_rolls_forward_from_holiday():
    # Good Friday 2024-03-29: next session is Monday 2024-04-01.
    assert next_session(pd.Timestamp("2024-03-29")) == pd.Timestamp("2024-04-01")
    # Weekend month-end: Sat 2024-06-29 -> Mon 2024-07-01.
    assert next_session(pd.Timestamp("2024-06-29")) == pd.Timestamp("2024-07-01")


def test_unscheduled_closures():
    s = trading_sessions(pd.Timestamp("2001-09-01").date(), pd.Timestamp("2001-09-30").date())
    for closed in ("2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14"):
        assert pd.Timestamp(closed) not in s    # post-9/11 closure
    sandy = trading_sessions(pd.Timestamp("2012-10-25").date(), pd.Timestamp("2012-11-02").date())
    assert pd.Timestamp("2012-10-29") not in sandy  # Hurricane Sandy


def test_mode_calendar_selection():
    real = get_calendar("real", pd.Timestamp("2023-01-01").date(), pd.Timestamp("2023-12-31").date())
    synth = get_calendar("synthetic", pd.Timestamp("2023-01-01").date(), pd.Timestamp("2023-12-31").date())
    assert len(real) == 250
    assert len(synth) == 260                    # plain b-days keep holidays


# ---------------------------------------------------------- security master --
def test_symbol_resolves_by_date():
    assert resolve("FB", pd.Timestamp("2015-06-01").date()).sid == "SEC_META"
    assert resolve("META", pd.Timestamp("2023-06-01").date()).sid == "SEC_META"
    assert resolve("FB", pd.Timestamp("2023-06-01").date()) is None  # renamed away
    assert resolve("JAVA", pd.Timestamp("2008-01-01").date()).sid == "SEC_SUNW"


def test_no_overlapping_symbol_claims():
    assert audit_duplicates() == []


def test_successors_recorded():
    m = load_master()
    assert m["SEC_XLNX"].successor["sid"] == "SEC_AMD"
    assert m["SEC_EMC"].successor["sid"] == "SEC_DELL"
    assert m["SEC_SUNW"].successor["sid"] == "SEC_ORCL"


def test_stitch_refuses_fabricated_returns():
    """A rename must not splice frames across a long gap (no fake returns)."""
    idx1 = pd.bdate_range("2007-01-02", "2007-08-23")
    idx2 = pd.bdate_range("2007-08-24", "2009-12-31")
    f1 = pd.DataFrame({"close": 5.0, "adj_close": 5.0}, index=idx1)
    f2 = pd.DataFrame({"close": 5.1, "adj_close": 5.1}, index=idx2)
    out = stitch_history("SEC_SUNW", {"SUNW": f1, "JAVA": f2})
    assert len(out) == len(idx1) + len(idx2)          # contiguous rename: OK
    # Same frames but with a 60-day hole between spans -> refuse.
    f2_gap = f2.iloc[40:]
    with pytest.raises(ValueError, match="gap"):
        stitch_history("SEC_SUNW", {"SUNW": f1, "JAVA": f2_gap})
    # Ticker spans clip frames to their validity windows, so a vendor frame
    # bleeding past the rename date cannot double-count either:
    f1_bleed = pd.DataFrame({"close": 5.0},
                            index=pd.bdate_range("2007-01-02", "2007-12-31"))
    out2 = stitch_history("SEC_SUNW", {"SUNW": f1_bleed, "JAVA": f2})
    assert not out2.index.has_duplicates
