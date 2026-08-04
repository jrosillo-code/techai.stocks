"""Evidence tiers for the first real-data study.

Tier A — higher confidence: complete adjusted-price histories, no revised-
         macro dependence, no earnings-event assumptions, no missing ticker
         spans, adequate liquidity/cost data.
Tier B — limited confidence: revised macro inputs, incomplete historical
         constituents (survivorship-limited cross-sections), partial
         fundamentals, approximate tax treatment, non-fatal provider
         disagreements.
Tier C — unavailable: required data failed or is absent; the experiment must
         not run (or its result must not be shown).

Tiers are assigned per experiment from (a) the strategy's data dependencies
and (b) the actual quality-gate findings of the run. Tier A and Tier B are
NEVER mixed in one leaderboard.
"""
from __future__ import annotations

# Families whose cross-sectional selection ranges over the full universe —
# survivorship-limited whenever delisted coverage is incomplete.
_UNIVERSE_CROSS_SECTIONAL = {"xsmom", "meanrev", "breakout", "ml"}
# Families that consume macro series (revised, not vintage).
_MACRO_DEPENDENT = {"regime"}
# Families that consume fundamentals.
_FUNDAMENTAL_DEPENDENT = {"fundamental"}


_KNOWN_FAMILIES = (_UNIVERSE_CROSS_SECTIONAL | _MACRO_DEPENDENT
                   | _FUNDAMENTAL_DEPENDENT
                   | {"benchmark", "tsmom", "riskmanaged", "target_holdings"})


def gate_flags(gate: dict | None) -> dict:
    """Extract tiering-relevant facts from a quality-gate document."""
    lims = " ".join((gate or {}).get("limitations", []))
    return {
        "delisted_incomplete": "delisted names" in lims,
        "split_only_prices": "split-adjusted-only" in lims,
        "fundamentals_partial": "fundamentals cover only" in lims,
        "macro_revised": True,   # FRED series are revised, never vintage
        "no_vix": "no VIX" in lims,
    }


def assign_tier(record: dict, gate: dict | None) -> tuple[str, str]:
    """(tier, reason) for one experiment registry record.

    FAIL-CLOSED (audit finding AUD-009): unknown or missing metadata can
    never promote — a record with no recognizable family lands in Tier B at
    best, and a missing/failed record in Tier C.
    """
    family = record.get("family", "")
    flags = gate_flags(gate)

    if record.get("status") in ("failed", "deprecated", None):
        return "C", f"not run ({record.get('status')})"
    if not family or family not in _KNOWN_FAMILIES:
        return "B", (f"unrecognized family '{family}' — tier metadata missing, "
                     "cannot verify Tier A data dependencies")

    if family in _MACRO_DEPENDENT:
        if flags["no_vix"]:
            return "C", "requires VIX/macro series absent from the store"
        return "B", "macro inputs are revised series, not real-time vintages"

    if family in _FUNDAMENTAL_DEPENDENT:
        if flags["fundamentals_partial"]:
            return "B", "point-in-time fundamentals cover only part of the universe"
        return "B", "fundamental data quality not yet independently audited"

    if family in _UNIVERSE_CROSS_SECTIONAL and flags["delisted_incomplete"]:
        return "B", ("cross-sectional selection over a universe missing "
                     "delisted names (current-constituent bias)")

    # Basket / benchmark / overlay strategies on long-history live names.
    if flags["split_only_prices"]:
        # Only a limitation if the canonical series for involved names is
        # split-only; the gate reports affected tickers, treat as B globally
        # to stay conservative when any canonical series lacks dividends.
        return "B", "one or more canonical price series lack dividends"
    return "A", ("complete adjusted histories; no revised-macro, "
                 "earnings-event, or missing-span dependence")


def tier_table(records: list[dict], gate: dict | None) -> list[dict]:
    out = []
    for rec in records:
        tier, reason = assign_tier(rec, gate)
        out.append({"strategy": rec.get("strategy"), "family": rec.get("family"),
                    "tier": tier, "tier_reason": reason})
    return out
