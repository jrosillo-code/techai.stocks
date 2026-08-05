"""Research freeze: an immutable, hashed specification of the first real study.

The freeze document captures everything that defines the study BEFORE any
real-data result is seen:

  * every active strategy variant and parameter grid (with core/exploratory/
    deprecated status and stated hypotheses),
  * universe, benchmark and basket definitions,
  * cost scenarios and portfolio-construction rules,
  * split conventions (train/validation/holdout) and subperiod windows,
  * ranking and rejection criteria (as source fingerprints + prose),
  * minimum data-quality requirements,
  * sha256 fingerprints of the code modules that implement strategy logic,
    portfolio construction, the engine, and the ranking.

``verify_freeze`` recomputes the canonical document from the CURRENT configs
and code and compares hashes. Real-mode experiment runs refuse to start on a
mismatch — the first real run cannot silently drift from what was frozen, and
no parameter optimization beyond the frozen grids is possible because the
grids themselves are part of the hash.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import CONFIG_DIR, PROJECT_ROOT, load_yaml
from .utils import get_logger, stable_hash

log = get_logger("freeze")

# v1 (hash c8b99c3fb795d3e19f64cc197283321f) is SUPERSEDED by the adversarial
# audit of 2026-08-04: it bound only strategy/engine/ranking code, leaving
# metrics, statistical validation, evidence tiers, the calendar, the data
# layer and the registry logic changeable without detection (finding AUD-001),
# and its engine/metrics contained fixed defects (AUD-006 CAGR annualization,
# AUD-011 missing invariant hooks). configs/research_freeze_v1.json is
# preserved unmodified for the record; no real-data results were ever produced
# under v1, so nothing is invalidated retroactively.
#
# v2 (hash 49767ea3efc44cead711d72946c3fe31) is SUPERSEDED as of 2026-08-05 by
# a deliberate REDEFINITION of the study, not by a defect in v2's controls:
#
#   * the universe grew from 38 securities to 120 (81 live, 39 delisted). The
#     delisted roster went from 5 names to 39, which is the single largest
#     reduction in survivorship bias the study has made. A backtest over 33
#     live megacaps and a backtest over 81 live names plus 39 dead ones are
#     different experiments; running them under one freeze would have implied
#     otherwise.
#   * two strategy families were added (factor, allocation) to test v2's
#     central negative finding — that its 24 families were one bet in disguise.
#   * finding AUD-016 was fixed: experiment_id did not include the universe, so
#     re-running a strategy against an expanded roster would have reused the
#     old roster's cached curve. That defect could not fire while one freeze
#     held the universe fixed, but it fires exactly at a freeze boundary like
#     this one.
#
# configs/research_freeze_v1.json and _v2.json are preserved unmodified. No
# real-data results were ever produced under either, so nothing is invalidated
# retroactively; the synthetic v2 cohort stays in the registry and is filtered
# out of the leaderboard by universe hash rather than deleted.
# v3 (hash 7edd4b56d181956d1d68645b08e03e04) governed the FIRST REAL STUDY:
# 208 variants on real prices, 624 experiments, gate PASS WITH LIMITATIONS,
# zero robust candidates. That study stands on the permanent record and is not
# altered by what follows.
#
# v4 (2026-08-05) supersedes it for the NEXT study, for two reasons:
#
#   * A structural gap. All 208 v3 variants were long-only, so their high
#     correlation to the index was a property of the study's own constraint,
#     not a finding about technology stocks. The engine has supported shorts
#     and charged borrow since v1 and nothing had used them. The `longshort`
#     family lifts that constraint.
#   * New data. The EDGAR concept set went from 5 fields to 13, adding gross
#     profit, total assets, equity, net income, R&D, debt and cash — all free,
#     all from a request already being made. R&D intensity is the canonical
#     technology-specific quality signal and could not previously be measured
#     at all; gross profitability was being approximated by a free-cash-flow
#     margin that conflates it with capital intensity.
#
# HONEST ACCOUNTING. Both changes RAISE the bar rather than lower it: every
# added variant enters the deflated-Sharpe trial count for the whole study.
# And the holdout has already been opened once, under v3 — any v4 result on it
# is a second look, which is development evidence however it is labelled.
# Genuinely out-of-sample evidence now requires forward paper trading.
# v4 (hash 384dc48714fc530de3f3a4a9168df12f) ran a real study: 666 experiments,
# 2 robust candidates, the first the project has produced on real prices —
# BetaHedgedBasket, which was also the only construction whose correlation to
# the index was near zero rather than 0.6-0.9. That study stands on the record.
#
# v5 (2026-08-05) adds a `chartable` family: strategies designed single-symbol
# from the start, so the rule tested in Python and the rule running on a
# TradingView chart are the same rule. Everything exported to a chart before
# this was portable by accident, and half of those exports were gauges rather
# than entry rules. Same accounting as always: the additions raise the trial
# count and therefore the bar, and the holdout has been opened, so any v5
# result on it is development evidence.
#
# v5 NEVER RAN AGAINST REAL PRICES. It was superseded the same day by v6, after
# a synthetic validation run confirmed the three new strategies executed
# end-to-end but before any real-data study was commissioned under it. The
# synthetic v5 cohort stays in the registry and is filtered out of the
# leaderboard by (universe_hash, freeze_version) rather than deleted. Nothing
# is invalidated retroactively because nothing was ever concluded from it.
#
# v6 (2026-08-05) extends the same family to close the two most conspicuous
# gaps in what this study has ever tested:
#
#   * VOLUME. Across every freeze from v1 to v5, volume appeared in exactly one
#     place — the participation cap that stops a simulated fill from consuming
#     an implausible share of a day's turnover — and never once as a signal.
#     Every breakout rule in the study treats a new high on half-normal
#     turnover and a new high on triple turnover as the same event.
#     VolumeConfirmedBreakout is the first strategy here to read it, and it
#     ships with a CONTROL ARM (vol_mult=1.0, the identical rule with the
#     filter off) so the filter's contribution can be read off rather than
#     inferred. Measured in dollars, which is `volume * close` — a quantity
#     Pine computes exactly rather than approximating.
#   * A BET THAT IS NOT THE SAME BET. The study's central negative finding,
#     unchanged since v2, is that its families are one bet in disguise: all
#     long the same names at broadly the same times, all correlating 0.6-0.9
#     with the index. Every attempt to break that so far has been another way
#     of reading prices. TurnOfMonth reads no market data at all — its exposure
#     for any future year is already determined — so whatever else is wrong
#     with it, it cannot be that bet. Windows are in CALENDAR days, not trading
#     days, because "the third trading day before month end" is not knowable on
#     the day itself without looking forward, and this family exists precisely
#     so the tested rule and the chart rule are the same rule.
#
# HONEST ACCOUNTING, unchanged and unimproved by any of this: both additions
# enter the deflated-Sharpe trial count for the whole study, so they raise the
# bar rather than lower it. The control arm is a real trial and counts as one.
# The holdout has been opened once already, under v3 and again under v4, so any
# v6 result on it is a third look and is development evidence however it is
# labelled. Genuinely out-of-sample evidence still requires forward paper
# trading, and nothing here changes that.
#
# v6 (hash 2f5d0c554f0fed8ba47a95adc5659bdc) NEVER RAN AGAINST REAL PRICES
# either. Its synthetic validation run exposed a defect in its own
# specification, which is exactly what that run is for:
#
#   The VolumeConfirmedBreakout CONTROL ARM did not run. It was marked
#   `status: deprecated` on the reasoning that a control is not a candidate,
#   but `deprecated` in this project means WITHDRAWN — run_experiments.py
#   records such an entry in the registry with its stated reason and never
#   executes it. Six filtered variants ran and the unfiltered comparison was
#   silently empty, so the one thing the control existed to establish — whether
#   the volume filter does anything — could not be established. The frozen v6
#   config additionally asserted in a comment that the entry "still runs",
#   which was false.
#
# v7 (2026-08-05) is v6 with `status: exploratory` on that control arm, so it
# runs and is ranked alongside everything else. Nothing else changed. This is a
# correction to a specification defect caught BEFORE any real-data result
# existed under v6 — the freeze was not edited, because a freeze is never
# edited; a new one was cut, which is what the immutability rule is for.
# configs/research_freeze_v5.json and _v6.json are preserved unmodified, and
# their synthetic cohorts stay in the registry, filtered out of the leaderboard
# by (universe_hash, freeze_version) rather than deleted.
#
# v7 (hash e5a13c8887f98ae693d2eaaa51044e41) ran a synthetic validation and was
# superseded before any real study, like v5 and v6 before it.
#
# v8 (2026-08-05) answers a direct question: which rules suit the AI complex,
# which suit technology broadly, and what exactly do they say to DO. It adds
# five documented indicators the study had never tested, all single-symbol with
# explicit entry and exit:
#
#   GaussianTrendBands       buy an ATR-band break, sell back to the centreline
#   GaussianTrendHold        buy a rising centreline, sell on a wide band break
#   Supertrend               the ratcheting mid-price band, flip long/flat
#   ADXTrendStrength         Wilder's trend STRENGTH, which nothing here measured
#   RelativeStrengthNewHigh  the ratio to the index at a new high (O'Neil line)
#
# Three points of method, none of them optional:
#
#   * EVERY ONE IS GRIDDED OVER AN AI BASKET *AND* THE FULL 81-NAME ROSTER.
#     "Good for AI" versus "good for tech generally" is a question the data can
#     answer by running the same rule over both. Asserting it in a docstring
#     would be a preference wearing a finding's clothes.
#   * GaussianTrendBands and GaussianTrendHold are a deliberate PAIR: identical
#     filter, opposite aggression. Comparing them isolates parameterisation
#     from the indicator, the same way v6/v7's control arm isolates the volume
#     filter from the breakout it rides on.
#   * The Gaussian filter is Ehlers' cascaded-pole form, NOT a centred kernel.
#     A centred Gaussian weights bars on both sides of the point it smooths,
#     which on a price series reads the future; it is the most common way a
#     good-looking band script is silently wrong, and the output looks entirely
#     normal. tests/test_no_lookahead.py pins the causal behaviour on a unit
#     step: ~0.008 at the step bar, half-way only 8 bars later.
#
# Accounting, unchanged: 307 variants means a larger trial battery and a
# harsher deflated-Sharpe correction for the WHOLE study, so this raises the
# bar. The holdout has been opened three times. Genuinely out-of-sample still
# means forward paper trading and nothing else.
FREEZE_VERSION = 8
FREEZE_PATH = CONFIG_DIR / f"research_freeze_v{FREEZE_VERSION}.json"

# Code whose behavior defines the study. Any edit to these invalidates the
# freeze (that is the point). Pure presentation modules (reporting.py, chart
# code, HTML templates) are deliberately excluded so cosmetic report fixes do
# not block a frozen run — but everything that computes, tiers, gates or
# records a number is bound.
_FROZEN_MODULES = [
    "src/aitb/strategies/base.py",
    "src/aitb/strategies/benchmarks.py",
    "src/aitb/strategies/tsmom.py",
    "src/aitb/strategies/xsmom.py",
    "src/aitb/strategies/meanrev.py",
    "src/aitb/strategies/breakout.py",
    "src/aitb/strategies/fundamental.py",
    "src/aitb/strategies/regime.py",
    "src/aitb/strategies/riskmanaged.py",
    "src/aitb/strategies/ml.py",
    # bound since v3 (new families):
    "src/aitb/strategies/factor.py",
    "src/aitb/strategies/allocation.py",
    # bound since v4 (long-short / hedged family):
    "src/aitb/strategies/longshort.py",
    # bound since v5 (single-symbol family):
    "src/aitb/strategies/chartable.py",
    "src/aitb/portfolio.py",
    "src/aitb/features.py",
    "src/aitb/backtest/engine.py",
    "src/aitb/costs.py",
    "src/aitb/ranking.py",
    "src/aitb/universe.py",
    # bound since v2 (audit finding AUD-001):
    "src/aitb/metrics.py",
    "src/aitb/validation.py",
    "src/aitb/tiers.py",
    "src/aitb/calendar.py",
    "src/aitb/experiments.py",
    "src/aitb/holdout.py",
    "src/aitb/tax.py",
    "src/aitb/data/loader.py",
    "src/aitb/data/import_bundle.py",
    "src/aitb/data/quality.py",
    "src/aitb/data/security_master.py",
    "src/aitb/data/providers.py",
    "src/aitb/data/providers_ext.py",
]

REJECTION_CRITERIA = [
    "fails to beat the best simple diversified benchmark score by the margin",
    "negative holdout Sharpe",
    "cost-fragile: base-cost edge collapses under the stressed scenario",
    ">50% of P&L from a single security (concentration penalty)",
    "performance dependent on one calendar year (leave-one-year-out)",
    "performance existing only post-2023 (AI-rally dependence)",
    "bootstrap 90% CI on Sharpe includes zero by a wide margin",
    "requires data that failed the quality gate (assigned Tier C, not run)",
]

MIN_DATA_QUALITY = {
    "min_universe_names": 10,
    "required_benchmarks": ["SPY", "QQQ"],
    "max_missing_session_pct": 0.03,
    "fundamental_leak_tolerance": 0,
    "gate_statuses_permitted": ["PASS FOR RESEARCH", "PASS WITH LIMITATIONS"],
}


def _module_hash(rel: str) -> str:
    p = PROJECT_ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, cwd=PROJECT_ROOT).stdout.strip()
    except Exception:
        return "unknown"


def build_canonical() -> dict:
    """The canonical study definition, rebuilt from current configs + code."""
    return {
        "freeze_version": FREEZE_VERSION,
        "experiments": load_yaml("experiments.yaml"),
        "universe": load_yaml("universe.yaml"),
        "costs": load_yaml("costs.yaml"),
        "backtest": load_yaml("backtest.yaml"),
        "security_master": load_yaml("security_master.yaml"),
        "code_fingerprints": {m: _module_hash(m) for m in _FROZEN_MODULES},
        "ranking_criteria": (
            "composite score = 2*holdout_sharpe + dev_sharpe + 0.5*min(calmar,3) "
            "+ stability + (PSR-0.5) - degradation - turnover/concentration/"
            "complexity/regime penalties + max_dd; verdict relative to best "
            "diversified benchmark score + 0.25 margin (see ranking.py fingerprint)"),
        "rejection_criteria": REJECTION_CRITERIA,
        "min_data_quality": MIN_DATA_QUALITY,
        "no_optimization_clause": (
            "The first real run executes exactly the frozen grids. No parameter "
            "search, selection, or re-ranking outside them is permitted; "
            "walk-forward selection in robustness reporting is diagnostic only "
            "and does not alter the frozen specification."),
    }


def freeze_hash(canonical: dict) -> str:
    return stable_hash(canonical, 32)


def create_freeze() -> Path:
    if FREEZE_PATH.exists():
        raise FileExistsError(
            f"{FREEZE_PATH} already exists — a freeze is immutable. "
            f"Increment FREEZE_VERSION for a new study specification.")
    canonical = build_canonical()
    doc = {
        "hash": freeze_hash(canonical),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "canonical": canonical,
    }
    FREEZE_PATH.write_text(json.dumps(doc, indent=2, default=str, sort_keys=True))
    log.info("research freeze v%d created: hash=%s", FREEZE_VERSION, doc["hash"])
    return FREEZE_PATH


def load_freeze() -> dict:
    if not FREEZE_PATH.exists():
        raise FileNotFoundError(
            f"no research freeze found at {FREEZE_PATH} — create it with "
            "`python -m aitb.freeze` BEFORE any real-data run")
    return json.loads(FREEZE_PATH.read_text())


def verify_freeze() -> dict:
    """Raise unless the current configs+code match the frozen hash exactly."""
    doc = load_freeze()
    current = freeze_hash(build_canonical())
    if current != doc["hash"]:
        frozen = doc["canonical"]
        live = build_canonical()
        diffs = []
        for key in live:
            if stable_hash(live[key]) != stable_hash(frozen.get(key)):
                diffs.append(key)
        raise RuntimeError(
            f"RESEARCH FREEZE VIOLATION: current spec hash {current} != frozen "
            f"{doc['hash']} (changed sections: {diffs}). The first real-data "
            "study must run the frozen specification. Either revert the "
            "changes or, if this is deliberately a NEW study, increment "
            "FREEZE_VERSION and create a new freeze.")
    log.info("freeze verified: v%d hash=%s (frozen %s)",
             doc["canonical"]["freeze_version"], doc["hash"], doc["created_at"][:10])
    return doc


def registry_freeze_record(doc: dict, data_mode: str) -> dict:
    """Registry record announcing which freeze governs the runs that follow."""
    return {
        "id": f"freeze_v{doc['canonical']['freeze_version']}_{data_mode}",
        "status": "freeze",
        "freeze_hash": doc["hash"],
        "freeze_created_at": doc["created_at"],
        "freeze_git_commit": doc["git_commit"],
        "data_mode": data_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    path = create_freeze()
    print(f"Freeze written: {path}")
    print(f"Hash: {json.loads(path.read_text())['hash']}")
