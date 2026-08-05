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
FREEZE_VERSION = 4
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
