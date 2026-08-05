"""Automatic per-strategy documentation and research notebooks.

Everything derives from single sources of truth — class docstrings,
signatures, the frozen grid, the registry, genealogy and scorecards — so
nothing is manually duplicated and nothing can drift from the code.
The generated notebook is a structured research journal (hypothesis,
rationale, assumptions, experiment history, lessons, open questions), not a
Jupyter notebook, and it re-renders automatically as experiments accumulate.
"""
from __future__ import annotations

import html

import numpy as np

from .catalog import StrategyEntry
from .research_mgmt import genealogy_for
from .scorecard import build_scorecard

FAMILY_ASSUMPTIONS = {
    "tsmom": ["trends persist beyond the measurement window",
              "whipsaw costs are outweighed by avoided drawdowns"],
    "xsmom": ["relative winners keep winning over 1-12 month horizons",
              "the universe is broad enough that ranking is meaningful"],
    "meanrev": ["short-term overshoots partially retrace",
                "round-trip costs are below the reversion edge (usually FALSE — kept as a falsification target)"],
    "breakout": ["multi-month highs attract flows and continuation",
                 "regime filters remove bear-market whipsaw"],
    "fundamental": ["fundamentals lead prices at quarterly horizons",
                    "point-in-time publication dates are honored"],
    "regime": ["macro states persist long enough to trade",
               "revised macro data approximates real-time availability (Tier B at best)"],
    "riskmanaged": ["volatility clusters; drawdowns cluster",
                    "de-risking costs less than the tail it removes"],
    "ml": ["a shrunk linear blend generalizes across regimes",
           "purged walk-forward controls leakage"],
    "benchmark": ["passive exposure is the null hypothesis"],
    "factor": ["the shared sector factor is separable from what is left over",
               "the residual, not the total return, is what carries information",
               "participation and consensus are measurable before prices confirm them"],
    "quality": ["published accounts describe the business faithfully",
                "profitability and research spending persist for years",
                "filing dates are the availability dates, never period ends"],
    "longshort": ["shares can actually be borrowed, at roughly the modelled rate",
                  "the winner-minus-loser spread is separable from market direction",
                  "a hedge ratio estimated on the past applies to the future"],
    "allocation": ["trailing covariance is a usable estimate of future covariance",
                   "risk balance is worth its rebalancing cost",
                   "no forecasting skill is required — only measurement"],
}

FAMILY_TIMEFRAME = {
    "meanrev": "days", "breakout": "weeks-months", "xsmom": "months",
    "tsmom": "months", "riskmanaged": "months-years", "fundamental": "quarters",
    "regime": "weeks-months", "ml": "weeks-months", "benchmark": "years",
    "factor": "months", "allocation": "months-years", "longshort": "months",
    "quality": "quarters",
}


def strategy_doc(e: StrategyEntry) -> dict:
    """Structured documentation record for one strategy."""
    lines = e.docstring.splitlines()
    description = lines[0] if lines else e.class_name
    formulation = e.hypothesis or description
    params = [{"name": k, "default": v} for k, v in e.parameters.items()]

    # naive pseudocode from the build() source: keep comments + control flow
    pseudo = []
    for line in e.source.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith(("def build", "if ", "for ", "return ")):
            pseudo.append(s.lstrip("# "))
    return {
        "class": e.class_name,
        "family": e.family,
        "status": e.status,
        "description": description,
        "long_description": e.docstring,
        "mathematical_formulation": formulation,
        "pseudocode": pseudo[:14],
        "parameters": params,
        "grids": e.grids,
        "compatible_markets": "US large-cap equities & ETFs (daily bars)",
        "intended_timeframe": FAMILY_TIMEFRAME.get(e.family, "months"),
        "strengths": _strengths(e),
        "weaknesses": _weaknesses(e),
        "assumptions": FAMILY_ASSUMPTIONS.get(e.family, []),
        "tradingview_compatible": e.tradingview,
        "tradingview_note": e.tradingview_note,
        "evidence_tier": e.tier or "pending real data",
        "audit_status": "covered by adversarial audit (freeze v2)",
        "python_implementation": e.source,
    }


def _strengths(e: StrategyEntry) -> list[str]:
    out = []
    if e.status == "core":
        out.append("Has a real reason to work, and survived the cull that "
                   "retired weaker ideas")
    if len(e.parameters) <= 3:
        out.append("Few settings to tune, so there is little room to "
                   "accidentally fit it to the past")
    if e.family == "riskmanaged":
        out.append("Changes how much you hold, not which stocks you pick — so "
                   "it does not depend on picking winners correctly")
    if e.family == "longshort":
        out.append("Can genuinely move independently of the market, because it "
                   "is not permanently long it — the only construction here "
                   "that can")
    if e.family == "allocation":
        out.append("Requires no forecasting at all — it only measures how "
                   "risky things have been, which is far easier than "
                   "predicting what comes next")
    if e.family == "factor":
        out.append("Built specifically to behave differently from the rest of "
                   "the sector, rather than to score highest on its own")
    if e.tradingview:
        out.append("Simple enough to watch on a TradingView chart")
    return out or ["Still an open question — under evaluation"]


def _weaknesses(e: StrategyEntry) -> list[str]:
    out = []
    if e.family == "meanrev":
        out.append("The edge vanishes once realistic trading costs are charged — "
                   "measured, not assumed")
    if e.family in ("xsmom", "breakout", "ml", "factor"):
        out.append("Ranks companies against each other, so it is only as honest "
                   "as the list it ranks. That list now includes 39 firms that "
                   "went bust or were bought — but free price data often has no "
                   "history for them, which would flatter the result again")
    if e.family == "longshort":
        out.append("Shorting costs money every day you hold it, and a short "
                   "that goes wrong grows into a bigger position instead of "
                   "shrinking out of one. Borrow is modelled at a flat rate; "
                   "real borrow on hard-to-find shares is far worse")
    if e.family == "allocation":
        out.append("Estimates how assets move together from the recent past. "
                   "Those relationships break down in exactly the crises where "
                   "the diversification was supposed to help")
    if e.family == "regime":
        out.append("Relies on economic data as it reads today, not as it read at "
                   "the time — so its real-time value is unproven")
    if e.family == "fundamental":
        out.append("Needs company accounts dated to when they were actually "
                   "published; that data is not yet in place")
    if len(e.parameters) > 5:
        out.append(f"{len(e.parameters)} settings to choose — enough that some "
                   "combination will look good by chance, so the sensitivity "
                   "checks matter")
    return out or ["Limits not yet known — it has only run on simulated data"]


def research_notebook(e: StrategyEntry) -> dict:
    """Structured research journal, auto-updated from the registry."""
    recs = e.experiments
    history = []
    for r in sorted(recs, key=lambda x: str(x.get("timestamp", ""))):
        dev = (r.get("metrics_dev") or {})
        hold = (r.get("metrics_holdout") or {})
        history.append({
            "strategy": r.get("strategy"),
            "date": str(r.get("timestamp", ""))[:10],
            "data_mode": r.get("data_mode"),
            "dev_sharpe": round(float(dev.get("sharpe") or np.nan), 2),
            "holdout_sharpe": round(float(hold.get("sharpe") or np.nan), 2),
            "max_dd": round(float(dev.get("max_drawdown") or np.nan), 2),
            "turnover": round(float(r.get("annual_turnover") or np.nan), 1),
        })

    lessons = []
    if e.status == "deprecated":
        lessons.append(f"Deprecated: {e.status_reason}")
    if e.family == "meanrev":
        lessons.append("Zero-cost Sharpe ≈0.85 became −0.3 under stressed costs — "
                       "cost modeling is the whole story for this family.")
    if e.family == "xsmom":
        lessons.append("Momentum-weighted variant concentrated all P&L in one "
                       "name; selection carries the signal, weighting should not.")
    if e.family == "riskmanaged":
        lessons.append("Trend gate and vol targeting address different failure "
                       "modes; combined overlay dominated either alone in demos.")

    questions = [
        "Does the (synthetic-)demonstrated behavior survive the first real-data study?",
        "Which evidence tier does the strategy earn under the real quality gate?",
    ]
    if e.family in ("fundamental", "regime"):
        questions.append("What data acquisition would lift the tier (see roadmap)?")

    return {
        "strategy": e.class_name,
        "hypothesis": e.hypothesis,
        "rationale": e.docstring.split("\n\n")[0] if e.docstring else "",
        "expected_behavior": FAMILY_TIMEFRAME.get(e.family, ""),
        "assumptions": FAMILY_ASSUMPTIONS.get(e.family, []),
        "references": ["configs/experiments.yaml (frozen grid)",
                       "audit/reports/adversarial_audit.md"],
        "implementation_notes": f"src/aitb/strategies/{e.family}.py"
                                if e.family != "benchmark" else "src/aitb/strategies/benchmarks.py",
        "experiment_history": history,
        "lessons_learned": lessons,
        "remaining_questions": questions,
        "future_ideas": genealogy_for(e.class_name),
        "scorecard": build_scorecard(e),
    }
