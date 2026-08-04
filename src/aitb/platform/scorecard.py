"""Research-quality scorecard — NOT a performance score.

Eleven 0–5 dimensions with explicit reasons, so the card answers "why is this
strategy trusted (or not)?" rather than "did it go up?". Derived entirely
from recorded artifacts: the frozen grid status, registry metrics (already
computed — no new holdout access), tier labels, audit findings and code
introspection.
"""
from __future__ import annotations

import numpy as np

from .catalog import StrategyEntry


def _dim(score: float, reason: str) -> dict:
    return {"score": round(max(0.0, min(5.0, score)), 1), "reason": reason}


def build_scorecard(e: StrategyEntry) -> dict:
    recs = e.experiments
    dev = [(r.get("metrics_dev") or {}) for r in recs]
    hold = [(r.get("metrics_holdout") or {}) for r in recs]
    psrs = [r.get("psr_dev") for r in recs if r.get("psr_dev") is not None]

    dims: dict[str, dict] = {}

    dims["implementation_quality"] = _dim(
        4.0 if e.status in ("core", "exploratory") else 2.0,
        "engine-level accounting verified by hand ledgers + per-bar invariant "
        "(audit 2026-08); strategy expressed through audited primitives"
        if e.status != "unlisted" else "not part of the frozen, audited grid")

    dims["audit_confidence"] = _dim(
        4.0,
        "covered by the adversarial audit (no CRITICAL findings; engine cleared); "
        "residual: AUD-010 verdict-layer false discovery, AUD-012 dividend "
        "reinvestment assumption")

    tier_scores = {"A": 4.5, "B": 2.5, "C": 0.5, "": 1.5}
    dims["data_quality"] = _dim(
        tier_scores.get(e.tier, 1.5),
        e.tier_reason or "tier not yet assigned (no real-data run)")
    dims["evidence_tier"] = _dim(
        tier_scores.get(e.tier, 1.5),
        f"Tier {e.tier or '? (synthetic only)'}")

    if psrs:
        p = float(np.nanmean([float(x) for x in psrs]))
        dims["statistical_confidence"] = _dim(
            5 * p, f"mean probabilistic Sharpe {p:.2f} across {len(psrs)} variants "
                   "(deflated-Sharpe correction applied at family level)")
    else:
        dims["statistical_confidence"] = _dim(0.5, "no recorded experiments")

    if dev and hold:
        degr = [max(float((d.get('sharpe') or 0)) - float((h.get('sharpe') or 0)), 0.0)
                for d, h in zip(dev, hold)]
        m = float(np.nanmean(degr)) if degr else 1.0
        dims["robustness"] = _dim(
            4.5 - 2.0 * min(m, 2.0),
            f"mean dev→holdout Sharpe degradation {m:.2f} "
            "(subperiod stability also enters the composite ranking)")
    else:
        dims["robustness"] = _dim(0.5, "no dev/holdout record")

    n_params = len(e.parameters)
    dims["interpretability"] = _dim(
        5.0 - 0.4 * max(n_params - 2, 0),
        f"{n_params} parameters; hypothesis stated: "
        f"{'yes' if e.hypothesis else 'no'}")

    dims["capacity"] = _dim(
        3.5 if e.family in ("riskmanaged", "benchmark", "tsmom", "regime") else 2.5,
        "large-cap universe, √participation impact model; see capacity.csv for "
        "size-scaled estimates (real-data capacity unverified)")

    dims["tradingview_compatibility"] = _dim(
        4.5 if e.tradingview else 0.5, e.tradingview_note)

    doc_len = len(e.docstring) + len(e.hypothesis)
    dims["documentation_quality"] = _dim(
        min(5.0, 1.0 + doc_len / 120),
        f"docstring+hypothesis {doc_len} chars; full auto-generated doc page")

    dims["reproducibility"] = _dim(
        5.0 if e.status != "unlisted" else 2.0,
        "deterministic seeds, stable experiment IDs, append-only registry, "
        "freeze-v2-fingerprinted implementation" if e.status != "unlisted"
        else "not fingerprint-bound")

    overall = round(float(np.mean([d["score"] for d in dims.values()])), 2)
    trusted = (overall >= 3.5 and e.status == "core"
               and dims["statistical_confidence"]["score"] >= 3.0)
    return {
        "strategy": e.class_name,
        "overall": overall,
        "dimensions": dims,
        "verdict": ("trusted for continued research" if trusted else
                    "not yet trusted — see low-scoring dimensions"),
        "note": ("Research-quality assessment only; says nothing about future "
                 "returns. Real-data tiers pending until the first real study."),
    }
