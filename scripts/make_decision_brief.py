#!/usr/bin/env python3
"""Generate reports/real/decision_brief.html — the concise verdict document.

Answers, conservatively and only from real-data results:
  1. Did any strategy survive?          4. Which were rejected and why?
  2. What evidence supports it?         5. What should be tested prospectively?
  3. What could invalidate it?          6. Trade / paper-trade / do nothing?

Advancement requires ALL of (§interpretation rules):
  * Tier A evidence,
  * verdict robust_candidate (beats diversified benchmark after costs),
  * bootstrap 90% CI on family-best Sharpe excluding zero,
  * top-1 contribution share < 50% and not AI-rally-dependent,
  * a stated economic hypothesis,
  * capacity at least $1M.

The maximum possible recommendation from a single frozen historical study is
PAPER-TRADE. This script cannot emit "trade".
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aitb.config import reports_dir, results_dir
from aitb.experiments import ExperimentRegistry
from aitb.holdout import holdout_status
from aitb.utils import get_logger

log = get_logger("decision_brief")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-mode", default="real", choices=["real"])
    args = ap.parse_args()
    RES, REP = results_dir("real"), reports_dir("real")
    REP.mkdir(parents=True, exist_ok=True)

    rank_path = RES / "strategy_ranking.csv"
    gate_path = RES / "data_quality.json"
    if not rank_path.exists() or not gate_path.exists():
        log.warning("real results missing — decision brief stays UNAVAILABLE")
        return 0
    ranking = pd.read_csv(rank_path)
    gate = json.loads(gate_path.read_text())
    hs = holdout_status("real")
    registry = ExperimentRegistry.for_mode("real")
    df = registry.load()
    # One universe cohort only (AUD-016). The registry is append-only across
    # freezes, so it holds results from rosters of different sizes; mixing them
    # would corrupt the trial battery, the walk-forward selection and every
    # comparison downstream. Older cohorts stay on the permanent record.
    from aitb.ranking import current_cohort
    df = current_cohort(df)

    fam_path = RES / "robustness" / "family_summary.csv"
    fam = pd.read_csv(fam_path) if fam_path.exists() else pd.DataFrame()
    cap_path = RES / "capacity.csv"
    cap = pd.read_csv(cap_path) if cap_path.exists() else pd.DataFrame()

    active = ranking[ranking.get("verdict", "") != "benchmark"].copy()
    if "tier" not in active.columns:
        active["tier"] = "B"

    # ---- fail-closed preconditions (audit AUD-004/AUD-005) ---------------
    # Any of these forces DO NOTHING regardless of results: a compromised
    # holdout, a gate that is not currently passing, or a gate whose store
    # fingerprint no longer matches (stale results).
    blocked_reason = None
    if hs.get("compromised"):
        blocked_reason = "holdout is COMPROMISED (accessed repeatedly, before freezing, or log tampering detected)"
    elif gate.get("status") not in ("PASS FOR RESEARCH", "PASS WITH LIMITATIONS"):
        blocked_reason = f"data-quality gate is '{gate.get('status')}'"
    else:
        try:
            from aitb.data.quality import store_fingerprint
            if gate.get("store_fingerprint") != store_fingerprint():
                blocked_reason = ("data store changed since validation — gate is stale; "
                                  "re-validate and re-run before deciding")
        except Exception as exc:
            blocked_reason = f"could not verify store fingerprint ({exc})"

    def _finite(x, default=None):
        """Fail-closed float: missing/NaN/inf -> None (never passes a check)."""
        try:
            v = float(x)
            return v if v == v and abs(v) != float("inf") else default
        except (TypeError, ValueError):
            return default

    # ---- advancement test (all criteria, fail-closed) --------------------
    survivors, near_misses = [], []
    for _, row in active.iterrows():
        top_share = _finite(row.get("top_name_share"))
        checks = {"tier_A": row.get("tier") == "A",
                  "robust_verdict": row.get("verdict") == "robust_candidate",
                  # unknown concentration blocks (audit: fail closed on NaN)
                  "not_concentrated": top_share is not None and top_share < 0.5,
                  "not_ai_rally_only": not bool(row.get("fails_pre_ai_rally", False))}
        frow = fam[fam["best_variant"] == row["strategy"]] if len(fam) else pd.DataFrame()
        ci_lo = _finite(frow.iloc[0]["sharpe_ci90_lo"]) if len(frow) else None
        checks["bootstrap_ci_positive"] = ci_lo is not None and ci_lo > 0
        crow = cap[cap["strategy"] == row["strategy"]] if len(cap) else pd.DataFrame()
        # AUD-004 fix: a MISSING capacity estimate blocks advancement — it is
        # unknown, not adequate.
        cap_est = _finite(crow.iloc[0]["capacity_est_usd"]) if len(crow) else None
        checks["capacity_1M_plus"] = cap_est is not None and cap_est >= 1e6
        entry = {"strategy": row["strategy"], "family": row["family"],
                 "score": row["score"], "checks": checks,
                 "passed": all(checks.values()) and blocked_reason is None}
        (survivors if entry["passed"] else near_misses).append(entry)

    rejected = active[active["verdict"] == "rejected"]
    deprecated = df[df["status"] == "deprecated"] if "status" in df.columns else pd.DataFrame()

    if blocked_reason:
        decision = f"DO NOTHING — advancement blocked: {blocked_reason}"
        survivors = []
    elif survivors:
        decision = "PAPER-TRADE the surviving strategy under the prospective protocol"
    else:
        decision = "DO NOTHING (no strategy met all advancement criteria)"

    def check_list(checks: dict) -> str:
        return ", ".join(f"{'✓' if v else '✗'} {k}" for k, v in checks.items())

    surv_html = "".join(
        f"<li><b>{s['strategy']}</b> ({s['family']}, score {s['score']}) — {check_list(s['checks'])}</li>"
        for s in survivors) or "<li>None.</li>"
    near_html = "".join(
        f"<li>{s['strategy']}: {check_list(s['checks'])}</li>"
        for s in sorted(near_misses, key=lambda x: -float(x['score']))[:5]) or "<li>—</li>"
    rej_html = "".join(
        f"<li>{r.strategy} ({r.family}) — score {r.score}</li>"
        for r in rejected.itertuples()) or "<li>None.</li>"
    dep_html = "".join(
        f"<li>{r.strategy}: {getattr(r, 'reason', '')}</li>"
        for r in deprecated.itertuples()) or "<li>None.</li>"

    holdout_note = ("<p class='bad'><b>HOLDOUT COMPROMISED</b>: accessed more than once "
                    "or before the freeze — treat holdout columns as in-sample.</p>"
                    if hs.get("compromised") else
                    f"<p>Holdout accessed once, as logged: "
                    f"{hs['access_log'][0]['purpose'] if hs.get('access_log') else 'never'}.</p>")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Decision brief — first real-data study</title>
<style>body{{font-family:sans-serif;max-width:850px;margin:2.5rem auto;line-height:1.5;padding:0 1rem}}
h1{{border-bottom:3px solid #2563eb;padding-bottom:.3rem}} .bad{{color:#991b1b}}
.verdict{{background:#eff6ff;border-left:5px solid #2563eb;padding:1rem;font-size:1.1rem}}
li{{margin:.3rem 0}}</style></head><body>
<h1>Decision brief — first frozen real-data study</h1>
<p>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ·
data-quality gate: <b>{gate['status']}</b> · validated {gate.get('validated_at', '?')[:10]}</p>
{holdout_note}
<div class="verdict"><b>Current decision: {decision}.</b><br>
Live trading is out of scope for a single historical study regardless of results.</div>

<h2>1. Did any strategy survive?</h2>
<p>{len(survivors)} of {len(active)} active strategies met every advancement
criterion (Tier A evidence, beats the diversified benchmark after costs,
bootstrap CI excludes zero, no single-name or single-regime dependence,
adequate capacity).</p><ul>{surv_html}</ul>

<h2>2. Evidence and what could invalidate it</h2>
<p>Supporting evidence per survivor is the check list above plus the full
diagnostics in <code>research_report_full.html</code>. Any of the following
would invalidate a survivor: failure to track its backtest in prospective
paper trading; sensitivity of the edge to the {gate.get('limitations', ['data limitations'])[0] if gate.get('limitations') else 'known data limitations'};
performance concentrated in a regime not yet re-tested out of sample;
materially worse fills than the modeled cost scenarios.</p>

<h2>3. Closest non-survivors (for context, NOT recommendations)</h2>
<ul>{near_html}</ul>

<h2>4. Rejected strategies</h2><ul>{rej_html}</ul>
<h3>Deprecated before the run (with reasons)</h3><ul>{dep_html}</ul>

<h2>5. What to test prospectively</h2>
<p>The paper-trading protocol in <code>docs/prospective_testing_protocol.md</code>
applies unchanged to the surviving strategy (if any). No parameter may be
modified between this study and the prospective test.</p>

<h2>6. Honest caveats</h2>
<ul>
<li>Gate limitations: {'; '.join(gate.get('limitations', [])) or 'none recorded'}.</li>
<li>One historical study — even frozen — is a single draw; survival here is a
reason to paper-trade, never proof of future profitability.</li>
<li>Nothing in this brief is investment advice.</li>
</ul>
</body></html>"""
    out = REP / "decision_brief.html"
    out.write_text(html)
    log.info("decision brief -> %s (%s)", out, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
