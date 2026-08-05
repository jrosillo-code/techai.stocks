"""Narrow bets: per-cluster and per-name results, with the luck subtracted.

WHY THIS PAGE IS DANGEROUS, AND WHAT IS DONE ABOUT IT
-----------------------------------------------------
Narrowing a strategy to one cluster or one company is the single most reliable
way to manufacture a backtest that means nothing. The mechanism is not subtle:
run N variants against a target and the BEST of them looks good even when
every one of them is worthless, because the maximum of N random draws is not
zero. Do that across 15 clusters and 18 shortlist names and you are running
hundreds of searches while reporting only the winners.

The usual response is a paragraph of warning next to the numbers. That does not
work — the numbers are concrete and the warning is not.

So this module quantifies it instead. For every target it computes what the
BEST of that target's own trials would have scored if none of the strategies
had any edge at all, using the expected maximum Sharpe of N trials under the
null (Bailey & Lopez de Prado). The page then reports:

    excess = best observed Sharpe  -  expected best under the null

A target whose excess is at or below zero has produced exactly what luck
produces. That is not a caveat attached to the result; it IS the result, and it
is shown in the same table cell.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No per-target parameter search. Every number on this page comes from the
frozen grid, evaluated identically across all targets — the SAME variants
against each cluster. Choosing different parameters per cluster is the actual
overfitting the freeze exists to prevent, and it would make the null-model
correction below meaningless, because the trial count would no longer be known.

Nothing here writes to any registry, runs a backtest, or touches the holdout.
It reads recorded results and does arithmetic on them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import load_universe_config
from ..utils import get_logger
from ..validation import expected_max_sharpe

log = get_logger("platform.focus")

_BASKET_RE = re.compile(r"basket=([^,)]+)")
_TICKER_RE = re.compile(r"ticker=([^,)]+)")


@dataclass
class FocusTarget:
    """One cluster or one company, and what the study found on it."""
    key: str
    kind: str                 # "cluster" | "company"
    label: str
    n_members: int
    n_trials: int
    best_strategy: str
    best_sharpe: float
    median_sharpe: float
    expected_best_under_null: float
    excess: float
    best_verdict: str
    worst_drawdown: float
    sharpe_dispersion: float   # sd / |mean| across this target's trials

    @property
    def survives_luck(self) -> bool:
        return self.excess > 0

    @property
    def correction_is_weak(self) -> bool:
        """True when the trials are near-duplicates of each other.

        The null model scales with the SPREAD of the trial Sharpes. Thirty-two
        variants that all score within a whisker of each other produce almost
        no expected maximum, so the correction subtracts almost nothing —
        which is arithmetically right (choosing the best of 32 near-identical
        rules is barely a choice) but means this column is not protecting the
        reader from anything on that target. It has to be said out loud,
        because a large `excess` looks like strong evidence either way.

        This is not hypothetical here: on the real v4 study the megacap_ai
        cluster ran 32 trials with a Sharpe spread of 0.11 against a mean of
        1.09 — a dispersion ratio under 0.10 — so the whole battery was
        effectively one strategy wearing 32 parameter sets.
        """
        return self.n_trials >= 5 and self.sharpe_dispersion < 0.25

    @property
    def reading(self) -> str:
        """Plain English, and never more confident than the arithmetic."""
        if self.n_trials < 3:
            return ("too few trials here to say anything about luck — the "
                    "correction needs a battery, not a couple of runs")
        if self.excess <= 0:
            return ("this is what luck looks like: the best result on this "
                    "target is no better than the best of the same number of "
                    "worthless strategies")
        if self.correction_is_weak:
            return ("the variants tested here are near-duplicates of each "
                    "other, so this correction subtracts almost nothing — "
                    "read the excess as untested, not as passed")
        if self.excess < 0.15:
            return ("marginally above what luck produces, which is not the "
                    "same as evidence")
        return ("clears what luck alone would produce on this many trials — "
                "still development evidence, not out-of-sample")


def _target_of(strategy: str) -> tuple[str, str] | None:
    """Which cluster or company a variant was pointed at, if any.

    basket=None means the whole universe, which is not a narrow bet and is
    excluded — this page is about what happens when you aim.
    """
    m = _BASKET_RE.search(strategy)
    if m and m.group(1) not in ("None", "null", ""):
        return ("cluster", m.group(1))
    m = _TICKER_RE.search(strategy)
    if m:
        return ("company", m.group(1))
    return None


def build_focus(ranking: pd.DataFrame,
                min_trials: int = 2) -> list[FocusTarget]:
    """Group recorded results by the cluster or company they were aimed at."""
    if ranking.empty or "strategy" not in ranking.columns:
        return []

    ucfg = load_universe_config()
    rows: list[FocusTarget] = []
    tagged = []
    for _, r in ranking.iterrows():
        t = _target_of(str(r["strategy"]))
        if t:
            tagged.append((t[0], t[1], r))
    if not tagged:
        return []

    df = pd.DataFrame([{"kind": k, "key": v, **row.to_dict()}
                       for k, v, row in tagged])

    for (kind, key), grp in df.groupby(["kind", "key"]):
        # Buy & hold on a single ticker is a reference point, not a strategy
        # aimed at that company — including it would count the benchmark as a
        # trial and understate the correction.
        active = grp[grp.get("family", "") != "benchmark"]
        if len(active) < min_trials:
            continue
        sr = pd.to_numeric(active["dev_sharpe"], errors="coerce").dropna()
        if sr.empty:
            continue

        # Expected best of N under the null, in the same annualised units as
        # the observed Sharpes. The dispersion is taken from THIS target's own
        # trials: a target whose variants disagree wildly has a higher bar,
        # which is the correct behaviour — spread is what generates lucky maxima.
        per_period = sr.to_numpy() / np.sqrt(252)
        e_max = expected_max_sharpe(len(per_period),
                                    float(np.var(per_period))) * np.sqrt(252)

        best_row = active.loc[sr.idxmax()]
        n_members = (len(ucfg.baskets.get(key, [])) if kind == "cluster" else 1)
        rows.append(FocusTarget(
            key=key, kind=kind,
            label=key.replace("_", " "),
            n_members=n_members,
            n_trials=int(len(sr)),
            best_strategy=str(best_row["strategy"]),
            best_sharpe=float(sr.max()),
            median_sharpe=float(sr.median()),
            expected_best_under_null=float(e_max),
            excess=float(sr.max() - e_max),
            best_verdict=str(best_row.get("verdict", "")),
            worst_drawdown=float(pd.to_numeric(
                active.get("max_drawdown"), errors="coerce").min()),
            sharpe_dispersion=float(sr.std() / abs(sr.mean()))
            if sr.mean() and len(sr) > 1 else 0.0,
        ))

    rows.sort(key=lambda t: t.excess, reverse=True)
    return rows


def focus_summary(targets: list[FocusTarget]) -> dict:
    """Headline counts, phrased so the failure case cannot be skimmed past."""
    if not targets:
        return {"n_targets": 0}
    survived = [t for t in targets if t.survives_luck and t.n_trials >= 3]
    return {
        "n_targets": len(targets),
        "n_clusters": sum(1 for t in targets if t.kind == "cluster"),
        "n_companies": sum(1 for t in targets if t.kind == "company"),
        "total_trials": sum(t.n_trials for t in targets),
        "n_beating_luck": len(survived),
        "n_indistinguishable": len(targets) - len(survived),
        # Targets where the correction subtracts almost nothing because the
        # trials are near-duplicates. Counted separately so a page cannot
        # report "3 of 3 beat luck" without also reporting that the test was
        # barely a test.
        "n_weak_correction": sum(1 for t in targets if t.correction_is_weak),
    }
