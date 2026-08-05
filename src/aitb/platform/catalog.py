"""Strategy catalog: one merged view per strategy class.

Aggregates — strictly read-only — code introspection (docstring, params,
source), registry records (experiments, verdicts, tiers, metrics), genealogy,
scorecard inputs and TradingView compatibility. This is the backbone the
dashboard, strategy pages, comparison engine and documentation generator all
draw from. It never writes to any registry and never triggers a holdout
access (it only reads metrics already recorded).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import pandas as pd

from ..config import load_yaml
from ..experiments import ExperimentRegistry
from ..strategies import STRATEGY_CLASSES
from ..utils import get_logger

log = get_logger("platform.catalog")

# Strategies expressible as a self-contained TradingView Pine v5 script
# (single-instrument logic, no cross-sectional ranking, no fundamentals).
TRADINGVIEW_PORTABLE = {
    "QQQMovingAverage": "trend filter on one symbol — fully portable",
    "TrendFollowCash": "per-symbol SMA gate — portable per symbol (cash leg manual)",
    "DonchianBreakout": "channel breakout — portable per symbol",
    "VolCompressionBreakout": "squeeze breakout — portable per symbol",
    "RSIReversion": "RSI(2) reversion w/ trend filter — portable per symbol",
    "BollingerReversion": "band reversion — portable per symbol",
    "AbsoluteMomentum": "absolute momentum gate — portable per symbol",
    "DrawdownDeRisk": "equity-curve de-risking — approximable on one symbol",
    "VolTargetedBasket": "vol targeting — approximable on one symbol",
    "TrendPlusVolTarget": "trend gate + vol scaling — approximable on one symbol",
}
NOT_PORTABLE_REASON = ("requires cross-sectional ranking, multi-asset portfolio "
                       "state, or point-in-time fundamentals — not expressible "
                       "as a standalone TradingView script")

# Rules that are single-symbol but still cannot port, because the DATA is not on
# a chart. Kept separate from the reason above so the page does not tell a
# reader that a single-symbol rule needs cross-sectional ranking.
NOT_PORTABLE_DATA = {
    "ShortSqueezeCandidate":
        "single-symbol, but reads FINRA daily short-sale volume, which is not "
        "a TradingView data feed. Approximating it from price and volume would "
        "be a different strategy wearing this one's name.",
}


@dataclass
class StrategyEntry:
    class_name: str
    family: str
    hypothesis: str
    docstring: str
    parameters: dict          # name -> default
    source: str
    status: str               # core / exploratory / deprecated / unlisted
    status_reason: str
    grids: list
    experiments: list = field(default_factory=list)   # registry records (base scen)
    best_verdict: str = ""
    best_score: float | None = None
    tier: str = ""
    tier_reason: str = ""
    tradingview: bool = False
    tradingview_note: str = ""

    @property
    def n_experiments(self) -> int:
        return len(self.experiments)



def load_registry(mode: str) -> pd.DataFrame:
    """Registry records, de-duplicated by experiment id (first write wins).

    The registry is append-only by design. Re-running a study whose equity
    curves were deleted re-appends identical records for the same ids — the
    results are byte-identical (deterministic seeds), but counting the rows
    twice would overstate how much research was done. The engine-side registry
    is fingerprinted by the research freeze and must not be edited, so the
    display layer de-duplicates instead.
    """
    df = ExperimentRegistry.for_mode(mode).load()
    if df.empty or "id" not in df.columns:
        return df
    return df.drop_duplicates("id", keep="first")


def current_registry(mode: str) -> pd.DataFrame:
    """Registry records for the CURRENT universe cohort only.

    Freeze v3 widened the universe from 38 securities to 120, so the registry
    now holds results from two incompatible rosters. A strategy's track record,
    its run count and every headline number must come from one of them — a
    Sharpe earned picking among 33 megacaps is not comparable to one earned
    picking among 81 live names plus 39 dead ones. The older cohort stays in
    the file (nothing is ever deleted) and is reported separately as superseded.
    """
    from ..ranking import current_cohort
    return current_cohort(load_registry(mode))


def _grid_status() -> dict[str, tuple[str, str, list]]:
    """class -> (status, reason, grids) from experiments.yaml (the frozen grid)."""
    spec = load_yaml("experiments.yaml")
    out: dict[str, tuple[str, str, list]] = {}
    for family, entries in spec.items():
        for e in entries:
            cls = e["class"]
            status = e.get("status", "exploratory")
            reason = e.get("reason", "")
            prev = out.get(cls)
            # A class can appear with several statuses (e.g. core grid + one
            # deprecated variant): keep the strongest active status, collect
            # all grids.
            rank = {"core": 3, "exploratory": 2, "deprecated": 1}
            if prev is None or rank.get(status, 0) > rank.get(prev[0], 0):
                out[cls] = (status, reason, (prev[2] if prev else []) + [e.get("grid", {})])
            else:
                out[cls] = (prev[0], prev[1], prev[2] + [e.get("grid", {})])
    return out


def build_catalog(mode: str = "synthetic",
                  ranking: pd.DataFrame | None = None) -> dict[str, StrategyEntry]:
    registry = ExperimentRegistry.for_mode(mode)
    # Current cohort only: a strategy page showing a track record must not
    # interleave runs from two different universes.
    df = current_registry(mode)
    ok = df[(df.get("status") == "ok") & (df.get("scenario") == "base")] \
        if not df.empty else pd.DataFrame()
    grid_info = _grid_status()

    if ranking is None:
        rank_path = registry.root / "strategy_ranking.csv"
        ranking = pd.read_csv(rank_path) if rank_path.exists() else pd.DataFrame()

    catalog: dict[str, StrategyEntry] = {}
    for cls_name, cls in sorted(STRATEGY_CLASSES.items()):
        sig = inspect.signature(cls.__init__)
        params = {k: (v.default if v.default is not inspect.Parameter.empty else None)
                  for k, v in sig.parameters.items() if k != "self"}
        status, reason, grids = grid_info.get(cls_name, ("unlisted", "not in the frozen grid", []))
        entry = StrategyEntry(
            class_name=cls_name,
            family=getattr(cls, "family", "unspecified"),
            hypothesis=getattr(cls, "hypothesis", ""),
            # The class's OWN docstring, never an inherited one. inspect.getdoc
            # walks the MRO, so any strategy without a docstring silently
            # inherited abc.ABC's — "Helper class that provides a standard way
            # to create an ABC using inheritance" was published as the
            # description of ten strategies. The stated hypothesis is the
            # honest fallback; it is what the strategy claims to do.
            docstring=(cls.__doc__ or "").strip() or getattr(cls, "hypothesis", ""),
            parameters=params,
            source=inspect.getsource(cls),
            status=status, status_reason=reason, grids=grids,
            tradingview=cls_name in TRADINGVIEW_PORTABLE,
            tradingview_note=TRADINGVIEW_PORTABLE.get(
                cls_name, NOT_PORTABLE_DATA.get(cls_name, NOT_PORTABLE_REASON)),
        )
        if not ok.empty:
            mine = ok[ok["strategy"].str.startswith(cls_name + "(")]
            entry.experiments = mine.to_dict("records")
        if not ranking.empty:
            rmine = ranking[ranking["strategy"].str.startswith(cls_name + "(")]
            if len(rmine):
                best = rmine.sort_values("score", ascending=False).iloc[0]
                entry.best_verdict = str(best.get("verdict", ""))
                entry.best_score = float(best.get("score", float("nan")))
                entry.tier = str(best.get("tier", "")) if "tier" in rmine.columns else ""
                entry.tier_reason = str(best.get("tier_reason", "")) if "tier_reason" in rmine.columns else ""
        catalog[cls_name] = entry
    return catalog


def platform_stats(mode: str = "synthetic") -> dict:
    """Headline numbers for the dashboard."""
    registry = ExperimentRegistry.for_mode(mode)
    all_df = load_registry(mode)
    df = current_registry(mode)
    cat = build_catalog(mode)
    rank_path = registry.root / "strategy_ranking.csv"
    ranking = pd.read_csv(rank_path) if rank_path.exists() else pd.DataFrame()

    def n_status(s):
        return sum(1 for e in cat.values() if e.status == s)

    stats = {
        "total_strategies": len([e for e in cat.values() if e.status != "unlisted"]),
        "core": n_status("core"), "exploratory": n_status("exploratory"),
        "deprecated": n_status("deprecated"),
        "experiments_total": int(len(df)) if not df.empty else 0,
        "experiments_ok": int((df.get("status") == "ok").sum()) if not df.empty else 0,
        "experiments_failed": int((df.get("status") == "failed").sum()) if not df.empty else 0,
        "holdout_events": int((df.get("status") == "holdout_event").sum()) if not df.empty else 0,
        # Runs from earlier universe definitions: still on the permanent record,
        # deliberately excluded from every comparison on the site.
        "experiments_superseded": int(max(len(all_df) - len(df), 0)),
    }
    if not ranking.empty:
        vc = ranking["verdict"].value_counts().to_dict()
        stats["robust_candidates"] = int(vc.get("robust_candidate", 0))
        stats["rejected"] = int(vc.get("rejected", 0))
        stats["inconclusive"] = int(vc.get("inconclusive", 0))
        stats["benchmarks"] = int(vc.get("benchmark", 0))
        if "tier" in ranking.columns:
            stats["tiers"] = ranking["tier"].value_counts().to_dict()
    if not df.empty and "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
        if len(ts):
            stats["first_experiment"] = str(ts.min().date())
            stats["last_experiment"] = str(ts.max().date())
            stats["experiments_by_day"] = (
                ts.dt.date.value_counts().sort_index().astype(int)
                .rename(str).to_dict())
    return stats
