"""Strategies added under research freeze v4.

NOTE ON FAMILIES. Only MarketNeutralMomentum and BetaHedgedBasket are in the
`longshort` family. The others live in this module for authorship reasons but
are long-only, and are labelled `quality` or `factor` accordingly. That
distinction is not cosmetic: `family` is part of every experiment's identity,
groups the deflated-Sharpe trial batteries, and drives the correlation-by-family
chart on the site. Labelling a long-only screen "long-short" made the site claim
hedged strategies correlate 0.83 with the index when the actual hedged ones
correlate about zero.

WHY THESE, AND HOW THEY WERE CHOSEN
-----------------------------------
The first real study tested 208 variants and produced zero robust candidates,
with almost everything correlating 0.7–0.9 to simply holding the Nasdaq-100.

That correlation was never a finding about technology stocks. It was a
mechanical consequence of the study's own design: **every one of the 208
variants was long-only.** A portfolio that is always long the same market must
co-move with it. The families added in v3 to "find something that behaves
differently" could not possibly have succeeded, because the constraint that
forced the co-movement was never relaxed.

The engine has supported shorts and charged borrow costs since v1. Nothing had
used them. That is the gap these fill, and it is derived from the STRUCTURE of
the study — not from reading the leaderboard and building something that would
have scored well on it. The latter would be fitting to an answer sheet that has
already been opened.

WHAT THIS COSTS, STATED UP FRONT
--------------------------------
Adding these raises the total trial count, and the deflated Sharpe ratio
corrects for every trial ever run. These strategies therefore make the bar
HIGHER for themselves and for all 208 that came before. That is the correct
accounting and it is not free.

More importantly: the holdout period has already been opened once. Any result
these produce on it is a SECOND look, which is development evidence, not
out-of-sample evidence. They are recorded and reported as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import MarketData
from ..features import momentum, pit_fundamental_panel, realized_vol, xs_zscore
from ..portfolio import equal_weight, rebalance_schedule, top_n_selection
from ..universe import investable_mask
from .base import Strategy


class MarketNeutralMomentum(Strategy):
    """Long the strongest names, short the weakest, dollar-neutral.

    The long book and the short book carry the same market exposure, so the
    index component cancels and what remains is the spread between winners and
    losers. This is the only construction in the study capable of a low
    correlation to the market by design rather than by accident.

    It is also the most expensive: two books to trade, borrow charged daily on
    the short side at the scenario rate (50bp base, 150bp stressed), and shorts
    that lose money grow into the position rather than shrinking out of it.

    Two ways the modelled result is NOT conservative, both worth stating:
    borrow is a flat rate, where real borrow on hard-to-locate shares runs to
    5-50% and some names cannot be shorted at all; and the study contains 39
    companies that died, so the short book has hindsight access to names a
    lender would very likely have bought you in on. Against that, the engine
    marks a delisted short at its last quoted price rather than at zero, which
    forgoes the full gain a real bankruptcy short would have earned.
    """
    family = "longshort"
    hypothesis = ("The winner-minus-loser spread within technology is a real,"
                  " tradeable return stream that survives borrow costs, and"
                  " being dollar-neutral removes the market exposure that made"
                  " every long-only variant a repackaged index fund. Expected"
                  " to have far lower correlation to QQQ and a far lower raw"
                  " return; the open question is whether the spread clears its"
                  " own costs.")

    def __init__(self, lookback_days: int = 126, skip_days: int = 21,
                 top_n: int = 10, gross: float = 1.0, rebalance: str = "ME",
                 basket: str | None = None):
        super().__init__(lookback_days=lookback_days, skip_days=skip_days,
                         top_n=top_n, gross=gross, rebalance=rebalance,
                         basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        px = md.adj_close[mask.columns]
        mom = momentum(px, p["lookback_days"], p["skip_days"]).where(mask)

        longs = top_n_selection(mom, p["top_n"])
        shorts = top_n_selection(-mom, p["top_n"])
        # A name cannot be in both books (possible when fewer than 2N names are
        # investable — early history, or a narrow basket).
        both = (longs > 0) & (shorts > 0)
        longs = longs.where(~both, 0.0)
        shorts = shorts.where(~both, 0.0)

        half = p["gross"] / 2.0
        w = equal_weight(longs) * half - equal_weight(shorts) * half
        # Only trade on dates where BOTH books exist; a one-sided book is a
        # directional bet this strategy does not claim to make.
        ok = (longs.sum(axis=1) > 0) & (shorts.sum(axis=1) > 0)
        w = w.where(ok, 0.0)
        return rebalance_schedule(w, p["rebalance"])


class BetaHedgedBasket(Strategy):
    """Hold the basket; short the index at the basket's trailing beta.

    Keeps the stock-selection decision and subtracts the market. Where
    MarketNeutralMomentum hedges by holding an offsetting book of stocks, this
    hedges with the index directly — cheaper to trade and more precise, but it
    only removes the part of the risk the index explains.
    """
    family = "longshort"
    hypothesis = ("Once the market component is hedged away, whatever remains"
                  " in a megacap technology basket is either genuine selection"
                  " value or nothing. This measures which, and it is the"
                  " cleanest available test of whether owning these specific"
                  " companies beats owning the sector.")

    def __init__(self, basket: str = "megacap_ai", hedge: str = "QQQ",
                 beta_window: int = 126, max_hedge: float = 1.5,
                 rebalance: str = "W-FRI"):
        super().__init__(basket=basket, hedge=hedge, beta_window=beta_window,
                         max_hedge=max_hedge, rebalance=rebalance)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        mask = investable_mask(md, md.universe.baskets[p["basket"]])
        px = md.adj_close[mask.columns]
        w = equal_weight(mask)

        if p["hedge"] not in md.adj_close.columns:
            raise ValueError(f"hedge instrument {p['hedge']} not in the data")
        bench = md.adj_close[p["hedge"]].pct_change()
        port = (w.shift(1) * px.pct_change()).sum(axis=1)

        # Trailing beta of the basket to the index. Rolling cov/var only —
        # every input is through T, so the hedge ratio used on T+1 is knowable.
        cov = port.rolling(p["beta_window"]).cov(bench)
        var = bench.rolling(p["beta_window"]).var()
        beta = (cov / var.replace(0, np.nan)).clip(0.0, p["max_hedge"])

        out = w.copy()
        # No hedge until a beta exists — an unhedged stub is honest; a
        # fabricated beta of 1.0 would not be.
        out[p["hedge"]] = (-beta).fillna(0.0)
        return rebalance_schedule(out.fillna(0.0), p["rebalance"])


class GrossProfitability(Strategy):
    """Rank on profitability, not growth (Novy-Marx 2013).

    Gross profitability is the most robust quality signal in the literature and
    is close to orthogonal to the growth screens already tested: the companies
    that earn the most per dollar of revenue are frequently not the ones
    growing fastest. Uses gross profit over total assets — the actual ratio —
    from the point-in-time panel, so a quarter enters only on its filing date.
    Falls back to the free-cash-flow margin only where a data store predates
    the v4 concept set and lacks the proper inputs.
    """
    family = "quality"
    hypothesis = ("Profitability predicts returns as strongly as growth does"
                  " and is close to independent of it, so a profitability"
                  " screen should select a materially different set of"
                  " companies than the growth screens already tested — and"
                  " selecting differently is the precondition for behaving"
                  " differently.")

    def __init__(self, top_n: int = 12, rebalance: str = "QE",
                 min_history_q: int = 8, basket: str | None = None):
        super().__init__(top_n=top_n, rebalance=rebalance,
                         min_history_q=min_history_q, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        if md.fundamentals.empty:
            raise ValueError("GrossProfitability requires point-in-time fundamentals")

        cols_all = md.fundamentals.columns
        if "gross_profit" in cols_all and "assets" in cols_all:
            # The real Novy-Marx ratio: gross profit over total assets.
            num = pit_fundamental_panel(md, "gross_profit", "ttm")
            den = pit_fundamental_panel(md, "assets", "level")
        else:
            # Fallback for stores collected before these fields were requested.
            # Free-cash-flow margin conflates profitability with capital
            # intensity — a large error in semiconductors — so it is used only
            # when the proper inputs are absent, never in preference to them.
            num = pit_fundamental_panel(md, "fcf", "ttm")
            den = pit_fundamental_panel(md, "revenue", "ttm")

        cols = [c for c in num.columns if c in mask.columns and c in den.columns]
        margin = (num[cols] / den[cols].replace(0, np.nan)).where(mask[cols])

        sel = top_n_selection(margin, p["top_n"]).where(margin > 0, 0.0)
        w = equal_weight(sel).reindex(columns=mask.columns, fill_value=0.0)
        return rebalance_schedule(w, p["rebalance"])


class PostEarningsDrift(Strategy):
    """Buy revenue surprises, hold for a fixed window (Bernard & Thomas 1989).

    The study has always excluded event-driven strategies for lack of
    intraday-precision earnings timestamps. That is the right call for
    same-day trading, but it is too strict for a multi-week drift: the SEC
    filing date is known exactly, and holding for a quarter makes the
    open-versus-close ambiguity irrelevant.

    Surprise is measured against the company's OWN trailing growth trend, not
    against analyst estimates (which are not in the data and are not
    point-in-time available for free).
    """
    family = "quality"
    hypothesis = ("Prices under-react to earnings news and drift for weeks"
                  " afterwards. This is among the most replicated anomalies in"
                  " finance, has never been tested here, and can be tested"
                  " honestly with filing dates alone — no intraday timestamps"
                  " required for a 63-day hold.")

    def __init__(self, hold_days: int = 63, top_n: int = 10,
                 rebalance: str = "W-FRI", basket: str | None = None):
        super().__init__(hold_days=hold_days, top_n=top_n,
                         rebalance=rebalance, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        if md.fundamentals.empty:
            raise ValueError("PostEarningsDrift requires point-in-time fundamentals")

        # Year-on-year TTM revenue growth, and how much it accelerated. Both
        # step only on publication dates by construction of the PIT panel.
        yoy = pit_fundamental_panel(md, "revenue", "yoy_growth")
        cols = [c for c in yoy.columns if c in mask.columns]
        yoy = yoy[cols]

        # A "surprise" is a step change in the published figure: the panel is
        # flat between filings, so a non-zero diff marks a fresh filing and its
        # size is the change versus what was previously known.
        step = yoy.diff()
        fresh = step.where(step.abs() > 1e-12)
        # Carry the surprise forward for the holding window, then let it lapse.
        surprise = fresh.ffill(limit=p["hold_days"])
        # Only count a surprise while it is still inside its window.
        age = (~fresh.isna()).astype(int)
        held = age.rolling(p["hold_days"], min_periods=1).max().astype(bool)
        score = surprise.where(held).where(mask[cols])

        sel = top_n_selection(score, p["top_n"]).where(score > 0, 0.0)
        w = equal_weight(sel).reindex(columns=mask.columns, fill_value=0.0)
        return rebalance_schedule(w, p["rebalance"])


class DispersionTimedSelection(Strategy):
    """Be selective only when selection can pay.

    Stock-picking earns nothing when every name moves together — there is no
    spread to capture, only costs to pay. Cross-sectional dispersion measures
    how much spread is available. This holds a concentrated momentum book when
    dispersion is high and reverts to the whole basket when it is not, so the
    turnover is spent only in the conditions where selection has an edge.
    """
    family = "factor"
    hypothesis = ("The payoff to stock selection is proportional to"
                  " cross-sectional dispersion, which is measurable in advance"
                  " and strongly autocorrelated. Conditioning selection on it"
                  " should keep most of the upside while avoiding the cost drag"
                  " of trading through low-dispersion regimes.")

    def __init__(self, lookback_days: int = 126, top_n: int = 8,
                 disp_window: int = 63, disp_pct: float = 0.6,
                 rebalance: str = "ME", basket: str | None = None):
        super().__init__(lookback_days=lookback_days, top_n=top_n,
                         disp_window=disp_window, disp_pct=disp_pct,
                         rebalance=rebalance, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        px = md.adj_close[mask.columns]

        # Dispersion: cross-sectional stdev of trailing returns, smoothed.
        rets = px.pct_change().where(mask)
        disp = rets.rolling(p["disp_window"]).std().std(axis=1)
        # Expanding quantile — the threshold on day T uses only history to T,
        # so it cannot encode the future distribution of dispersion.
        thresh = disp.expanding(min_periods=252).quantile(p["disp_pct"])
        selective = (disp >= thresh).fillna(False)

        mom = momentum(px, p["lookback_days"], skip_days=21).where(mask)
        picked = equal_weight(top_n_selection(mom, p["top_n"]).where(mom > 0, 0.0))
        broad = equal_weight(mask)

        w = picked.where(selective, broad)
        return rebalance_schedule(w.fillna(0.0), p["rebalance"])


class ResearchIntensity(Strategy):
    """Rank on R&D spending relative to revenue.

    Accounting expenses research immediately while its payoff arrives over
    years, so a research-heavy company reports lower earnings than its
    economics justify. The gap is a documented and persistent mispricing
    (Lev & Sougiannis 1996; Chan, Lakonishok & Sougiannis 2001) — and it is
    specifically a TECHNOLOGY effect, which makes its absence from a study of
    technology companies conspicuous.

    Requires the R&D field, which the data layer only began collecting in v4.
    Where a filer does not tag it the name is simply not rankable that quarter,
    never imputed as zero — zero R&D is a claim, absence is not.
    """
    family = "quality"
    hypothesis = ("R&D-intensive technology companies are systematically"
                  " undervalued because accounting expenses their main"
                  " investment immediately, so ranking on R&D-to-revenue"
                  " selects companies whose reported earnings understate their"
                  " economics. This is the tech-specific quality signal the"
                  " study has been missing.")

    def __init__(self, top_n: int = 12, rebalance: str = "QE",
                 require_profitable: bool = False, basket: str | None = None):
        super().__init__(top_n=top_n, rebalance=rebalance,
                         require_profitable=require_profitable, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        if md.fundamentals.empty or "rnd" not in md.fundamentals.columns:
            raise ValueError("ResearchIntensity requires the 'rnd' field; "
                             "re-download fundamentals with the v4 concept set")

        rnd = pit_fundamental_panel(md, "rnd", "ttm")
        rev = pit_fundamental_panel(md, "revenue", "ttm")
        cols = [c for c in rnd.columns if c in mask.columns]
        score = (rnd[cols] / rev[cols].replace(0, np.nan)).where(mask[cols])
        if p["require_profitable"]:
            ni = pit_fundamental_panel(md, "net_income", "ttm")
            score = score.where(ni[cols] > 0)

        sel = top_n_selection(score, p["top_n"]).where(score > 0, 0.0)
        w = equal_weight(sel).reindex(columns=mask.columns, fill_value=0.0)
        return rebalance_schedule(w, p["rebalance"])


class AccrualQuality(Strategy):
    """Prefer earnings backed by cash (Sloan 1996).

    Accruals are the part of reported profit not yet collected in cash.
    Companies with high accruals systematically underperform: the accrual
    component of earnings is less persistent than the cash component, and
    prices behave as if investors do not distinguish them. This holds the
    LOWEST-accrual names — earnings least dependent on estimates.

    A falsification target as much as a candidate: the effect has weakened
    materially since it was published, which is itself worth measuring.
    """
    family = "quality"
    hypothesis = ("Earnings backed by cash persist; earnings backed by accruals"
                  " do not, and prices do not distinguish them. Ranking on the"
                  " cash-to-earnings gap should therefore predict returns —"
                  " though the effect is widely reported to have decayed since"
                  " publication, so a null result here is informative.")

    def __init__(self, top_n: int = 12, rebalance: str = "QE",
                 basket: str | None = None):
        super().__init__(top_n=top_n, rebalance=rebalance, basket=basket)

    def build(self, md: MarketData) -> pd.DataFrame:
        p = self.params
        tickers = md.universe.baskets[p["basket"]] if p["basket"] else None
        mask = investable_mask(md, tickers)
        need = {"net_income", "assets"}
        if md.fundamentals.empty or not need <= set(md.fundamentals.columns):
            raise ValueError("AccrualQuality requires net_income and assets; "
                             "re-download fundamentals with the v4 concept set")

        ni = pit_fundamental_panel(md, "net_income", "ttm")
        fcf = pit_fundamental_panel(md, "fcf", "ttm")
        assets = pit_fundamental_panel(md, "assets", "level")
        cols = [c for c in ni.columns if c in mask.columns]
        # Accruals = the non-cash part of profit, scaled by size. Low is good,
        # so the score is negated before ranking.
        accruals = (ni[cols] - fcf[cols]) / assets[cols].replace(0, np.nan)
        score = (-accruals).where(mask[cols])

        sel = top_n_selection(score, p["top_n"])
        w = equal_weight(sel).reindex(columns=mask.columns, fill_value=0.0)
        return rebalance_schedule(w, p["rebalance"])
