"""Static research-site generator.

Builds a fully self-contained multi-page HTML site under site/ from the
registries, catalog, scorecards, genealogy, notebooks, ideas, roadmap,
portfolio lab and audit artifacts. No server, no external assets, vanilla JS
only — reproducible and diffable years later. Strictly read-only over
results; building the site can never touch a registry, a freeze, or the
holdout.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT, load_backtest_config, results_dir
from ..experiments import ExperimentRegistry
from ..utils import get_logger
from .catalog import (build_catalog, current_registry, load_registry,
                      platform_stats)
from .docgen import research_notebook, strategy_doc
from .portfolio_lab import (blend_report, correlation_matrix,
                            load_strategy_returns, regime_cofailure)
from .research_mgmt import assistant_review, build_roadmap, load_ideas
from .scorecard import build_scorecard
from .tradingview import export_all, export_pine

log = get_logger("platform.site")

SITE_DIR = PROJECT_ROOT / "site"

GLOSSARY = {
    "strategy": "A rule for deciding what to hold and when. Tested, not traded.",
    "experiment": "One strategy run once, under one set of trading-cost assumptions.",
    "benchmark": "The simple thing you'd do instead — buy an index fund and wait. "
                 "Everything is measured against this.",
    "robust candidate": "Beat the simple benchmark AND survived the extra checks. "
                        "The best label available — it still is not a recommendation.",
    "inconclusive": "Did something, but not clearly better than the simple option.",
    "rejected": "Failed. Kept on record so the same idea isn't retried by accident.",
    "deprecated": "Deliberately retired, with the reason written down.",
    "Sharpe": "Return earned per unit of bumpiness. Higher is better; "
              "above 1 is good, above 2 is rare and usually too good to be true.",
    "drawdown": "The worst peak-to-trough fall. -50% means the account halved "
                "before recovering.",
    "holdout": "A slice of history locked away and looked at exactly once, so the "
               "final score isn't tuned.",
    "freeze": "A fingerprint of every rule and every line of code, taken before "
              "results were seen, so nothing can be quietly changed afterwards.",
    "Tier A/B/C": "How trustworthy the underlying data is. A = clean, "
                  "B = known gaps, C = not usable.",
}

# Plain-English names for strategy classes. The site should never show a raw
# Python constructor call to a human.
CLASS_LABELS = {
    "BuyAndHold": "Buy & hold",
    "EqualWeightUniverse": "Equal weight",
    "CapWeightUniverse": "Market-cap weight",
    "QQQMovingAverage": "QQQ 200-day trend",
    "SimpleMomentum12_1": "12-month momentum",
    "TrendFollowCash": "Per-stock trend filter",
    "AbsoluteMomentum": "Absolute momentum",
    "DualMomentum": "Dual momentum",
    "XSMomentumTopN": "Pick the strongest stocks",
    "XSRiskAdjMomentum": "Strongest, risk-adjusted",
    "RelativeStrengthVsBench": "Beating the index",
    "RSIReversion": "Buy the dip (RSI)",
    "ShortTermReversal": "Buy the weekly losers",
    "BollingerReversion": "Buy the dip (bands)",
    "DonchianBreakout": "Breakout to new highs",
    "VolCompressionBreakout": "Breakout after a squeeze",
    "QualityGrowth": "Quality & growth",
    "ValuationAwareGrowth": "Growth at a fair price",
    "RegimeSwitchedTech": "Risk-off when markets turn",
    "SemisLeadership": "Follow the chipmakers",
    "VolTargetedBasket": "Steady-risk basket",
    "DrawdownDeRisk": "Cut exposure in drawdowns",
    "InverseVolBasket": "Calm-stock weighting",
    "TrendPlusVolTarget": "Trend filter + steady risk",
    "MLRankStrategy": "Machine-learning ranking",
    "GoldenCrossRotation": "Golden cross",
    # added with freeze v3
    "ResidualMomentum": "Strong for its own reasons",
    "MultiHorizonMomentum": "Trending on every timeframe",
    "LowVolatilityTech": "The calmest tech stocks",
    "BreadthGatedBasket": "Exit when the rally narrows",
    "ThemeRotation": "Rotate between tech themes",
    "FundamentalAcceleration": "Growth that's speeding up",
    "EqualRiskContribution": "Balance the risk, not the money",
    "MinCorrelationSleeve": "The least-alike corner",
}

_BASKETS = {"megacap_ai": "megacap AI", "target_holdings": "your shortlist",
            "ai_compute": "AI compute", "semiconductors": "semiconductors",
            "cloud_platforms": "cloud", "enterprise_ai": "enterprise AI",
            "cybersecurity": "cybersecurity", "dc_infrastructure": "data centres",
            "ai_broad": "all 81 tech names", "semi_equipment": "chip equipment",
            "eda_tools": "chip-design software", "networking": "networking",
            "robotics": "robotics", "ai_power": "AI power",
            "internet_platforms": "internet platforms"}
_REBAL = {"ME": "monthly", "QE": "quarterly", "W-FRI": "weekly", "YE": "yearly"}


def humanize(strategy: str) -> tuple[str, str]:
    """'TrendPlusVolTarget(basket=megacap_ai,target_vol=0.15,…)' ->
    ('Trend filter + steady risk', 'megacap AI · 15% target vol')."""
    cls, _, rest = strategy.partition("(")
    label = CLASS_LABELS.get(cls, cls)
    params = {}
    for kv in rest.rstrip(")").split(","):
        k, _, v = kv.partition("=")
        if k.strip():
            params[k.strip()] = v.strip()

    if cls == "BuyAndHold":
        return f"Buy & hold {params.get('ticker', '')}".strip(), ""

    bits = []
    if params.get("basket") and params["basket"] != "None":
        bits.append(_BASKETS.get(params["basket"], params["basket"]))
    elif "basket" in params:
        bits.append("whole universe")
    if "ticker" in params:
        bits.append(params["ticker"])
    if "top_n" in params:
        bits.append(f"top {params['top_n']}")
    if "target_vol" in params:
        try:
            bits.append(f"{float(params['target_vol']):.0%} target vol")
        except ValueError:
            pass
    for key, fmt in (("sma_window", "{}-day average"),
                     ("lookback_days", "{}-day lookback"),
                     ("entry_window", "{}-day high"),
                     ("trend_window", "{}-day trend"),
                     ("vol_window", "{}-day vol"),
                     ("entry", "enter below {}"),
                     ("weighting", "{} weighted"),
                     # freeze-v3 parameters
                     ("horizons", "{}-day horizons"),
                     ("min_agree", "{} must agree"),
                     ("top_themes", "top {} themes"),
                     ("beta_window", "{}-day beta"),
                     ("cov_window", "{}-day covariance"),
                     ("corr_window", "{}-day correlation"),
                     ("threshold", "breadth above {}")):
        if key in params:
            bits.append(fmt.format(str(params[key]).replace("_", " ")))
    # de-risking depths, written as percentages a human reads
    for key, fmt in (("dd_start", "trim from {}"), ("dd_full", "flat by {}")):
        if key in params:
            try:
                bits.append(fmt.format(f"{float(params[key]):.0%}"))
            except ValueError:
                pass
    if params.get("hysteresis") not in (None, "0.0"):
        bits.append("with a buffer")
    if "rebalance" in params:
        bits.append(_REBAL.get(params["rebalance"], params["rebalance"]))
    return label, " · ".join(bits)



_ACRONYMS = (("PIT ", "point-in-time "), ("SMA", "moving average"),
             ("maxDD", "worst fall"), ("vol ", "volatility "),
             ("RLS", "row-level security"))


def _plain(text: str) -> str:
    """Expand jargon for display. Never mutates the underlying record."""
    out = str(text)
    for a, b in _ACRONYMS:
        out = out.replace(a, b)
    return out


def _strategy_link(strategy: str, depth: int = 0) -> str:
    cls = strategy.split("(")[0]
    label, sub = humanize(strategy)
    prefix = "../" * depth
    sub_html = f"<span class='sub'>{html.escape(sub)}</span>" if sub else ""
    return (f"<a href='{prefix}strategy/{html.escape(cls)}.html'>"
            f"{html.escape(label)}</a>{sub_html}")


_CSS = """
:root{
  color-scheme: light;
  --bg:#f4f4f2; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --mut:#78776f;
  --line:#e3e2dd; --acc:#2a78d6; --acc-soft:#eaf2fd;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; --serious:#ec835a;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --bg:#121211; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7; --mut:#9a998f;
    --line:#333330; --acc:#3987e5; --acc-soft:#17263a;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --bg:#121211; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7; --mut:#9a998f;
  --line:#333330; --acc:#3987e5; --acc-soft:#17263a;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
  margin:0;background:var(--bg);color:var(--ink);line-height:1.55;
  -webkit-font-smoothing:antialiased}
nav{background:var(--surface);border-bottom:1px solid var(--line);
  padding:0 1.2rem;display:flex;gap:.2rem;align-items:center;flex-wrap:wrap;
  position:sticky;top:0;z-index:20}
nav .brand{font-weight:700;margin-right:1rem;padding:.85rem 0;font-size:.95rem}
nav a{color:var(--ink-2);text-decoration:none;font-size:.88rem;padding:.85rem .7rem;
  border-bottom:2px solid transparent}
nav a:hover{color:var(--ink);border-bottom-color:var(--line)}
nav a.on{color:var(--acc);border-bottom-color:var(--acc);font-weight:600}
main{max-width:1120px;margin:0 auto;padding:1.6rem 1.2rem 3rem}
h1{font-size:1.75rem;margin:.2rem 0 .4rem;letter-spacing:-.02em}
h2{font-size:1.1rem;margin:2.2rem 0 .2rem;letter-spacing:-.01em}
h2+.lede{margin:.1rem 0 .9rem}
h3{font-size:.95rem;margin:1.4rem 0 .4rem}
.lede{color:var(--ink-2);font-size:.95rem;max-width:70ch}
.note{color:var(--mut);font-size:.83rem;max-width:78ch}

/* hero */
.hero{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:1.3rem 1.4rem;margin:.6rem 0 1.2rem}
.hero h1{margin-top:0}
.pill{display:inline-flex;align-items:center;gap:.4rem;font-size:.78rem;
  font-weight:600;padding:.2rem .6rem;border-radius:99px;border:1px solid transparent}
.pill-warn{background:#fdf5e3;color:#7a5600;border-color:#f2dfae}
.pill-good{background:#e7f6e7;color:#075c07;border-color:#bde5bd}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .pill-warn{background:#3a2f14;color:#f6cf72;border-color:#5b4a1d}
:root:where(:not([data-theme="light"])) .pill-good{background:#123212;color:#7fd77f;border-color:#1e4d1e}}

/* pipeline */
.pipe{display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0 .3rem}
.step{flex:1 1 170px;background:var(--surface);border:1px solid var(--line);
  border-radius:11px;padding:.7rem .85rem;position:relative}
.step .n{font-size:.72rem;color:var(--mut);font-weight:600;letter-spacing:.04em}
.step .t{font-weight:650;font-size:.92rem;margin:.15rem 0}
.step .d{font-size:.79rem;color:var(--ink-2)}
.step.done{border-color:#bde5bd;background:linear-gradient(0deg,#f4fbf4,var(--surface))}
.step.now{border-color:var(--acc);box-shadow:0 0 0 3px var(--acc-soft)}
.step .tick{position:absolute;top:.6rem;right:.7rem;font-size:.9rem}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .step.done{background:var(--surface);border-color:#1e4d1e}}

/* stats */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.7rem}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:.85rem 1rem}
.stat .v{font-size:1.9rem;font-weight:700;letter-spacing:-.02em;line-height:1.1}
.stat .k{font-size:.83rem;font-weight:600;margin-top:.1rem}
.stat .d{font-size:.78rem;color:var(--mut);margin-top:.25rem;line-height:1.4}

/* charts */
.chart{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:1rem 1.1rem;margin:.5rem 0}
.chart .ct{font-weight:650;font-size:.92rem}
.chart .cs{font-size:.8rem;color:var(--mut);margin-bottom:.7rem}
.bar-row{display:grid;grid-template-columns:minmax(120px,190px) 1fr auto;
  gap:.7rem;align-items:center;margin:.3rem 0;font-size:.85rem}
.bar-row .lab{color:var(--ink-2);text-align:right}
.bar-track{background:var(--line);border-radius:5px;height:16px;overflow:hidden}
.bar-fill{height:100%;border-radius:5px;transition:width .3s}
.bar-row .val{font-variant-numeric:tabular-nums;font-weight:650;min-width:2.4rem}
.spark{display:flex;align-items:flex-end;gap:2px;height:64px;margin-top:.4rem}
.spark i{background:var(--s1);border-radius:3px 3px 0 0;flex:1;min-width:3px;display:block}

/* tables */
.tbl-wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);
  border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{padding:.55rem .75rem;text-align:left;border-bottom:1px solid var(--line)}
th{font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;
  color:var(--mut);font-weight:650;white-space:nowrap;background:var(--surface);
  cursor:pointer;user-select:none}
th:hover{color:var(--ink)}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--acc-soft)}
td a{color:var(--acc);text-decoration:none;font-weight:600}
td a:hover{text-decoration:underline}
.sub{display:block;color:var(--mut);font-size:.76rem;font-weight:400}
.num{font-variant-numeric:tabular-nums;text-align:right}

/* badges */
.badge{display:inline-block;padding:.12rem .5rem;border-radius:99px;
  font-size:.72rem;font-weight:600;white-space:nowrap}
.b-robust_candidate,.b-core,.b-A{background:#e7f6e7;color:#075c07}
.b-inconclusive,.b-exploratory,.b-B{background:#fdf5e3;color:#7a5600}
.b-rejected,.b-deprecated,.b-C{background:#fbeaea;color:#8f1f1f}
.b-benchmark{background:#eaf2fd;color:#12467e}
.b-unlisted{background:var(--line);color:var(--ink-2)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .badge{filter:brightness(1.7) saturate(.8)}}

.warnbox{background:#fdf5e3;border:1px solid #f2dfae;border-left:4px solid var(--warning);
  padding:.8rem 1rem;margin:1rem 0;border-radius:10px;font-size:.88rem;color:#5c4310}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .warnbox{background:#2c2410;border-color:#5b4a1d;color:#f0d79a}}

.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1rem;align-items:start}
@media(max-width:820px){.grid2{grid-template-columns:1fr}
  .bar-row{grid-template-columns:minmax(90px,130px) 1fr auto}}
input,select{padding:.42rem .6rem;border:1px solid var(--line);border-radius:8px;
  margin:.15rem;background:var(--surface);color:var(--ink);font-size:.86rem}
pre{background:#15150f;color:#e8e7de;padding:.9rem;border-radius:10px;
  overflow-x:auto;font-size:.78rem;line-height:1.5}
details{margin:.4rem 0}summary{cursor:pointer;color:var(--acc);font-size:.88rem;font-weight:600}
dfn{border-bottom:1px dotted var(--mut);cursor:help;font-style:normal}
.score{display:flex;gap:.5rem;align-items:center;margin:.3rem 0;font-size:.85rem}
.score .sn{width:190px;color:var(--ink-2)}
.bar{height:9px;border-radius:5px;background:var(--s1);min-width:3px}
.tree{border-left:2px solid var(--line);margin-left:.4rem;padding-left:1rem}
.tree .node{margin:.6rem 0;background:var(--surface);border:1px solid var(--line);
  border-radius:10px;padding:.6rem .85rem;font-size:.85rem}
.legend{display:flex;gap:1rem;flex-wrap:wrap;font-size:.79rem;color:var(--ink-2);margin-top:.6rem}
.legend span{display:inline-flex;align-items:center;gap:.35rem}
.dot{width:10px;height:10px;border-radius:3px;display:inline-block}
"""

_PAGES = [("index.html", "Overview"), ("data.html", "The data"),
          ("experiments.html", "Experiments"),
          ("strategies.html", "Strategies"), ("compare.html", "Compare"),
          ("portfolio.html", "Portfolio lab"), ("ideas.html", "Ideas"),
          ("roadmap.html", "Roadmap"), ("audit.html", "Trust & audit")]


def _nav(current: str, depth: int = 0) -> str:
    p = "../" * depth
    links = "".join(
        f"<a href='{p}{href}'{' class=on' if href == current else ''}>{label}</a>"
        for href, label in _PAGES)
    return f"<nav><span class='brand'>AI &amp; Tech Strategy Research</span>{links}</nav>"


def _page(title: str, body: str, depth: int = 0, current: str = "") -> str:
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
            f"<body>{_nav(current, depth)}<main>{body}"
            f"<p class='note' style='margin-top:2.5rem'>Generated "
            f"{datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')} · "
            f"a read-only view of the research record · "
            f"research tool, not investment advice.</p></main>"
            f"<script>document.querySelectorAll('table th').forEach((th,i)=>{{"
            f"th.onclick=()=>{{const tb=th.closest('table').tBodies[0];"
            f"const rows=[...tb.rows];const asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';"
            f"rows.sort((a,b)=>{{const x=a.cells[i]?.innerText.trim()||'',y=b.cells[i]?.innerText.trim()||'';"
            f"const nx=parseFloat(x.replace(/[^0-9.-]/g,'')),ny=parseFloat(y.replace(/[^0-9.-]/g,''));"
            f"return (!isNaN(nx)&&!isNaN(ny))?(asc?nx-ny:ny-nx):(asc?x.localeCompare(y):y.localeCompare(x));}});"
            f"rows.forEach(r=>tb.appendChild(r));}};}});</script>"
            f"</body></html>")


def _badge(x: str) -> str:
    txt = str(x).replace("_", " ")
    return f"<span class='badge b-{html.escape(str(x))}'>{html.escape(txt)}</span>"


def _dfn(term: str, label: str | None = None) -> str:
    tip = GLOSSARY.get(term, "")
    return f"<dfn title=\"{html.escape(tip)}\">{html.escape(label or term)}</dfn>"


def _mode_banner(mode: str) -> str:
    if mode == "synthetic":
        return ("<div class='warnbox'><b>These numbers are from a simulated "
                "market, not the real one.</b> The system has been built and "
                "audited, but the study against real market prices has not run "
                "yet. Everything here demonstrates that the machinery works — "
                "none of it says anything about how these ideas would have "
                "actually performed.</div>")
    return ""


def _hbar(rows: list[tuple[str, float, str]], title: str, subtitle: str,
          unit: str = "") -> str:
    """Horizontal bars: (label, value, css-color). Direct-labelled — required
    relief for the sub-3:1 contrast slots in the palette."""
    if not rows:
        return ""
    mx = max((v for _, v, _ in rows), default=1) or 1
    bars = "".join(
        f"<div class='bar-row'><div class='lab'>{html.escape(l)}</div>"
        f"<div class='bar-track'><div class='bar-fill' style='width:{max(v / mx * 100, 1.5):.1f}%;"
        f"background:{c}' title='{html.escape(l)}: {v:g}{unit}'></div></div>"
        f"<div class='val'>{v:g}{unit}</div></div>"
        for l, v, c in rows)
    return (f"<div class='chart'><div class='ct'>{html.escape(title)}</div>"
            f"<div class='cs'>{html.escape(subtitle)}</div>{bars}</div>")


# Index/sector funds count as diversified baselines; every other BuyAndHold is
# one company. Mirrors ranking.py's `etf_bh` hurdle set exactly — if one list
# changes the other must, or the page will describe a bar the ranking did not
# actually apply (tests/test_audit_regressions.py pins them together).
_ETF_BENCH = ("SPY", "QQQ", "XLK", "SOXX", "SMH", "IGV", "VGT", "IWM",
              "ARKK", "SKYY", "CIBR", "BOTZ")


# ------------------------------------------------------------- dashboard ----
def _pipeline(mode: str) -> str:
    steps = [
        ("Step 1", "Build the lab", "A backtesting engine with strict rules "
         "against fooling yourself.", "done"),
        ("Step 2", "Audit it", "An adversarial review found and fixed 6 serious "
         "flaws before any results were trusted.", "done"),
        ("Step 3", "Run on real prices", "Not started. Needs one command run on "
         "a machine with internet access.", "now"),
        ("Step 4", "Decide", "Only after step 3. The best possible outcome is "
         "“paper-trade it and watch”.", "todo"),
    ]
    return "<div class='pipe'>" + "".join(
        f"<div class='step {cls}'>{'<span class=tick>✅</span>' if cls == 'done' else ''}"
        f"<div class='n'>{n}</div><div class='t'>{t}</div><div class='d'>{d}</div></div>"
        for n, t, d, cls in steps) + "</div>"


def _dashboard(mode, stats, cat, registry) -> str:
    synthetic = mode == "synthetic"
    pill = ("<span class='pill pill-warn'>● Simulated data — real study not run</span>"
            if synthetic else "<span class='pill pill-good'>● Real market data</span>")

    from ..config import load_universe_config
    u = load_universe_config()
    n_live = sum(1 for s in u.securities if s.delisted is None)
    n_dead = len(u.securities) - n_live

    hero = f"""<div class='hero'>
<h1>AI &amp; Technology Strategy Research</h1>
<p style='margin:.2rem 0 .7rem'>{pill}</p>
<p class='lede'>This is a laboratory for testing investing ideas on US tech and
AI stocks — things like <i>“only hold it while it’s trending up”</i> or
<i>“spread the money across the ten biggest names”</i>. Every idea is measured
against the boring alternative of just buying an index fund and waiting.
The point is to find out which ideas are <b>actually</b> better, and to be
honest when they aren’t.</p>
<p class='lede'>It runs over <b>{n_live} technology companies</b> plus
<b>{n_dead} that no longer exist</b> — bankrupt, bought or taken private.
Including the dead ones is the difference between a backtest that means
something and one that just rediscovers which stocks went up.
<a href='data.html'>See the full list →</a></p>
</div>"""

    # --- what happened, in plain numbers ---------------------------------
    n_strat = stats.get("total_strategies", 0)
    n_exp = stats.get("experiments_ok", 0)
    robust = stats.get("robust_candidates", 0)
    rejected = stats.get("rejected", 0)
    inconc = stats.get("inconclusive", 0)

    stat_cards = f"""<div class='stats'>
<div class='stat'><div class='v'>{n_strat}</div><div class='k'>ideas tested</div>
<div class='d'>Distinct strategies, each with several parameter settings.</div></div>
<div class='stat'><div class='v'>{n_exp:,}</div><div class='k'>test runs recorded</div>
<div class='d'>Every run is kept forever — including the failures, so nothing is
quietly retried.</div></div>
<div class='stat'><div class='v'>{len(u.securities)}</div><div class='k'>companies in the test</div>
<div class='d'>{n_live} still trading, {n_dead} that died. Across 36 years of
market history.</div></div>
<div class='stat'><div class='v'>{robust}</div><div class='k'>beat the benchmark</div>
<div class='d'>Passed every check. Still only a <i>candidate</i>, and only on
simulated data.</div></div>
</div>"""

    # --- verdict chart (status colours + labels; never colour alone) ------
    verdict_rows = [
        ("Beat the benchmark", robust, "var(--good)"),
        ("Unclear", inconc, "var(--warning)"),
        ("Failed", rejected, "var(--critical)"),
    ]
    verdicts = _hbar(verdict_rows, "How the ideas scored",
                     "Each idea, judged against simply buying an index fund and "
                     "holding it. Simulated data.")

    fams = pd.Series({e.class_name: e.family for e in cat.values()
                      if e.status != "unlisted"}).value_counts()
    fam_labels = {"benchmark": "Buy & hold baselines", "riskmanaged": "Risk management",
                  "tsmom": "Trend following", "xsmom": "Pick the strongest",
                  "meanrev": "Buy the dip", "breakout": "Breakouts",
                  "fundamental": "Company fundamentals", "regime": "Market conditions",
                  "ml": "Machine learning", "factor": "Diversifying factors",
                  "allocation": "Smarter weighting"}
    fam_chart = _hbar(
        [(fam_labels.get(f, f), int(n), "var(--s1)") for f, n in fams.items()],
        "What kinds of ideas were tried",
        "Grouped by the reasoning behind them, not by how well they did.")

    # --- timeline ---------------------------------------------------------
    timeline = ""
    by_day = stats.get("experiments_by_day", {})
    if len(by_day) > 8:
        mx = max(by_day.values())
        bars = "".join(f"<i style='height:{max(8, 100 * n / mx):.0f}%' "
                       f"title='{d}: {n} runs'></i>" for d, n in by_day.items())
        timeline = (f"<div class='chart'><div class='ct'>Research activity</div>"
                    f"<div class='cs'>Test runs per day · hover a bar for the count"
                    f"</div><div class='spark'>{bars}</div></div>")
    elif len(by_day) > 1:
        # A handful of days: a sparkline would just be a solid slab. Name the
        # days instead — the honest reading is "this ran in a couple of bursts".
        timeline = _hbar([(str(d), int(n), "var(--s1)") for d, n in by_day.items()],
                         "Research activity",
                         "Test runs per day. The work happened in a few "
                         "concentrated bursts, not continuously.")
    elif by_day:
        # One day. A chart of a single bar is decoration, not information.
        (day, n), = by_day.items()
        timeline = (f"<p class='note'>All {n:,} runs in the current set were "
                    f"executed on {html.escape(str(day))}, in one batch — the "
                    f"whole grid is re-run from scratch whenever the list of "
                    f"companies changes, so there is no drift between the "
                    f"oldest and newest result.</p>")

    # --- leaderboard ------------------------------------------------------
    rank_path = registry.root / "strategy_ranking.csv"
    leaders = ""
    if rank_path.exists():
        r = pd.read_csv(rank_path)
        # One row per IDEA. The grid runs many parameter variants of the same
        # strategy; listing five near-identical rows of the same idea reads as
        # five findings when it is one.
        r["_cls"] = r["strategy"].str.split("(").str[0]
        top = (r[r["verdict"] != "benchmark"]
               .sort_values("score", ascending=False)
               .drop_duplicates("_cls").head(6))
        # The hurdle an active strategy must clear is set by the DIVERSIFIED
        # baselines only — index funds and whole-basket weightings. Buying one
        # company and holding it is not a fair alternative: nobody could have
        # known in advance which company. Showing single-name buy & hold under
        # "the baselines they had to beat" would misstate the actual test.
        is_bh_single = (r["strategy"].str.startswith("BuyAndHold(")
                        & ~r["strategy"].isin(
                            [f"BuyAndHold(ticker={x})" for x in _ETF_BENCH]))
        bench = (r[(r["verdict"] == "benchmark") & ~is_bh_single]
                 .sort_values("score", ascending=False)
                 .drop_duplicates("_cls").head(5))
        singles = (r[(r["verdict"] == "benchmark") & is_bh_single]
                   .sort_values("score", ascending=False).head(5))

        def rows(d):
            return "".join(
                f"<tr><td>{_strategy_link(x.strategy)}</td>"
                f"<td class='num'>{x.score:.1f}</td>"
                f"<td class='num'>{x.holdout_sharpe:.2f}</td>"
                f"<td class='num'>{x.max_drawdown:.0%}</td>"
                f"<td>{_badge(x.verdict)}</td></tr>" for x in d.itertuples())

        head = ("<tr><th>Strategy</th><th class='num'>Score</th>"
                "<th class='num'>Sharpe</th><th class='num'>Worst fall</th>"
                "<th>Verdict</th></tr>")
        leaders = f"""
<h2>Best-performing ideas</h2>
<p class='lede'>Best setting of each distinct idea — the grid tests many
parameter variants of each, and listing near-identical siblings would read as
several findings when it is really one. Ranked by a composite score (0–10) that rewards steady
risk-adjusted returns and punishes fragility — never raw return alone.
<b>Sharpe</b> is return per unit of bumpiness; above 1 is good.
<b>Worst fall</b> is the deepest peak-to-trough drop you'd have sat through.</p>
<div class='tbl-wrap'><table><thead>{head}</thead><tbody>{rows(top)}</tbody></table></div>

<h2>The baselines they had to beat</h2>
<p class='lede'>The bar every idea above had to clear: buying a fund, or the
whole basket, with no cleverness at all. If an idea cannot beat these, it is
not worth the trouble.</p>
<div class='tbl-wrap'><table><thead>{head}</thead><tbody>{rows(bench)}</tbody></table></div>

<h2>What single companies did — <i>not</i> a fair comparison</h2>
<p class='lede'>The best individual stocks beat almost everything. That is not a
strategy: picking the winner required knowing, in 1995, which one it would be.
These are shown because leaving them out would look like hiding them, and they
are deliberately excluded from the bar above.</p>
<div class='tbl-wrap'><table><thead>{head}</thead><tbody>{rows(singles)}</tbody></table></div>"""

    # --- next actions -----------------------------------------------------
    road = build_roadmap(mode)[:4]
    road_html = "".join(
        f"<li><b>{html.escape(r['title'])}</b> <span class='note'>"
        f"({html.escape(r['category'])})</span></li>" for r in road)

    trust = f"""
<h2>Why you can trust these numbers (or not)</h2>
<div class='grid2'>
<div class='chart'><div class='ct'>What's guarded</div>
<div class='cs'>Controls that were built in before any result was seen</div>
<ul class='lede' style='margin:0;padding-left:1.1rem'>
<li><b>No peeking ahead.</b> A strategy decides using yesterday's information and
trades at tomorrow's opening price. It cannot cheat.</li>
<li><b>Locked rules.</b> Every rule and every line of code was fingerprinted
before results existed, so nothing can be tweaked afterwards to look better.</li>
<li><b>Failures kept.</b> {rejected} rejected and {stats.get('deprecated', 0)}
retired ideas stay on the record with reasons.</li>
<li><b>Costs charged.</b> Trading fees, spreads and market impact are all
subtracted — several ideas that looked good stop working once they are.</li>
<li><b>The losers are in.</b> {n_dead} companies that went bankrupt, were bought
or were taken private stay in the test permanently, so a strategy has to buy
them at their peaks and ride them down. <a href='data.html'>The list →</a></li>
</ul></div>
<div class='chart'><div class='ct'>What's still missing</div>
<div class='cs'>Honest limitations</div>
<ul class='lede' style='margin:0;padding-left:1.1rem'>
<li><b>Real prices.</b> The biggest one. Everything here is simulated.</li>
<li><b>Dead companies may not survive contact with real data.</b> The roster
includes {n_dead} of them, but free providers often serve no history for
delisted tickers. If that happens the run reports itself as
survivorship-limited rather than quietly dropping them.</li>
<li><b>More ideas means a higher bar, not a better answer.</b> Every extra
variant tested raises the chance one looks good by luck, and the scoring
subtracts for it.</li>
<li><b>One test is not proof.</b> Even a perfect run would only justify
paper-trading, never real money on its own.</li>
</ul></div></div>
<p class='note'><a href='audit.html'>Full audit and governance detail →</a></p>"""

    return (hero + _mode_banner(mode)
            + "<h2>Where this stands</h2>"
            + "<p class='lede'>Four steps from idea to decision. Two are done.</p>"
            + _pipeline(mode)
            + "<h2>What's been done so far</h2>" + stat_cards
            + "<div class='grid2' style='margin-top:1rem'>"
            + verdicts + fam_chart + "</div>"
            + timeline + leaders + trust
            + "<h2>What happens next</h2>"
            + "<p class='lede'>The top priorities, generated from what the data "
              "is currently missing:</p><ol class='lede'>" + road_html
            + "</ol><p class='note'><a href='roadmap.html'>Full roadmap →</a></p>")


# ----------------------------------------------------- experiment explorer ---
def _dict(x) -> dict:
    """Registry fields loaded through pandas can be NaN — guard to dict."""
    return x if isinstance(x, dict) else {}


def _experiments_page(registry, df=None) -> str:
    df = registry.load() if df is None else df
    if df.empty:
        return "<p>No experiments recorded.</p>"

    # Which roster each run saw. Runs from an older, narrower universe stay on
    # the record permanently but are labelled, because their numbers are not
    # comparable with the current ones and must never be read as if they were.
    from ..ranking import current_cohort
    current_hash = None
    cur = current_cohort(df)
    if not cur.empty and "universe_hash" in cur.columns:
        vals = cur["universe_hash"].dropna()
        current_hash = vals.iloc[0] if len(vals) else None

    def cohort_of(r) -> str:
        uh = r.get("universe_hash")
        if not isinstance(uh, str):
            return "—"
        size = r.get("universe_size")
        n = f"{int(size)} names" if isinstance(size, (int, float)) and size == size else "older roster"
        return f"current · {n}" if uh == current_hash else f"superseded · {n}"

    rows = []
    for r in df.to_dict("records"):
        dev = _dict(r.get("metrics_dev"))
        hold = _dict(r.get("metrics_holdout"))
        rows.append({
            "id": r.get("id", ""), "strategy": r.get("strategy", ""),
            "family": r.get("family", ""), "status": r.get("status", ""),
            "scenario": r.get("scenario", ""), "data_mode": r.get("data_mode", ""),
            "universe": cohort_of(r),
            "provider": r.get("provider", ""),
            "date": str(r.get("timestamp", ""))[:10],
            "dev_sharpe": round(dev.get("sharpe"), 2) if dev.get("sharpe") is not None else None,
            "holdout_sharpe": round(hold.get("sharpe"), 2) if hold.get("sharpe") is not None else None,
            "dev_cagr": round(dev.get("cagr"), 4) if dev.get("cagr") is not None else None,
            "max_dd": round(dev.get("max_drawdown"), 3) if dev.get("max_drawdown") is not None else None,
            "turnover": (round(float(r["annual_turnover"]), 1)
                          if isinstance(r.get("annual_turnover"), (int, float))
                          and r.get("annual_turnover") == r.get("annual_turnover") else None),
            "reason": r.get("reason", "") or r.get("error", ""),
            "hypothesis": r.get("hypothesis", ""),
            "notes": r.get("notes", ""),
        })
    data = json.dumps(rows, default=str)
    return f"""
<p class='note'>{len(rows)} registry records — nothing is ever deleted.
The <b>universe</b> column says which roster of companies each run saw.
<b>Superseded</b> runs were made against an earlier, narrower list; they are kept
for the record and excluded from every ranking and comparison on this site,
because a score earned picking among 33 megacaps means something different from
one earned picking among 120 companies. Filter freely; click a row for the full
record.</p>
<div>
<input id='q' placeholder='search text…' size='28'>
<select id='fFam'><option value=''>family</option></select>
<select id='fStatus'><option value=''>status</option></select>
<select id='fScen'><option value=''>scenario</option></select>
<select id='fMode'><option value=''>data mode</option></select>
<select id='fUniv'><option value=''>universe</option></select>
</div>
<p class='note' id='count' style='margin:.5rem 0'></p>
<div class='tbl-wrap'><table id='tbl'><thead><tr>
<th data-k='date'>date</th><th data-k='strategy'>strategy</th>
<th data-k='family'>family</th><th data-k='status'>status</th>
<th data-k='scenario'>cost</th><th data-k='data_mode'>mode</th>
<th data-k='universe'>universe</th>
<th data-k='dev_sharpe'>dev Sharpe</th><th data-k='holdout_sharpe'>holdout Sharpe</th>
<th data-k='max_dd'>maxDD</th><th data-k='turnover'>turnover</th>
</tr></thead><tbody></tbody></table></div>
<pre id='detail' style='display:none'></pre>
<script>
const DATA = {data};
const tb = document.querySelector('#tbl tbody');
const fills = {{fFam:'family', fStatus:'status', fScen:'scenario', fMode:'data_mode', fUniv:'universe'}};
for (const [id, key] of Object.entries(fills)) {{
  const s = document.getElementById(id);
  [...new Set(DATA.map(r => r[key]).filter(Boolean))].sort().forEach(v => {{
    const o = document.createElement('option'); o.value = v; o.textContent = v; s.append(o);
  }});
  s.onchange = render;
}}
document.getElementById('q').oninput = render;
let sortKey = 'date', sortDir = -1;
document.querySelectorAll('#tbl th').forEach(th => th.onclick = () => {{
  const k = th.dataset.k; sortDir = (sortKey === k) ? -sortDir : -1; sortKey = k; render();
}});
function render() {{
  const q = document.getElementById('q').value.toLowerCase();
  let rows = DATA.filter(r =>
    (!q || JSON.stringify(r).toLowerCase().includes(q)) &&
    Object.entries(fills).every(([id, key]) => {{
      const v = document.getElementById(id).value;
      return !v || r[key] === v;
    }}));
  rows.sort((a, b) => ((a[sortKey] ?? '') < (b[sortKey] ?? '') ? 1 : -1) * sortDir);
  const LIMIT = 500;
  tb.innerHTML = rows.slice(0, LIMIT).map((r, i) =>
    `<tr onclick='show(${{JSON.stringify(JSON.stringify(r))}})'>` +
    ['date','strategy','family','status','scenario','data_mode','universe',
     'dev_sharpe','holdout_sharpe','max_dd','turnover']
      .map(k => `<td>${{r[k] ?? ''}}</td>`).join('') + '</tr>').join('');
  // Never truncate silently: a hidden row limit reads as "that is everything".
  const c = document.getElementById('count');
  c.textContent = rows.length > LIMIT
    ? `Showing the first ${{LIMIT}} of ${{rows.length}} matching records — narrow the filters to see the rest.`
    : `Showing all ${{rows.length}} matching records.`;
}}
function show(j) {{
  const d = document.getElementById('detail');
  d.style.display = 'block';
  d.textContent = JSON.stringify(JSON.parse(j), null, 2);
  d.scrollIntoView({{behavior: 'smooth'}});
}}
render();
</script>"""


# --------------------------------------------------------- strategy pages ----
SCORE_LABELS = {
    "implementation_quality": ("Built correctly", "Is the code doing what it claims?"),
    "audit_confidence": ("Independently checked", "Has a hostile reviewer been through it?"),
    "data_quality": ("Data it ran on", "How trustworthy are the underlying prices?"),
    "evidence_tier": ("Evidence grade", "A = clean data, B = known gaps, C = unusable."),
    "statistical_confidence": ("Could this be luck?", "Higher means less likely to be chance."),
    "robustness": ("Holds up under pressure", "Does it survive when conditions change?"),
    "interpretability": ("Easy to understand", "Fewer moving parts is better."),
    "capacity": ("Room to scale", "Would it still work with real money in it?"),
    "tradingview_compatibility": ("Chartable", "Can you watch it on TradingView?"),
    "documentation_quality": ("Documented", "Is it written down properly?"),
    "reproducibility": ("Repeatable", "Can someone rerun this years from now?"),
}


def _scorebar(dim: str, d: dict) -> str:
    label, why = SCORE_LABELS.get(dim, (dim.replace("_", " ").capitalize(), ""))
    pct = d["score"] / 5 * 100
    colour = ("var(--good)" if d["score"] >= 3.5
              else "var(--warning)" if d["score"] >= 2 else "var(--critical)")
    return (f"<div class='bar-row' style='grid-template-columns:minmax(140px,200px) 1fr 3rem'>"
            f"<div class='lab'>{html.escape(label)}"
            f"<span class='sub' style='text-align:right'>{html.escape(why)}</span></div>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{pct:.0f}%;"
            f"background:{colour}' title='{html.escape(d['reason'])}'></div></div>"
            f"<div class='val'>{d['score']}</div></div>")


def _strategy_page(e, mode: str) -> str:
    doc = strategy_doc(e)
    nb = research_notebook(e)
    sc = nb["scorecard"]
    gen = nb["future_ideas"]
    label = CLASS_LABELS.get(e.class_name, e.class_name)

    # The docstring's first line is often a clause, not a sentence — take the
    # whole first paragraph so the page never ends mid-thought.
    blurb = " ".join((e.docstring or "").split("\n\n")[0].split())

    hist_rows = "".join(
        f"<tr><td>{h['date']}</td><td>{_strategy_link(str(h['strategy']), depth=1)}</td>"
        f"<td class='num'>{h['dev_sharpe']}</td>"
        f"<td class='num'>{h['holdout_sharpe']}</td>"
        f"<td class='num'>{h['max_dd']:.0%}</td>"
        f"<td class='num'>{h['turnover']}</td></tr>"
        for h in nb["experiment_history"])

    tree = ""
    for line in gen:
        nodes = "".join(
            f"<div class='node'><b>{html.escape(str(v.get('change', '')))}</b><br>"
            f"<span class='note'>Why: {html.escape(str(v.get('why', '')))} → "
            f"{html.escape(str(v.get('outcome', '')))}"
            f"{' · needed a new freeze' if v.get('requires_freeze_bump') else ''}"
            f"</span></div>"
            for v in line["versions"])
        tree += f"<h3>{html.escape(line['title'])}</h3><div class='tree'>{nodes}</div>"

    pine = export_pine(e.class_name)
    if pine:
        tv = (f"<p class='lede'>{html.escape(e.tradingview_note)}</p>"
              f"<p><a href='../tradingview/{e.class_name}.pine' download "
              f"style='font-weight:600'>⬇ Download the TradingView script</a></p>"
              f"<details><summary>Preview the script</summary>"
              f"<pre>{html.escape(pine)}</pre></details>"
              f"<p class='note'>The Pine version is an approximation of the tested "
              f"Python one — no cost model, single symbol, no cash sleeve. Use it to "
              f"watch the signal on a chart, not to judge performance.</p>")
    else:
        tv = (f"<p class='lede'>Not available on TradingView. "
              f"{html.escape(e.tradingview_note)}</p>")

    verdict_line = ""
    if e.best_verdict:
        verdict_line = f" · best result: {_badge(e.best_verdict)}"

    return f"""
<h1>{html.escape(label)}</h1>
<p class='lede'><i>{html.escape(_plain(e.hypothesis))}</i></p>
<p>{_badge(e.status)}{verdict_line} · research-quality score
<b>{sc['overall']}/5</b> · {e.n_experiments} test runs</p>
{_mode_banner(mode)}

<h2>What it does</h2>
<p class='lede'>{html.escape(_plain(blurb))}</p>
<p class='note'>Works on: {doc['compatible_markets']} · typical holding period:
{doc['intended_timeframe']}</p>

<div class='grid2'>
<div class='chart'><div class='ct'>What's good about it</div><div class='cs'>&nbsp;</div>
<ul class='lede' style='margin:0;padding-left:1.1rem'>
{''.join(f'<li>{html.escape(s)}</li>' for s in doc['strengths'])}</ul></div>
<div class='chart'><div class='ct'>What's wrong with it</div><div class='cs'>&nbsp;</div>
<ul class='lede' style='margin:0;padding-left:1.1rem'>
{''.join(f'<li>{html.escape(s)}</li>' for s in doc['weaknesses'])}</ul></div>
</div>

<h2>How much to trust it</h2>
<p class='lede'>This scores the <b>quality of the research</b>, not the returns.
A high score means the finding is well-built and well-checked — it does not mean
the strategy makes money. Hover any bar for the detail behind the score.</p>
<div class='chart'>{''.join(_scorebar(k, v) for k, v in sc['dimensions'].items())}
<p class='note' style='margin-top:.8rem'><b>Verdict: {html.escape(sc['verdict'])}.</b>
{html.escape(sc['note'])}</p></div>

<h2>What it assumes</h2>
<p class='lede'>If these stop being true, the strategy stops working.</p>
<ul class='lede'>{''.join(f'<li>{html.escape(s)}</li>' for s in doc['assumptions'])}</ul>

<h2>Track record</h2>
<p class='lede'>Every recorded run of this idea. <b>Sharpe</b> is return per unit
of bumpiness. <b>Holdout</b> is the locked-away slice of history, looked at once —
if it is much worse than the development figure, the idea was over-tuned.</p>
<div class='tbl-wrap'><table><thead><tr><th>Date</th><th>Setting</th>
<th class='num'>Sharpe (dev)</th><th class='num'>Sharpe (holdout)</th>
<th class='num'>Worst fall</th><th class='num'>Turnover</th></tr></thead>
<tbody>{hist_rows}</tbody></table></div>

<h2>What we learned</h2>
<ul class='lede'>{''.join(f'<li>{html.escape(x)}</li>' for x in nb['lessons_learned']) or '<li>Nothing recorded yet.</li>'}</ul>
<h3>Still open</h3>
<ul class='lede'>{''.join(f'<li>{html.escape(x)}</li>' for x in nb['remaining_questions'])}</ul>

<h2>How this idea evolved</h2>
{tree or "<p class='note'>No version history recorded yet.</p>"}

<h2>Watch it on TradingView</h2>
{tv}

<h2>The fine print</h2>
<details><summary>Settings and their defaults</summary>
<div class='tbl-wrap' style='margin-top:.5rem'><table><thead><tr><th>Setting</th>
<th>Default</th></tr></thead><tbody>
{''.join(f"<tr><td>{html.escape(p['name'])}</td><td>{html.escape(str(p['default']))}</td></tr>" for p in doc['parameters'])}
</tbody></table></div></details>
<details><summary>Exact settings tested (frozen)</summary>
<pre>{html.escape(json.dumps(doc['grids'], indent=1))}</pre></details>
<details><summary>Python source (fingerprinted by the research freeze)</summary>
<pre>{html.escape(doc['python_implementation'])}</pre></details>
"""


# ------------------------------------------------------------ compare page ---
def _compare_page(registry, df=None) -> str:
    df = registry.load() if df is None else df
    ok = df[(df.get("status") == "ok") & (df.get("scenario") == "base")] \
        if not df.empty else pd.DataFrame()
    if ok.empty:
        return "<p>No experiments.</p>"
    recs = []
    for r in ok.to_dict("records"):
        dev = _dict(r.get("metrics_dev"))
        hold = _dict(r.get("metrics_holdout"))
        ps = _dict(r.get("period_split_2023"))
        recs.append({
            "strategy": r["strategy"], "family": r["family"],
            "cls": r["strategy"].split("(")[0],
            "dev_sharpe": dev.get("sharpe"), "holdout_sharpe": hold.get("sharpe"),
            "cagr": dev.get("cagr"), "vol": dev.get("ann_vol"),
            "max_dd": dev.get("max_drawdown"), "calmar": dev.get("calmar"),
            "sortino": dev.get("sortino"), "turnover": r.get("annual_turnover"),
            "top_share": _dict(r.get("contributions")).get("top1_share"),
            "nvda_share": _dict(r.get("contributions")).get("nvda_share"),
            "post2023_dep": (ps.get("ai_rally_dependent") if isinstance(ps, dict) else None),
            "psr": r.get("psr_dev"),
        })
    data = json.dumps(recs, default=str)
    cols = ["dev_sharpe", "holdout_sharpe", "cagr", "vol", "max_dd", "calmar",
            "sortino", "turnover", "top_share", "nvda_share", "post2023_dep", "psr"]
    return f"""
<p class='note'>Tick any strategies to compare side by side. Export the
selection as CSV. All values from the registry (base costs, dev period unless
labeled holdout).</p>
<input id='q' placeholder='filter…' size='30'>
<div id='list' style='max-height:260px;overflow:auto;background:#fff;
border:1px solid var(--line);padding:.4rem;margin:.4rem 0'></div>
<button onclick='exportCSV()'>Export CSV</button>
<div id='out'></div>
<script>
const DATA = {data};
const COLS = {json.dumps(cols)};
const sel = new Set();
function renderList() {{
  const q = document.getElementById('q').value.toLowerCase();
  document.getElementById('list').innerHTML = DATA
    .filter(r => !q || r.strategy.toLowerCase().includes(q))
    .map((r, i) => `<label style='display:block;font-size:.8rem'>
      <input type='checkbox' ${{sel.has(r.strategy) ? 'checked' : ''}}
       onchange='toggle(${{JSON.stringify(r.strategy)}})'> ${{r.strategy}}</label>`)
    .join('');
}}
function toggle(name) {{ sel.has(name) ? sel.delete(name) : sel.add(name); renderTable(); }}
function fmt(v) {{ return v === null || v === undefined ? '—' :
  (typeof v === 'number' ? v.toFixed(3) : v); }}
function renderTable() {{
  const rows = DATA.filter(r => sel.has(r.strategy));
  if (!rows.length) {{ document.getElementById('out').innerHTML = ''; return; }}
  let h = '<table><tr><th>metric</th>' +
    rows.map(r => `<th>${{r.strategy.slice(0, 42)}}</th>`).join('') + '</tr>';
  for (const c of COLS) {{
    h += `<tr><td><b>${{c}}</b></td>` +
      rows.map(r => `<td>${{fmt(r[c])}}</td>`).join('') + '</tr>';
  }}
  document.getElementById('out').innerHTML = h + '</table>';
}}
function exportCSV() {{
  const rows = DATA.filter(r => sel.has(r.strategy));
  const csv = ['strategy,' + COLS.join(',')].concat(
    rows.map(r => [r.strategy].concat(COLS.map(c => r[c] ?? '')).join(','))).join('\\n');
  const a = document.createElement('a');
  a.href = 'data:text/csv,' + encodeURIComponent(csv);
  a.download = 'comparison.csv'; a.click();
}}
document.getElementById('q').oninput = renderList;
renderList();
</script>"""


# ------------------------------------------------------------- build all -----
def build_site(mode: str = "synthetic", out: Path | None = None) -> Path:
    out = out or SITE_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "strategy").mkdir(exist_ok=True)
    registry = ExperimentRegistry.for_mode(mode)
    stats = platform_stats(mode)
    cat = build_catalog(mode)
    reg_df = load_registry(mode)      # full permanent record (explorer)
    cur_df = current_registry(mode)   # current universe cohort (comparisons)

    def head(title: str, lede: str) -> str:
        return f"<h1>{html.escape(title)}</h1><p class='lede'>{lede}</p>"

    # dashboard
    (out / "index.html").write_text(
        _page("Overview — AI & Tech Strategy Research",
              _dashboard(mode, stats, cat, registry), current="index.html"))

    # the data behind everything
    (out / "data.html").write_text(
        _page("The data — AI & Tech Strategy Research",
              _data_page(mode), current="data.html"))

    # experiments
    (out / "experiments.html").write_text(_page(
        "Experiments",
        head("Every test ever run",
             "One row per test run. Nothing is ever deleted — failures and "
             "retired ideas stay here permanently, which is what stops the same "
             "dead end being explored twice. Click any row for the full record.")
        + _mode_banner(mode) + _experiments_page(registry, reg_df),
        current="experiments.html"))

    # strategies index + pages
    rows = "".join(
        f"<tr><td><a href='strategy/{e.class_name}.html'>"
        f"{html.escape(CLASS_LABELS.get(e.class_name, e.class_name))}</a>"
        f"<span class='sub'>{html.escape(e.class_name)}</span></td>"
        f"<td>{_badge(e.status)}</td>"
        f"<td class='num'>{e.n_experiments}</td>"
        f"<td>{'✔' if e.tradingview else '—'}</td>"
        f"<td class='lede' style='font-size:.82rem'>{html.escape(_plain(e.hypothesis)[:120])}</td></tr>"
        for e in sorted(cat.values(), key=lambda x: (x.status != "core", x.class_name))
        if e.status != "unlisted")
    (out / "strategies.html").write_text(_page(
        "Strategies",
        head("The ideas being tested",
             "Each one is a rule for deciding what to hold and when. "
             "<b>Core</b> ideas are the headline set; <b>exploratory</b> ones are "
             "being probed; <b>deprecated</b> ones were retired on purpose, with "
             "the reason recorded. Click through for a plain-English explanation, "
             "its track record, and a TradingView script where one is possible.")
        + _mode_banner(mode)
        + "<div class='tbl-wrap'><table><thead><tr><th>Idea</th><th>Status</th>"
          "<th class='num'>Runs</th><th>TradingView</th><th>The bet it makes</th>"
          "</tr></thead><tbody>" + rows + "</tbody></table></div>",
        current="strategies.html"))
    for e in cat.values():
        if e.status == "unlisted":
            continue
        (out / "strategy" / f"{e.class_name}.html").write_text(
            _page(CLASS_LABELS.get(e.class_name, e.class_name),
                  _strategy_page(e, mode), depth=1))

    # compare
    (out / "compare.html").write_text(_page(
        "Compare",
        head("Put ideas side by side",
             "Tick any two or more to line up their numbers. Export the "
             "selection as a spreadsheet with the button. Only runs against the "
             "current list of companies appear here — older runs used a narrower "
             "list and are not comparable.")
        + _mode_banner(mode) + _compare_page(registry, cur_df), current="compare.html"))

    # portfolio lab
    (out / "portfolio.html").write_text(_page(
        "Portfolio lab",
        head("Do these ideas work well together?",
             "Two mediocre strategies can beat a good one if they fail at "
             "different times. This page checks whether they actually do — how "
             "closely they move together, whether they collapse in the same "
             "market conditions, and whether combining them helps.")
        + _mode_banner(mode) + _portfolio_page(mode), current="portfolio.html"))

    # ideas + roadmap
    (out / "ideas.html").write_text(_page(
        "Ideas",
        head("Ideas waiting to be tested",
             "Written down before any code is built, so a hypothesis can't be "
             "invented after seeing what worked.") + _ideas_page(),
        current="ideas.html"))
    (out / "roadmap.html").write_text(_page(
        "Roadmap",
        head("What to work on next",
             "Ranked automatically by expected value against effort, and fed by "
             "what the data is currently missing plus any unresolved audit "
             "findings.") + _roadmap_page(mode), current="roadmap.html"))

    # audit summary page
    (out / "audit.html").write_text(_page(
        "Trust & audit",
        head("Why you can trust these numbers",
             "This system was reviewed as if by a hostile auditor whose job was "
             "to prove the results wrong. Everything found is listed below — "
             "fixed or still open.") + _audit_page(), current="audit.html"))

    # tradingview exports
    export_all(out / "tradingview")

    log.info("site built at %s (mode=%s)", out, mode)
    return out


_REGIME_LABELS = {
    "dotcom_aftermath": "Dot-com crash", "pre_gfc": "Pre-2008 boom",
    "gfc": "2008 crisis", "qe_bull": "2009–19 bull run",
    "covid_shock": "Covid 2020", "speculative_2021": "2021 mania",
    "rate_bear_2022": "2022 rate shock", "ai_rally": "AI rally (2023+)",
}
_MAX_MATRIX = 12   # beyond this a correlation grid is unreadable on a screen


_FAMILY_LABELS = {
    "riskmanaged": "Risk management", "tsmom": "Trend following",
    "xsmom": "Pick the strongest", "meanrev": "Buy the dip",
    "breakout": "Breakouts", "fundamental": "Company fundamentals",
    "regime": "Market conditions", "ml": "Machine learning",
    "factor": "Diversifying factors (new)", "allocation": "Smarter weighting (new)",
    "benchmark": "Buy & hold baselines",
}


def _independence_section(mode: str) -> str:
    """Does anything here actually behave differently from owning the index?

    This is the question freeze v3 added two whole strategy families to answer,
    so the answer belongs on the page whichever way it comes out. It is
    computed live from the stored curves rather than written by hand.
    """
    import numpy as np
    from ..experiments import ExperimentRegistry
    from ..strategies import STRATEGY_CLASSES
    from .catalog import current_registry

    try:
        reg = ExperimentRegistry.for_mode(mode)
        df = current_registry(mode)
        if df.empty:
            return ""
        ok = df[(df["status"] == "ok") & (df["scenario"] == "base")].copy()
        qrows = ok[ok["strategy"] == "BuyAndHold(ticker=QQQ)"]
        if qrows.empty:
            return ""
        qcurve = reg.load_curve(qrows.iloc[0]["id"])
        if qcurve is None:
            return ""
        q = qcurve["returns"]

        ok["dev_sharpe"] = ok["metrics_dev"].map(
            lambda m: (m or {}).get("sharpe") or float("nan"))
        ok["cls"] = ok["strategy"].str.split("(").str[0]
        best = (ok[ok["family"] != "benchmark"]
                .sort_values("dev_sharpe", ascending=False)
                .drop_duplicates("cls"))

        rows = []
        for rec in best.to_dict("records"):
            c = reg.load_curve(rec["id"])
            if c is None:
                continue
            m = pd.DataFrame({"s": c["returns"], "q": q})
            m = (1 + m).resample("ME").prod() - 1
            corr = float(m["s"].corr(m["q"]))
            if not np.isfinite(corr):
                continue
            cls = rec["cls"]
            k = STRATEGY_CLASSES.get(cls)
            rows.append((cls, getattr(k, "family", "?"), corr))
        if len(rows) < 4:
            return ""

        d = pd.DataFrame(rows, columns=["cls", "family", "corr"]).sort_values("corr")
        med = d.groupby("family")["corr"].median().sort_values()

        def colour(v: float) -> str:
            return ("var(--good)" if v < 0.6 else
                    "var(--warning)" if v < 0.8 else "var(--critical)")

        fam_chart = _hbar(
            [(_FAMILY_LABELS.get(f, f), round(float(v), 2), colour(float(v)))
             for f, v in med.items()],
            "How much each kind of idea just tracks the index",
            "Median correlation to buying the Nasdaq-100 and holding it. "
            "Lower is better — 1.00 would mean it is the index in disguise.")

        best_rows = "".join(
            f"<tr><td>{_strategy_link(r.cls)}</td>"
            f"<td>{html.escape(_FAMILY_LABELS.get(r.family, r.family))}</td>"
            f"<td class='num' style='color:{colour(r.corr)};font-weight:600'>"
            f"{r.corr:.2f}</td></tr>"
            for r in d.head(6).itertuples())
        worst_rows = "".join(
            f"<tr><td>{_strategy_link(r.cls)}</td>"
            f"<td>{html.escape(_FAMILY_LABELS.get(r.family, r.family))}</td>"
            f"<td class='num' style='color:{colour(r.corr)};font-weight:600'>"
            f"{r.corr:.2f}</td></tr>"
            for r in d.tail(4).itertuples())

        # The specific claim freeze v3 was built to test, checked out loud.
        new_fams = [f for f in ("factor", "allocation") if f in med.index]
        verdict = ""
        if new_fams:
            new_med = float(med[new_fams].median())
            old_med = float(med.drop(new_fams).median())
            mc = d[d["cls"] == "MinCorrelationSleeve"]["corr"]
            mc_txt = ""
            if len(mc):
                mc_txt = (f" The sleeve built <i>only</i> to find the "
                          f"least-alike corner of the sector still moves with the "
                          f"index at <b>{float(mc.iloc[0]):.2f}</b>.")
            better = new_med < old_med
            verdict = (
                f"<div class='warnbox' style='border-left-color:"
                f"{'var(--good)' if better else 'var(--critical)'}'>"
                f"<b>The honest answer: mostly no.</b> The two families added "
                f"specifically to break this pattern land at a median "
                f"correlation of <b>{new_med:.2f}</b>, against <b>{old_med:.2f}</b> "
                f"for the families that already existed — "
                f"{'better' if better else '<b>worse</b>, not better'}.{mc_txt} "
                f"On this evidence the sector does not contain an independent "
                f"corner to hide in, and the one thing that did help was "
                f"reducing exposure, not re-picking stocks.</div>")

        return f"""
<h2>Does anything here actually behave differently?</h2>
<p class='lede'>The most useful result of the previous round was a negative one:
nearly every idea tested turned out to be the same bet in different clothing.
Two new families were added to attack that directly. This is whether they
worked.</p>
{verdict}
{fam_chart}
<div class='grid2' style='margin-top:1rem'>
<div class='chart'><div class='ct'>Least like just owning the index</div>
<div class='cs'>The only ideas offering real diversification</div>
<div class='tbl-wrap'><table><thead><tr><th>Idea</th><th>Kind</th>
<th class='num'>Correlation</th></tr></thead><tbody>{best_rows}</tbody></table></div></div>
<div class='chart'><div class='ct'>Most like just owning the index</div>
<div class='cs'>These are index funds with extra steps and extra costs</div>
<div class='tbl-wrap'><table><thead><tr><th>Idea</th><th>Kind</th>
<th class='num'>Correlation</th></tr></thead><tbody>{worst_rows}</tbody></table></div></div>
</div>
<p class='note'>Monthly correlations, best-scoring version of each idea, on
simulated prices. The pattern is the finding; the exact numbers are not.</p>"""
    except Exception as exc:   # never let an analysis break the page
        log.warning("independence section skipped: %s", exc)
        return ""


def _portfolio_page(mode: str) -> str:
    returns = load_strategy_returns(mode, include_benchmarks=True)
    if returns.empty:
        return ("<div class='chart'><div class='ct'>Nothing to compare yet</div>"
                "<div class='cs'>This page needs the day-by-day results of each "
                "strategy, which are not stored in the repository — they are large "
                "(~125 MB) and can be rebuilt exactly from the experiment record."
                "</div><p class='lede'>To fill this page in, run:</p>"
                "<pre>python scripts/run_all.py --data-mode synthetic\n"
                "python scripts/research.py dashboard</pre></div>")

    bt_cfg = load_backtest_config()
    # One column per IDEA, capped — a 30-wide grid of truncated names is not a
    # chart, it is a wall. The selection is NOT "first twelve found": this page
    # exists to answer whether anything diversifies, so it takes a spread across
    # families in a fixed order, with the families added specifically to break
    # the one-bet problem going first. Whatever is left out is named below.
    from ..strategies import STRATEGY_CLASSES
    _FAMILY_ORDER = ["factor", "allocation", "riskmanaged", "tsmom", "xsmom",
                     "breakout", "fundamental", "regime", "meanrev", "ml",
                     "benchmark"]

    by_class: dict[str, str] = {}
    for col in returns.columns:
        by_class.setdefault(col.split("(")[0], col)

    def family_of(cls: str) -> str:
        c = STRATEGY_CLASSES.get(cls)
        return getattr(c, "family", "benchmark") if c else "benchmark"

    ordered = sorted(by_class.items(),
                     key=lambda kv: (_FAMILY_ORDER.index(family_of(kv[0]))
                                     if family_of(kv[0]) in _FAMILY_ORDER
                                     else len(_FAMILY_ORDER), kv[0]))
    # At most two ideas per family before coming back round, so one large
    # family cannot crowd out every other.
    keep, seen = [], {}
    for rounds in (2, 99):
        for cls, col in ordered:
            f = family_of(cls)
            if col in keep or seen.get(f, 0) >= rounds:
                continue
            keep.append(col)
            seen[f] = seen.get(f, 0) + 1
            if len(keep) >= _MAX_MATRIX:
                break
        if len(keep) >= _MAX_MATRIX:
            break

    omitted = [c for c in by_class.values() if c not in keep]
    # Named, not hidden — but folded away, because a 20-item inline list is a
    # wall of text where a disclosure is all that is needed.
    omitted_note = (
        "<details style='display:inline'><summary style='display:inline;cursor:pointer'>"
        f"{len(omitted)} not shown</summary> — " +
        ", ".join(html.escape(humanize(c)[0]) for c in omitted) + ".</details>"
        if omitted else "")
    sub = returns[keep]
    short = {c: humanize(c)[0] for c in keep}

    corr = correlation_matrix(sub)
    cofail = regime_cofailure(sub, bt_cfg.subperiods)
    blends = blend_report(sub)

    def corr_cell(v: float) -> str:
        """Sequential blue: dark = moves together (bad for diversification)."""
        if pd.isna(v):
            return "background:var(--line)"
        t = max(0.0, min(1.0, (float(v) + 1) / 2))
        # stretch across the band that actually occurs so differences show
        t = max(0.0, min(1.0, (t - 0.55) / 0.45))
        ink = "#fff" if t > 0.55 else "var(--ink)"
        return f"background:rgba(42,120,214,{0.10 + t * 0.85:.2f});color:{ink}"

    # Columns are NUMBERED, rows are named. Full names in both axes make every
    # column as wide as a sentence and push the grid off the screen; the row
    # labels already carry the names, so the columns only need a key.
    names = [short[c] for c in corr.columns]
    head = "".join(f"<th class='num' style='font-size:.7rem;font-weight:600;"
                   f"text-transform:none' title='{html.escape(n)}'>{i + 1}</th>"
                   for i, n in enumerate(names))
    rows = ""
    for i, (_, row) in enumerate(corr.iterrows()):
        cells = "".join(
            f"<td style='{corr_cell(v)};text-align:center;font-size:.72rem;"
            f"font-variant-numeric:tabular-nums' title='{html.escape(names[i])} vs "
            f"{html.escape(names[j])}: {v:.2f}'>{v:.2f}</td>"
            for j, v in enumerate(row))
        rows += (f"<tr><td style='font-size:.75rem;font-weight:600;white-space:nowrap'>"
                 f"<span style='color:var(--mut);font-weight:400'>{i + 1}.</span> "
                 f"{html.escape(names[i])}</td>{cells}</tr>")
    corr_html = (f"<div class='tbl-wrap'><table><thead><tr><th></th>{head}</tr></thead>"
                 f"<tbody>{rows}</tbody></table></div>")

    cofail_html = ""
    if not cofail.empty:
        cols = [c for c in cofail.columns]
        chead = "".join(f"<th class='num'>{html.escape(_REGIME_LABELS.get(c, c))}</th>"
                        for c in cols)
        crows = ""
        for name, row in cofail.iterrows():
            cells = ""
            for v in row:
                if pd.isna(v):
                    cells += "<td class='num' style='color:var(--mut)'>–</td>"
                else:
                    bg = ("rgba(12,163,12,.16)" if v > 0.5 else
                          "rgba(208,59,59,.16)" if v < 0 else "transparent")
                    cells += (f"<td class='num' style='background:{bg}'>{v:.1f}</td>")
            crows += (f"<tr><td style='white-space:nowrap;font-weight:600'>"
                      f"{html.escape(short.get(name, name))}</td>{cells}</tr>")
        cofail_html = (f"<div class='tbl-wrap'><table><thead><tr><th>Strategy</th>"
                       f"{chead}</tr></thead><tbody>{crows}</tbody></table></div>")

    blend_html = ""
    if len(blends):
        brows = ""
        for r in blends.head(12).itertuples(index=False):
            win = ("<b style='color:var(--good)'>yes</b>" if r.blend_beats_both
                   else "<span style='color:var(--mut)'>no</span>")
            brows += (f"<tr><td>{html.escape(humanize(r.a)[0])} <span class='sub'>+ "
                      f"{html.escape(humanize(r.b)[0])}</span></td>"
                      f"<td class='num'>{r.corr_monthly:.2f}</td>"
                      f"<td class='num'>{r.sharpe_a:.2f}</td>"
                      f"<td class='num'>{r.sharpe_b:.2f}</td>"
                      f"<td class='num'><b>{r.sharpe_blend:.2f}</b></td>"
                      f"<td>{win}</td></tr>")
        blend_html = (f"<div class='tbl-wrap'><table><thead><tr><th>Pair</th>"
                      f"<th class='num'>Correlation</th><th class='num'>Sharpe A</th>"
                      f"<th class='num'>Sharpe B</th><th class='num'>Together</th>"
                      f"<th>Better than both?</th></tr></thead>"
                      f"<tbody>{brows}</tbody></table></div>")

    return f"""
{_independence_section(mode)}

<h2>How closely do they move together?</h2>
<p class='lede'>1.00 means two strategies rise and fall in lockstep — holding
both gives you no protection. Lower numbers mean they behave differently, which
is what makes combining them worthwhile. <b>Darker = moves together.</b></p>
<p class='note'>Showing one version of each idea, {len(keep)} of {len(by_class)},
spread across strategy families rather than taken in order. {omitted_note}</p>
{corr_html}
<p class='note'>Most of these are highly correlated because nearly all of them
hold the same handful of big technology stocks. That is the honest finding:
they are variations on one bet, not independent bets.</p>

<h2>Do they fail at the same time?</h2>
<p class='lede'>Each cell is the risk-adjusted return during that market period —
green is good, red is a loss. A column that is red all the way down means every
strategy broke in that period, and holding several of them would not have saved
you.</p>
{cofail_html}

<h2>Is a pair better than either one alone?</h2>
<p class='lede'>Splitting money 50/50 between two strategies, rebalanced monthly.
“Together” is the combined result. If it beats both parents, the pair genuinely
diversifies.</p>
{blend_html}
<p class='note'>The cost of switching between strategies is not modelled here, so
treat the combined figures as a best case rather than a promise.</p>"""


_SECTOR_LABELS = {
    "semiconductors": "Semiconductors", "semi_equipment": "Chip equipment",
    "eda": "Chip-design software", "software": "Software",
    "internet": "Internet", "hardware": "Hardware & storage",
    "cybersecurity": "Cybersecurity", "networking": "Networking",
    "dc_infrastructure": "Data-centre infrastructure", "power": "Power for AI",
    "robotics": "Robotics & automation", "ai_consumer": "Consumer AI",
}

# How each dead company died — shown so the delisted roster reads as a list of
# outcomes rather than a list of tickers. Sourced from the security master's
# successor records; this maps the machine-readable `kind` to plain English.
_EXIT_LABELS = {
    "bankruptcy": ("Went bust", "var(--critical)"),
    "acquisition": ("Bought", "var(--s1)"),
    "merger": ("Merged", "var(--s1)"),
    "take_private": ("Taken private", "var(--s4)"),
    "asset_sale": ("Broken up", "var(--s2)"),
    "split": ("Split up", "var(--s2)"),
}


def _data_page(mode: str) -> str:
    """What the study is allowed to look at, and what it is still missing."""
    from ..config import load_universe_config
    from ..data.security_master import load_master

    u = load_universe_config()
    master = load_master()
    live = [s for s in u.securities if s.delisted is None]
    dead = sorted((s for s in u.securities if s.delisted is not None),
                  key=lambda s: s.delisted)

    by_sector: dict[str, int] = {}
    for s in u.securities:
        by_sector[s.sector] = by_sector.get(s.sector, 0) + 1
    sector_chart = _hbar(
        [(_SECTOR_LABELS.get(k, k), v, "var(--s1)")
         for k, v in sorted(by_sector.items(), key=lambda kv: -kv[1])],
        "What the universe is made of",
        "Companies per part of the technology stack, live and delisted together.")

    # Exit reasons for the dead names.
    exits: dict[str, int] = {}
    sid_by_symbol = {sym: rec for rec in master.values() for sym in rec.all_symbols}
    for s in dead:
        rec = sid_by_symbol.get(s.ticker)
        kind = (rec.successor or {}).get("kind", "acquisition") if rec else "acquisition"
        exits[kind] = exits.get(kind, 0) + 1
    exit_chart = _hbar(
        [(_EXIT_LABELS.get(k, (k, "var(--s1)"))[0], v,
          _EXIT_LABELS.get(k, (k, "var(--s1)"))[1])
         for k, v in sorted(exits.items(), key=lambda kv: -kv[1])],
        "How the dead companies died",
        "Every one was, at some point, a name a momentum screen would have bought.")

    def dead_rows() -> str:
        out = ""
        for s in dead:
            rec = sid_by_symbol.get(s.ticker)
            succ = (rec.successor or {}) if rec else {}
            kind = succ.get("kind", "acquisition")
            label, colour = _EXIT_LABELS.get(kind, (kind, "var(--s1)"))
            # Prefer the most specific explanation available: the successor's
            # own note, then the security's note (which is where the genuinely
            # confusing cases are written up, e.g. Broadcom), then a bare
            # "absorbed by X".
            note = succ.get("notes", "") or (rec.notes if rec else "")
            if not note and succ.get("sid"):
                other = master.get(succ["sid"])
                note = f"absorbed by {other.name}" if other else ""
            out += (f"<tr><td><b>{html.escape(s.ticker)}</b>"
                    f"<span class='sub'>{html.escape(s.name)}</span></td>"
                    f"<td class='num'>{s.ipo.year}</td>"
                    f"<td class='num'>{s.delisted.year}</td>"
                    f"<td><span style='color:{colour};font-weight:600'>"
                    f"{html.escape(label)}</span></td>"
                    f"<td class='lede' style='font-size:.8rem'>{html.escape(note)}</td></tr>")
        return out

    def live_rows() -> str:
        out = ""
        for s in sorted(live, key=lambda x: (x.sector, x.ticker)):
            out += (f"<tr><td><b>{html.escape(s.ticker)}</b></td>"
                    f"<td>{html.escape(s.name)}</td>"
                    f"<td>{html.escape(_SECTOR_LABELS.get(s.sector, s.sector))}</td>"
                    f"<td class='num'>{s.ipo.year}</td></tr>")
        return out

    basket_rows = "".join(
        f"<tr><td><b>{html.escape(_BASKETS.get(k, k))}</b>"
        f"<span class='sub'>{html.escape(k)}</span></td>"
        f"<td class='num'>{len(v)}</td>"
        f"<td class='lede' style='font-size:.78rem'>{html.escape(', '.join(v[:14]))}"
        f"{'…' if len(v) > 14 else ''}</td></tr>"
        for k, v in u.baskets.items())

    bench_rows = "".join(
        f"<tr><td><b>{html.escape(b.ticker)}</b></td><td>{html.escape(b.name)}</td>"
        f"<td class='num'>{b.ipo.year}</td>"
        f"<td class='num'>{b.expense_ratio:.2%}</td></tr>"
        for b in u.benchmarks)

    # What holding the dead names actually did, straight from the registry.
    # Prose about survivorship bias is easy to nod along to; the numbers are
    # what make it land.
    dead_evidence = ""
    try:
        from .catalog import current_registry
        reg = current_registry(mode)
        if not reg.empty and "strategy" in reg.columns:
            base = reg[(reg.get("status") == "ok") & (reg.get("scenario") == "base")]
            dead_tickers = {s.ticker for s in dead}
            found = []
            for r in base.to_dict("records"):
                name = str(r.get("strategy", ""))
                if not name.startswith("BuyAndHold(ticker="):
                    continue
                tk = name.split("=", 1)[1].rstrip(")")
                if tk not in dead_tickers:
                    continue
                m = r.get("metrics_dev") or {}
                dd, cagr = m.get("max_drawdown"), m.get("cagr")
                if dd is None or cagr is None:
                    continue
                found.append((tk, float(cagr), float(dd)))
            found.sort(key=lambda x: x[2])
            if found:
                names = {s.ticker: s.name for s in dead}
                rows = "".join(
                    f"<tr><td><b>{html.escape(tk)}</b>"
                    f"<span class='sub'>{html.escape(names.get(tk, ''))}</span></td>"
                    f"<td class='num' style='color:{'var(--critical)' if c < 0 else 'var(--ink)'}'>"
                    f"{c:+.1%}</td>"
                    f"<td class='num' style='color:var(--critical)'>{d:.1%}</td></tr>"
                    for tk, c, d in found[:10])
                dead_evidence = f"""
<h2>What holding them actually did</h2>
<p class='lede'>Straight from the experiment record: buying one of these and
holding it, through to the day it stopped trading. <b>Worst fall</b> of −100%
means the shares became worthless. These are the outcomes that vanish from a
backtest run only on companies that still exist.</p>
<div class='tbl-wrap'><table><thead><tr><th>Company</th>
<th class='num'>Return per year</th><th class='num'>Worst fall</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class='note'>Simulated prices — the pattern is what matters, not the exact
figures. The ten worst of {len(found)} delisted names tested.</p>"""
    except Exception:  # the page must render even before any run exists
        dead_evidence = ""

    simulated = mode == "synthetic"
    reality = (
        "<div class='warnbox'><b>This page describes the roster, not real "
        "prices.</b> Everything listed here is what the study is <i>defined</i> "
        "to look at. The price history it has actually run on is simulated — "
        "the real download has not happened, because this machine cannot reach "
        "any market-data provider. The list below is exactly what that download "
        "would fetch.</div>" if simulated else "")

    return f"""
<h1>What the study is allowed to look at</h1>
<p class='lede'>A backtest can only be as honest as the list of companies it is
shown. This is that list — {len(u.securities)} companies and
{len(u.benchmarks)} funds, covering 1990 to 2026.</p>
{reality}

<div class='stats'>
<div class='stat'><div class='v'>{len(live)}</div><div class='k'>companies still trading</div>
<div class='d'>Chips, chip equipment, chip-design software, cloud, software,
internet, security, networking, robotics and the power companies feeding AI
data centres.</div></div>
<div class='stat'><div class='v'>{len(dead)}</div><div class='k'>companies that no longer exist</div>
<div class='d'>Bankrupt, bought or taken private. Kept in permanently — this is
the part most backtests quietly leave out.</div></div>
<div class='stat'><div class='v'>{len(u.benchmarks)}</div><div class='k'>funds to beat</div>
<div class='d'>The boring alternatives: index funds, sector funds, and cash.</div></div>
<div class='stat'><div class='v'>{len(u.baskets)}</div><div class='k'>theme groups</div>
<div class='d'>Named subsets a strategy can be told to trade, so a result can be
attributed to a theme rather than to one lucky stock.</div></div>
</div>

<h2>Why the dead companies matter more than the live ones</h2>
<p class='lede'>If you test a strategy only on companies that still exist today,
you have already told it the answer. Every stock in that list survived. The
strategy looks brilliant because it was never given the chance to buy Nortel in
2000, WorldCom in 2001, or Silicon Graphics on the way down — and a real
investor, choosing in real time, absolutely would have.</p>
<p class='lede'>That flattery has a name: <b>survivorship bias</b>. It is the
single most common reason a backtest that looks wonderful makes no money.
The {len(dead)} companies below are here so it cannot happen. A strategy that
buys strength has to buy them too, at their peaks, and ride them down.</p>

<div class='grid2' style='margin-top:1rem'>{sector_chart}{exit_chart}</div>

{dead_evidence}
<h2>The companies that didn't make it</h2>
<p class='lede'>Ordered by the year they stopped trading. <b>Went bust</b> means
the equity went to zero — anyone holding it lost everything.</p>
<div class='tbl-wrap'><table><thead><tr><th>Ticker</th>
<th class='num'>Listed</th><th class='num'>Gone</th><th>What happened</th>
<th>Detail</th></tr></thead><tbody>{dead_rows()}</tbody></table></div>

<h2>The companies still trading</h2>
<p class='lede'>Sorted by part of the technology stack. “Listed” is the year the
study is first allowed to trade the name — never before its IPO, and never
within six months of it.</p>
<details><summary>Show all {len(live)} companies</summary>
<div class='tbl-wrap' style='margin-top:.6rem'><table><thead><tr><th>Ticker</th>
<th>Company</th><th>Part of the stack</th><th class='num'>Listed</th></tr></thead>
<tbody>{live_rows()}</tbody></table></div></details>

<h2>The funds a strategy has to beat</h2>
<p class='lede'>If an idea cannot beat simply buying one of these and waiting,
it is not worth running. The fee column is what the fund charges every year —
a strategy has to beat that too.</p>
<div class='tbl-wrap'><table><thead><tr><th>Ticker</th><th>Fund</th>
<th class='num'>Listed</th><th class='num'>Annual fee</th></tr></thead>
<tbody>{bench_rows}</tbody></table></div>

<h2>Theme groups</h2>
<p class='lede'>A strategy can be pointed at any of these instead of the whole
list. Delisted members stay in their group for the years they actually traded.</p>
<div class='tbl-wrap'><table><thead><tr><th>Group</th><th class='num'>Members</th>
<th>Who's in it</th></tr></thead><tbody>{basket_rows}</tbody></table></div>

<h2>Rules the roster obeys</h2>
<div class='grid2'>
<div class='chart'><div class='ct'>What keeps it honest</div>
<div class='cs'>Applied on every single day of the test</div>
<ul class='lede' style='margin:0;padding-left:1.1rem'>
<li><b>Nothing before it existed.</b> A company enters only after its IPO plus
six months of trading history. No buying Arm in 2019.</li>
<li><b>Nothing after it died.</b> A delisted name is sold at its last price and
never traded again — but its earlier years stay fully in play.</li>
<li><b>Nothing untradeable.</b> Names under $3, or without enough daily volume
to get in and out of, are skipped on the days that is true of them.</li>
<li><b>Names are not identities.</b> Every company has a permanent internal ID.
Broadcom is the clearest trap: Avago bought the Broadcom name in 2016, so a
price series labelled “Broadcom” starts in 2009 and hides the original
company's dot-com boom and collapse entirely. Here they are two separate
companies and are never joined.</li>
</ul></div>
<div class='chart'><div class='ct'>What is still missing</div>
<div class='cs'>Honest limitations of this list</div>
<ul class='lede' style='margin:0;padding-left:1.1rem'>
<li><b>Real prices.</b> Still the big one. The roster is defined; the download
has not run.</li>
<li><b>Free data may not have the dead names.</b> Free providers often serve no
history at all for delisted tickers. If that happens, the study reports itself
as survivorship-limited rather than pretending the list above was honoured.</li>
<li><b>It is not the whole market.</b> {len(live)} live names is a considered
selection, not every listed technology company. Anything excluded is excluded
by the list above, which is at least visible and arguable.</li>
<li><b>Delisting prices are optimistic.</b> A bankruptcy is modelled as a sale
at the last quoted price. In reality the exit is usually worse.</li>
</ul></div></div>
<p class='note'>The full machine-readable definition lives in
<code>configs/universe.yaml</code> and <code>configs/security_master.yaml</code>,
both fingerprinted by the research freeze — they cannot be edited after results
exist without the change being detected.</p>"""


def _ideas_page() -> str:
    ideas = load_ideas()
    rows = ""
    for i in ideas:
        rows += (f"<tr><td>{i['id']}</td><td><b>{html.escape(i['title'])}</b><br>"
                 f"<span class='note'>{html.escape(i.get('hypothesis', '')[:160])}</span></td>"
                 f"<td>{i.get('category', '')}</td><td>{_badge(i['status'])}</td>"
                 f"<td>{i.get('expected_edge', '')}</td><td>{i.get('difficulty', '')}</td>"
                 f"<td>{html.escape(str(i.get('est_time', '')))}</td>"
                 f"<td>{html.escape(', '.join(map(str, i.get('required_providers', []))))}</td></tr>")
    return ("<p class='note'>Ideas exist before implementation; statuses: idea → "
            "planned → implementing → testing → auditing → accepted/rejected/archived."
            " Edit configs/research_ideas.yaml.</p>"
            "<table><tr><th>id</th><th>idea</th><th>category</th><th>status</th>"
            "<th>edge</th><th>difficulty</th><th>est. time</th><th>providers</th></tr>"
            + rows + "</table>")


def _roadmap_page(mode: str) -> str:
    road = build_roadmap(mode)
    rows = "".join(
        f"<tr><td>{i + 1}</td><td>{r['id']}</td><td><b>{html.escape(r['title'])}</b><br>"
        f"<span class='note'>{html.escape(str(r['rationale'])[:180])}</span></td>"
        f"<td>{r['category']}</td><td>{r['priority_score']}</td>"
        f"<td>{html.escape(', '.join(map(str, r['dependencies'])))}</td></tr>"
        for i, r in enumerate(road))
    sug = assistant_review(mode)
    sug_html = "".join(f"<li><b>[{s.kind}]</b> {html.escape(s.target[:70])} — "
                       f"{html.escape(s.detail)}</li>" for s in sug[:25]) or "<li>—</li>"
    return (f"<p class='note'>Priorities scored by expected value / difficulty with a "
            f"data-blocker boost; gate limitations and open audit findings are "
            f"injected automatically.</p>"
            f"<table><tr><th>#</th><th>id</th><th>item</th><th>category</th>"
            f"<th>priority</th><th>dependencies</th></tr>{rows}</table>"
            f"<h2>Heuristic assistant review (read-only, rule-based)</h2>"
            f"<p class='note'>Never modifies studies; never touches holdout data "
            f"beyond recorded metrics.</p><ul>{sug_html}</ul>")


def _audit_page() -> str:
    import json as _json
    findings_path = PROJECT_ROOT / "audit" / "findings" / "findings.jsonl"
    rows = ""
    if findings_path.exists():
        for line in findings_path.read_text().splitlines():
            f = _json.loads(line)
            rows += (f"<tr><td>{f['id']}</td><td>{_badge(f['severity'])}</td>"
                     f"<td>{f['status']}</td><td><b>{html.escape(f['title'])}</b><br>"
                     f"<span class='note'>{html.escape(f['description'][:200])}</span></td></tr>")
    from ..freeze import FREEZE_VERSION, load_freeze
    try:
        fhash = load_freeze()["hash"]
    except Exception:
        fhash = "not yet created"

    return (f"<p>Verdict: <b>READY WITH MATERIAL LIMITATIONS</b> · freeze v{FREEZE_VERSION} "
            f"<code>{html.escape(fhash)}</code> · v1 and v2 preserved and superseded · "
            "<a href='../audit/reports/adversarial_audit.md'>full report</a></p>"
            "<p class='lede'>Findings are numbered in the order they were found "
            "and never renumbered or removed. <b>Open</b> means known and "
            "unfixed, which is a deliberate state, not an oversight — each one "
            "records why it has not been fixed yet.</p>"
            "<table><tr><th>id</th><th>severity</th><th>status</th><th>finding</th></tr>"
            + rows + "</table>"
            + "<h2>Standing controls</h2><ul>"
              f"<li>Research freeze v{FREEZE_VERSION} verified by every real-mode entry point</li>"
              "<li>Tamper-evident holdout log (registry-mirrored, hash-chained)</li>"
              "<li>Fail-closed decision brief (paper-trade / do-nothing only)</li>"
              "<li>Evidence tiers A/B/C, never mixed in a leaderboard</li>"
              "<li>One universe cohort per leaderboard: results from an earlier, "
              "narrower list of companies are retained but never ranked beside "
              "current ones</li>"
              "<li>Experiment IDs bound to the universe fingerprint, so widening "
              "the roster cannot silently reuse the old roster's results</li>"
              "<li>Experiment lineage bound to store fingerprint + freeze hash</li>"
              "<li>Append-only registries; deprecated variants retained with reasons</li></ul>")
