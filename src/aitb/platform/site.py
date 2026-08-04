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
from .catalog import build_catalog, platform_stats
from .docgen import research_notebook, strategy_doc
from .portfolio_lab import (blend_report, correlation_matrix,
                            load_strategy_returns, regime_cofailure)
from .research_mgmt import assistant_review, build_roadmap, load_ideas
from .scorecard import build_scorecard
from .tradingview import export_all, export_pine

log = get_logger("platform.site")

SITE_DIR = PROJECT_ROOT / "site"

_CSS = """
:root{--bg:#f8fafc;--card:#fff;--ink:#0f172a;--mut:#64748b;--acc:#2563eb;
--good:#166534;--warn:#92400e;--bad:#991b1b;--line:#e2e8f0}
*{box-sizing:border-box}body{font-family:-apple-system,Segoe UI,Helvetica,sans-serif;
margin:0;background:var(--bg);color:var(--ink);line-height:1.5}
nav{background:#0f172a;color:#e2e8f0;padding:.7rem 1.2rem;display:flex;gap:1.1rem;
flex-wrap:wrap;position:sticky;top:0;z-index:5}
nav a{color:#cbd5e1;text-decoration:none;font-size:.92rem}nav a:hover{color:#fff}
nav .brand{color:#fff;font-weight:700}
main{max-width:1180px;margin:1.4rem auto;padding:0 1.2rem}
h1{font-size:1.5rem;margin:.4rem 0 1rem}h2{font-size:1.15rem;margin-top:1.8rem;
border-bottom:1px solid var(--line);padding-bottom:.25rem}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:.7rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:.7rem .9rem}.card .v{font-size:1.45rem;font-weight:700}
.card .k{color:var(--mut);font-size:.78rem}
table{border-collapse:collapse;width:100%;font-size:.82rem;background:var(--card)}
th,td{border:1px solid var(--line);padding:.3rem .5rem;text-align:left}
th{background:#f1f5f9;cursor:pointer;position:sticky;top:46px}
tr:hover{background:#f8fafc}
.badge{display:inline-block;padding:.05rem .45rem;border-radius:99px;font-size:.72rem}
.b-core{background:#dcfce7;color:var(--good)}.b-exploratory{background:#fef9c3;color:var(--warn)}
.b-deprecated{background:#fee2e2;color:var(--bad)}.b-benchmark{background:#e0e7ff;color:#3730a3}
.b-A{background:#dcfce7;color:var(--good)}.b-B{background:#fef9c3;color:var(--warn)}
.b-C{background:#fee2e2;color:var(--bad)}.b-unlisted{background:#e2e8f0;color:#475569}
.warnbox{background:#fef3c7;border-left:4px solid #f59e0b;padding:.6rem .9rem;margin:.8rem 0;font-size:.88rem}
.note{color:var(--mut);font-size:.82rem}
input,select{padding:.3rem .45rem;border:1px solid var(--line);border-radius:6px;margin:.15rem}
pre{background:#0f172a;color:#e2e8f0;padding:.9rem;border-radius:8px;overflow-x:auto;font-size:.78rem}
img{max-width:100%;border:1px solid var(--line);border-radius:8px;margin:.4rem 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:800px){.grid2{grid-template-columns:1fr}}
.score{display:flex;gap:.4rem;align-items:center;margin:.2rem 0;font-size:.85rem}
.bar{height:8px;border-radius:4px;background:var(--acc);min-width:2px}
details{margin:.3rem 0}summary{cursor:pointer;color:var(--acc)}
.tree{border-left:3px solid var(--line);margin-left:.5rem;padding-left:1rem}
.tree .node{margin:.6rem 0;background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:.5rem .8rem;font-size:.85rem}
"""

_NAV = """<nav><span class="brand">AITB Research</span>
<a href="{p}index.html">Dashboard</a><a href="{p}experiments.html">Experiments</a>
<a href="{p}strategies.html">Strategies</a><a href="{p}compare.html">Compare</a>
<a href="{p}portfolio.html">Portfolio lab</a><a href="{p}ideas.html">Ideas</a>
<a href="{p}roadmap.html">Roadmap</a><a href="{p}audit.html">Audit</a></nav>"""


def _page(title: str, body: str, depth: int = 0) -> str:
    prefix = "../" * depth
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
            f"<body>{_NAV.format(p=prefix)}<main><h1>{html.escape(title)}</h1>"
            f"{body}<p class='note'>Generated "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
            f"read-only view over the registries · research tool, not "
            f"investment advice.</p></main></body></html>")


def _badge(x: str) -> str:
    return f"<span class='badge b-{html.escape(str(x))}'>{html.escape(str(x))}</span>"


def _mode_banner(mode: str) -> str:
    if mode == "synthetic":
        return ("<div class='warnbox'><b>SYNTHETIC DATA MODE.</b> All numbers on "
                "this site are demonstrations of the machinery on generated data "
                "— not market history. The real-data study has not run yet.</div>")
    return ""


# ------------------------------------------------------------- dashboard ----
def _dashboard(mode, stats, cat, registry) -> str:
    cards = [
        ("Strategy classes", stats.get("total_strategies", 0)),
        ("Core", stats.get("core", 0)), ("Exploratory", stats.get("exploratory", 0)),
        ("Deprecated", stats.get("deprecated", 0)),
        ("Robust candidates", stats.get("robust_candidates", "—")),
        ("Rejected", stats.get("rejected", "—")),
        ("Experiments OK", stats.get("experiments_ok", 0)),
        ("Failed runs", stats.get("experiments_failed", 0)),
        ("Holdout events", stats.get("holdout_events", 0)),
        ("Last experiment", stats.get("last_experiment", "—")),
        ("Freeze", "v2"), ("Audit", "READY w/ LIMITATIONS"),
        ("Real study", "NOT RUN"), ("Data mode", mode),
    ]
    tiers = stats.get("tiers", {})
    for t in ("A", "B", "C"):
        if tiers:
            cards.append((f"Tier {t}", tiers.get(t, 0)))
    cards_html = "".join(f"<div class='card'><div class='v'>{v}</div>"
                         f"<div class='k'>{k}</div></div>" for k, v in cards)

    fams = pd.Series({e.class_name: e.family for e in cat.values()
                      if e.status != "unlisted"}).value_counts()
    fam_rows = "".join(f"<tr><td>{f}</td><td>{n}</td></tr>" for f, n in fams.items())

    rank_path = registry.root / "strategy_ranking.csv"
    leaders = ""
    if rank_path.exists():
        r = pd.read_csv(rank_path)
        bench = r[r["verdict"] == "benchmark"].head(5)
        top = r[r["verdict"] != "benchmark"].head(5)
        def rows(d):
            return "".join(
                f"<tr><td><a href='strategy/{html.escape(x.strategy.split('(')[0])}.html'>"
                f"{html.escape(x.strategy[:60])}</a></td><td>{x.score}</td>"
                f"<td>{_badge(x.verdict)}</td></tr>" for x in d.itertuples())
        leaders = (f"<h2>Benchmark leaders</h2><table><tr><th>benchmark</th>"
                   f"<th>score</th><th>verdict</th></tr>{rows(bench)}</table>"
                   f"<h2>Top active strategies</h2><table><tr><th>strategy</th>"
                   f"<th>score</th><th>verdict</th></tr>{rows(top)}</table>")

    timeline = ""
    by_day = stats.get("experiments_by_day", {})
    if by_day:
        mx = max(by_day.values())
        bars = "".join(
            f"<div title='{d}: {n}' style='display:inline-block;width:10px;"
            f"height:{6 + 54 * n / mx}px;background:var(--acc);margin-right:2px;"
            f"vertical-align:bottom'></div>" for d, n in by_day.items())
        timeline = f"<h2>Cumulative experiment timeline</h2><div>{bars}</div>" \
                   f"<p class='note'>experiments per day (hover for counts)</p>"

    road = build_roadmap(mode)[:5]
    road_html = "".join(f"<li><b>{html.escape(r['title'])}</b> "
                        f"<span class='note'>(priority {r['priority_score']}, "
                        f"{html.escape(r['category'])})</span></li>" for r in road)

    return (_mode_banner(mode) + f"<div class='cards'>{cards_html}</div>"
            + timeline
            + "<div class='grid2'><div><h2>Strategy families</h2><table>"
              f"<tr><th>family</th><th>classes</th></tr>{fam_rows}</table></div>"
              f"<div><h2>Current roadmap (top 5)</h2><ol>{road_html}</ol>"
              f"<p><a href='roadmap.html'>full roadmap →</a></p></div></div>"
            + leaders)


# ----------------------------------------------------- experiment explorer ---
def _dict(x) -> dict:
    """Registry fields loaded through pandas can be NaN — guard to dict."""
    return x if isinstance(x, dict) else {}


def _experiments_page(registry) -> str:
    df = registry.load()
    if df.empty:
        return "<p>No experiments recorded.</p>"
    rows = []
    for r in df.to_dict("records"):
        dev = _dict(r.get("metrics_dev"))
        hold = _dict(r.get("metrics_holdout"))
        rows.append({
            "id": r.get("id", ""), "strategy": r.get("strategy", ""),
            "family": r.get("family", ""), "status": r.get("status", ""),
            "scenario": r.get("scenario", ""), "data_mode": r.get("data_mode", ""),
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
Filter freely; click a row for the full record.</p>
<div>
<input id='q' placeholder='search text…' size='28'>
<select id='fFam'><option value=''>family</option></select>
<select id='fStatus'><option value=''>status</option></select>
<select id='fScen'><option value=''>scenario</option></select>
<select id='fMode'><option value=''>data mode</option></select>
</div>
<table id='tbl'><thead><tr>
<th data-k='date'>date</th><th data-k='strategy'>strategy</th>
<th data-k='family'>family</th><th data-k='status'>status</th>
<th data-k='scenario'>cost</th><th data-k='data_mode'>mode</th>
<th data-k='dev_sharpe'>dev Sharpe</th><th data-k='holdout_sharpe'>holdout Sharpe</th>
<th data-k='max_dd'>maxDD</th><th data-k='turnover'>turnover</th>
</tr></thead><tbody></tbody></table>
<pre id='detail' style='display:none'></pre>
<script>
const DATA = {data};
const tb = document.querySelector('#tbl tbody');
const fills = {{fFam:'family', fStatus:'status', fScen:'scenario', fMode:'data_mode'}};
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
  tb.innerHTML = rows.slice(0, 500).map((r, i) =>
    `<tr onclick='show(${{JSON.stringify(JSON.stringify(r))}})'>` +
    ['date','strategy','family','status','scenario','data_mode',
     'dev_sharpe','holdout_sharpe','max_dd','turnover']
      .map(k => `<td>${{r[k] ?? ''}}</td>`).join('') + '</tr>').join('');
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
def _scorebar(dim: str, d: dict) -> str:
    return (f"<div class='score'><div style='width:220px'>{html.escape(dim)}</div>"
            f"<div class='bar' style='width:{d['score'] * 28}px'></div>"
            f"<b>{d['score']}</b><span class='note'> — {html.escape(d['reason'])}"
            f"</span></div>")


def _strategy_page(e, mode: str) -> str:
    doc = strategy_doc(e)
    nb = research_notebook(e)
    sc = nb["scorecard"]
    gen = nb["future_ideas"]

    hist_rows = "".join(
        f"<tr><td>{h['date']}</td><td>{html.escape(str(h['strategy'])[:70])}</td>"
        f"<td>{h['data_mode']}</td><td>{h['dev_sharpe']}</td>"
        f"<td>{h['holdout_sharpe']}</td><td>{h['max_dd']}</td><td>{h['turnover']}</td></tr>"
        for h in nb["experiment_history"])

    tree = ""
    for line in gen:
        nodes = "".join(
            f"<div class='node'><b>{v['id']}</b> · freeze {v.get('freeze')} · "
            f"{html.escape(str(v.get('change', '')))}<br>"
            f"<span class='note'>why: {html.escape(str(v.get('why', '')))} → "
            f"{html.escape(str(v.get('outcome', '')))}"
            f"{' · required freeze bump' if v.get('requires_freeze_bump') else ''}"
            f"</span></div>"
            for v in line["versions"])
        tree += f"<h3>{html.escape(line['title'])}</h3><div class='tree'>{nodes}</div>"

    pine = export_pine(e.class_name)
    tv = (f"<p>{_badge('portable') if e.tradingview else _badge('not-portable')} "
          f"{html.escape(e.tradingview_note)}</p>")
    if pine:
        tv += (f"<p><a href='../tradingview/{e.class_name}.pine' download>"
               f"Download {e.class_name}.pine</a></p><details><summary>view "
               f"script</summary><pre>{html.escape(pine)}</pre></details>")

    body = f"""
{_mode_banner(mode)}
<p>{_badge(e.status)} {_badge(e.family)}
{_badge(e.tier) if e.tier else ''} · best verdict:
{_badge(e.best_verdict or 'n/a')} · scorecard <b>{sc['overall']}/5</b></p>
<p><i>{html.escape(e.hypothesis)}</i></p>

<h2>Scorecard (research quality, not performance)</h2>
{''.join(_scorebar(k, v) for k, v in sc['dimensions'].items())}
<p class='note'>{html.escape(sc['verdict'])} — {html.escape(sc['note'])}</p>

<h2>Documentation</h2>
<p>{html.escape(doc['description'])}</p>
<p class='note'>Markets: {doc['compatible_markets']} · timeframe:
{doc['intended_timeframe']} · audit: {doc['audit_status']}</p>
<div class='grid2'>
<div><h3>Strengths</h3><ul>{''.join(f'<li>{html.escape(s)}</li>' for s in doc['strengths'])}</ul></div>
<div><h3>Weaknesses</h3><ul>{''.join(f'<li>{html.escape(s)}</li>' for s in doc['weaknesses'])}</ul></div>
</div>
<h3>Assumptions</h3><ul>{''.join(f'<li>{html.escape(s)}</li>' for s in doc['assumptions'])}</ul>
<h3>Parameters</h3><table><tr><th>name</th><th>default</th></tr>
{''.join(f"<tr><td>{html.escape(p['name'])}</td><td>{html.escape(str(p['default']))}</td></tr>" for p in doc['parameters'])}</table>
<h3>Frozen grids</h3><pre>{html.escape(json.dumps(doc['grids'], indent=1))}</pre>

<h2>Research notebook</h2>
<p><b>Rationale:</b> {html.escape(nb['rationale'])}</p>
<h3>Experiment history ({len(nb['experiment_history'])})</h3>
<table><tr><th>date</th><th>variant</th><th>mode</th><th>dev Sharpe</th>
<th>holdout Sharpe</th><th>maxDD</th><th>turnover</th></tr>{hist_rows}</table>
<h3>Lessons learned</h3><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in nb['lessons_learned']) or '<li>—</li>'}</ul>
<h3>Remaining questions</h3><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in nb['remaining_questions'])}</ul>

<h2>Genealogy</h2>{tree or "<p class='note'>no recorded version history yet</p>"}

<h2>TradingView</h2>{tv}

<h2>Python implementation (freeze-v2 fingerprinted)</h2>
<details><summary>view source</summary><pre>{html.escape(doc['python_implementation'])}</pre></details>
"""
    return body


# ------------------------------------------------------------ compare page ---
def _compare_page(registry) -> str:
    df = registry.load()
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

    # dashboard
    (out / "index.html").write_text(
        _page("Research dashboard", _dashboard(mode, stats, cat, registry)))

    # experiments
    (out / "experiments.html").write_text(
        _page("Experiment explorer", _mode_banner(mode) + _experiments_page(registry)))

    # strategies index + pages
    rows = "".join(
        f"<tr><td><a href='strategy/{e.class_name}.html'>{e.class_name}</a></td>"
        f"<td>{e.family}</td><td>{_badge(e.status)}</td>"
        f"<td>{_badge(e.tier) if e.tier else '—'}</td>"
        f"<td>{e.n_experiments}</td>"
        f"<td>{_badge('TV') if e.tradingview else '—'}</td>"
        f"<td>{html.escape(e.hypothesis[:90])}</td></tr>"
        for e in cat.values() if e.status != "unlisted")
    (out / "strategies.html").write_text(_page(
        "Strategy catalog", _mode_banner(mode) +
        "<table><tr><th>class</th><th>family</th><th>status</th><th>tier</th>"
        "<th>experiments</th><th>TradingView</th><th>hypothesis</th></tr>"
        + rows + "</table>"))
    for e in cat.values():
        if e.status == "unlisted":
            continue
        (out / "strategy" / f"{e.class_name}.html").write_text(
            _page(e.class_name, _strategy_page(e, mode), depth=1))

    # compare
    (out / "compare.html").write_text(
        _page("Comparison engine", _mode_banner(mode) + _compare_page(registry)))

    # portfolio lab
    (out / "portfolio.html").write_text(
        _page("Portfolio laboratory", _mode_banner(mode) + _portfolio_page(mode)))

    # ideas + roadmap
    (out / "ideas.html").write_text(_page("Idea backlog", _ideas_page()))
    (out / "roadmap.html").write_text(_page("Research roadmap", _roadmap_page(mode)))

    # audit summary page
    (out / "audit.html").write_text(_page("Audit & governance", _audit_page()))

    # tradingview exports
    export_all(out / "tradingview")

    log.info("site built at %s (mode=%s)", out, mode)
    return out


def _portfolio_page(mode: str) -> str:
    returns = load_strategy_returns(mode, include_benchmarks=True)
    if returns.empty:
        return "<p>No curves available.</p>"
    corr = correlation_matrix(returns)
    bt_cfg = load_backtest_config()
    cofail = regime_cofailure(returns, bt_cfg.subperiods)
    blends = blend_report(returns)

    def heat(v):
        if pd.isna(v):
            return "#f1f5f9"
        t = max(-1, min(1, float(v)))
        return (f"rgba(37,99,235,{abs(t) * 0.75:.2f})" if t > 0
                else f"rgba(220,38,38,{abs(t) * 0.75:.2f})")

    names = [c[:26] for c in corr.columns]
    corr_html = "<table><tr><th></th>" + "".join(f"<th style='font-size:.62rem'>{html.escape(n)}</th>" for n in names) + "</tr>"
    for i, row in enumerate(corr.itertuples(index=False)):
        corr_html += f"<tr><td style='font-size:.62rem'><b>{html.escape(names[i])}</b></td>"
        corr_html += "".join(f"<td style='background:{heat(v)};text-align:center;"
                             f"font-size:.62rem'>{v:.2f}</td>" for v in row)
        corr_html += "</tr>"
    corr_html += "</table>"

    cofail_html = ""
    if not cofail.empty:
        cofail_html = "<table><tr><th>strategy</th>" + "".join(
            f"<th>{html.escape(c)}</th>" for c in cofail.columns) + "</tr>"
        for name, row in cofail.iterrows():
            cofail_html += (f"<tr><td style='font-size:.7rem'>{html.escape(name[:40])}</td>"
                            + "".join(f"<td style='background:{heat(v/2 if pd.notna(v) else v)};"
                                      f"text-align:center'>{'' if pd.isna(v) else f'{v:.1f}'}</td>"
                                      for v in row) + "</tr>")
        cofail_html += "</table><p class='note'>Sharpe per regime window — red columns shared across rows = co-failure.</p>"

    blend_html = ""
    if len(blends):
        blend_html = "<table><tr>" + "".join(f"<th>{c}</th>" for c in blends.columns) + "</tr>"
        for r in blends.itertuples(index=False):
            blend_html += "<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>"
        blend_html += ("</table><p class='note'>50/50 monthly-rebalanced blends of "
                       "net strategy returns. Switching costs between strategies "
                       "are NOT modeled — treat blends as upper bounds.</p>")

    return (f"<h2>Strategy correlation (monthly returns)</h2>{corr_html}"
            f"<h2>Regime co-failure map</h2>{cofail_html}"
            f"<h2>Pairwise blends: can two strategies beat both parents?</h2>{blend_html}")


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
    return ("<p>Verdict: <b>READY WITH MATERIAL LIMITATIONS</b> · freeze v2 "
            "<code>49767ea3efc44cead711d72946c3fe31</code> · v1 preserved/superseded · "
            "<a href='../audit/reports/adversarial_audit.md'>full report</a></p>"
            "<table><tr><th>id</th><th>severity</th><th>status</th><th>finding</th></tr>"
            + rows + "</table>"
            + "<h2>Standing controls</h2><ul>"
              "<li>Research freeze v2 verified by every real-mode entry point</li>"
              "<li>Tamper-evident holdout log (registry-mirrored, hash-chained)</li>"
              "<li>Fail-closed decision brief (paper-trade / do-nothing only)</li>"
              "<li>Evidence tiers A/B/C, never mixed in a leaderboard</li>"
              "<li>Experiment lineage bound to store fingerprint + freeze hash</li>"
              "<li>Append-only registries; deprecated variants retained with reasons</li></ul>")
