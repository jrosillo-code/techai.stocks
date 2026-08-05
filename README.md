# AI & Technology Strategy Research and Backtesting System

A modular Python research platform for systematically developing, testing and
ranking trading/investment strategies on the US technology, semiconductor,
cloud, software, cybersecurity, robotics and AI sectors — and for answering the
harder question underneath: **does any of this trading actually beat simply
owning the leading AI/technology companies?**

> **This is a research tool, not investment advice.** Historical results —
> and *a fortiori* synthetic results — do not predict future performance.

---

## Data modes: real vs synthetic (strictly separated)

The system runs in exactly one of two modes; they share code but nothing else:

| | `--data-mode real` | `--data-mode synthetic` |
|---|---|---|
| Data source | canonical validated store `data/real/` only | deterministic generator (or cached free providers) |
| Calendar | NYSE sessions (`exchange_calendars`, holidays + 9/11/Sandy closures) | plain business days |
| Results | `results/real/` | `results/synthetic/` |
| Reports | `reports/real/` | `reports/synthetic/` |
| Missing data | **hard failure** — never substituted | n/a |
| Gate | requires a passing `validate_real_data.py` run for the exact current store contents (fingerprinted) | none, but every output carries a synthetic warning |

Real and synthetic experiments live in separate registries and the ranking
code **refuses** to rank across modes (tested). Reports are labeled with their
mode in the title and header.

### Current status of real data in this repository

**NO REAL DATA HAS BEEN ACQUIRED.** The build environment blocks all
market-data hosts (Stooq, Yahoo, FRED, SEC EDGAR, Tiingo, Polygon — verified
2026-08-04). Everything up to the download step is implemented and tested;
`reports/real/` contains an explicit UNAVAILABLE placeholder, not synthetic
numbers. See `data/real_data_manifest.md` for exactly what to run elsewhere.

## The research platform

The project is a long-horizon research platform, not a one-off backtest. A
static, fully self-contained research site (no server, no external assets) is
generated from the registries and governance artifacts:

```bash
python scripts/research.py dashboard      # build site/ -> open site/index.html
python scripts/research.py update         # incremental experiments + site refresh
python scripts/research.py validate       # real-data quality gate
python scripts/research.py compare A B    # side-by-side metric table
python scripts/research.py roadmap        # ranked research priorities
python scripts/research.py ideas          # idea backlog (configs/research_ideas.yaml)
python scripts/research.py assistant      # rule-based evidence review (read-only)
python scripts/research.py export         # markdown research summary
python scripts/research.py tradingview    # Pine v5 exports for portable strategies
```

Site pages: **dashboard** (state-of-the-research cards, timeline, leaders,
roadmap) · **experiment explorer** (every registry record ever, filterable,
nothing deleted) · **strategy catalog + per-strategy pages** (auto docs,
research notebook, scorecard, genealogy tree, frozen grids, Pine export,
fingerprinted source) · **comparison engine** (any selection, CSV export) ·
**portfolio lab** (strategy correlation heatmap, regime co-failure map,
pairwise blend analysis) · **ideas** · **roadmap** (auto-generated from
backlog + gate limitations + open audit findings) · **audit & governance**.

Platform guarantees: the site builder and assistant are **read-only** over
registries, freezes, locks and findings (tested byte-for-byte); scorecards
measure research quality, never future returns; deprecated strategies stay
visible with reasons; every numeric page carries its data-mode banner.

**Plugins:** drop `plugins/<name>/strategy.py` with a `@register`-decorated
Strategy subclass (see `plugins/example_golden_cross/`) — it appears in the
catalog/site automatically. A plugin cannot enter a REAL study without being
added to the frozen grid and fingerprint list, which forces a freeze bump
(tested).

### Publishing the site (Vercel)

`site/` is committed, fully self-contained (no external assets, no CDN, no
build step) and is what gets served. `vercel.json` configures it:

1. [vercel.com](https://vercel.com) → **Add New → Project** → import
   `jrosillo-code/techai.stocks`.
2. Leave **Root Directory** at the repository root (`./`). `vercel.json`
   already sets Framework = *Other*, an empty Build Command, and Output
   Directory = `site`. There is nothing to install and no Python runs on
   Vercel — the site is static HTML generated locally and committed.
3. **Deploy.** No environment variables are required.

The config sends `X-Robots-Tag: noindex` on every page. **Keep it that way
until the first real-data study has run** — every number on the site is
currently generated from a simulated market, and the pages say so. Remove
that header only when the site shows real results you intend to publish.

To refresh the site after new experiments:

```bash
python scripts/research.py dashboard   # rewrites site/
git add site results/synthetic/experiments.jsonl && git commit && git push
```

Vercel redeploys on push, so the published dashboard always matches the
committed research record.

### What is version-controlled, and why

`results/synthetic/experiments.jsonl` (the append-only experiment registry)
and the derived summary tables **are committed** — they are the research
record, and without them a fresh clone cannot reconstruct what was tested or
reproduce a ranking. Equity curves (`results/*/curves/`, ~125 MB) are **not**
committed: they are fully regenerable from the registry plus the frozen code
via `python scripts/run_all.py --data-mode synthetic`. Until you regenerate
them, the portfolio-lab page renders empty on a fresh clone; everything else
works from the registry.

## The universe (freeze v3)

**120 securities: 81 still trading, 39 delisted.** Semiconductors, chip
equipment, chip-design software (EDA), cloud, enterprise software, internet,
cybersecurity, networking, robotics and the power companies feeding AI data
centres — plus 14 benchmark funds and 15 theme baskets.

The 39 delisted names are the point. A universe of survivors is a universe that
has already been told the answer: every name in it went on to exist. The roster
therefore keeps the dot-com bankruptcies (Nortel, WorldCom, Silicon Graphics,
JDS Uniphase, Lucent), the acquisition wave (Altera, Broadcom Corp, Red Hat,
VMware, Activision, Splunk, Ansys, Juniper) and the take-privates (Citrix,
Zendesk, Proofpoint, Twitter). A momentum screen has to buy them at their peaks
and ride them into the delisting, exactly as a real investor would have.

The security master carries the ticker traps that come with them — most
sharply, **Broadcom**: Avago bought the Broadcom *name* in 2016, so a vendor
series labelled "Broadcom" begins in 2009 and hides the original company's
dot-com boom and collapse entirely. `BRCM` and `AVGO` are separate securities
here and are never joined.

*Caveat that has not gone away:* the roster is defined, but free providers
frequently serve no history for delisted tickers. A real run that cannot fetch
them reports itself as survivorship-limited rather than pretending the list was
honoured.

## The first frozen real-data study (one command)

The first real run is governed by a **research freeze**
(`configs/research_freeze_v3.json`, hash `f173a689adb51ed3be8ccb0b96e615af`):
every strategy grid, benchmark, cost scenario, split, ranking rule, rejection
rule, and the fingerprints of the 31 code modules that compute, tier, gate or
record results — hashed before any real result exists. Every real-mode entry
point (experiments, robustness, capacity, company analysis, report) verifies
the hash and aborts on any drift — no tuning is possible during the first
study.

**Freeze v2 is superseded** (preserved unmodified at
`configs/research_freeze_v2.json`, hash `49767ea3efc44cead711d72946c3fe31`) —
by a deliberate redefinition of the study, not by a defect in its controls: the
universe grew from 38 securities to 120, and two strategy families (`factor`,
`allocation`) were added to test v2's central negative finding, that its 24
families were one bet in disguise. One real defect was fixed alongside it
(AUD-016): `experiment_id` did not bind the universe, so re-running a strategy
against a wider roster would have reused the narrower roster's cached results.
Every leaderboard, scorecard and comparison is now restricted to a single
universe cohort; the v2 cohort stays in the registry and is excluded by hash
rather than deleted.

**Freeze v1 is superseded** (preserved unmodified at
`configs/research_freeze_v1.json`, tag `pre-real-data-freeze-v1`): the
2026-08 adversarial audit (`audit/reports/adversarial_audit.md`, verdict
`READY WITH MATERIAL LIMITATIONS`) found v1 bound only strategy/engine/
ranking code, its holdout log was not tamper-evident, the decision brief had
fail-open paths, experiment records lacked data lineage, and CAGR was
mis-annualized. All CRITICAL/HIGH findings are fixed with regression tests
(87 tests); no real-data results ever existed under v1, so nothing was
retroactively invalidated.

On a network-enabled macOS/Linux machine:

```bash
./scripts/run_first_real_study.sh          # optionally: --start 1998-01-01
```

This validates the environment, verifies the freeze, downloads (resumable),
imports with checksum verification, runs the quality gate (hard stop on FAIL
with recovery instructions), executes the frozen study, and produces:

* `reports/real/research_report_full.html` — full methodology + diagnostics,
  with strategies separated into **evidence tiers** (A: complete adjusted
  histories, no revised-macro/event dependence · B: revised macro, partial
  fundamentals, survivorship-limited universes · C: unavailable — never mixed
  in one leaderboard);
* `reports/real/decision_brief.html` — did anything survive, what supports
  it, what invalidates it, and a decision capped at **paper-trade / do
  nothing** (the brief is structurally incapable of recommending live
  trading);
* `first_real_study_<stamp>.tar.gz` — shareable bundle: reports, registry,
  quality gate, reconciliation, capacity, holdout log, environment
  fingerprint, freeze file. No API keys, no licensed raw data.

Holdout discipline: the runner freezes the selection, logs the single
sanctioned holdout access, and every later access flips a public
`compromised` flag that reports must display. The prospective paper-trading
protocol (`docs/prospective_testing_protocol.md`) is frozen alongside and
applies unchanged to any survivor.

## The real-data workflow

```bash
# 1. On any network-enabled machine (only step that needs internet):
python scripts/download_real_data.py --providers yahoo stooq fred sec \
       --start 1998-01-01 --output data/export_bundle
#    -> portable bundle: per-provider parquet + manifest + sha256 checksums,
#       resumable, rate-limited, per-symbol failure log

# 2. Import (verifies checksums, normalizes schemas, reconciles providers,
#    trims rows outside IPO/delisting windows, records provider metadata):
python scripts/import_data_bundle.py --input data/export_bundle
#    or, for your own CSV/JSON/Parquet files: see data/import/README.md
python scripts/import_data_bundle.py --loose

# 3. Validate — MANDATORY. Writes the quality gate (PASS FOR RESEARCH /
#    PASS WITH LIMITATIONS / FAIL — DO NOT BACKTEST). Real-mode runs refuse
#    to start without a passing gate for the current store fingerprint:
python scripts/validate_real_data.py

# 4. Research run + report:
python scripts/run_all.py --data-mode real
```

Other commands:

```bash
python scripts/run_experiments.py --data-mode real --families benchmarks   # one family
python scripts/run_experiments.py --data-mode real --families riskmanaged  # one family
python scripts/run_robustness.py  --data-mode real                         # walk-forward, DSR, bootstrap
python scripts/run_capacity.py    --data-mode real --top 5                 # $100k..$100M scaling
python scripts/run_company_analysis.py --data-mode real                    # own-it vs trade-it
python scripts/make_report.py     --data-mode real                         # report only
python scripts/run_all.py         --data-mode synthetic                    # offline demonstration
python -m pytest tests/ -q                                                 # 64 tests
```

Reproduce a specific experiment: look up its record in
`results/<mode>/experiments.jsonl` (each has a stable `id`, full `spec`,
provider, git commit); `aitb.strategies.from_spec(record["spec"])` rebuilds
the exact strategy object.

## Architecture

```
ai-tech-backtest/
  configs/
    universe.yaml          PIT universe (81 live + 39 delisted) + theme
                           baskets + target_holdings
    security_master.yaml   permanent sids, ticker histories (FB→META,
                           SUNW→JAVA), successors (XLNX→AMD, EMC→DELL)
    costs.yaml             zero/low/base/stressed scenarios
    backtest.yaml          conventions, splits, regime windows
    experiments.yaml       grids with core/exploratory/deprecated status +
                           stated hypotheses + deprecation reasons
  src/aitb/
    calendar.py            NYSE session calendar (real) / b-days (synthetic)
    config.py              typed config + per-mode results/reports dirs
    data/
      providers.py         stooq / yahoo / fred adapters
      providers_ext.py     tiingo / alphavantage / SEC EDGAR companyfacts
                           (point-in-time fundamentals w/ filing dates)
      synthetic.py         deterministic offline generator (tests/demos ONLY)
      store.py             synthetic-mode parquet cache
      realstore.py         canonical real store + metadata + checksums
      import_bundle.py     bundle/loose import, schema normalization,
                           cross-provider reconciliation (flag, never average)
      quality.py           validation gate: PASS / PASS WITH LIMITATIONS / FAIL
      security_master.py   symbol resolution by date; gap-refusing stitching
      loader.py            mode-gated panel assembly; PIT fundamentals accessor
      validation.py (data) gap/duplicate/jump/stale/OHLC checks
    universe.py            point-in-time investable mask
    features.py            causal signal building blocks
    portfolio.py           weighting schemes, caps, vol targeting, schedules
    costs.py               bps + √participation impact + borrow model
    backtest/engine.py     next-open execution engine
    metrics.py             performance/risk metrics
    validation.py          walk-forward, block+stationary bootstrap, PSR/DSR,
                           MC, NVDA/top-N dependence, pre/post-2023 splits,
                           leave-one-year-out
    tax.py                 approximate taxable/deferred overlay (documented
                           approximation — turnover-based, no lot accounting)
    holdout.py             holdout lock: freeze-before-look, access log,
                           compromise flag surfaced in reports
    experiments.py         append-only per-mode registry + runner
    ranking.py             composite score, relative benchmark hurdle,
                           refuses cross-mode ranking
    reporting.py           charts + self-contained HTML reports
    strategies/            benchmark, tsmom, xsmom, meanrev, breakout,
                           fundamental, regime, riskmanaged (incl. combined
                           TrendPlusVolTarget with hysteresis), ml,
                           factor (residual momentum, multi-horizon consensus,
                           low-vol, breadth gating, theme rotation, fundamental
                           acceleration), allocation (equal-risk contribution,
                           minimum-correlation sleeve)
  scripts/                 download_real_data, import_data_bundle,
                           validate_real_data, run_experiments, run_robustness,
                           run_capacity, run_company_analysis, make_report,
                           run_all, download_data (synthetic)
  data/real_data_manifest.md   exact commands + datasets still required
  data/import/README.md        user-supplied file formats
  tests/                   64 tests incl. bias-regression, mode-separation,
                           calendar, symbol-change, gate, holdout, tax
```

## Methodology and bias prevention

| Bias | Countermeasure |
|---|---|
| Look-ahead | Signals dated T fill at the **open of T+1**, structurally. Regression test: perfect-foresight signals cannot beat buy & hold. |
| Feature leakage | All features trailing; truncation test proves it. |
| Fundamental leakage | Quarterly rows enter only at publication/filing date. EDGAR adapter uses SEC `filed` dates; the gate **fails fatally** if any `published < period_end`. |
| Survivorship | PIT universe with IPO seasoning and delisting enforcement; delisted names retained; importer trims vendor rows outside listing windows; remaining current-constituent bias is disclosed by the gate, not hidden. |
| Symbol traps | Security master with dated ticker spans; stitching refuses gaps/overlaps that would fabricate returns (FB→META, SUNW→JAVA tested). |
| Bad data | Import-time schema validation; cross-provider return reconciliation (differences flagged, never averaged); OHLC/jump/stale/gap checks; NYSE-session gap detection; checksummed transport. |
| Unrealistic fills | Cost scenarios + √participation impact vs trailing ADV; 5%-of-ADV cap; borrow on shorts; capacity analysis at $100k–$100M. |
| Overfitting / multiple testing | Untouched holdout with a **lock** (freeze-specs-then-look, access log, public compromise flag); walk-forward selection; block + stationary bootstrap; PSR/deflated Sharpe over full trial batteries; parameter sensitivity; every failed AND deprecated variant stays in the registry with reasons. |
| Winner/regime dependence | Per-strategy NVDA share, top-1/3/5 contribution shares, pre/post-2023 splits, leave-one-year-out — recorded on every experiment. |
| Taxes | Optional overlay comparing taxable vs deferred outcomes; high-turnover edges are re-judged after estimated taxes. |

### Series roles (documented convention)

Signals use total-return `adj_close`; fills and marks use ratio-adjusted OHLC
(`adj_close/close × raw OHLC`), so dividends are embedded continuously in the
price path; raw `close` is retained and never silently mixed. Stooq data is
split-adjusted only — the importer records this per ticker and the gate
reports the affected names as a limitation.

## Data sources

| Source | Status | Notes / licensing |
|---|---|---|
| Yahoo Finance | adapter ready | unofficial endpoint; total-return adjclose; fallback-quality |
| Stooq | adapter ready | free EOD; split-adjusted only (no dividends); research use |
| FRED | adapter ready | public domain; revised series, not vintages (disclosed) |
| SEC EDGAR | adapter ready | public domain; PIT fundamentals via companyfacts + filing dates; set `SEC_USER_AGENT` |
| Tiingo | adapter ready | key in `.env`; total-return quality + dividends/splits |
| Alpha Vantage | adapter ready | key in `.env`; heavily rate-limited free tier |
| Polygon / FMP / EODHD / Nasdaq Data Link | planned | subclass `PriceProvider`; nothing else changes |
| Synthetic | active here | deterministic; tests/demos/offline dev ONLY |

**Free-data gaps** (disclosed in every real report): no delisted-name history,
no historical index constituents, no macro vintages, no earnings timestamps.
Paid sources (CRSP, Norgate, EODHD) fix these; their files can be dropped into
`data/import/`.

## Limitations (read before believing any number)

* No real data could be acquired in this environment — all shipped result
  artifacts are synthetic-mode demonstrations of the machinery.
* The real universe will be partially current-constituent biased until a
  delisted-history source is imported (the gate labels this).
* Macro strategies using CPI/UNRATE carry revision look-ahead; excluded from
  the first real run by default.
* Tax module is a documented approximation (no lot accounting).
* Earnings-event strategies stay disabled until timestamped event data exists.
* Borrow costs are scenario constants; no options data; no intraday data.
