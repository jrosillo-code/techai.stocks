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

## The universe (freeze v4)

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
(`configs/research_freeze_v11.json`, hash `6d942efe888bde823a96f0590df6ff25`):
every strategy grid, benchmark, cost scenario, split, ranking rule, rejection
rule, and the fingerprints of the 34 code modules that compute, tier, gate or
record results — hashed before any real result exists. Every real-mode entry
point (experiments, robustness, capacity, company analysis, report) verifies
the hash and aborts on any drift — no tuning is possible during the first
study.

## The first real study — and what it found

Run 2026-08-05 on real market data under freeze v3. **113 of 120** universe
names and **33 of 39** delisted names had usable history; the quality gate
returned `PASS WITH LIMITATIONS`. 208 variants produced 624 experiments.

**Zero robust candidates.** 183 rejected, 162 inconclusive. The bar was set by
equal-weighting the target shortlist monthly (score 4.24); the best active
strategy — a 250-day trend filter on that same shortlist — scored 4.36, short
of the required 0.25 margin by 0.13. It did cut the worst drawdown from −74% to
−31%, which is a real difference, just not one that clears a bar built to
resist wishful thinking.

The structural finding was more useful than the scoreboard: nearly everything
correlated 0.7–0.9 with simply holding QQQ, and that was **not** a fact about
technology stocks. All 208 variants were long-only, so the co-movement was a
property of the study's own constraint. Freeze v4 lifts it.

## The second real study — the first thing that survived

Run 2026-08-05 under freeze v4. 666 experiments. **Two robust candidates**, the
first this project has produced on real prices, and both the same strategy at
two parameter settings:

| | |
|---|---|
| Strategy | `BetaHedgedBasket(basket=target_holdings, hedge=QQQ, beta_window=252, max_hedge=1.5, rebalance=W-FRI)` |
| Score / hurdle | 4.93 against 4.24 + 0.25 margin |
| Sharpe | 0.85 development, 1.62 holdout |
| CAGR | 17.0% development, 28.5% holdout |
| Worst drawdown | −33.8% |
| Survives stressed costs | yes (stressed Sharpe 0.75) |
| Turnover | 7.8× a year |
| Evidence tier | **B** |

It owns the shortlist and shorts QQQ against it at the basket's trailing beta,
capped at 1.5×. That it is the *only* construction to clear the bar, and also
the only one whose correlation to the index is near zero rather than 0.6–0.9,
is the same finding twice: what the study kept rediscovering was the market
factor, and the one thing that removed it is the one thing that scored.

**Five reasons to hold this loosely**, none of them optional reading:

* **Tier B, not A.** Its inputs include revised macro series and a
  survivorship-limited universe. It is not clean-data evidence.
* **The holdout is not out-of-sample any more.** It was opened under v3 and
  again under v4. Whatever the label says, this is development evidence.
* **The short side is priced with a flat borrow assumption.** QQQ is cheap and
  deep to borrow so the error is probably small here, but it is an assumption,
  not a measurement — and it is the top item on the roadmap for that reason.
* **7.8× annual turnover** means the result lives or dies on the cost model,
  which is a model.
* **Two settings of one strategy is one finding**, not two.

**Freeze v3 is superseded** (preserved at `configs/research_freeze_v3.json`,
hash `7edd4b56d181956d1d68645b08e03e04`) by freeze v4, which adds:

* a **`longshort` family** — market-neutral momentum, beta-hedged baskets,
  gross profitability, R&D intensity, accrual quality, post-earnings drift and
  dispersion-timed selection. On simulated data the market-neutral book
  correlates **−0.005** with QQQ against 0.63–0.84 for every long-only family.
* **eight more fundamental fields**, all free and all from an EDGAR request
  already being made: gross profit, cost of revenue, total assets, equity, net
  income, **R&D**, debt and cash. R&D intensity is the canonical
  technology-specific quality signal and was previously unmeasurable.

Both changes *raise* the bar — every added variant enters the deflated-Sharpe
correction for the whole study — and the v3 holdout has been opened, so any v4
result on it is development evidence, not out-of-sample evidence. Genuinely
out-of-sample now means forward paper trading.

## Freeze v6 — strategies you can actually watch on a chart

**Freeze v5 never ran against real prices.** It added a `chartable` family —
strategies designed single-symbol from the start, so the rule tested in Python
and the rule running on a TradingView chart are the *same rule* rather than an
approximation of one. Everything exported to a chart before that was portable
by accident: a leftover of a family built for portfolios, and half of them
gauges rather than entry rules. v5 was validated on simulated data and then
superseded the same day by v6, before any real study was commissioned under it,
so nothing is retroactively invalidated — nothing was ever concluded from it.

**Freeze v6** (`configs/research_freeze_v6.json`, hash
`2f5d0c554f0fed8ba47a95adc5659bdc`) extended that family to close the two most
conspicuous gaps in anything this study has ever tested — and **v6 never ran
against real prices either**. Its synthetic validation run exposed a defect in
its own specification, which is what that run exists for: the control arm below
was marked `status: deprecated`, and `deprecated` in this project means
*withdrawn* — the runner records such an entry with its reason and never
executes it. Six filtered variants ran and the unfiltered comparison was
silently empty. **Freeze v7** (`configs/research_freeze_v7.json`, hash
`e5a13c8887f98ae693d2eaaa51044e41`) is v6 with that one status corrected to
`exploratory`; nothing else differs. The freeze was not edited — a freeze is
never edited — a new one was cut, which is what the immutability rule is for.

The two gaps both freezes address:

* **Volume.** From v1 through v5, volume appeared in exactly one place — the
  participation cap that stops a simulated fill from consuming an implausible
  share of a day's turnover — and never once as a signal. Every breakout rule
  in the study treats a new high on half-normal turnover and a new high on
  triple turnover as the same event. `VolumeConfirmedBreakout` is the first
  strategy here to read it, and it ships with a **control arm**
  (`vol_mult=1.0`, the identical rule with the filter switched off) so the
  filter's contribution can be measured rather than inferred. Turnover is
  measured in dollars — `volume * close` — which a Pine script computes
  exactly rather than approximating.
* **A bet that is not the same bet.** The study's central negative finding,
  unchanged since v2, is that its families are one bet in disguise: all long
  the same names at broadly the same times, all correlating 0.6–0.9 with the
  index. Every attempt to break that so far has been another way of reading
  prices. `TurnOfMonth` reads no market data at all — its exposure for any
  future year is already determined — so whatever else is wrong with it, it
  cannot be that bet. Windows are in **calendar** days, not trading days,
  because "the third trading day before month end" is not knowable on the day
  itself without looking forward, and this family exists precisely so the
  tested rule and the chart rule are the same rule. (`IDEA-011` records what a
  forward session calendar would buy back.)

The five `chartable` strategies export as real Pine `strategy()` scripts, so
TradingView's Strategy Tester runs them and marks every trade, and they carry a
different header from the rest: the *rule* is exact, only the accounting is an
approximation — and that approximation is large, because Pine charges no
commission, no spread, no slippage, no impact and no borrow.

Same accounting as always, and it does not improve: both additions enter the
deflated-Sharpe trial count for the whole study, so they raise the bar rather
than lower it, the control arm is a real trial and counts as one, and the
holdout has now been opened twice. Any v6/v7 result on it is a third look and
is development evidence however it is labelled.

## Freeze v8 — "which rule for AI, which for tech, and what does it say to do"

Five documented indicators the study had never tested, each expressible on one
chart with explicit BUY and SELL marks:

| Strategy | Buy | Sell |
|---|---|---|
| `GaussianTrendBands` | close above the upper ATR band | back to the centreline (fast) |
| `GaussianTrendHold` | close above a *rising* centreline | close below the lower band (wide) |
| `Supertrend` | close above the ratcheting band | close below it |
| `ADXTrendStrength` | +DI over −DI **and** ADX above the floor | either condition fails |
| `RelativeStrengthNewHigh` | price/index ratio at a new N-bar high | ratio at an N-bar low |

**Every one is gridded over an AI basket *and* the full 81-name roster.**
Whether a rule suits the AI complex or technology broadly is a question the
data can answer by running the same rule over both; asserting it in a docstring
would be a preference wearing a finding's clothes. `basket: null` is all 81
names, against `megacap_ai`, `ai_compute` and `semiconductors`.

`GaussianTrendBands` and `GaussianTrendHold` are a deliberate **pair**:
identical filter, opposite aggression — enter hard and exit fast, versus enter
easily and exit late. Comparing them isolates parameterisation from the
indicator, the same way v7's control arm isolates the volume filter from the
breakout it rides on.

### The centred-kernel trap

A "Gaussian filter" in signal processing is a **centred** kernel: it weights
bars on both sides of the point it smooths. Applied to a price series that
reads the future — the centreline bends into a reversal *before* the reversal,
every band entry looks prescient, and the equity curve is fiction. It is the
most common way a good-looking band script is silently wrong, and nothing about
the output looks wrong.

This uses Ehlers' cascaded-pole form instead, which is causal by construction,
and the behaviour is pinned by a test rather than asserted: on a unit step the
filter reads **0.008 at the step bar** and does not reach half until **8 bars
later**. A centred kernel reads ~0.5 *on* the step.

### What data actually constrains this

Worth being straight about, since it is the obvious next question: for these
five, data was **not** the constraint. Every one runs on OHLCV that has been in
the store since v3 — they had simply never been built. Where data genuinely
binds is elsewhere, and each of these blocks a class of strategy rather than a
single rule:

* **FINRA daily short-sale volume** — free, daily, per symbol. The only free
  read on positioning rather than price. (`IDEA-012`)
* **Earnings dates with BMO/AMC timing** — blocks the entire event-driven
  family, which is Tier C by design until it exists. (`IDEA-001`)
* **Delisting returns** — would move the universe from Tier B to Tier A.
  (`IDEA-002`)
* **Options-implied volatility** — would let volatility regimes be measured
  forward-looking rather than trailing.

### One hypothesis already refuted, before any real run

`TurnOfMonth` was added to supply a return stream uncorrelated with the
price-driven families. **On the synthetic cohort it correlates +0.53 with QQQ**,
against −0.002 for the market-neutral `longshort` family. The arithmetic is
structural rather than an artifact of simulated data: the rule is in the market
about 34% of sessions and *fully long the basket* when it is, so it inherits
roughly √0.34 of the market's correlation. Its **timing** is uncorrelated with
every price signal here; its **exposure** is still long-the-market.

That is the freeze-v3 lesson recurring — co-movement with the index was a
property of the long-only *constraint*, not of the signal — and the v6/v7
design walked back into it.

The stated hypothesis is left **unedited** in the frozen docstring and in
`configs/experiments.yaml`. It was registered before the result, and rewriting
it to match the outcome is precisely what the freeze exists to prevent. The
correction lives here and in `IDEA-013` (hedge the in-window exposure with a
short index leg), which needs a freeze bump because it is a new strategy rather
than a parameter change.

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

# Re-running under a NEW FREEZE that needs no new data — the common case, since
# a freeze bump usually adds strategies rather than inputs. Reuses the store in
# data/real and starts at the quality gate. Refuses if that store is empty
# rather than backtesting nothing; does NOT skip the gate.
./scripts/run_first_real_study.sh --skip-download
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
