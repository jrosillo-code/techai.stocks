# Real-data manifest — what the research run still needs

Status: **NO REAL DATA HAS BEEN ACQUIRED.** This build environment blocks all
market-data hosts (Stooq, Yahoo, FRED, SEC EDGAR, Tiingo, Polygon — re-verified
2026-08-05: every one returns `CONNECT tunnel failed, 403`). The pipeline,
importer, validation gate and real-mode runner are complete and tested; only the
download step requires a network-enabled machine.

Everything currently published — every Sharpe ratio, drawdown and verdict — is
simulated. Swapping in real prices requires **no code change**. It requires
running the commands below somewhere with internet access.

---

## Route A — do everything on one networked machine (simplest)

```bash
git clone https://github.com/jrosillo-code/techai.stocks
cd techai.stocks
python3 -m pip install -e '.[dev]'                  # needs Python >= 3.11

export SEC_USER_AGENT="Your Name your@email.com"    # SEC requires a real contact
./scripts/run_first_real_study.sh
```

That single script runs all nine phases: environment check → freeze
verification → download → checksum-verified import → **quality gate (hard stop
on FAIL)** → frozen backtests → robustness → capacity → company analysis →
reports. It never substitutes synthetic data, and it never tunes a parameter —
the freeze hash is verified before any backtest runs, so the study that executes
is exactly the study specified in advance.

Expect roughly 30–90 minutes, most of it downloading. It is resumable: re-run
after an interruption and completed symbols are skipped.

## Route B — download elsewhere, analyse here

Use this when the analysis machine has no internet (as here). The bundle is a
portable directory with per-file checksums.

On the networked machine:

```bash
python scripts/download_real_data.py \
    --providers yahoo stooq fred sec \
    --start 1998-01-01 \
    --output data/export_bundle
```

Copy `data/export_bundle/` across, then:

```bash
python scripts/import_data_bundle.py --input data/export_bundle
python scripts/validate_real_data.py      # must PASS before any backtest
python scripts/run_all.py --data-mode real
python scripts/research.py dashboard      # rebuild the site on real results
```

`validate_real_data.py` is the gate. It returns one of three statuses, and the
runner refuses to proceed on the third:

| Status | Meaning |
|---|---|
| `PASS FOR RESEARCH` | coverage and integrity checks clean |
| `PASS WITH LIMITATIONS` | usable, with named gaps recorded in every report |
| `FAIL — DO NOT BACKTEST` | hard stop; nothing downstream runs |

---

## Required datasets

Universe as of freeze v3: **120 securities** (81 live, 39 delisted) plus
**14 benchmark funds** — see `configs/universe.yaml`.

| Dataset | Symbols | Source (free) | Notes |
|---|---|---|---|
| Daily OHLCV + adjusted close | 120 universe names + 14 benchmark ETFs | Yahoo (total-return adj) + Stooq (cross-check) | Yahoo `adjclose` includes dividends; Stooq is split-only. The importer prefers Yahoo, keeps both, and records the difference — they are never averaged together. |
| Dividends & splits | same | Yahoo events / Tiingo | carried alongside prices where the provider returns them |
| Macro series | FEDFUNDS, DGS2, DGS10, T10Y2Y, CPIAUCSL→CPIYOY, UNRATE, VIXCLS, BAA10Y | FRED (keyless CSV) | revised data, not vintages — the gate records this and the regime family is labelled accordingly |
| PIT fundamentals | universe names (best-effort) | SEC EDGAR companyfacts | filing dates are the availability dates, never fiscal period ends; set `SEC_USER_AGENT` |
| Earnings events | universe names | none free with timestamps | optional; event strategies stay disabled without it |

Setting `TIINGO_API_KEY` adds Tiingo as the preferred price source
automatically. Its delisted coverage is better than Yahoo's and it has a free
tier.

---

## Known gaps no free provider fills

* **Delisted-name history — now the biggest gap by far.** The universe
  deliberately contains **39 companies that no longer trade**: NT, WCOM, SGI,
  JDSU, LU, CPQ, GTW, PALM, ALTR, BRCM, ATML, LSI, FCS, IDTI, CY, RHT, VMW,
  ATVI, SPLK, ANSS, MENT, CA, NOVL, NUAN, LNKD, TWTR, JNPR, BRCD, TLAB, MFE,
  FEYE, PFPT, CTXS, ZEN, SUNW, EMC, YHOO, MXIM, XLNX. Free providers generally
  serve no history for them at all.

  This matters more than every other gap combined. Those names are the only
  thing stopping the backtest from being a study of companies that happened to
  survive. Without them the gate labels the run `PASS WITH LIMITATIONS`, every
  report discloses it, and the results are optimistic by an unknown amount.

  **Fix:** a licensed source carrying delisting returns — CRSP, Norgate Data or
  EODHD. Drop their files in `data/import/prices/` and the importer picks them
  up. Norgate is the cheapest realistic option for an individual.

* **The Broadcom trap.** Do not accept a single "Broadcom" price series. Avago
  acquired Broadcom Corp in 2016 and took the name, so a vendor series labelled
  Broadcom typically begins at Avago's 2009 IPO and silently omits the original
  company's dot-com boom and collapse. `BRCM` and `AVGO` are separate
  securities in `configs/security_master.yaml` and must stay separate. The same
  pattern applies to JDSU (split into VIAV and LITE in 2015) and SGI (whose
  symbol was later recycled by an unrelated company).

* **Historical index constituents** (Nasdaq-100 membership by date): not freely
  available in reliable form; universe sensitivity analysis relies on the
  broad-list approach instead.

* **Macro vintages** (ALFRED is free but per-series; not wired yet). Until it
  is, regime strategies use revised data and their results are upper bounds.

* **Earnings timestamps** with BMO/AMC precision. Event-driven strategies stay
  disabled rather than run on a guess.

---

## What a real run cannot do

Real prices raise the study from "the machinery works" to "here is evidence".
They do not raise it to "trade this". The frozen decision rules cap the best
possible outcome at *paper-trade it and watch* — see
`docs/prospective_protocol.md`. One historical study, however clean, is one
sample.
