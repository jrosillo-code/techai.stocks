# Real-data manifest — what the research run still needs

Status: **NO REAL DATA HAS BEEN ACQUIRED.** This build environment blocks all
market-data hosts (Stooq, Yahoo, FRED, SEC EDGAR, Tiingo, Polygon — verified
2026-08-04). The pipeline, importer, validation gate and real-mode runner are
complete and tested; only the download step requires a network-enabled machine.

## One command to run elsewhere

```bash
python scripts/download_real_data.py \
    --providers yahoo stooq fred sec \
    --start 1998-01-01 \
    --output data/export_bundle
```

Then move `data/export_bundle/` to this machine (or run everything there) and:

```bash
python scripts/import_data_bundle.py --input data/export_bundle
python scripts/validate_real_data.py          # must print PASS before backtests
python scripts/run_all.py --data-mode real
```

## Required datasets

| Dataset | Symbols | Source (free) | Notes |
|---|---|---|---|
| Daily OHLCV + adjusted close | 38 universe names + 8 benchmark ETFs (see `configs/universe.yaml`) | Yahoo (total-return adj) + Stooq (cross-check) | Yahoo `adjclose` includes dividends; Stooq is split-only — importer prefers Yahoo, records the difference |
| Dividends & splits | same | Yahoo events / Tiingo | carried alongside prices where the provider returns them |
| Macro series | FEDFUNDS, DGS2, DGS10, T10Y2Y, CPIAUCSL→CPIYOY, UNRATE, VIXCLS, BAA10Y | FRED (keyless CSV) | revised data, not vintages — gate records this limitation |
| PIT fundamentals | universe names (best-effort) | SEC EDGAR companyfacts | filing dates = availability dates; set `SEC_USER_AGENT` in .env |
| Earnings events | universe names | none free with timestamps | optional; event strategies stay disabled without it |

## Known gaps no free provider fills

* **Delisted-name history** (SUNW, EMC, YHOO, MXIM, XLNX): free providers do
  not serve delisted series. Without them the real universe is partially
  current-constituent biased — the gate will label the run
  `PASS WITH LIMITATIONS` and the report discloses it. CRSP/Norgate/EODHD
  fix this if a license is available (drop their files in `data/import/prices/`).
* **Historical index constituents** (Nasdaq-100 membership by date): not
  freely available in reliable form; universe sensitivity analysis relies on
  the broad-list approach instead.
* **Macro vintages** (ALFRED is free but per-series; not wired yet).
* **Earnings timestamps** with BMO/AMC precision.
