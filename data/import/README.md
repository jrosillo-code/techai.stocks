# User-supplied data imports

Drop files here and run `python scripts/import_data_bundle.py --loose`.

```
data/import/
  prices/          <TICKER>.csv|.json|.parquet — flexible columns; needs at
                   least a date column and a close column. Recognized names:
                   date/timestamp, open, high, low, close, adj_close (or
                   "adjusted close"), volume. Without adj_close the series is
                   treated as split-adjusted-only and flagged.
  corporate_actions/  reserved (dividends/splits as columns on price files
                      are already supported: `dividend`, `split`)
  fundamentals/    <TICKER>.csv|.parquet with columns:
                   period_end, published, revenue, eps, fcf, shares
                   `published` MUST be the real filing/announcement date —
                   rows where published < period_end are rejected.
  macro/           <SERIES>.csv|.parquet with a date column + one value column
  universes/       reserved for historical constituent lists
  earnings/        reserved: ann_date, timing (bmo/amc), eps, eps_est, ...
```

Every import is normalized, trimmed to the security's configured listing
window, written to the canonical store (`data/real/`) with provider metadata,
and reported in `results/real/import_report.json`.

After ANY import you must re-run `python scripts/validate_real_data.py` —
the quality gate fingerprints the store and real-mode backtests refuse to run
on unvalidated data.
