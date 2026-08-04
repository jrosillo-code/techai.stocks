"""Bundle import/reconciliation, schema validation, tax overlay, dependence."""
import json

import numpy as np
import pandas as pd
import pytest

from aitb.data import realstore
from aitb.data.import_bundle import (import_bundle, normalize_price_frame,
                                     reconcile)


def _bundle(tmp_path, with_manifest=True, tamper=False):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2019-01-02", periods=1000)
    close = 50 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 1000)))
    base = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "adj_close": close * 1.03,
                         "volume": 2e7}, index=idx)
    base.index.name = "date"
    b = tmp_path / "bundle"
    for prov, frame in (("yahoo", base), ("stooq", base.assign(adj_close=base["close"]))):
        d = b / "prices" / prov
        d.mkdir(parents=True)
        frame.to_parquet(d / "NVDA.parquet")
    if with_manifest:
        files = {str(p.relative_to(b)): realstore.file_sha256(p)
                 for p in b.rglob("*.parquet")}
        if tamper:
            k = next(iter(files))
            files[k] = "0" * 64
        (b / "manifest.json").write_text(json.dumps(
            {"created_at": "2026-08-04T00:00:00Z", "files": files}))
    return b


def test_import_prefers_total_return_provider(tmp_path):
    b = _bundle(tmp_path)
    report = import_bundle(b, root=tmp_path / "real")
    assert "prices/NVDA" in report.imported
    meta = realstore.read_meta("prices", "NVDA", root=tmp_path / "real")
    assert meta["provider"] == "yahoo"           # ranked above stooq
    assert meta["adjustment"] == "total_return"
    rec = report.reconciliation[0]
    assert rec["chosen"] == "yahoo" and "vs_stooq" in rec


def test_import_rejects_tampered_bundle(tmp_path):
    b = _bundle(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="integrity"):
        import_bundle(b, root=tmp_path / "real")


def test_import_trims_outside_listing_window(tmp_path):
    """ARM IPO'd 2023-09-14: earlier vendor rows must be trimmed."""
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2022-01-03", periods=700)   # starts pre-IPO
    close = pd.Series(60 * np.exp(np.cumsum(rng.normal(0, 0.02, 700))), index=idx)
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "adj_close": close, "volume": 1e6}, index=idx)
    df.index.name = "date"
    d = tmp_path / "bundle" / "prices" / "yahoo"
    d.mkdir(parents=True)
    df.to_parquet(d / "ARM.parquet")
    report = import_bundle(tmp_path / "bundle", root=tmp_path / "real",
                           strict_checksums=False)
    assert any(t["ticker"] == "ARM" for t in report.trimmed)
    stored = realstore.read("prices", "ARM", root=tmp_path / "real")
    assert stored.index.min() >= pd.Timestamp("2023-09-14")


def test_normalize_flexible_columns():
    df = pd.DataFrame({"Date": ["2024-01-02", "2024-01-03"],
                       "Close": [10.0, 10.5], "Adjusted Close": [9.9, 10.4],
                       "Vol": [100, 200]})
    out = normalize_price_frame(df, "test.csv")
    assert list(out.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert out["adj_close"].iloc[0] == 9.9
    assert out["open"].iloc[0] == 10.0            # backfilled from close


def test_normalize_requires_close():
    with pytest.raises(ValueError, match="close"):
        normalize_price_frame(pd.DataFrame({"date": ["2024-01-02"], "x": [1]}), "bad")


def test_reconcile_flags_disagreement():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(3)
    a = pd.DataFrame({"adj_close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))}, index=idx)
    noisy = a.copy()
    noisy["adj_close"] = noisy["adj_close"] * (1 + rng.normal(0, 0.01, 300))  # big diffs
    chosen, stats = reconcile("XXX", {"yahoo": a, "stooq": noisy})
    assert chosen == "yahoo"
    assert stats["vs_stooq"]["return_diffs_gt_50bp"] > 0
    assert "warning" in stats


# ------------------------------------------------------------------- tax ----
def test_tax_overlay_reduces_high_turnover_more():
    from aitb.tax import TaxConfig, apply_tax_overlay
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2015-01-01", periods=2520)
    r = pd.Series(rng.normal(0.0006, 0.012, 2520), index=idx)
    low_to = pd.Series(0.0005, index=idx)    # ~0.13x/yr => LT treatment
    high_to = pd.Series(0.02, index=idx)     # ~5x/yr => ST treatment
    res_low = apply_tax_overlay(r, low_to)
    res_high = apply_tax_overlay(r, high_to)
    assert res_low.tax_drag_annual >= 0
    assert res_high.tax_drag_annual > res_low.tax_drag_annual
    assert res_high.after_tax_cagr < res_high.pre_tax_cagr


def test_tax_deferred_beats_taxable_for_high_turnover():
    from aitb.tax import TaxConfig, apply_tax_overlay
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=2520)
    r = pd.Series(rng.normal(0.0006, 0.012, 2520), index=idx)
    to = pd.Series(0.02, index=idx)
    taxable = apply_tax_overlay(r, to)
    deferred = apply_tax_overlay(r, to, cfg=TaxConfig(deferred=True))
    assert deferred.after_tax_cagr > taxable.after_tax_cagr


# ------------------------------------------------------------- dependence ---
def test_period_split_and_loyo():
    from aitb.validation import leave_one_year_out, period_split_metrics
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2018-01-01", "2025-12-31", freq="B")
    r = pd.Series(rng.normal(0.0, 0.01, len(idx)), index=idx)
    # Make 2023+ carry all the performance -> ai_rally_dependent flag.
    r.loc["2023":] += 0.002
    ps = period_split_metrics(r, "2023-01-01")
    assert ps["post"]["sharpe"] > ps["pre"]["sharpe"]
    assert ps["ai_rally_dependent"] is True
    loyo = leave_one_year_out(r)
    assert loyo["min"] <= loyo["full"] <= loyo["max"] + 1e-9
    assert loyo["most_supportive_year"] in (2023, 2024, 2025)


def test_contribution_shares_nvda():
    from aitb.validation import contribution_shares
    idx = pd.bdate_range("2020-01-01", periods=500)
    w = pd.DataFrame({"NVDA": 0.5, "MSFT": 0.5}, index=idx)
    ar = pd.DataFrame({"NVDA": 0.002, "MSFT": 0.0002}, index=idx)
    out = contribution_shares(w, ar)
    assert out["nvda_share"] > 0.85
    assert out["top1_share"] == pytest.approx(out["nvda_share"])


def test_stationary_bootstrap_brackets():
    from aitb.validation import stationary_bootstrap_ci
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0005, 0.01, 1200),
                  index=pd.bdate_range("2018-01-01", periods=1200))
    point, lo, hi = stationary_bootstrap_ci(r, n_boot=200, seed=8)
    assert lo <= point <= hi
