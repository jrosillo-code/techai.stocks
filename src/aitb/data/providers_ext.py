"""Extended real-data provider adapters: Tiingo, Alpha Vantage, SEC EDGAR.

These adapters are exercised by scripts/download_real_data.py in a
network-enabled environment. In this repository's build environment all data
hosts are blocked, so they are covered by schema-level unit tests only —
treat the first networked run as the integration test and check the
download manifest for per-symbol failures.

Licensing:
  * Tiingo — free tier: EOD with dividends/splits; API key required;
    redistribution prohibited.
  * Alpha Vantage — free tier: rate-limited (25 req/day as of 2025); key
    required.
  * SEC EDGAR — public domain; requires a descriptive User-Agent and
    <=10 req/s. companyfacts gives XBRL values WITH filing dates, which is
    what makes genuine point-in-time fundamentals possible for free.
"""
from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

from ..utils import get_logger
from .providers import (PRICE_COLUMNS, NoDataError, PriceProvider, ProviderError,
                        _http_get)

log = get_logger("data.providers_ext")

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "aitb-research admin@example.com (update SEC_USER_AGENT in .env)")


class TiingoProvider(PriceProvider):
    name = "tiingo"

    def __init__(self):
        self.key = os.environ.get("TIINGO_API_KEY", "")
        if not self.key:
            raise ProviderError("TIINGO_API_KEY not set")

    def fetch_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        resp = _http_get(url, params={
            "startDate": str(start), "endDate": str(end), "format": "json",
            "token": self.key})
        rows = resp.json()
        if not rows:
            raise NoDataError(f"tiingo returned no data for {ticker}")
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df.set_index("date").sort_index()
        out = pd.DataFrame({
            "open": df["open"], "high": df["high"], "low": df["low"],
            "close": df["close"], "adj_close": df["adjClose"],
            "volume": df["volume"],
        })
        # Corporate actions carried alongside (canonical store keeps them).
        out["dividend"] = df.get("divCash", 0.0)
        out["split"] = df.get("splitFactor", 1.0)
        return out


class AlphaVantageProvider(PriceProvider):
    name = "alphavantage"

    def __init__(self):
        self.key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
        if not self.key:
            raise ProviderError("ALPHAVANTAGE_API_KEY not set")

    def fetch_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        resp = _http_get("https://www.alphavantage.co/query", params={
            "function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": ticker,
            "outputsize": "full", "apikey": self.key})
        payload = resp.json()
        series = payload.get("Time Series (Daily)")
        if not series:
            raise ProviderError(f"alphavantage: {payload.get('Note') or payload.get('Information') or 'no data'}")
        df = pd.DataFrame(series).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().astype(float)
        out = pd.DataFrame({
            "open": df["1. open"], "high": df["2. high"], "low": df["3. low"],
            "close": df["4. close"], "adj_close": df["5. adjusted close"],
            "volume": df["6. volume"],
            "dividend": df["7. dividend amount"],
            "split": df["8. split coefficient"],
        })
        out.index.name = "date"
        return out.loc[str(start):str(end)]


# ------------------------------------------------------------------- EDGAR --
# XBRL concept preference lists (first match wins). Values are kept exactly as
# originally filed; restatements/amendments arrive as later facts with later
# `filed` dates and are dropped unless keep_amendments=True.
# XBRL tags, in preference order — the first one a filer actually uses wins.
# Companies tag the same economic quantity differently and change tags between
# years, which is why each field lists alternatives rather than one concept.
_CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet"],
    "eps": ["EarningsPerShareDiluted"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],

    # ---- added for freeze v4 -------------------------------------------
    # All free, all from the same companyfacts call already being made, so
    # they cost no extra requests — only the fields were never asked for.
    #
    # gross_profit + assets are the actual Novy-Marx profitability ratio.
    # The study has been approximating it with the free-cash-flow margin
    # because these were not collected; that proxy conflates profitability
    # with capital intensity, which in semiconductors is a large error.
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    # R&D intensity is the tech-specific quality signal. Accounting expenses
    # R&D immediately while its benefit is multi-year, so a research-heavy
    # company looks less profitable than it is — a documented and persistent
    # mispricing (Lev & Sougiannis 1996, Chan/Lakonishok/Sougiannis 2001).
    # Nothing in a technology study should be missing this.
    "rnd": ["ResearchAndDevelopmentExpense"],
    # Balance-sheet quality: leverage and the accrual component of earnings
    # (Sloan 1996 — accruals predict returns negatively, robustly).
    "debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
}


def _opt(parts: dict, field: str, idx) -> pd.Series:
    """A field that may be absent: NaN, never zero.

    Zero would be a claim (this company had no R&D), NaN is the truth (this
    filer did not tag it). Downstream ranking drops NaN rather than ranking a
    fabricated zero as the worst value.
    """
    if field not in parts:
        return pd.Series(np.nan, index=range(len(idx)))
    return parts[field].reindex(idx).reset_index(drop=True)


class EdgarFundamentals:
    """Point-in-time quarterly fundamentals from SEC companyfacts.

    `published` = the SEC `filed` date of the filing containing the fact —
    the honest information-availability date. Duration facts of ~one quarter
    are used directly; cash-flow tags that only appear year-to-date are
    differenced within the fiscal year. Original values are preserved: for
    each (concept, period) the EARLIEST filing wins; later amendments are
    recorded separately when keep_amendments=True.
    """

    name = "edgar"
    _HEADERS = {"User-Agent": SEC_USER_AGENT}

    def _cik(self, ticker: str) -> int:
        resp = _http_get("https://www.sec.gov/files/company_tickers.json",
                         headers=self._HEADERS)
        for row in resp.json().values():
            if row["ticker"].upper() == ticker.upper():
                return int(row["cik_str"])
        raise NoDataError(f"no CIK found for {ticker}")

    def _facts(self, concept_names: list[str], facts: dict) -> pd.DataFrame:
        gaap = facts.get("facts", {}).get("us-gaap", {})
        for concept in concept_names:
            if concept in gaap:
                units = gaap[concept]["units"]
                unit = "USD/shares" if "USD/shares" in units else \
                       ("shares" if "shares" in units else "USD")
                if unit not in units:
                    continue
                df = pd.DataFrame(units[unit])
                df["concept"] = concept
                return df
        return pd.DataFrame()

    def fetch_fundamentals(self, ticker: str,
                           keep_amendments: bool = False) -> pd.DataFrame:
        cik = self._cik(ticker)
        resp = _http_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                         headers=self._HEADERS)
        facts = resp.json()

        parts: dict[str, pd.Series] = {}
        pub: dict[str, pd.Series] = {}
        for field, concepts in _CONCEPTS.items():
            df = self._facts(concepts, facts)
            if df.empty:
                continue
            df["end"] = pd.to_datetime(df["end"])
            df["filed"] = pd.to_datetime(df["filed"])
            if "start" in df.columns:
                df["start"] = pd.to_datetime(df["start"])
                df["days"] = (df["end"] - df["start"]).dt.days
            else:
                df["days"] = np.nan
            df = df[df.get("form", pd.Series("10-Q", index=df.index))
                    .isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])]
            if not keep_amendments:
                # earliest filing per period end = originally reported value
                df = df.sort_values("filed").drop_duplicates("end", keep="first")

            quarterly = df[(df["days"].isna()) | (df["days"].between(75, 100))]
            if len(quarterly) >= 4:
                use = quarterly
            else:
                # YTD-only tag: difference consecutive YTD values in-year.
                ytd = df.sort_values("end").copy()
                if "start" in ytd.columns:
                    ytd["fy"] = ytd["start"].dt.year
                    ytd["val"] = ytd.groupby("fy")["val"].diff().fillna(ytd["val"])
                use = ytd
            s = use.set_index("end")["val"].astype(float)
            parts[field] = s[~s.index.duplicated(keep="first")]
            f = use.set_index("end")["filed"]
            pub[field] = f[~f.index.duplicated(keep="first")]

        if "revenue" not in parts:
            raise NoDataError(f"EDGAR: no usable revenue facts for {ticker}")
        idx = parts["revenue"].index
        out = pd.DataFrame({
            "period_end": idx,
            "published": pub["revenue"].reindex(idx),
            "revenue": parts["revenue"].reindex(idx),
            "eps": parts.get("eps", pd.Series(dtype=float)).reindex(idx),
            "fcf": (parts.get("ocf", pd.Series(dtype=float)).reindex(idx)
                    - parts.get("capex", pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)),
            "shares": parts.get("shares", pd.Series(dtype=float)).reindex(idx),
            # v4 additions — absent for filers that do not tag them, which is
            # recorded as NaN rather than imputed. A strategy needing a field
            # simply has fewer names to choose from on those dates.
            "gross_profit": _opt(parts, "gross_profit", idx),
            "cost_of_revenue": _opt(parts, "cost_of_revenue", idx),
            "assets": _opt(parts, "assets", idx),
            "equity": _opt(parts, "equity", idx),
            "net_income": _opt(parts, "net_income", idx),
            "rnd": _opt(parts, "rnd", idx),
            "debt": _opt(parts, "debt", idx),
            "cash": _opt(parts, "cash", idx),
        }).reset_index(drop=True)
        # Conservative fallback: if any published date is missing, assume a
        # 90-day lag rather than dropping the row silently.
        missing = out["published"].isna()
        out.loc[missing, "published"] = out.loc[missing, "period_end"] + pd.Timedelta(days=90)
        return out.sort_values("period_end").reset_index(drop=True)


def get_extended_provider(name: str):
    registry = {"tiingo": TiingoProvider, "alphavantage": AlphaVantageProvider}
    if name in registry:
        return registry[name]()
    raise KeyError(name)
