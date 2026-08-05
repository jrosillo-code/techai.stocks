"""Market-data provider adapters.

Every adapter implements the same narrow interface so higher layers never
depend on a specific vendor:

    fetch_daily(ticker, start, end)  -> DataFrame indexed by date with columns
                                        [open, high, low, close, adj_close, volume]
    fetch_series(series_id, start)   -> macro Series (FRED-style)

Adjusted and unadjusted prices are carried side by side and are never merged
silently: ``close`` is the raw traded price, ``adj_close`` is the
split- and dividend-adjusted total-return price. Ratio-adjusting the OHLC
columns for backtest execution happens explicitly in the loader.

Licensing notes (verify before production use):
  * Stooq — free EOD CSVs; personal/research use.
  * Yahoo Finance — unofficial endpoint; terms restrict redistribution.
    Used strictly as a fallback.
  * FRED — public domain data; the fredgraph CSV endpoint needs no key.
  * Tiingo/Polygon/AlphaVantage/FMP — commercial; adapters can be added by
    subclassing ``PriceProvider`` without touching the rest of the system.

The ``SyntheticProvider`` (see synthetic.py) implements the same interface and
is used when the environment has no network access to data hosts.
"""
from __future__ import annotations

import io
import time
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd
import requests

from ..utils import get_logger

log = get_logger("data.providers")

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class ProviderError(RuntimeError):
    pass


class NoDataError(ProviderError):
    """The provider answered, and the answer is "this does not exist here".

    Distinct from ProviderError, which means the request itself failed and may
    succeed later. Retrying this one is guaranteed to waste time — which is
    exactly what happened when the universe grew to 39 delisted names that no
    free provider carries: four attempts with 2/4/8/16s backoff each, for a
    symbol that could never resolve, is 30 seconds of sleeping per name per
    provider. Callers record it as a coverage gap and move on immediately.
    """


class PriceProvider(ABC):
    """Abstract price/macro provider."""

    name: str = "abstract"

    @abstractmethod
    def fetch_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Return daily bars indexed by DatetimeIndex, columns PRICE_COLUMNS."""

    def fetch_series(self, series_id: str, start: date, end: date) -> pd.Series:
        raise NotImplementedError(f"{self.name} has no macro series support")


# A definitive "no" from the server. Retrying cannot change the answer:
#   400 the request is malformed for this symbol
#   403 not entitled (Yahoo does this for some delisted/foreign symbols)
#   404 no such symbol
#   410 gone
# 429 and 5xx are deliberately absent — those are worth retrying.
_PERMANENT_STATUS = frozenset({400, 403, 404, 410})


def _http_get(url: str, params: dict | None = None, retries: int = 4,
              timeout: int = 30, headers: dict | None = None) -> requests.Response:
    """GET with exponential backoff, rate-limit handling, and fast failure.

    Backoff applies only to failures that might resolve on a retry. A hard
    "no such symbol" returns immediately: waiting 30 seconds to be told 404
    four times is pure cost, and with 39 permanently-delisted names across
    three providers it added roughly an hour to every download.
    """
    delay = 2.0
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=headers)
            if resp.status_code == 429:
                log.warning("rate limited by %s, sleeping %.0fs", url, delay)
                time.sleep(delay)
                delay *= 2
                continue
            if resp.status_code in _PERMANENT_STATUS:
                raise NoDataError(
                    f"{url} returned {resp.status_code} — no data for this "
                    "symbol at this provider (not retried)")
            resp.raise_for_status()
            return resp
        except NoDataError:
            raise
        except requests.RequestException as exc:  # includes HTTPError
            last = exc
            log.warning("attempt %d/%d failed for %s: %s", attempt + 1, retries, url, exc)
            if attempt < retries - 1:      # never sleep after the last attempt
                time.sleep(delay)
                delay *= 2
    raise ProviderError(f"failed to fetch {url}: {last}")


class StooqProvider(PriceProvider):
    """Free EOD data from stooq.com (US tickers use the `.us` suffix).

    Stooq serves split-adjusted prices; dividends are NOT included, so
    ``adj_close`` equals ``close`` here and the limitation is recorded by the
    loader. Prefer Tiingo/Yahoo when total-return series are required.
    """

    name = "stooq"

    def fetch_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        symbol = f"{ticker.lower()}.us"
        url = "https://stooq.com/q/d/l/"
        resp = _http_get(url, params={"s": symbol, "i": "d",
                                      "d1": start.strftime("%Y%m%d"),
                                      "d2": end.strftime("%Y%m%d")})
        df = pd.read_csv(io.StringIO(resp.text))
        if "Close" not in df.columns or df.empty:
            raise NoDataError(f"stooq returned no data for {ticker}")
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["adj_close"] = df["close"]  # split-adjusted only; no dividends
        if "volume" not in df.columns:
            df["volume"] = float("nan")
        return df[PRICE_COLUMNS]


class YahooProvider(PriceProvider):
    """Fallback provider using Yahoo's public chart endpoint (no key)."""

    name = "yahoo"
    _HEADERS = {"User-Agent": "Mozilla/5.0 (research; aitb backtester)"}

    def fetch_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "period1": int(pd.Timestamp(start).timestamp()),
            "period2": int(pd.Timestamp(end).timestamp()) + 86400,
            "interval": "1d",
            "events": "div,splits",
        }
        resp = _http_get(url, params=params, headers=self._HEADERS)
        payload = resp.json()
        result = payload.get("chart", {}).get("result")
        if not result:
            raise NoDataError(f"yahoo returned no data for {ticker}")
        r = result[0]
        ts = pd.to_datetime(r["timestamp"], unit="s").normalize()
        quote = r["indicators"]["quote"][0]
        adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose")
        df = pd.DataFrame(
            {
                "open": quote["open"], "high": quote["high"], "low": quote["low"],
                "close": quote["close"], "volume": quote["volume"],
                "adj_close": adj if adj is not None else quote["close"],
            },
            index=ts,
        ).dropna(subset=["close"])
        df.index.name = "date"
        return df[PRICE_COLUMNS].sort_index()


class FredProvider(PriceProvider):
    """Macro series from FRED via the keyless fredgraph CSV endpoint."""

    name = "fred"

    def fetch_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError("FRED serves macro series, not equity bars")

    def fetch_series(self, series_id: str, start: date, end: date) -> pd.Series:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        resp = _http_get(url, params={"id": series_id})
        df = pd.read_csv(io.StringIO(resp.text), na_values=["."])
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["value"].astype(float).dropna()
        return s.loc[str(start):str(end)].rename(series_id)


def get_provider(name: str) -> PriceProvider:
    from .synthetic import SyntheticProvider  # local import avoids cycle

    registry: dict[str, type[PriceProvider]] = {
        "stooq": StooqProvider,
        "yahoo": YahooProvider,
        "fred": FredProvider,
        "synthetic": SyntheticProvider,
    }
    if name not in registry:
        raise KeyError(f"unknown provider '{name}' (have {sorted(registry)})")
    return registry[name]()
