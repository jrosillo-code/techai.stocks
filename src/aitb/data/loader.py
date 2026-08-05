"""Panel assembly: aligned price/volume matrices plus point-in-time helpers.

The loader produces a ``MarketData`` bundle used by every strategy and the
engine:

  * ``open/high/low/close``   — split-adjusted execution prices (ratio-adjusted
                                so simulated fills at these prices compound
                                consistently with ``adj_close``),
  * ``adj_close``             — total-return series (signals & performance),
  * ``dollar_volume``         — for liquidity screens and impact modeling,
  * ``macro``                 — macro series frame,
  * ``fundamentals``          — long frame keyed by (ticker, published) with a
                                strict as-of accessor so quarterly data can
                                never leak before its publication date.

Series roles (documented contract):
  * signal construction .......... adj_close (total-return)
  * trade execution .............. open, ratio-adjusted = adj_close/close × raw open
  * position accounting .......... ratio-adjusted marks (dividends embedded
                                   continuously; see README §dividends)
  * benchmark total returns ...... adj_close
Raw ``close`` is retained unmodified; adjusted and unadjusted series are never
mixed implicitly anywhere else.

DATA MODES
----------
  * ``synthetic`` — provider-backed (deterministic generator or cached free
    providers); business-day calendar; results under results/synthetic.
  * ``real`` — reads ONLY the canonical validated store under data/real/
    (populated by scripts/download_real_data.py or import_data_bundle.py);
    NYSE session calendar. A missing dataset raises ``RealDataMissing`` —
    there is no fallback to synthetic data, ever.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ..config import UniverseConfig, load_universe_config
from ..utils import get_logger
from . import realstore
from .store import DataStore
from .providers import get_provider
from .validation import Issue, validate_prices

log = get_logger("data.loader")

MACRO_SERIES = ["FEDFUNDS", "DGS10", "DGS2", "T10Y2Y", "CPIYOY", "UNRATE",
                "VIXCLS", "BAA10Y"]
MIN_REAL_NAMES = 10
REQUIRED_REAL_BENCHMARKS = ("SPY", "QQQ")


@dataclass
class MarketData:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame          # raw (unadjusted) close
    adj_close: pd.DataFrame      # total-return close
    dollar_volume: pd.DataFrame
    macro: pd.DataFrame
    fundamentals: pd.DataFrame
    universe: UniverseConfig
    # FINRA short-sale volume as a share of the session's tape, date x ticker.
    # EMPTY unless the store has it. Optional on purpose: the series begins
    # 2009-07-31, so a strategy reading it is blind across the dot-com collapse
    # and 2008, and making it mandatory would impose that blindness on the
    # whole study. A strategy that needs it must refuse explicitly when it is
    # absent, the way ResearchIntensity refuses without `rnd`.
    short_share: pd.DataFrame = field(default_factory=pd.DataFrame)
    issues: list[Issue] = field(default_factory=list)
    provider_name: str = ""
    data_mode: str = "synthetic"

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self.adj_close.index

    def fundamentals_asof(self, ticker: str, asof: pd.Timestamp) -> pd.DataFrame:
        """All quarterly rows for `ticker` PUBLISHED strictly before `asof`.

        This is the only sanctioned access path for fundamentals inside a
        backtest; it keys on the publication date, not the fiscal period end.
        """
        f = self.fundamentals
        if f.empty:
            return f
        rows = f[(f["ticker"] == ticker) & (f["published"] < asof)]
        return rows.sort_values("period_end")


def _assemble(frames: dict[str, pd.DataFrame], macro: pd.DataFrame,
              fundamentals: pd.DataFrame, ucfg: UniverseConfig,
              issues: list[Issue], provider_name: str,
              data_mode: str) -> MarketData:
    def panel(col: str) -> pd.DataFrame:
        return pd.DataFrame({t: f[col] for t, f in frames.items()}).sort_index()

    raw_close = panel("close")
    adj_close = panel("adj_close")
    # Ratio-adjust OHLC so simulated fills compound like the TR series.
    ratio = adj_close / raw_close
    open_ = panel("open") * ratio
    high = panel("high") * ratio
    low = panel("low") * ratio
    dollar_volume = panel("volume") * raw_close

    macro = macro.reindex(adj_close.index).ffill() if not macro.empty \
        else pd.DataFrame(index=adj_close.index)

    md = MarketData(
        open=open_, high=high, low=low, close=raw_close, adj_close=adj_close,
        dollar_volume=dollar_volume, macro=macro, fundamentals=fundamentals,
        universe=ucfg, issues=issues, provider_name=provider_name,
        data_mode=data_mode,
    )
    n_err = sum(1 for i in issues if i.severity == "error")
    log.info("loaded %d tickers, %d validation warnings, %d errors [%s/%s]",
             len(frames), len(issues) - n_err, n_err, data_mode, provider_name)
    return md


def load_market_data(provider_name: str = "synthetic",
                     start: date = date(1990, 1, 2),
                     end: date = date(2026, 7, 31),
                     refresh: bool = False,
                     mode: str | None = None) -> MarketData:
    """Load the research dataset for a data mode.

    mode='real' ignores provider_name and reads the canonical validated store;
    mode='synthetic' (default) uses the provider/cache path. Passing
    provider_name='synthetic' with mode='real' is an error by construction.
    """
    mode = mode or "synthetic"
    if mode == "real":
        return _load_real(start, end)
    if mode != "synthetic":
        raise ValueError(f"unknown data mode '{mode}'")
    log.warning("SYNTHETIC DATA MODE — results are demonstrations of the "
                "machinery, not market history.")
    return _load_provider(provider_name, start, end, refresh)


# ------------------------------------------------------------ synthetic path --
def _load_provider(provider_name: str, start: date, end: date,
                   refresh: bool) -> MarketData:
    ucfg = load_universe_config()
    provider = get_provider(provider_name)
    store = DataStore(provider)

    tickers = ucfg.tickers + ucfg.benchmark_tickers
    frames: dict[str, pd.DataFrame] = {}
    issues: list[Issue] = []
    for t in tickers:
        try:
            df = store.get_prices(t, start, end, refresh=refresh)
        except Exception as exc:
            log.warning("no data for %s from %s: %s", t, provider_name, exc)
            issues.append(Issue(t, "fetch_failed", str(exc), "error"))
            continue
        issues.extend(validate_prices(t, df))
        frames[t] = df

    macro = pd.DataFrame()
    macro_cols = {}
    for sid in MACRO_SERIES:
        try:
            macro_cols[sid] = store.get_series(sid, start, end, refresh=refresh)
        except Exception as exc:
            log.warning("macro series %s unavailable: %s", sid, exc)
    if macro_cols:
        macro = pd.DataFrame(macro_cols)

    fund_frames = []
    for t in ucfg.tickers:
        f = store.get_fundamentals(t, refresh=refresh)
        if not f.empty:
            fund_frames.append(f.assign(ticker=t))
    fundamentals = (pd.concat(fund_frames, ignore_index=True)
                    if fund_frames else pd.DataFrame())

    md = _assemble(frames, macro, fundamentals, ucfg, issues,
                   provider_name, "synthetic")
    # Mirror the real store's optional short-volume panel so strategies that
    # read it can be exercised before a real run. Only for the synthetic
    # provider — a cached free-provider run has no business inventing one.
    if provider_name == "synthetic":
        from .synthetic import synthetic_short_share
        md.short_share = synthetic_short_share(list(md.adj_close.columns),
                                               md.calendar)
    return md


# ----------------------------------------------------------------- real path --
def _load_real(start: date, end: date) -> MarketData:
    """Load from the canonical real store. Hard-fails when coverage is
    insufficient; never substitutes synthetic data."""
    ucfg = load_universe_config()
    have = set(realstore.available("prices"))

    missing_bench = [b for b in REQUIRED_REAL_BENCHMARKS if b not in have]
    if missing_bench:
        raise realstore.RealDataMissing(
            f"required benchmarks missing from data/real/prices: {missing_bench}")
    univ_have = [t for t in ucfg.tickers if t in have]
    if len(univ_have) < MIN_REAL_NAMES:
        raise realstore.RealDataMissing(
            f"only {len(univ_have)} universe names in the real store "
            f"(need >= {MIN_REAL_NAMES}); run the downloader or import a bundle")

    frames: dict[str, pd.DataFrame] = {}
    issues: list[Issue] = []
    for t in sorted(have):
        if t not in ucfg.tickers and t not in ucfg.benchmark_tickers:
            continue
        df = realstore.read("prices", t)
        df = df.loc[str(start):str(end)]
        if df.empty:
            continue
        issues.extend(validate_prices(t, df))
        frames[t] = df[realstore.PRICE_COLUMNS]

    macro_cols = {}
    for sid in realstore.available("macro"):
        s = realstore.read("macro", sid)
        macro_cols[sid] = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    macro = pd.DataFrame(macro_cols) if macro_cols else pd.DataFrame()
    if "FEDFUNDS" not in macro.columns:
        log.warning("real store has no FEDFUNDS series — cash will earn 0%%")

    fund_frames = []
    for t in realstore.available("fundamentals"):
        f = realstore.read("fundamentals", t)
        if not f.empty:
            fund_frames.append(f.assign(ticker=t))
    fundamentals = (pd.concat(fund_frames, ignore_index=True)
                    if fund_frames else pd.DataFrame())

    # FINRA short-sale volume, if the store has it. Optional throughout: the
    # series starts 2009-07-31 and an absent panel must read as "no data",
    # never as zero short activity. The RATIO is derived here rather than
    # stored, so a change to its definition stays repairable from raw counts.
    sv_frames = {}
    for t in realstore.available("short_volume"):
        df = realstore.read("short_volume", t)
        if df.empty or "total_volume" not in df.columns:
            continue
        d = df.set_index(pd.to_datetime(df["date"]))
        denom = d["total_volume"].where(d["total_volume"] > 0)
        sv_frames[t] = d["short_volume"] / denom
    short_share = (pd.DataFrame(sv_frames).sort_index()
                   if sv_frames else pd.DataFrame())
    if not short_share.empty:
        log.info("short-sale volume: %d tickers, %s to %s",
                 short_share.shape[1], short_share.index.min().date(),
                 short_share.index.max().date())

    md = _assemble(frames, macro, fundamentals, ucfg, issues,
                   provider_name="real-store", data_mode="real")
    if not short_share.empty:
        md.short_share = short_share.reindex(md.calendar)
    return md
