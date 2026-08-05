"""Deterministic synthetic market generator.

The execution environment for this project blocks all market-data hosts, so
this module generates a *stylized* but statistically realistic history for the
configured universe:

  * shared market / sector factor structure with regime-dependent drift & vol,
  * regime schedule shaped like the last ~35 years (dot-com boom/bust, GFC,
    QE bull, COVID crash, 2021 speculation, 2022 rate bear, 2023+ AI rally),
  * per-name profiles (beta, idiosyncratic vol, era alphas) so cross-sectional
    dispersion resembles reality (a few huge winners, many mediocre names,
    boom-bust names, and delistings),
  * IPO and delisting dates from the universe config are honored exactly,
  * stylized macro paths (fed funds, 10y yield, CPI YoY, unemployment) and a
    VIX proxy derived from realized factor vol,
  * quarterly fundamentals with realistic publication lags (45 days).

Everything is a pure function of (global seed, ticker), so runs are exactly
reproducible. THIS IS NOT HISTORICAL DATA — any backtest on it demonstrates
the research machinery, not actual market performance. Swap the provider to
``stooq``/``yahoo`` in a networked environment for real data.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..config import load_universe_config
from ..utils import stable_hash
from .providers import PRICE_COLUMNS, PriceProvider

GLOBAL_SEED = 20260804
START = date(1990, 1, 2)
END = date(2026, 7, 31)

# ---------------------------------------------------------------- regimes ---
# (start, end, market drift, market vol, tech extra drift, semis extra drift)
REGIMES = [
    ("early90s",    "1990-01-01", "1994-12-31", 0.07, 0.13, 0.02, 0.02),
    ("dotcom_boom", "1995-01-01", "2000-03-10", 0.16, 0.15, 0.18, 0.22),
    ("dotcom_bust", "2000-03-11", "2002-10-09", -0.22, 0.26, -0.38, -0.42),
    ("mid00s_bull", "2002-10-10", "2007-10-09", 0.12, 0.13, 0.04, 0.05),
    ("gfc",         "2007-10-10", "2009-03-09", -0.45, 0.42, -0.05, -0.10),
    ("qe_bull",     "2009-03-10", "2020-02-19", 0.13, 0.15, 0.05, 0.07),
    ("covid_crash", "2020-02-20", "2020-03-23", -1.60, 0.75, 0.10, 0.05),
    ("covid_boom",  "2020-03-24", "2021-12-31", 0.30, 0.22, 0.12, 0.18),
    ("rate_bear",   "2022-01-01", "2022-12-28", -0.20, 0.26, -0.15, -0.18),
    ("ai_rally",    "2022-12-29", "2026-07-31", 0.14, 0.16, 0.10, 0.22),
]

# Per-name profile: beta, idio vol, dict of regime -> annual alpha.
# Names not listed get sector defaults. Alphas are calibrated to produce
# realistic dispersion (NVDA-like mega-winner, SMCI-like boom/bust, dot-com
# casualties), not to reproduce actual returns.
PROFILES: dict[str, dict] = {
    "NVDA": {"beta": 1.6, "ivol": 0.42, "alpha": {"qe_bull": 0.10, "covid_boom": 0.25, "rate_bear": -0.20, "ai_rally": 0.55}},
    "AMD":  {"beta": 1.7, "ivol": 0.50, "alpha": {"dotcom_bust": -0.25, "qe_bull": 0.06, "covid_boom": 0.15, "ai_rally": 0.05}},
    "AVGO": {"beta": 1.3, "ivol": 0.30, "alpha": {"qe_bull": 0.10, "ai_rally": 0.25}},
    "TSM":  {"beta": 1.2, "ivol": 0.30, "alpha": {"qe_bull": 0.04, "ai_rally": 0.18}},
    "ASML": {"beta": 1.3, "ivol": 0.32, "alpha": {"qe_bull": 0.09, "covid_boom": 0.15, "rate_bear": -0.10, "ai_rally": 0.05}},
    "ARM":  {"beta": 1.6, "ivol": 0.55, "alpha": {"ai_rally": 0.20}},
    "MU":   {"beta": 1.5, "ivol": 0.45, "alpha": {"ai_rally": 0.10}},
    "MRVL": {"beta": 1.6, "ivol": 0.45, "alpha": {"dotcom_bust": -0.30, "ai_rally": 0.10}},
    "INTC": {"beta": 1.1, "ivol": 0.30, "alpha": {"dotcom_boom": 0.10, "qe_bull": -0.02, "covid_boom": -0.10, "ai_rally": -0.25}},
    "QCOM": {"beta": 1.3, "ivol": 0.35, "alpha": {"dotcom_boom": 0.30, "dotcom_bust": -0.30, "ai_rally": 0.0}},
    "AMAT": {"beta": 1.4, "ivol": 0.36, "alpha": {"ai_rally": 0.12}},
    "LRCX": {"beta": 1.4, "ivol": 0.36, "alpha": {"ai_rally": 0.12}},
    "MSFT": {"beta": 1.1, "ivol": 0.22, "alpha": {"dotcom_boom": 0.15, "dotcom_bust": -0.10, "qe_bull": 0.08, "ai_rally": 0.12}},
    "GOOGL": {"beta": 1.1, "ivol": 0.26, "alpha": {"qe_bull": 0.06, "ai_rally": 0.08}},
    "AMZN": {"beta": 1.3, "ivol": 0.32, "alpha": {"dotcom_boom": 0.40, "dotcom_bust": -0.45, "qe_bull": 0.14, "rate_bear": -0.15, "ai_rally": 0.06}},
    "META": {"beta": 1.3, "ivol": 0.34, "alpha": {"qe_bull": 0.10, "rate_bear": -0.35, "ai_rally": 0.25}},
    "AAPL": {"beta": 1.1, "ivol": 0.26, "alpha": {"qe_bull": 0.14, "covid_boom": 0.15, "ai_rally": 0.02}},
    "TSLA": {"beta": 1.8, "ivol": 0.55, "alpha": {"qe_bull": 0.20, "covid_boom": 0.60, "rate_bear": -0.45, "ai_rally": -0.05}},
    "ORCL": {"beta": 1.0, "ivol": 0.24, "alpha": {"dotcom_boom": 0.25, "dotcom_bust": -0.25, "ai_rally": 0.15}},
    "CRM":  {"beta": 1.3, "ivol": 0.32, "alpha": {"qe_bull": 0.10, "rate_bear": -0.15, "ai_rally": 0.02}},
    "ADBE": {"beta": 1.2, "ivol": 0.28, "alpha": {"qe_bull": 0.10, "ai_rally": -0.08}},
    "NOW":  {"beta": 1.3, "ivol": 0.32, "alpha": {"qe_bull": 0.12, "rate_bear": -0.15, "ai_rally": 0.10}},
    "SNOW": {"beta": 1.5, "ivol": 0.48, "alpha": {"covid_boom": 0.10, "rate_bear": -0.35, "ai_rally": -0.08}},
    "PLTR": {"beta": 1.7, "ivol": 0.55, "alpha": {"covid_boom": 0.05, "rate_bear": -0.40, "ai_rally": 0.45}},
    "CRWD": {"beta": 1.4, "ivol": 0.40, "alpha": {"covid_boom": 0.25, "rate_bear": -0.25, "ai_rally": 0.15}},
    "PANW": {"beta": 1.2, "ivol": 0.32, "alpha": {"qe_bull": 0.08, "ai_rally": 0.10}},
    "FTNT": {"beta": 1.2, "ivol": 0.34, "alpha": {"qe_bull": 0.10, "ai_rally": 0.02}},
    "ANET": {"beta": 1.3, "ivol": 0.36, "alpha": {"qe_bull": 0.12, "ai_rally": 0.22}},
    "SMCI": {"beta": 1.5, "ivol": 0.55, "alpha": {"qe_bull": 0.02, "ai_rally": 0.35, "smci_crash": -0.9}},
    "DELL": {"beta": 1.1, "ivol": 0.30, "alpha": {"ai_rally": 0.12}},
    "VRT":  {"beta": 1.4, "ivol": 0.42, "alpha": {"rate_bear": -0.25, "ai_rally": 0.35}},
    "CEG":  {"beta": 0.9, "ivol": 0.30, "alpha": {"ai_rally": 0.30}},
    "VST":  {"beta": 0.9, "ivol": 0.32, "alpha": {"ai_rally": 0.35}},
    # ---- live names added with the freeze-v3 universe expansion ----
    "TXN":  {"beta": 1.0, "ivol": 0.26, "alpha": {"dotcom_boom": 0.15, "dotcom_bust": -0.20, "qe_bull": 0.04}},
    "ADI":  {"beta": 1.1, "ivol": 0.28, "alpha": {"dotcom_boom": 0.18, "dotcom_bust": -0.25, "qe_bull": 0.04}},
    "NXPI": {"beta": 1.3, "ivol": 0.34, "alpha": {"qe_bull": 0.08}},
    "ON":   {"beta": 1.5, "ivol": 0.45, "alpha": {"dotcom_bust": -0.35, "gfc": -0.30, "ai_rally": -0.05}},
    "MCHP": {"beta": 1.2, "ivol": 0.32, "alpha": {"qe_bull": 0.05, "ai_rally": -0.10}},
    "SWKS": {"beta": 1.4, "ivol": 0.40, "alpha": {"qe_bull": 0.12, "ai_rally": -0.15}},
    "QRVO": {"beta": 1.4, "ivol": 0.42, "alpha": {"ai_rally": -0.18}},
    "MPWR": {"beta": 1.3, "ivol": 0.36, "alpha": {"qe_bull": 0.14, "ai_rally": 0.05}},
    "GFS":  {"beta": 1.3, "ivol": 0.38, "alpha": {"ai_rally": -0.05}},
    "WOLF": {"beta": 1.5, "ivol": 0.50, "alpha": {"dotcom_boom": 0.35, "dotcom_bust": -0.45, "covid_boom": 0.20, "ai_rally": -0.45}},
    "ALAB": {"beta": 1.7, "ivol": 0.60, "alpha": {"ai_rally": 0.25}},
    "CRDO": {"beta": 1.7, "ivol": 0.58, "alpha": {"ai_rally": 0.30}},
    "KLAC": {"beta": 1.3, "ivol": 0.34, "alpha": {"qe_bull": 0.06, "ai_rally": 0.14}},
    "TER":  {"beta": 1.3, "ivol": 0.36, "alpha": {"dotcom_boom": 0.20, "dotcom_bust": -0.35, "ai_rally": 0.05}},
    "ENTG": {"beta": 1.3, "ivol": 0.36, "alpha": {"qe_bull": 0.08, "rate_bear": -0.20}},
    "SNPS": {"beta": 1.0, "ivol": 0.26, "alpha": {"qe_bull": 0.08, "ai_rally": 0.10}},
    "CDNS": {"beta": 1.0, "ivol": 0.28, "alpha": {"dotcom_bust": -0.25, "qe_bull": 0.10, "ai_rally": 0.12}},
    "IBM":  {"beta": 0.9, "ivol": 0.22, "alpha": {"dotcom_boom": 0.05, "qe_bull": -0.06, "ai_rally": 0.08}},
    "NFLX": {"beta": 1.3, "ivol": 0.42, "alpha": {"qe_bull": 0.28, "covid_boom": 0.05, "rate_bear": -0.35, "ai_rally": 0.15}},
    "INTU": {"beta": 1.0, "ivol": 0.26, "alpha": {"qe_bull": 0.10, "ai_rally": 0.0}},
    "WDAY": {"beta": 1.3, "ivol": 0.34, "alpha": {"qe_bull": 0.08, "rate_bear": -0.20, "ai_rally": -0.05}},
    "DDOG": {"beta": 1.5, "ivol": 0.46, "alpha": {"covid_boom": 0.35, "rate_bear": -0.35, "ai_rally": 0.05}},
    "MDB":  {"beta": 1.5, "ivol": 0.48, "alpha": {"covid_boom": 0.30, "rate_bear": -0.35, "ai_rally": -0.10}},
    "NET":  {"beta": 1.6, "ivol": 0.52, "alpha": {"covid_boom": 0.45, "rate_bear": -0.45, "ai_rally": 0.05}},
    "TEAM": {"beta": 1.4, "ivol": 0.40, "alpha": {"covid_boom": 0.25, "rate_bear": -0.35, "ai_rally": -0.10}},
    "PATH": {"beta": 1.6, "ivol": 0.55, "alpha": {"rate_bear": -0.55, "ai_rally": -0.15}},
    "IOT":  {"beta": 1.5, "ivol": 0.50, "alpha": {"rate_bear": -0.40, "ai_rally": 0.10}},
    "SHOP": {"beta": 1.6, "ivol": 0.52, "alpha": {"qe_bull": 0.25, "covid_boom": 0.50, "rate_bear": -0.60, "ai_rally": 0.10}},
    "UBER": {"beta": 1.4, "ivol": 0.42, "alpha": {"covid_boom": 0.05, "rate_bear": -0.20, "ai_rally": 0.15}},
    "ABNB": {"beta": 1.3, "ivol": 0.40, "alpha": {"rate_bear": -0.25, "ai_rally": 0.0}},
    "APP":  {"beta": 1.8, "ivol": 0.62, "alpha": {"rate_bear": -0.60, "ai_rally": 0.60}},
    "TTD":  {"beta": 1.5, "ivol": 0.46, "alpha": {"qe_bull": 0.20, "covid_boom": 0.35, "rate_bear": -0.35, "ai_rally": 0.0}},
    "ZS":   {"beta": 1.5, "ivol": 0.46, "alpha": {"covid_boom": 0.35, "rate_bear": -0.35, "ai_rally": 0.05}},
    "OKTA": {"beta": 1.5, "ivol": 0.46, "alpha": {"covid_boom": 0.35, "rate_bear": -0.45, "ai_rally": -0.15}},
    "CYBR": {"beta": 1.3, "ivol": 0.38, "alpha": {"qe_bull": 0.05, "ai_rally": 0.15}},
    "S":    {"beta": 1.6, "ivol": 0.55, "alpha": {"rate_bear": -0.55, "ai_rally": -0.05}},
    "CSCO": {"beta": 1.1, "ivol": 0.28, "alpha": {"dotcom_boom": 0.45, "dotcom_bust": -0.55, "qe_bull": 0.0, "ai_rally": 0.05}},
    "KEYS": {"beta": 1.1, "ivol": 0.30, "alpha": {"qe_bull": 0.06}},
    "HPE":  {"beta": 1.1, "ivol": 0.32, "alpha": {"qe_bull": -0.05, "ai_rally": 0.15}},
    "NTAP": {"beta": 1.2, "ivol": 0.34, "alpha": {"dotcom_boom": 0.50, "dotcom_bust": -0.60, "qe_bull": 0.0}},
    "PSTG": {"beta": 1.4, "ivol": 0.42, "alpha": {"rate_bear": -0.25, "ai_rally": 0.20}},
    "WDC":  {"beta": 1.4, "ivol": 0.42, "alpha": {"dotcom_bust": -0.30, "gfc": -0.20, "ai_rally": 0.10}},
    "ISRG": {"beta": 1.1, "ivol": 0.32, "alpha": {"mid00s_bull": 0.35, "qe_bull": 0.12, "ai_rally": 0.05}},
    "ROK":  {"beta": 1.0, "ivol": 0.26, "alpha": {"qe_bull": 0.04}},
    "SYM":  {"beta": 1.7, "ivol": 0.60, "alpha": {"ai_rally": 0.10}},
    "NRG":  {"beta": 1.0, "ivol": 0.34, "alpha": {"gfc": -0.20, "ai_rally": 0.30}},
    "GEV":  {"beta": 1.1, "ivol": 0.36, "alpha": {"ai_rally": 0.40}},
    "PWR":  {"beta": 1.1, "ivol": 0.34, "alpha": {"qe_bull": 0.08, "ai_rally": 0.25}},

    # ---- Delisted / fallen names — boom-bust paths. ----
    # These exist so survivorship bias is VISIBLE rather than assumed away:
    # each is a name a momentum or quality screen would plausibly have bought
    # near its peak and then ridden into the delisting.
    "SUNW": {"beta": 1.5, "ivol": 0.40, "alpha": {"dotcom_boom": 0.45, "dotcom_bust": -0.65, "mid00s_bull": -0.15, "gfc": -0.25}},
    "EMC":  {"beta": 1.3, "ivol": 0.32, "alpha": {"dotcom_boom": 0.35, "dotcom_bust": -0.45}},
    "YHOO": {"beta": 1.5, "ivol": 0.42, "alpha": {"dotcom_boom": 0.55, "dotcom_bust": -0.60, "qe_bull": -0.02}},
    "MXIM": {"beta": 1.3, "ivol": 0.32, "alpha": {}},
    "XLNX": {"beta": 1.3, "ivol": 0.32, "alpha": {"covid_boom": 0.10}},
    "NT":   {"beta": 1.7, "ivol": 0.50, "alpha": {"dotcom_boom": 0.70, "dotcom_bust": -0.95, "mid00s_bull": -0.25, "gfc": -0.75}},
    "WCOM": {"beta": 1.4, "ivol": 0.45, "alpha": {"dotcom_boom": 0.35, "dotcom_bust": -1.20}},
    "SGI":  {"beta": 1.4, "ivol": 0.48, "alpha": {"dotcom_boom": 0.10, "dotcom_bust": -0.55, "mid00s_bull": -0.35, "gfc": -0.60}},
    "JDSU": {"beta": 1.8, "ivol": 0.62, "alpha": {"dotcom_boom": 0.95, "dotcom_bust": -1.05, "qe_bull": -0.05}},
    "LU":   {"beta": 1.5, "ivol": 0.45, "alpha": {"dotcom_boom": 0.45, "dotcom_bust": -0.85, "mid00s_bull": -0.05}},
    "GTW":  {"beta": 1.4, "ivol": 0.48, "alpha": {"dotcom_boom": 0.30, "dotcom_bust": -0.70, "mid00s_bull": -0.20}},
    "PALM": {"beta": 1.7, "ivol": 0.60, "alpha": {"dotcom_bust": -0.85, "mid00s_bull": -0.15, "qe_bull": -0.20}},
    "CPQ":  {"beta": 1.3, "ivol": 0.38, "alpha": {"dotcom_boom": 0.20, "dotcom_bust": -0.45}},
    "ALTR": {"beta": 1.3, "ivol": 0.34, "alpha": {"dotcom_boom": 0.35, "dotcom_bust": -0.45, "qe_bull": 0.04}},
    "BRCM": {"beta": 1.6, "ivol": 0.50, "alpha": {"dotcom_boom": 0.85, "dotcom_bust": -0.95, "qe_bull": 0.10}},
    "ATML": {"beta": 1.5, "ivol": 0.48, "alpha": {"dotcom_boom": 0.45, "dotcom_bust": -0.75, "qe_bull": 0.0}},
    "LSI":  {"beta": 1.4, "ivol": 0.44, "alpha": {"dotcom_boom": 0.35, "dotcom_bust": -0.65, "qe_bull": -0.02}},
    "FCS":  {"beta": 1.4, "ivol": 0.42, "alpha": {"dotcom_bust": -0.30, "gfc": -0.35}},
    "IDTI": {"beta": 1.4, "ivol": 0.42, "alpha": {"dotcom_boom": 0.40, "dotcom_bust": -0.70, "qe_bull": 0.05}},
    "CY":   {"beta": 1.4, "ivol": 0.42, "alpha": {"dotcom_boom": 0.35, "dotcom_bust": -0.60, "qe_bull": 0.0}},
    "RHT":  {"beta": 1.4, "ivol": 0.42, "alpha": {"dotcom_boom": 0.60, "dotcom_bust": -0.85, "qe_bull": 0.10}},
    "VMW":  {"beta": 1.2, "ivol": 0.34, "alpha": {"qe_bull": 0.02, "ai_rally": 0.10}},
    "ATVI": {"beta": 1.0, "ivol": 0.32, "alpha": {"mid00s_bull": 0.20, "qe_bull": 0.08}},
    "SPLK": {"beta": 1.4, "ivol": 0.44, "alpha": {"qe_bull": 0.05, "covid_boom": 0.10, "rate_bear": -0.30}},
    "ANSS": {"beta": 1.0, "ivol": 0.28, "alpha": {"qe_bull": 0.10, "ai_rally": 0.05}},
    "MENT": {"beta": 1.2, "ivol": 0.36, "alpha": {"dotcom_bust": -0.30, "qe_bull": 0.04}},
    "CA":   {"beta": 1.0, "ivol": 0.30, "alpha": {"dotcom_boom": 0.20, "dotcom_bust": -0.40, "qe_bull": -0.02}},
    "NOVL": {"beta": 1.2, "ivol": 0.40, "alpha": {"dotcom_boom": 0.15, "dotcom_bust": -0.55, "mid00s_bull": -0.20}},
    "NUAN": {"beta": 1.3, "ivol": 0.42, "alpha": {"mid00s_bull": 0.15, "qe_bull": 0.02}},
    "LNKD": {"beta": 1.5, "ivol": 0.48, "alpha": {"qe_bull": 0.08}},
    "TWTR": {"beta": 1.4, "ivol": 0.48, "alpha": {"qe_bull": -0.10, "covid_boom": 0.20, "rate_bear": -0.10}},
    "JNPR": {"beta": 1.5, "ivol": 0.45, "alpha": {"dotcom_boom": 0.80, "dotcom_bust": -1.00, "qe_bull": -0.06}},
    "BRCD": {"beta": 1.6, "ivol": 0.52, "alpha": {"dotcom_boom": 0.75, "dotcom_bust": -0.95, "qe_bull": -0.04}},
    "TLAB": {"beta": 1.5, "ivol": 0.46, "alpha": {"dotcom_boom": 0.55, "dotcom_bust": -0.85, "mid00s_bull": -0.15, "qe_bull": -0.15}},
    "MFE":  {"beta": 1.3, "ivol": 0.42, "alpha": {"dotcom_boom": 0.35, "dotcom_bust": -0.55, "qe_bull": 0.05}},
    "FEYE": {"beta": 1.6, "ivol": 0.55, "alpha": {"qe_bull": -0.20, "covid_boom": 0.15, "rate_bear": -0.20}},
    "PFPT": {"beta": 1.4, "ivol": 0.44, "alpha": {"qe_bull": 0.12, "covid_boom": 0.20}},
    "CTXS": {"beta": 1.2, "ivol": 0.36, "alpha": {"dotcom_boom": 0.55, "dotcom_bust": -0.80, "qe_bull": 0.0}},
    "ZEN":  {"beta": 1.5, "ivol": 0.48, "alpha": {"covid_boom": 0.25, "rate_bear": -0.30}},
}

SECTOR_DEFAULTS = {"beta": 1.2, "ivol": 0.35, "alpha": {}}

SECTOR_FACTOR = {
    "semiconductors": "semis", "semi_equipment": "semis", "eda": "semis",
    "software": "tech", "internet": "tech", "hardware": "tech",
    "cybersecurity": "tech", "networking": "semis",
    "dc_infrastructure": "semis", "power": "market", "ai_consumer": "tech",
    "robotics": "tech",
}

# ETF definitions: (factor, beta to that factor, idio vol)
ETF_DEFS = {
    "SPY":  ("market", 1.00, 0.00),
    "QQQ":  ("tech_index", 1.00, 0.00),
    "XLK":  ("tech_index", 1.02, 0.02),
    "SOXX": ("semis_index", 1.00, 0.02),
    "SMH":  ("semis_index", 1.02, 0.02),
    "IGV":  ("tech_index", 1.05, 0.04),
    "IEF":  ("bonds", 1.00, 0.00),
    "BIL":  ("cash", 1.00, 0.00),
    # added with the freeze-v3 benchmark expansion
    "VGT":  ("tech_index", 1.00, 0.01),
    "IWM":  ("market", 1.05, 0.06),
    "ARKK": ("tech_index", 1.55, 0.22),
    "SKYY": ("tech_index", 1.10, 0.05),
    "CIBR": ("tech_index", 1.05, 0.06),
    "BOTZ": ("semis_index", 0.85, 0.06),
}

# Stylized macro paths: (date, value) knots, forward-filled. These roughly
# track the shape of the real series; they are NOT the actual data.
FEDFUNDS_KNOTS = [
    ("1990-01-01", 8.2), ("1992-09-01", 3.0), ("1994-02-01", 3.25),
    ("1995-02-01", 6.0), ("1998-11-01", 4.75), ("2000-05-01", 6.5),
    ("2001-01-03", 6.0), ("2001-12-01", 1.75), ("2003-06-01", 1.0),
    ("2004-06-01", 1.25), ("2006-06-01", 5.25), ("2007-09-01", 4.75),
    ("2008-12-16", 0.15), ("2015-12-16", 0.4), ("2018-12-19", 2.4),
    ("2019-10-30", 1.65), ("2020-03-15", 0.05), ("2022-03-16", 0.33),
    ("2022-12-14", 4.33), ("2023-07-26", 5.33), ("2024-09-18", 4.83),
    ("2025-06-01", 4.1), ("2026-03-01", 3.9),
]
DGS10_KNOTS = [
    ("1990-01-01", 8.0), ("1993-10-01", 5.3), ("1994-11-01", 8.0),
    ("1998-10-01", 4.4), ("2000-01-01", 6.6), ("2003-06-01", 3.3),
    ("2007-06-01", 5.2), ("2008-12-01", 2.1), ("2013-12-01", 3.0),
    ("2016-07-01", 1.4), ("2018-11-01", 3.2), ("2020-08-01", 0.55),
    ("2022-10-01", 4.2), ("2023-10-01", 4.9), ("2024-09-01", 3.7),
    ("2026-03-01", 4.2),
]
CPI_YOY_KNOTS = [
    ("1990-01-01", 5.2), ("1992-01-01", 3.0), ("1997-01-01", 2.3),
    ("2008-07-01", 5.5), ("2009-07-01", -1.5), ("2012-01-01", 2.9),
    ("2015-01-01", 0.1), ("2019-01-01", 1.9), ("2021-03-01", 2.6),
    ("2022-06-01", 9.0), ("2023-06-01", 3.0), ("2024-09-01", 2.4),
    ("2026-03-01", 2.6),
]
UNRATE_KNOTS = [
    ("1990-01-01", 5.4), ("1992-06-01", 7.8), ("2000-04-01", 3.8),
    ("2003-06-01", 6.3), ("2007-05-01", 4.4), ("2009-10-01", 10.0),
    ("2019-12-01", 3.5), ("2020-04-01", 14.7), ("2021-12-01", 3.9),
    ("2024-07-01", 4.3), ("2026-03-01", 4.2),
]


def trading_calendar(start: date = START, end: date = END) -> pd.DatetimeIndex:
    """Business-day calendar (US holidays not modeled; documented limitation)."""
    return pd.bdate_range(start, end, name="date")


def _knots_to_series(knots: list[tuple[str, float]], cal: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series({pd.Timestamp(d): v for d, v in knots})
    return s.reindex(cal.union(s.index)).interpolate(method="time").reindex(cal).ffill()


class _FactorMarket:
    """Singleton-ish factor return panel shared by all tickers."""

    _cache: dict[int, "_FactorMarket"] = {}

    def __init__(self, seed: int):
        self.cal = trading_calendar()
        n = len(self.cal)
        rng = np.random.default_rng(seed)
        ann = 252.0

        regime_id = np.zeros(n, dtype=int)
        drift = np.zeros(n)
        vol = np.zeros(n)
        tech_a = np.zeros(n)
        semi_a = np.zeros(n)
        for i, (name, s, e, mu, sig, ta, sa) in enumerate(REGIMES):
            mask = (self.cal >= s) & (self.cal <= e)
            regime_id[mask] = i
            drift[mask] = mu / ann
            vol[mask] = sig / np.sqrt(ann)
            tech_a[mask] = ta / ann
            semi_a[mask] = sa / ann
        self.regime_names = np.array([REGIMES[i][0] for i in regime_id])

        # Market factor: regime drift/vol + fat tails + EWMA vol clustering.
        z = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)
        cluster = np.ones(n)
        lam = 0.94
        for t in range(1, n):
            cluster[t] = np.sqrt(lam * cluster[t - 1] ** 2 + (1 - lam) * z[t - 1] ** 2)
        cluster = np.clip(cluster, 0.6, 2.5)
        self.market = drift + vol * cluster * z

        # Sector factors: correlated with market via their own shocks + alpha.
        self.tech = tech_a + 0.5 * vol * rng.standard_normal(n) + 0.25 * (self.market - drift)
        self.semis = semi_a + 0.8 * vol * rng.standard_normal(n) + 0.30 * (self.market - drift)

        # Bond factor: negatively loaded on 10y yield changes.
        dgs10 = _knots_to_series(DGS10_KNOTS, self.cal)
        dy = dgs10.diff().fillna(0.0).to_numpy()
        self.bonds = 0.03 / ann - 7.0 * dy / 100.0 / 5.0 + 0.002 * rng.standard_normal(n)

        fed = _knots_to_series(FEDFUNDS_KNOTS, self.cal)
        self.cash = (fed / 100.0 / ann).to_numpy()

        # Cap-weight style indices used by ETFs (market + sector tilt).
        self.tech_index = self.market + 0.55 * self.tech
        self.semis_index = self.market + 0.85 * self.semis

        self.fed = fed
        self.dgs10 = dgs10

    @classmethod
    def get(cls, seed: int = GLOBAL_SEED) -> "_FactorMarket":
        if seed not in cls._cache:
            cls._cache[seed] = cls(seed)
        return cls._cache[seed]


class SyntheticProvider(PriceProvider):
    name = "synthetic"

    def __init__(self, seed: int = GLOBAL_SEED):
        self.seed = seed
        self.market = _FactorMarket.get(seed)
        self._ucfg = load_universe_config()

    # ------------------------------------------------------------ equities --
    def fetch_daily(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        mkt = self.market
        cal = mkt.cal

        if ticker in ETF_DEFS:
            df = self._etf_bars(ticker)
        else:
            df = self._stock_bars(ticker)

        return df.loc[str(start):str(end)]

    def _rng(self, ticker: str, salt: str = "") -> np.random.Generator:
        return np.random.default_rng(int(stable_hash([self.seed, ticker, salt], 12), 16))

    def _listing_window(self, ticker: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        try:
            sec = self._ucfg.security(ticker)
            ipo = pd.Timestamp(sec.ipo)
            end = pd.Timestamp(sec.delisted) if sec.delisted else pd.Timestamp(END)
        except KeyError:
            bmk = {b.ticker: b for b in self._ucfg.benchmarks}.get(ticker)
            ipo = pd.Timestamp(bmk.ipo) if bmk else pd.Timestamp(START)
            end = pd.Timestamp(END)
        return max(ipo, pd.Timestamp(START)), min(end, pd.Timestamp(END))

    def _stock_bars(self, ticker: str) -> pd.DataFrame:
        mkt = self.market
        cal = mkt.cal
        prof = PROFILES.get(ticker, SECTOR_DEFAULTS)
        try:
            sector = self._ucfg.security(ticker).sector
        except KeyError:
            sector = "software"
        factor_name = SECTOR_FACTOR.get(sector, "tech")
        factor = {"tech": mkt.tech, "semis": mkt.semis, "market": np.zeros(len(cal))}[factor_name]

        rng = self._rng(ticker)
        n = len(cal)
        ann = 252.0

        alpha = np.zeros(n)
        for regime, a in prof.get("alpha", {}).items():
            if regime == "smci_crash":  # idiosyncratic accounting-scare style crash
                mask = (cal >= "2024-08-01") & (cal <= "2025-02-28")
            else:
                mask = mkt.regime_names == regime
            alpha[mask] += a / ann

        ivol = prof["ivol"] / np.sqrt(ann)
        idio = ivol * rng.standard_t(df=4, size=n) / np.sqrt(2.0)
        # Occasional earnings-style jumps (~4/yr).
        jumps = (rng.random(n) < 4 / ann) * rng.standard_normal(n) * 4 * ivol
        r = prof["beta"] * mkt.market + 0.8 * factor + alpha + idio + jumps
        r = np.clip(r, -0.45, 0.60)

        start_ts, end_ts = self._listing_window(ticker)
        live = (cal >= start_ts) & (cal <= end_ts)
        r = np.where(live, r, 0.0)

        p0 = 15.0 + 30.0 * rng.random()
        adj_close = p0 * np.exp(np.cumsum(np.log1p(r)))

        # Dividend yield for mature names -> close (price) drifts below
        # adj_close (total return). Growth names pay nothing.
        div_yield = {"AAPL": 0.008, "MSFT": 0.010, "ORCL": 0.012, "INTC": 0.02,
                     "QCOM": 0.02, "TSM": 0.015, "AVGO": 0.015, "MU": 0.004,
                     "EMC": 0.01, "MXIM": 0.02, "XLNX": 0.015}.get(ticker, 0.0)
        price_factor = np.exp(-div_yield / ann * np.arange(n))
        close = adj_close * price_factor

        gap = np.abs(rng.standard_normal(n)) * 0.4 * ivol
        intraday = np.abs(rng.standard_normal(n)) * 0.7 * ivol
        open_ = close / np.exp(rng.standard_normal(n) * 0.5 * ivol)
        high = np.maximum(open_, close) * np.exp(intraday)
        low = np.minimum(open_, close) / np.exp(gap + intraday * 0.5)

        # Dollar volume grows with era and spikes with |return|.
        base_adv = {"NVDA": 2e10, "TSLA": 2e10, "AAPL": 1.5e10, "MSFT": 1e10,
                    "AMZN": 8e9, "META": 8e9, "GOOGL": 7e9, "AMD": 6e9,
                    "SMCI": 3e9, "PLTR": 3e9}.get(ticker, 8e8)
        era_scale = np.linspace(0.05, 1.0, n) ** 1.5
        dollar_vol = base_adv * era_scale * np.exp(0.6 * rng.standard_normal(n)) * (1 + 8 * np.abs(r))
        volume = dollar_vol / np.maximum(close, 0.01)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close,
             "adj_close": adj_close, "volume": volume},
            index=cal,
        )
        return df.loc[live]

    def _etf_bars(self, ticker: str) -> pd.DataFrame:
        mkt = self.market
        cal = mkt.cal
        factor_name, beta, ivol_ann = ETF_DEFS[ticker]
        factor = getattr(mkt, factor_name)
        rng = self._rng(ticker)
        n = len(cal)
        ann = 252.0

        er = {b.ticker: b.expense_ratio for b in self._ucfg.benchmarks}.get(ticker, 0.0)
        idio = (ivol_ann / np.sqrt(ann)) * rng.standard_normal(n) if ivol_ann else np.zeros(n)
        r = beta * np.asarray(factor) + idio - er / ann

        start_ts, end_ts = self._listing_window(ticker)
        live = (cal >= start_ts) & (cal <= end_ts)
        r = np.where(live, r, 0.0)

        adj_close = 50.0 * np.exp(np.cumsum(np.log1p(np.clip(r, -0.5, 0.5))))
        # ETFs distribute ~1.3% (SPY-like) — tiny for QQQ, ~cash rate for BIL.
        div_yield = {"SPY": 0.015, "QQQ": 0.006, "XLK": 0.008, "IEF": 0.025,
                     "BIL": 0.0}.get(ticker, 0.005)
        close = adj_close * np.exp(-div_yield / ann * np.arange(n))
        spread = 0.0005
        open_ = close * np.exp(rng.standard_normal(n) * spread)
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        volume = 5e9 / np.maximum(close, 1.0) * np.exp(0.3 * rng.standard_normal(n))

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close,
             "adj_close": adj_close, "volume": volume},
            index=cal,
        )
        return df.loc[live]

    # ---------------------------------------------------------------- macro --
    def fetch_series(self, series_id: str, start: date, end: date) -> pd.Series:
        mkt = self.market
        cal = mkt.cal
        if series_id == "FEDFUNDS":
            s = mkt.fed
        elif series_id == "DGS10":
            s = mkt.dgs10
        elif series_id == "T10Y2Y":
            # Stylized curve slope: 10y minus a short rate tracking fed funds.
            s = mkt.dgs10 - (mkt.fed * 0.9 + 0.2)
        elif series_id == "CPIYOY":
            s = _knots_to_series(CPI_YOY_KNOTS, cal)
        elif series_id == "UNRATE":
            s = _knots_to_series(UNRATE_KNOTS, cal)
        elif series_id == "VIXCLS":
            rv = pd.Series(mkt.market, index=cal).rolling(21).std() * np.sqrt(252) * 100
            rng = np.random.default_rng(self.seed + 7)
            s = (rv * 1.25 + 4 + pd.Series(2.0 * rng.standard_normal(len(cal)), index=cal)).clip(9, 90)
            s = s.bfill()
        elif series_id == "BAA10Y":
            # Credit spread proxy: widens when market vol is high.
            rv = pd.Series(mkt.market, index=cal).rolling(63).std() * np.sqrt(252)
            s = (1.5 + 6.0 * (rv - 0.12).clip(lower=0)).bfill()
        else:
            raise KeyError(f"synthetic provider has no series '{series_id}'")
        s = pd.Series(np.asarray(s, dtype=float), index=cal, name=series_id)
        return s.loc[str(start):str(end)]

    # ---------------------------------------------------------- fundamentals --
    def fetch_fundamentals(self, ticker: str) -> pd.DataFrame:
        """Quarterly fundamentals with publication lag.

        Columns: period_end, published (period_end + ~45 days), revenue,
        eps, fcf, shares. `published` is the ONLY date at which the row may
        enter a backtest.
        """
        start_ts, end_ts = self._listing_window(ticker)
        q_ends = pd.date_range(start_ts, end_ts, freq="QE")
        if len(q_ends) == 0:
            return pd.DataFrame(columns=["period_end", "published", "revenue", "eps", "fcf", "shares"])
        rng = self._rng(ticker, "fundamentals")
        n = len(q_ends)

        prof = PROFILES.get(ticker, SECTOR_DEFAULTS)
        mkt = self.market
        # Quarterly revenue growth loosely mirrors the name's era alpha profile.
        g = np.full(n, 0.02)
        for i, qe in enumerate(q_ends):
            pos = mkt.cal.searchsorted(qe)
            regime = mkt.regime_names[min(pos, len(mkt.regime_names) - 1)]
            g[i] += prof.get("alpha", {}).get(regime, 0.0) / 3.0
        g += 0.03 * rng.standard_normal(n)
        rev0 = 100.0 * (1 + 4 * rng.random())
        revenue = rev0 * np.cumprod(1 + np.clip(g, -0.3, 0.5))
        margin = np.clip(0.12 + 0.05 * rng.standard_normal(n) + np.linspace(0, 0.08, n), -0.2, 0.45)
        fcf = revenue * margin
        shares = 1000.0 * np.cumprod(1 + np.full(n, 0.002) + 0.001 * rng.standard_normal(n))
        eps = fcf * 0.9 / shares * 1000

        lag = (40 + rng.integers(0, 21, size=n)).astype("timedelta64[D]")
        published = q_ends.to_numpy() + lag

        # v4 fields, mirroring what EDGAR now collects so the two modes carry
        # the same schema. Sector-plausible ratios with per-name dispersion:
        # semiconductor firms run lower gross margin and far heavier capital
        # and R&D intensity than software firms.
        try:
            sector = self._ucfg.security(ticker).sector
        except KeyError:
            sector = "software"
        hardware_like = sector in ("semiconductors", "semi_equipment", "hardware",
                                   "dc_infrastructure", "networking", "robotics")
        gm_base = 0.45 if hardware_like else 0.72
        rd_base = 0.16 if hardware_like else 0.20
        gm = np.clip(gm_base + 0.08 * rng.standard_normal(n), 0.10, 0.95)
        gross_profit = revenue * gm
        cost_of_revenue = revenue - gross_profit
        rnd = revenue * np.clip(rd_base + 0.05 * rng.standard_normal(n), 0.01, 0.5)
        net_income = fcf * np.clip(0.85 + 0.15 * rng.standard_normal(n), 0.2, 1.6)
        # Asset turnover ~0.6-0.9/yr on quarterly revenue.
        assets = revenue * np.clip(4.0 + 1.2 * rng.standard_normal(n), 1.2, 12.0)
        equity = assets * np.clip(0.55 + 0.12 * rng.standard_normal(n), 0.1, 0.95)
        debt = assets * np.clip(0.18 + 0.10 * rng.standard_normal(n), 0.0, 0.6)
        cash = assets * np.clip(0.22 + 0.10 * rng.standard_normal(n), 0.01, 0.7)

        return pd.DataFrame(
            {"period_end": q_ends, "published": pd.DatetimeIndex(published),
             "revenue": revenue, "eps": eps, "fcf": fcf, "shares": shares,
             "gross_profit": gross_profit, "cost_of_revenue": cost_of_revenue,
             "assets": assets, "equity": equity, "net_income": net_income,
             "rnd": rnd, "debt": debt, "cash": cash}
        )
