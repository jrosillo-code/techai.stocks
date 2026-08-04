import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aitb.config import load_universe_config
from aitb.data.loader import MarketData


def make_toy_md(n_days: int = 300, tickers=("AAA", "BBB", "CCC"),
                daily_ret: float = 0.001) -> MarketData:
    """Tiny deterministic market: geometric drift, huge liquidity, no macro."""
    cal = pd.bdate_range("2020-01-01", periods=n_days, name="date")
    px = {}
    for i, t in enumerate(tickers):
        r = daily_ret * (i + 1)
        px[t] = 100.0 * (1 + r) ** np.arange(n_days)
    close = pd.DataFrame(px, index=cal)
    open_ = close.shift(1).bfill()  # open = yesterday's close (simple, known)
    dollar_volume = pd.DataFrame(1e12, index=cal, columns=list(tickers))
    macro = pd.DataFrame(index=cal)
    return MarketData(
        open=open_, high=close * 1.01, low=open_ * 0.99, close=close,
        adj_close=close.copy(), dollar_volume=dollar_volume, macro=macro,
        fundamentals=pd.DataFrame(), universe=load_universe_config(),
        provider_name="toy",
    )


@pytest.fixture(scope="session")
def toy_md() -> MarketData:
    return make_toy_md()


@pytest.fixture(scope="session")
def synth_md() -> MarketData:
    from aitb.data.loader import load_market_data
    return load_market_data("synthetic")
