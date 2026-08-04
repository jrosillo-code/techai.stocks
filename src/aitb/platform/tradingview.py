"""TradingView Pine Script v5 exports for portable strategies.

Only single-instrument logic is exported; cross-sectional and fundamental
strategies are marked incompatible rather than approximated dishonestly.
Every script carries a header warning that the Pine port is an APPROXIMATION
of the audited Python implementation (daily bars, no cost model parity, no
Treasury sleeve) and is for chart study — not trading.
"""
from __future__ import annotations

from pathlib import Path

from ..utils import get_logger

log = get_logger("platform.tradingview")

_HEADER = """// {name} — research export from the aitb platform
// APPROXIMATION of the audited Python implementation (freeze v2).
// Differences vs the backtest engine: no next-open fill convention is
// enforceable in a Pine study, no transaction-cost model, no Treasury
// fallback sleeve, single symbol only. For chart study, NOT trading.
//@version=5
"""


def _pine_qqq_ma(params: dict) -> str:
    w = params.get("window", 200)
    return _HEADER.format(name="QQQMovingAverage") + f"""
indicator("AITB QQQ {w}d trend gate", overlay=true)
sma_len = input.int({w}, "SMA window")
ma = ta.sma(close, sma_len)
riskOn = close > ma
plot(ma, "SMA", color=riskOn ? color.green : color.red, linewidth=2)
bgcolor(riskOn ? color.new(color.green, 92) : color.new(color.red, 92))
alertcondition(ta.crossover(close, ma), "Risk ON", "Price crossed above SMA")
alertcondition(ta.crossunder(close, ma), "Risk OFF", "Price crossed below SMA")
"""


def _pine_trendfollow(params: dict) -> str:
    w = params.get("sma_window", 200)
    return _HEADER.format(name="TrendFollowCash") + f"""
indicator("AITB per-symbol trend filter ({w}d)", overlay=true)
sma_len = input.int({w}, "SMA window")
ma = ta.sma(close, sma_len)
inTrend = close > ma
plot(ma, "SMA", color=inTrend ? color.green : color.red, linewidth=2)
bgcolor(inTrend ? color.new(color.green, 92) : color.new(color.red, 92))
// Python version equal-weights all in-trend names and parks the rest in IEF.
"""


def _pine_donchian(params: dict) -> str:
    entry = params.get("entry_window", 55)
    exit_ = params.get("exit_window", 20)
    return _HEADER.format(name="DonchianBreakout") + f"""
indicator("AITB Donchian breakout {entry}/{exit_}", overlay=true)
entryLen = input.int({entry}, "Entry channel")
exitLen = input.int({exit_}, "Exit channel")
hi = ta.highest(high[1], entryLen)
lo = ta.lowest(low[1], exitLen)
plot(hi, "Entry high", color=color.teal)
plot(lo, "Exit low", color=color.orange)
longSignal = close > hi
exitSignal = close < lo
plotshape(longSignal, style=shape.triangleup, location=location.belowbar,
          color=color.green, size=size.tiny)
plotshape(exitSignal, style=shape.triangledown, location=location.abovebar,
          color=color.red, size=size.tiny)
// Python version additionally requires QQQ above its 200d SMA (regime filter)
// and caps concurrent positions.
"""


def _pine_rsi_reversion(params: dict) -> str:
    entry = params.get("entry", 10)
    exit_ = params.get("exit", 60)
    trend = params.get("trend_window", 200)
    return _HEADER.format(name="RSIReversion") + f"""
indicator("AITB RSI(2) reversion (entry<{entry}, exit>{exit_})", overlay=false)
r = ta.rsi(close, 2)
maTrend = ta.sma(close, {trend})
uptrend = close > maTrend
buyZone = r < {entry} and uptrend
exitZone = r > {exit_} or not uptrend
plot(r, "RSI(2)")
hline({entry}, "entry"), hline({exit_}, "exit")
bgcolor(buyZone ? color.new(color.green, 85) : exitZone ? color.new(color.red, 92) : na)
// AUDITED CAVEAT: this family's edge did NOT survive realistic costs in
// backtests (stressed-cost Sharpe negative). Study signal, not a system.
"""


def _pine_vol_target(params: dict) -> str:
    tv = params.get("target_vol", 0.20)
    win = params.get("vol_window", 63)
    return _HEADER.format(name="VolTargetedBasket") + f"""
indicator("AITB vol-target exposure gauge ({int(tv*100)}% ann)", overlay=false)
volWin = input.int({win}, "Vol window")
target = input.float({tv}, "Target annual vol")
ret = close / close[1] - 1
realized = ta.stdev(ret, volWin) * math.sqrt(252)
exposure = math.min(target / math.max(realized, 0.0001), 1.0)
plot(realized, "Realized vol", color=color.orange)
plot(exposure, "Suggested exposure (0-1)", color=color.blue, linewidth=2)
hline(1.0)
// Python version applies this at the BASKET level with an IEF sleeve.
"""


def _pine_trend_plus_vol(params: dict) -> str:
    tv = params.get("target_vol", 0.20)
    trend = params.get("trend_window", 200)
    hyst = params.get("hysteresis", 0.02)
    return _HEADER.format(name="TrendPlusVolTarget") + f"""
indicator("AITB trend+vol overlay gauge", overlay=false)
maLen = input.int({trend}, "Trend SMA")
hyst = input.float({hyst}, "Hysteresis band")
target = input.float({tv}, "Target annual vol")
ma = ta.sma(close, maLen)
var bool gateOn = true
if close < ma * (1 - hyst)
    gateOn := false
if close > ma * (1 + hyst)
    gateOn := true
ret = close / close[1] - 1
realized = ta.stdev(ret, 63) * math.sqrt(252)
volScale = math.min(target / math.max(realized, 0.0001), 1.0)
exposure = (gateOn ? 1.0 : 0.2) * volScale
plot(exposure, "Suggested exposure (0-1)", color=color.blue, linewidth=2)
plot(realized, "Realized vol", color=color.orange)
hline(1.0)
"""


_GENERATORS = {
    "QQQMovingAverage": _pine_qqq_ma,
    "TrendFollowCash": _pine_trendfollow,
    "DonchianBreakout": _pine_donchian,
    "RSIReversion": _pine_rsi_reversion,
    "VolTargetedBasket": _pine_vol_target,
    "TrendPlusVolTarget": _pine_trend_plus_vol,
}


def export_pine(class_name: str, params: dict | None = None) -> str | None:
    gen = _GENERATORS.get(class_name)
    return gen(params or {}) if gen else None


def export_all(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, gen in _GENERATORS.items():
        path = out_dir / f"{name}.pine"
        path.write_text(gen({}))
        written.append(str(path))
    log.info("wrote %d Pine scripts to %s", len(written), out_dir)
    return written
