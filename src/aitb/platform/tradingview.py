"""TradingView Pine Script v5 exports for portable strategies.

Only single-instrument logic is exported. Cross-sectional strategies (which
rank the whole universe against itself) and fundamental strategies (which need
point-in-time filings) are marked incompatible rather than approximated
dishonestly — a Pine script that quietly drops the ranking step is a different
strategy wearing the same name.

Entry/exit rules export as full ``strategy()`` scripts, not ``indicator()``, so
TradingView's Strategy Tester runs them and marks every trade on the chart.
Orders are placed on bar close and fill at the NEXT bar's open, which is the
same convention the Python engine enforces — that part does transfer.

What does NOT transfer, and is printed on every script:
  * no transaction-cost model (the study charges commission, half spread,
    slippage, market impact and borrow across four scenarios),
  * one symbol at a time, where the study holds a portfolio,
  * no Treasury sleeve — Pine sits in cash, the study earns the bill rate,
  * no participation cap against average daily volume.

Strategy Tester numbers will therefore NOT match the study, and are generally
more flattering. These are for watching a signal behave on a chart, not for
judging it.

Two headers, because two different things are being exported. Most of these
are portable LEFTOVERS of families built for portfolios: the signal itself is
an approximation, and they carry ``_HEADER``. The ``chartable`` family (freeze
v5/v6) was designed single-symbol from the start, so its signal is the tested
rule bar for bar and only the accounting differs; those carry
``_HEADER_EXACT``. Flattening the two into one notice would overstate the gap
for half of them, which is the direction that teaches a reader to stop reading
the notice at all.

Position-sizing and regime rules export as ``indicator()`` gauges instead:
they say how much to hold, not what to buy, and dressing them up as entry
signals would misrepresent them.
"""
from __future__ import annotations

from pathlib import Path

from ..utils import get_logger

log = get_logger("platform.tradingview")

_HEADER = """// {name} — research export from the aitb platform
// {tagline}
//
// APPROXIMATION of the audited Python implementation.
// It models no transaction costs at all: no commission, no spread, no
// slippage, no market impact, no borrow. It trades one symbol where the study
// holds a portfolio, has no Treasury sleeve for idle cash, and applies no
// participation cap against average daily volume. Strategy Tester results will
// not match the study and are usually more flattering.
// For chart study, NOT for trading.
//
// Fills: orders are placed on close and filled at the next bar's open, which
// matches the engine's execution convention.
//@version=5
"""

# The `chartable` family (freeze v5/v6) was designed single-symbol from the
# start, so its SIGNAL is not an approximation — it is the tested rule, bar for
# bar. Only the accounting differs. Saying "APPROXIMATION" flatly for these
# would be false in the direction that teaches a reader to stop reading the
# notice, which is the one direction that costs something.
_HEADER_EXACT = """// {name} — research export from the aitb platform
// {tagline}
//
// THE RULE HERE IS THE TESTED RULE, bar for bar — this family was designed
// single-symbol from the start, so nothing was dropped to fit it on a chart.
// The APPROXIMATION is in the accounting, and it is large:
// it models no transaction costs at all: no commission, no spread, no
// slippage, no market impact, no borrow. It trades one symbol where the study
// holds every name that qualifies, and has no Treasury sleeve for idle cash.
// Strategy Tester results will not match the study and are usually more
// flattering. For chart study, NOT for trading.
//
// Fills: orders are placed on close and filled at the next bar's open, which
// matches the engine's execution convention.
//@version=5
"""

# Provenance drawn on the chart, so a screenshot cannot lose the caveat.
_TABLE = """
var table info = table.new(position.top_right, 2, 4, border_width=1)
if barstate.islast
    table.cell(info, 0, 0, "aitb export", text_color=color.white,
               bgcolor=color.new(color.blue, 30), text_size=size.tiny)
    table.cell(info, 1, 0, "{label}", text_color=color.white,
               bgcolor=color.new(color.blue, 30), text_size=size.tiny)
    table.cell(info, 0, 1, "settings", text_size=size.tiny)
    table.cell(info, 1, 1, "{settings}", text_size=size.tiny)
    table.cell(info, 0, 2, "tested on", text_size=size.tiny)
    table.cell(info, 1, 2, "{tested}", text_size=size.tiny)
    table.cell(info, 0, 3, "costs", text_size=size.tiny)
    table.cell(info, 1, 3, "NOT modelled here", text_size=size.tiny,
               text_color=color.red)
"""

_MARKERS = """
plotshape(longEntry, title="Entry", style=shape.triangleup,
          location=location.belowbar, color=color.new(color.green, 0),
          size=size.small, text="BUY")
plotshape(longExit, title="Exit", style=shape.triangledown,
          location=location.abovebar, color=color.new(color.red, 0),
          size=size.small, text="EXIT")
"""


def _strategy_open(title: str) -> str:
    return (f'strategy("{title}", overlay=true, initial_capital=100000,\n'
            f'         default_qty_type=strategy.percent_of_equity,\n'
            f'         default_qty_value=100, process_orders_on_close=false,\n'
            f'         calc_on_every_tick=false)\n')


def _pine_qqq_ma(params: dict) -> str:
    w = params.get("window", 200)
    return (_HEADER.format(name="QQQMovingAverage",
                           tagline="Hold the index only while it is above its long moving average.")
            + _strategy_open(f"AITB QQQ {w}d trend gate") + f"""
sma_len = input.int({w}, "Moving-average window", minval=20)
ma = ta.sma(close, sma_len)
riskOn = close > ma

longEntry = riskOn and not riskOn[1]
longExit  = not riskOn and riskOn[1]

if longEntry
    strategy.entry("Long", strategy.long)
if longExit
    strategy.close("Long")

plot(ma, "SMA", color=riskOn ? color.green : color.red, linewidth=2)
bgcolor(riskOn ? color.new(color.green, 92) : color.new(color.red, 92), title="Regime")
{_MARKERS}
alertcondition(longEntry, "Risk ON", "Price crossed above its moving average")
alertcondition(longExit, "Risk OFF", "Price crossed below its moving average")
{_TABLE.format(label="QQQ trend gate", settings=f"{w}-day SMA",
               tested="SPY QQQ XLK SOXX SMH IGV")}
// The study parks exited capital in Treasuries (IEF); Pine sits in cash.
""")


def _pine_trendfollow(params: dict) -> str:
    w = params.get("sma_window", 200)
    return (_HEADER.format(name="TrendFollowCash",
                           tagline="Hold each stock only while it is above its own long moving average.")
            + _strategy_open(f"AITB per-symbol trend filter ({w}d)") + f"""
sma_len = input.int({w}, "Moving-average window", minval=20)
ma = ta.sma(close, sma_len)
inTrend = close > ma

longEntry = inTrend and not inTrend[1]
longExit  = not inTrend and inTrend[1]

if longEntry
    strategy.entry("Long", strategy.long)
if longExit
    strategy.close("Long")

plot(ma, "SMA", color=inTrend ? color.green : color.red, linewidth=2)
bgcolor(inTrend ? color.new(color.green, 92) : color.new(color.red, 92), title="In trend")
{_MARKERS}
alertcondition(longEntry, "Trend ON", "Closed above its moving average")
alertcondition(longExit, "Trend OFF", "Closed below its moving average")
{_TABLE.format(label="Per-stock trend", settings=f"{w}-day SMA",
               tested="every name in the universe, independently")}
// The study runs this on EVERY name at once, equal-weights whatever is in
// trend, and parks the remainder in Treasuries. One symbol shows the signal,
// not the portfolio.
""")


def _pine_donchian(params: dict) -> str:
    entry = params.get("entry_window", 55)
    exit_ = params.get("exit_window", 20)
    return (_HEADER.format(name="DonchianBreakout",
                           tagline="Buy a multi-month high, sell a multi-week low.")
            + _strategy_open(f"AITB Donchian breakout {entry}/{exit_}") + f"""
entryLen  = input.int({entry}, "Entry channel (days)", minval=5)
exitLen   = input.int({exit_}, "Exit channel (days)", minval=5)
useRegime = input.bool(true, "Require QQQ above its 200-day average")

hi = ta.highest(high[1], entryLen)
lo = ta.lowest(low[1], exitLen)

qqq   = request.security("NASDAQ:QQQ", "D", close)
qqqMa = request.security("NASDAQ:QQQ", "D", ta.sma(close, 200))
riskOn = not useRegime or qqq > qqqMa

longEntry = close > hi and riskOn
longExit  = close < lo or (useRegime and not riskOn)

if longEntry
    strategy.entry("Long", strategy.long)
if longExit
    strategy.close("Long")

plot(hi, "Entry high", color=color.teal)
plot(lo, "Exit low", color=color.orange)
bgcolor(riskOn ? na : color.new(color.red, 90), title="Risk-off")
{_MARKERS}
alertcondition(longEntry, "Breakout", "New high with a risk-on regime")
alertcondition(longExit, "Breakdown", "New low, or the regime turned off")
{_TABLE.format(label="Breakout", settings=f"{entry}d high / {exit_}d low",
               tested="each name, with a QQQ regime filter")}
// The UNFILTERED version was deprecated in the study: it whipsawed through
// bear markets and was dominated by this filtered twin at every entry window.
// The study also caps how many positions run at once.
""")


def _pine_rsi_reversion(params: dict) -> str:
    entry = params.get("entry", 10)
    exit_ = params.get("exit", 60)
    trend = params.get("trend_window", 200)
    return (_HEADER.format(name="RSIReversion",
                           tagline="Buy short-term weakness inside a long-term uptrend.")
            + _strategy_open(f"AITB RSI(2) reversion <{entry}") + f"""
entryLvl = input.int({entry}, "RSI entry below", minval=1, maxval=50)
exitLvl  = input.int({exit_}, "RSI exit above", minval=50, maxval=99)
trendLen = input.int({trend}, "Trend filter SMA", minval=20)

r = ta.rsi(close, 2)
maTrend = ta.sma(close, trendLen)
uptrend = close > maTrend

longEntry = r < entryLvl and uptrend and strategy.position_size == 0
longExit  = (r > exitLvl or not uptrend) and strategy.position_size > 0

if longEntry
    strategy.entry("Long", strategy.long)
if longExit
    strategy.close("Long")

plot(maTrend, "Trend SMA", color=uptrend ? color.green : color.gray)
bgcolor(uptrend ? na : color.new(color.red, 92), title="Downtrend")
{_MARKERS}
alertcondition(longEntry, "Oversold in uptrend", "RSI(2) below the entry level")
{_TABLE.format(label="Buy the dip", settings=f"RSI(2) < {entry}, exit > {exit_}",
               tested="each name, long-only")}
// AUDITED CAVEAT, AND THE MOST IMPORTANT LINE IN THIS FILE: this family's edge
// did NOT survive realistic costs in the study. Zero-cost Sharpe around 0.85
// became NEGATIVE under the stressed-cost scenario. Pine charges nothing, so
// the Strategy Tester will look far better than the truth. Kept as a worked
// example of a strategy that fails, not as a candidate.
""")


def _pine_vol_target(params: dict) -> str:
    # These fallbacks must equal the strategy class's own defaults — see
    # test_pine_defaults_match_the_python_strategy_defaults, which caught them
    # sitting 5 points apart, so the chart was drawing a rule the study never ran.
    tv = params.get("target_vol", 0.25)
    win = params.get("vol_window", 63)
    return (_HEADER.format(name="VolTargetedBasket",
                           tagline="Size the position so risk stays constant, not the dollar amount.")
            + f"""
indicator("AITB vol-target exposure gauge ({int(tv * 100)}% annual)", overlay=false)
volWin = input.int({win}, "Volatility window", minval=10)
target = input.float({tv}, "Target annual volatility", step=0.01)

ret = close / close[1] - 1
realized = ta.stdev(ret, volWin) * math.sqrt(252)
exposure = math.min(target / math.max(realized, 0.0001), 1.0)

plot(realized, "Realised volatility", color=color.orange, linewidth=2)
plot(exposure, "Suggested exposure (0-1)", color=color.blue, linewidth=2)
hline(1.0, "Fully invested", color=color.gray)
{_TABLE.format(label="Steady-risk sizing", settings=f"{int(tv * 100)}% target, {win}d window",
               tested="megacap AI basket")}
// A GAUGE, not a strategy: it says how much to hold, not what to buy, which is
// why it is an indicator rather than a backtestable strategy. The study applies
// it to a whole basket with a Treasury sleeve.
""")


def _pine_trend_plus_vol(params: dict) -> str:
    tv = params.get("target_vol", 0.20)
    trend = params.get("trend_window", 200)
    hyst = params.get("hysteresis", 0.02)
    min_exp = params.get("min_exposure", 0.2)
    vol_win = params.get("vol_window", 63)
    return (_HEADER.format(name="TrendPlusVolTarget",
                           tagline="Trend decides whether to hold; volatility decides how much.")
            + f"""
indicator("AITB trend + vol-target exposure gauge", overlay=false)
maLen  = input.int({trend}, "Trend SMA", minval=20)
hyst   = input.float({hyst}, "Hysteresis band", step=0.005)
target = input.float({tv}, "Target annual volatility", step=0.01)
minExp = input.float({min_exp}, "Exposure when the trend is off", step=0.05)
volWin = input.int({vol_win}, "Volatility window", minval=10)

ma = ta.sma(close, maLen)
var bool gateOn = true
if close < ma * (1 - hyst)
    gateOn := false
if close > ma * (1 + hyst)
    gateOn := true

ret = close / close[1] - 1
realized = ta.stdev(ret, volWin) * math.sqrt(252)
volScale = math.min(target / math.max(realized, 0.0001), 1.0)
exposure = (gateOn ? 1.0 : minExp) * volScale

plot(exposure, "Suggested exposure (0-1)", color=color.blue, linewidth=2)
plot(realized, "Realised volatility", color=color.orange)
hline(1.0, "Fully invested", color=color.gray)
bgcolor(gateOn ? na : color.new(color.red, 92), title="Trend off")
{_TABLE.format(label="Trend + steady risk",
               settings=f"{trend}d trend, {int(tv * 100)}% vol, {hyst:.0%} band",
               tested="megacap AI basket")}
// The hysteresis band is the point: exposure switches off only below
// (1-band) x SMA and back on only above (1+band) x SMA, which suppresses the
// whipsaw a bare moving-average cross produces right at the line.
""")


def _pine_beta_hedge(params: dict) -> str:
    bw = params.get("beta_window", 126)
    return (_HEADER.format(name="BetaHedgedBasket",
                           tagline="How much index to short against a holding, from its trailing beta.")
            + f"""
indicator("AITB beta-hedge ratio vs QQQ", overlay=false)
betaWin = input.int({bw}, "Beta window (days)", minval=20)
maxHedge = input.float(1.5, "Maximum hedge ratio", step=0.1)

r  = close / close[1] - 1
bq = request.security("NASDAQ:QQQ", "D", close)
rb = bq / bq[1] - 1

cov = ta.cov(r, rb, betaWin)
vb  = ta.variance(rb, betaWin)
beta = math.min(math.max(vb > 0 ? cov / vb : na, 0.0), maxHedge)

plot(beta, "Hedge ratio (short this much QQQ)", color=color.purple, linewidth=2)
hline(1.0, "One-for-one", color=color.gray)
{_TABLE.format(label="Beta hedge", settings=f"{bw}-day beta, capped at 1.5",
               tested="megacap AI and shortlist baskets")}
// Read it as: for every $1 held in this symbol, short this many dollars of
// QQQ. In the study this was the only construction that produced a robust
// candidate on real data — and the only one whose correlation to the index was
// near zero rather than 0.6-0.9.
//
// Pine cannot hold two instruments in one strategy, so this is a ratio gauge
// rather than a backtest. Reproducing it needs two positions in a real account,
// and rebalancing the hedge as beta drifts.
""")


def _pine_breadth_gate(params: dict) -> str:
    thr = params.get("threshold", 0.45)
    return (_HEADER.format(name="BreadthGatedBasket",
                           tagline="Exit when the rally narrows, not when the index finally turns.")
            + f"""
indicator("AITB breadth gate (8-name proxy)", overlay=false)
// APPROXIMATION OF AN APPROXIMATION. The study measures breadth across ~80
// technology names. Pine cannot poll a universe cheaply, so this samples eight
// large ones. The shape is roughly right; the LEVEL is not, and the threshold
// below does not mean the same thing it means in the study.
thr = input.float({thr}, "Breadth threshold", step=0.05)

f(sym) =>
    c = request.security(sym, "D", close)
    m = request.security(sym, "D", ta.sma(close, 200))
    c > m ? 1.0 : 0.0

breadth = (f("NASDAQ:NVDA") + f("NASDAQ:MSFT") + f("NASDAQ:AAPL") +
           f("NASDAQ:GOOGL") + f("NASDAQ:AMZN") + f("NASDAQ:META") +
           f("NASDAQ:AVGO") + f("NASDAQ:TSLA")) / 8.0
smooth = ta.sma(breadth, 21)
riskOn = smooth >= thr

plot(smooth, "Share above their 200-day average", color=color.blue, linewidth=2)
hline(thr, "Threshold", color=color.gray)
bgcolor(riskOn ? color.new(color.green, 92) : color.new(color.red, 90),
        title="Breadth regime")
alertcondition(smooth < thr and smooth[1] >= thr, "Breadth broke down",
               "Participation fell below the threshold")
{_TABLE.format(label="Breadth gate (proxy)", settings=f"threshold {thr}, 21-day smoothing",
               tested="~80-name universe, NOT this 8-name proxy")}
""")



def _pine_atr_stop(params: dict) -> str:
    entry = params.get("entry_window", 100)
    aw = params.get("atr_window", 22)
    am = params.get("atr_mult", 3.0)
    return (_HEADER_EXACT.format(name="ATRTrailingStop",
                           tagline="Ride the trend; exit on a stop that widens with volatility.")
            + _strategy_open(f"AITB ATR trailing stop ({entry}d entry, {am}x ATR)") + f"""
entryLen = input.int({entry}, "Entry: new N-day high", minval=10)
atrLen   = input.int({aw}, "ATR window", minval=5)
atrMult  = input.float({am}, "Stop distance (ATRs)", step=0.5, minval=0.5)

atr = ta.atr(atrLen)
breakout = close > ta.highest(close[1], entryLen)

var float peak = na
inPos = strategy.position_size > 0

// The stop tested today uses the peak as it stood BEFORE this bar. Updating
// the peak first and then testing against it means the stop can never fire.
stopLevel = na(peak) ? na : peak - atrMult * atr
longExit  = inPos and not na(stopLevel) and close < stopLevel
longEntry = not inPos and breakout

if longEntry
    strategy.entry("Long", strategy.long)
    peak := close
if longExit
    strategy.close("Long")
    peak := na
if inPos and not longExit
    peak := na(peak) ? close : math.max(peak, close)

plot(stopLevel, "Trailing stop", color=color.red, style=plot.style_linebr, linewidth=2)
// linebr on both: peak and stop are `na` while flat, and a plain plot would
// draw a straight line across the gap as if the stop had been live throughout.
plot(peak, "Peak since entry", color=color.new(color.green, 40),
     style=plot.style_linebr)
{_MARKERS}
alertcondition(longEntry, "Breakout entry", "New high — entry")
alertcondition(longExit, "Stop hit", "Closed below the ATR trailing stop")
{_TABLE.format(label="ATR trailing stop",
               settings=f"{entry}d high in, {am}x ATR({aw}) out",
               tested="each name in the universe")}
// The point of this one: every other trend exit in the study is a fixed
// lookback, which gives the same room in a calm market as in a violent one.
// This scales the room to the instrument's own volatility.
//
// One known divergence: the Python version can exit and re-enter on the same
// bar (it clears the stop, then re-tests the breakout). Pine cannot, because
// strategy.position_size still reads as open until the close fills. The
// difference only bites on a bar that both stops out and makes a new high.
""")


def _pine_52w_high(params: dict) -> str:
    win = params.get("window", 252)
    prox = params.get("min_proximity", 0.90)
    return (_HEADER_EXACT.format(name="FiftyTwoWeekHighProximity",
                           tagline="Hold what sits closest to its own 52-week high.")
            + _strategy_open(f"AITB 52-week-high proximity (>= {prox:.0%})") + f"""
win     = input.int({win}, "High window (bars)", minval=50)
minProx = input.float({prox}, "Minimum proximity to the high", step=0.01,
                      minval=0.5, maxval=1.0)

hi = ta.highest(close, win)
proximity = close / hi

longEntry = proximity >= minProx and proximity[1] < minProx
longExit  = proximity < minProx and proximity[1] >= minProx

if longEntry
    strategy.entry("Long", strategy.long)
if longExit
    strategy.close("Long")

plot(hi, "52-week high", color=color.new(color.blue, 40))
// The proximity band drawn on the price scale, so "within 10% of the high" is
// a visible line rather than a number. Plotting proximity itself would need a
// second scale, and an hline at 0 on an overlay squashes the price axis.
plot(hi * minProx, "Proximity band", color=color.new(color.orange, 30),
     style=plot.style_linebr)
bgcolor(proximity >= minProx ? color.new(color.green, 92) : na, title="Near the high")
{_MARKERS}
alertcondition(longEntry, "Near the high", "Within the proximity band of the 52-week high")
alertcondition(longExit, "Fell away", "Dropped out of the proximity band")
{_TABLE.format(label="52-week-high proximity",
               settings=f"{win}-bar high, within {prox:.0%}",
               tested="ranked across the universe, top 10 held")}
// A breakout is an EVENT; this is a STATE. George & Hwang (2004) found
// nearness to the 52-week high predicts returns better than past return does,
// because the high anchors expectations and traders under-react to news that
// pushes against it.
//
// DIFFERENCE THAT MATTERS: the study RANKS every name by proximity and holds
// the top 10. One chart can only apply the threshold, not the ranking, so this
// version will hold far more often than the tested one.
""")


def _pine_quiet_trend(params: dict) -> str:
    tw = params.get("trend_window", 200)
    vw = params.get("vol_window", 21)
    q = params.get("vol_quantile", 0.75)
    lb = params.get("lookback", 504)
    return (_HEADER_EXACT.format(name="QuietTrend",
                           tagline="Hold the trend only while the instrument is behaving calmly.")
            + _strategy_open(f"AITB quiet trend ({tw}d trend, vol under its {q:.0%})") + f"""
trendLen = input.int({tw}, "Trend SMA", minval=20)
volLen   = input.int({vw}, "Volatility window", minval=5)
q        = input.float({q}, "Volatility quantile ceiling", step=0.05,
                       minval=0.1, maxval=0.99)
lookback = input.int({lb}, "History for the quantile", minval=252)

ma = ta.sma(close, trendLen)
inTrend = close > ma

ret = close / close[1] - 1
vol = ta.stdev(ret, volLen) * math.sqrt(252)
// Rolling, not all-time: the ceiling on any bar must be computable from a
// window that ended on that bar.
ceiling = ta.percentile_linear_interpolation(vol, lookback, q * 100)
calm = vol <= ceiling

want = inTrend and calm
longEntry = want and not want[1]
longExit  = not want and want[1]

if longEntry
    strategy.entry("Long", strategy.long)
if longExit
    strategy.close("Long")

plot(ma, "Trend SMA", color=inTrend ? color.green : color.red)
bgcolor(not inTrend ? color.new(color.red, 92) :
        not calm ? color.new(color.orange, 90) : color.new(color.green, 94),
        title="Regime: red=downtrend, orange=too wild, green=hold")
{_MARKERS}
alertcondition(longExit, "Turned wild or down", "Left the quiet-uptrend regime")
{_TABLE.format(label="Quiet trend",
               settings=f"{tw}d trend, vol under its own {q:.0%} of {lb} bars",
               tested="each name in the universe")}
// The study tests volatility SIZING elsewhere (hold less when wild). This
// tests volatility SELECTION (do not hold at all when wild) — a different bet:
// one accepts the regime at a smaller size, the other declines it.
""")


def _pine_volume_breakout(params: dict) -> str:
    entry = params.get("entry_window", 55)
    exit_ = params.get("exit_window", 20)
    vw = params.get("vol_window", 50)
    vm = params.get("vol_mult", 1.5)
    return (_HEADER_EXACT.format(name="VolumeConfirmedBreakout",
                           tagline="Take the breakout only when the crowd showed up for it.")
            + _strategy_open(f"AITB volume-confirmed breakout ({entry}d high on {vm}x turnover)") + f"""
entryLen = input.int({entry}, "Entry: new N-day high", minval=10)
exitLen  = input.int({exit_}, "Exit: new N-day low", minval=5)
volLen   = input.int({vw}, "Turnover baseline window", minval=10)
volMult  = input.float({vm}, "Turnover multiple required", step=0.1, minval=1.0)

// Dollar turnover, not share count, so a stock that has doubled is compared
// against its own dollars. This is exactly the quantity the study uses — one
// of the few places a Pine script reproduces the Python rather than
// approximating it.
dv = volume * close
// The baseline excludes today (dv[1]), so a heavy day cannot lift the average
// it is being judged against.
baseline = ta.sma(dv[1], volLen)
heavy = dv > baseline * volMult

breakout  = close > ta.highest(close[1], entryLen)
breakdown = close < ta.lowest(close[1], exitLen)

inPos = strategy.position_size > 0
longEntry = not inPos and breakout and heavy
longExit  = inPos and breakdown

if longEntry
    strategy.entry("Long", strategy.long)
if longExit
    strategy.close("Long")

plot(ta.highest(close[1], entryLen), "Entry level", color=color.new(color.green, 40))
plot(ta.lowest(close[1], exitLen), "Exit level", color=color.new(color.red, 40))
plotshape(breakout and not heavy, title="Breakout on thin volume",
          style=shape.xcross, location=location.abovebar,
          color=color.new(color.gray, 20), size=size.tiny)
{_MARKERS}
alertcondition(longEntry, "Confirmed breakout", "New high on heavy turnover")
alertcondition(longExit, "Broke down", "New N-day low — exit")
{_TABLE.format(label="Volume-confirmed breakout",
               settings=f"{entry}d high, {vm}x {vw}d turnover, {exit_}d low out",
               tested="each name in the universe")}
// The grey x marks every breakout this rule REFUSED for thin turnover. That is
// the whole strategy in one plot: if the x's mostly precede failed moves, the
// filter is doing something; if they precede the best moves, it is a tax.
//
// The study uses volume in exactly one other place — the cap that stops a
// simulated fill from eating an implausible share of a day's turnover — and
// nowhere else as a signal. This is the only rule in the study that reads it.
//
// NOT a controlled comparison against the plain breakout export: that one also
// caps itself at six positions across the universe and this one holds every
// qualifier, because a cap is not expressible on one chart.
""")


def _pine_turn_of_month(params: dict) -> str:
    start = params.get("start_day", 26)
    end = params.get("end_day", 5)
    return (_HEADER_EXACT.format(name="TurnOfMonth",
                           tagline="Hold across the turn of the month; sit out the rest of it.")
            + _strategy_open(f"AITB turn of month (day {start} to day {end})") + f"""
startDay = input.int({start}, "Hold from this calendar day", minval=15, maxval=31)
endDay   = input.int({end}, "...through this day of the next month", minval=1, maxval=15)

// Calendar days, not trading days. The trading-day version of this effect is
// the more precise one, but "the third trading day before month end" cannot be
// identified on the day itself without knowing where the month ends — which is
// looking forward. This version is knowable in advance, so the chart rule and
// the tested rule are the same rule.
inWindow = dayofmonth >= startDay or dayofmonth <= endDay

longEntry = inWindow and not inWindow[1]
longExit  = not inWindow and inWindow[1]

if longEntry
    strategy.entry("Long", strategy.long)
if longExit
    strategy.close("Long")

bgcolor(inWindow ? color.new(color.green, 90) : na, title="In the window")
{_MARKERS}
alertcondition(longEntry, "Turn of month — in", "Entering the turn-of-month window")
alertcondition(longExit, "Turn of month — out", "Leaving the turn-of-month window")
{_TABLE.format(label="Turn of month",
               settings=f"calendar day {start} through {end}",
               tested="each name in the universe")}
// This reads a calendar and nothing else — no price, no volume, no trend. That
// is the point. Every other rule in the study is long the same names at
// broadly the same times, which is why they all correlate 0.6-0.9 with the
// index; a rule whose exposure was decided before the year began cannot be
// that same bet.
//
// Ariel (1987) and Lakonishok & Smidt (1988); the mechanism usually offered is
// flows — salary contributions, pension inflows and index rebalancing landing
// in a narrow window each month. It trades about 24 times a year, so costs
// matter more here than anywhere else on this page, and Pine charges none.
""")


_GENERATORS = {
    "QQQMovingAverage": _pine_qqq_ma,
    "TrendFollowCash": _pine_trendfollow,
    "DonchianBreakout": _pine_donchian,
    "RSIReversion": _pine_rsi_reversion,
    "VolTargetedBasket": _pine_vol_target,
    "TrendPlusVolTarget": _pine_trend_plus_vol,
    "BetaHedgedBasket": _pine_beta_hedge,
    "BreadthGatedBasket": _pine_breadth_gate,
    # freeze v5 — designed single-symbol, so the Python rule and the chart rule
    # are the same rule rather than an approximation of one.
    "ATRTrailingStop": _pine_atr_stop,
    "FiftyTwoWeekHighProximity": _pine_52w_high,
    "QuietTrend": _pine_quiet_trend,
    # freeze v6 — the study's first volume signal, and its first rule that
    # reads no price at all.
    "VolumeConfirmedBreakout": _pine_volume_breakout,
    "TurnOfMonth": _pine_turn_of_month,
}

# Which exports place orders (so TradingView's Strategy Tester runs them and
# marks trades) versus which only draw a signal. Position sizing, hedge ratios
# and regime gauges are not entry rules, and are not dressed up as if they were.
BACKTESTABLE = {"QQQMovingAverage", "TrendFollowCash", "DonchianBreakout",
                "RSIReversion", "ATRTrailingStop", "FiftyTwoWeekHighProximity",
                "QuietTrend", "VolumeConfirmedBreakout", "TurnOfMonth"}


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
