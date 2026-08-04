# Prospective (paper-trading) testing protocol

Applies to any strategy that survives the frozen first real-data study. The
strategy specification is the FROZEN one — no parameter, universe, weighting
or rebalance change is permitted between the historical study and this test.
This protocol defines observation only; it does not initiate live trading and
the system contains no broker-execution functionality.

## Protocol parameters

| Item | Value |
|---|---|
| Start date | the first trading day of the month following the decision brief |
| Signal computation | after the close of each trading day T, from data through T's close only |
| Signal publication | signal + intended weights written to `results/real/paper/signals/<date>.json` before the next open (timestamped, hash-chained to the previous file) |
| Intended execution | the official opening auction price of T+1 (matching the backtest convention) |
| Slippage logging | for every rebalance, record intended open price vs a realistic reference fill (open ± half-spread); log per-name in `results/real/paper/fills/<date>.csv` |
| Portfolio weights | exactly the frozen strategy's target weights; $1,000,000 notional paper account |
| Rebalance events | per the frozen spec (e.g. W-FRI / ME); a missed computation day is logged, never backfilled |
| Benchmark | QQQ total return, plus the equal-weight target_holdings basket |
| Minimum observation period | 12 months or 12 rebalance events, whichever is longer, before any evaluation against advancement criteria |

## Suspension criteria (stop the test, investigate)

* Realized tracking of the backtest engine's simultaneous paper run diverges
  by more than 2% cumulative (implementation bug).
* Realized drawdown exceeds 1.5× the strategy's historical maximum drawdown.
* A data-provider failure leaves signals uncomputable for 5 consecutive
  sessions.
* Any evidence of leakage in the signal pipeline (a signal that used same-day
  execution prices).

## Advancement criteria (to small-capital testing — a separate future decision)

After the minimum observation period, ALL of:

* Paper Sharpe within the strategy's bootstrap 90% CI from the frozen study.
* Realized costs ≤ the stressed-cost scenario assumptions.
* No suspension events, or all suspensions resolved as external.
* The decision to fund is made by the human owner, documented, with the same
  freeze discipline applied to the funded specification.

## Record keeping

Every signal file, fill log and monthly evaluation is append-only. The
evaluation at the end of the observation period is written once, referencing
the freeze hash and this protocol version.

Protocol version: 1 (frozen together with research_freeze_v1).
