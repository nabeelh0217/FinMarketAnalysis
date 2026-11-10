# Architecture

```
            raw LOBSTER CSVs (real)  or  synthetic generator (writes LOBSTER CSVs)
                                   │
                         data/lobster.py  ── exact parsing, row-count check
                                   │
                  data/normalize.py, data/validate.py, orderbook/validation.py
                     (sentinels, origin ids, quality checks, book-transition reconciliation)
                                   │
                    data/prepare.py ── Parquet per session + manifest (sha256, coverage, status)
                                   │
                              SessionData  (numpy arrays: messages + top-K book after each message)
                ┌──────────────────┼────────────────────────────┐
                │                  │                            │
     features/pipeline.py   labels/targets.py          execution/simulator.py
     (causal, observable    (future targets,            (queue tracking, fills;
      states only)           training only)              shares queue_step with
                │                  │                     models/fill_probability.py)
                └──── models/price_direction.py ────┐           │
                      (fit / calibrate / alpha)     │           │
                                                    ▼           │
                                       strategies.base.Signals  │
                                                    │           │
                     strategies/* ── QuoteIntent ──► backtest/engine.py ◄── risk.py
                                                    │   (event loop, latency, quote mgmt)
                                                    ▼
                                         accounting/ledger.py (cash, inventory, fees; reconcile)
                                                    │
                     evaluation/* (metrics, markouts, bootstrap) ── experiments/runner.py
                                                    │
                                     reporting/plots.py, reporting/report.py
```

## Module responsibilities

| module | responsibility | must not |
|---|---|---|
| `data` | parse sources, normalize, manifest, validate | let strategies see source-specific columns |
| `orderbook` | book-transition integrity checks | silently repair a corrupted book |
| `features` | causal features on observable states | use any row after the decision row |
| `labels` | future targets | be visible to a strategy |
| `models` | direction models, calibration, alpha, fill model | fit anything on validation/test rows (except the single calibration temperature on validation) |
| `strategies` | turn a `MarketView` + own state into a `QuoteIntent` | create fills, read future data, touch the ledger |
| `execution` | order lifecycle, latency, queue position, fills | know which strategy is running |
| `accounting` | cash, inventory, fees, equity, reconciliation | depend on strategy or model internals |
| `backtest` | chronological orchestration, splits | reorder historical messages |
| `evaluation` | metrics, markouts, statistics | annualize intraday ratios |
| `experiments` | protocol, selection on validation, registry | re-select on test results |
| `reporting` | figures and report from saved results | compute new results |

## Event ordering in the engine

For historical message i with timestamp ts_i:

1. Agent events with time **< ts_i** run first, in (time, priority, sequence) order with priority
   venue arrival (orders, cancels) < reports to the strategy < strategy wake-ups.
2. The exchange applies message i to every resting simulated order (fills at our limit price, maker).
3. If row i is a decision-eligible observable state and no wake-up is pending, a wake-up is scheduled at
   ts_i + market-data latency (respecting the optional minimum decision interval).

A wake-up uses the newest eligible state whose timestamp is ≤ wake time − market-data latency. Equal timestamps
favour the historical message (conservative). The engine is deterministic: the same inputs and config give
identical fills, decisions and P&L (tested).

## Interfaces

* `SessionData` — immutable-by-convention arrays; row i = message i + book after it.
* `FeatureTable.df` — one row per observable state: `row`, `ts_ns`, `feature_ts_ns`, features, `warm`, `valid_book`, `vol_ticks_sqrt_s`.
* `Signals` — per observable state: `vol`, optional `alpha` (ticks over the label horizon), `eligible`.
* `Strategy.quote(MarketView, StrategyState) -> QuoteIntent` — prices in price units, quantities in shares.
* `BacktestResult` — fills, equity snapshots, decision log, summary, and the full set of assumptions.

No global mutable state except the read-only `_CTX` handed to forked worker processes in `run_jobs`.
