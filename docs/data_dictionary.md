# Data dictionary

## Normalized events (`data/processed/<dataset_id>/<session_id>.parquet`)

One row per source message. Header metadata (Parquet schema key `amm_header`): schema version, session id,
ticker, date, source, levels, tick size, price scale, lot size, session open/close, normalization counters.

| field | type | units | semantics | source | observable at decision time? |
|---|---|---|---|---|---|
| `seq` | int64 | — | row index in the source file (stable ordering key) | derived | yes |
| `ts_ns` | int64 | ns after midnight | exchange event time, parsed exactly from LOBSTER's decimal seconds | LOBSTER `Time` | yes, from ts + market-data latency |
| `event_type` | int8 | — | 1 submit, 2 partial cancel, 3 delete, 4 visible execution, 5 hidden execution, 6 cross, 7 halt | LOBSTER `Type` | yes |
| `order_id` | int64 | — | exchange order reference | LOBSTER `Order ID` | yes |
| `size` | int64 | shares | size of the message (for type 7: as published) | LOBSTER `Size` | yes |
| `price` | int64 | $ × 10 000 | message price (type 7: halt code −1 / 0 / 1) | LOBSTER `Price` | yes |
| `side` | int8 | — | +1 buy, −1 sell limit order; for executions the *resting* side | LOBSTER `Direction` | yes |
| `origin_seq` | int64 | row | row of this order id's submission, −1 if never seen submitted (rested before the file or beyond depth) | derived | yes (only references earlier rows) |
| `ask_px_k`, `bid_px_k` | int64 | $ × 10 000 | k-th best price after the message; empty = ±9 999 999 999 | LOBSTER orderbook file | yes |
| `ask_sz_k`, `bid_sz_k` | int64 | shares | displayed size at that level; empty = 0 | LOBSTER orderbook file | yes |

Lossy or derived transformations (all counted in the manifest's `normalization` block):

* LOBSTER dummy prices and zero sizes → internal empty sentinels.
* Decimal digits beyond nanoseconds → truncated (count reported).
* Messages referencing an order before its submission → `origin_seq = −1` (count reported).

## Features (`data/interim/<dataset_id>/features_<session>.parquet`, built by `amm features build`)

One row per observable state. Every feature uses only messages up to and including `row`.

| column(s) | group | units | window |
|---|---|---|---|
| `row`, `ts_ns`, `feature_ts_ns` | key | row / ns | — |
| `spread_ticks`, `imb_l1`, `log_bid_sz1`, `log_ask_sz1` | book_l1 | ticks / ratio / log shares | instantaneous |
| `micro_minus_mid` | microprice | ticks | instantaneous |
| `imb_l{K}`, `log_depth_ratio_l{K}` | multi_level | ratio / log ratio | top K levels |
| `d_best_bid_e1`, `d_best_ask_e1`, `ret_e{n}`, `ret_t{W}` | price_history | ticks | n states / W seconds |
| `rv_t{W}` | volatility | ticks/√s | W seconds |
| `ofi_e{n}`, `ofi_t{W}` | order_flow | lots | n states / W seconds |
| `signed_vol_t{W}`, `trade_imb_t{W}`, `mo_rate_t{W}` | trade_flow | lots / ratio / per s | W seconds |
| `cancel_rate_t5`, `submit_rate_t5`, `update_rate_t5`, `touch_replenish_t5` | activity | per s / lots | 5 seconds |
| `secs_since_mid_change`, `secs_since_trade`, `session_frac` | timing | s / fraction | — |
| `mid_ticks`, `mid_ff_ticks` | aux | ticks | `mid_ff` forward-filled causally |
| `valid_book`, `warm` | aux | bool | warm = at least `warmup_s` after the first message |
| `vol_ticks_sqrt_s` | aux | ticks/√s | realized vol with floor; fed to strategies |

## Labels (`labels_<session>.parquet`, `amm labels build`) — never observable at decision time

| column | units | meaning |
|---|---|---|
| `ts_ns` | ns | decision time |
| `label_end_ns` | ns | last timestamp the label depends on (purging) |
| `dmid_ticks` | ticks | as-of mid change over the horizon |
| `y` | class | −1 / 0 / +1 (or 0 / 1 for binary) |

## Experiment artifacts (`reports/experiments/<id>/`)

| file | content |
|---|---|
| `config_snapshot.yaml` | configuration as run |
| `config_selected.yaml` | configuration with validation-selected parameters |
| `manifests.json` | data provenance for every session |
| `results.json` | every metric in the report |
| `trials.csv` | every backtest (validation, test, sensitivity, ablation) |
| `test_predictions.csv` | test-period probabilities and alpha of the selected model |
| `shadow_orders_test.csv` | test-period shadow orders with predicted fill probability |
| `backtests/` | equity curves, decision logs (subsampled), fills with markouts |
| `figures/`, `report.md` | generated report |
