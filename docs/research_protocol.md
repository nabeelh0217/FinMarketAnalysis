# Research protocol

Pre-declared before looking at test results; implemented in `experiments/runner.py`.

## Hierarchy

1. Verify data quality (`amm data validate`): critical failures stop the pipeline.
2. Deterministic accounting and execution tests pass (`pytest`).
3. Baselines (prior model; fixed-spread strategy) run first.
4. Predictive quality established out of sample before alpha touches a strategy.
5. One component at a time: A → B adds inventory control only; B → C adds alpha only.
6. Robustness: sensitivity and ablations, reported but never used to re-select.

## Splits

* **Sessions mode** (several sessions, e.g. synthetic): whole sessions per role in date order —
  train (4) → validation (2) → test (4) by default.
* **Intraday mode** (one LOBSTER session): after excluding the first and last 15 minutes,
  50% train → 30 s embargo → 20% validation → 30 s embargo → 30% test.

A sample belongs to a segment only if its decision time is inside it **and** its label window ends inside it
(purging). Scaling, imputation, model fitting and class-conditional means use training rows only.

## Model selection

* Fit every candidate on train.
* Temperature-calibrate on the first half of validation.
* Select by log loss on the second half of validation, with non-overlapping labels.
* Report test metrics for **every** candidate, not just the selected one, plus a block-bootstrap interval for
  the log-loss gain over the constant-probability prior.

## Strategy parameter selection

Small grids declared in the experiment config, evaluated on validation segments only, choosing the highest
total validation net P&L. The alpha strategy inherits the inventory strategy's selected parameters (γ, τ,
half-spread, …), so C − B measures the alpha component alone. The test segments are then run exactly once per
strategy with the frozen configuration (`config_selected.yaml`).

## Metrics

* **Primary:** net P&L after fees and rebates with terminal inventory liquidated at the far touch.
* Risk-adjusted: mean/std of per-interval (60 s) net P&L — not annualized.
* Secondary: mid-marked P&L, fills, fill rate, cancel rate, quote-to-fill, turnover, inventory (mean/max),
  drawdown, realized spread / adverse selection by horizon, P&L by causal regime.

## Uncertainty

Circular block bootstrap over 60 s intervals, blocks of 3 that wrap within a session and never span sessions; paired by interval for
strategy comparisons. Verdicts are mechanical: *supported* only when the 95% interval excludes zero in the
hypothesised direction.

## Multiple testing

Every backtest is appended to `trials.csv` (phase, strategy, parameters, segment, results) and every experiment
run to `reports/experiments/registry.jsonl` (config fingerprint, git commit, headline results). The report states
how many candidates and backtests were run. Only the pre-declared final comparisons are confirmatory.

## Planted-signal validation

The same protocol runs on two synthetic markets: `informed` (a real, known signal and real adverse selection)
and `null` (identical total aggressive flow, no informed traders). Mechanical predictability (a thinner queue
depletes first) exists in both, as in real markets, so the null scenario is expected to show *less*, not zero,
predictability.

## Placebo alpha

The sharpest bias check: strategy C is re-run with its own alpha series circularly shifted by half a session.
The placebo has the real signal's distribution and autocorrelation but no alignment with the market. If the
placebo beat B (no alpha), the engine would be rewarding something other than information (e.g. a mechanical
side effect of quoting asymmetrically), and the alpha result would not be trusted. The report states the
outcome explicitly.
