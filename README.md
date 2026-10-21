# Adaptive Market Maker (AMM)

A reproducible research platform that asks one question:

> **Does a market maker that adapts its quotes to short-horizon price forecasts, execution conditions and
> inventory risk outperform simpler market makers once realistic execution, latency, fees and adverse
> selection are accounted for?**

This is not a price-prediction toy or a bot that claims to beat the market. It is an event-driven
historical-replay simulator with a queue-aware execution model, causal features, leakage-controlled models,
exact accounting, and a pre-declared evaluation protocol whose verdicts are allowed to come out negative.

> **Status.** Everything below is implemented and tested (≈80 tests). The development sandbox could not
> reach lobsterdata.com, so **all results so far are on the project's synthetic market** and are labelled as
> such. They validate the machinery; they are not evidence about real markets. Running the identical
> protocol on the free LOBSTER sample takes the two steps under [Data](#data).

<!-- RESULTS -->

## What makes the evaluation credible

| design decision | why it matters |
|---|---|
| **Queue position tracked with order ids** (`execution/queue_model.py`) | LOBSTER messages carry order ids. Orders never seen being submitted rested before us (ahead); later ones are behind. That makes visible queue-ahead tracking exact under price-time priority, instead of "fill whenever the price trades". |
| **Three fill presets** (conservative / central / optimistic) | Brackets the unobservable part (hidden size ahead of us). The optimistic "fill on touch" preset is reported only to show how much naive assumptions inflate results. |
| **Latency on every hop**, historical messages win ties | An order can never trade against a book state from before it arrived; the strategy acts on stale information. |
| **Observable states only** | A sweep that prints several messages with one timestamp is one observable state; mid-burst books are never used. |
| **As-of labels + purging + embargo** | The target at t + h is the book prevailing at t + h, not the next print after it; no training label peeks into validation or test. |
| **Validation-only selection, single test pass** | Strategy C inherits B's parameters, so C − B isolates the alpha component. |
| **Liquidation-marked primary metric** | Residual inventory is closed at the far touch with a taker fee; mid-marking would flatter strategies that end long or short. |
| **Exact integer accounting** | Cash in micro-dollars; every backtest must reconcile cash, inventory, fees and equity or it fails. |
| **Planted-signal and placebo tests** | The same protocol runs on a synthetic market *with* informed flow and one *without* (`null`). Separately, strategy C is fed its own alpha shifted by half a session (same distribution, no information): if that placebo beat B, the engine would be rewarding noise. |
| **Block-bootstrap verdicts** | Paired, dependence-aware intervals; "supported" only when the 95% interval excludes zero. Every backtest is logged in `trials.csv`. |

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate     # Python >= 3.11 (tested on 3.11.17, Linux)
pip install -e ".[dev]"
pytest                                                    # ~80 tests, ~1-2 minutes
```

## Quick start (synthetic market, no downloads)

```bash
amm experiment run --config configs/experiments/smoke.yaml               # ~30 s end-to-end sanity run
amm experiment run --config configs/experiments/baseline_comparison.yaml # full protocol, ~15 min on 4 cores
amm experiment run --config configs/experiments/null_scenario.yaml       # planted-null check
amm report --experiment synthetic_informed_baseline                      # (re)build report.md + figures
```

Each run writes `reports/experiments/<id>/` (report, figures, `results.json`, `trials.csv`, configs, manifests)
and refuses to overwrite an existing run unless you pass `--overwrite`.

## Data

The default real dataset is the free LOBSTER sample (2012-06-21, 10 levels). LOBSTER's terms do not allow
redistribution, so raw files are never committed.

1. Download from <https://data.lobsterdata.com/info/DataSamples.php>, e.g. the **MSFT** and **AAPL** 10-level
   samples.
2. Unzip into `data/raw/lobster/` so that, for example, this file exists:
   `data/raw/lobster/MSFT_2012-06-21_34200000_57600000_message_10.csv` (plus the matching `_orderbook_10.csv`).

Then:

```bash
amm data validate --config configs/data.yaml                    # quality + book-transition integrity report
amm data prepare  --config configs/data.yaml                    # Parquet + manifest with sha256 checksums
amm experiment run --config configs/experiments/lobster_msft.yaml
amm experiment run --config configs/experiments/lobster_aapl.yaml
pytest -m needs_lobster                                         # real-data validation test (skipped if absent)
```

MSFT (~$30, one-tick spread, long queues) is the default: queue position and imbalance dominate large-tick
market making. AAPL (~$580 in 2012, multi-tick spreads) is the small-tick contrast. Optional sources (Nasdaq
ITCH samples, Databento MBO) are documented but not implemented; nothing here spends money.

## CLI reference

| command | what it does |
|---|---|
| `amm data synth --config configs/data_synthetic.yaml` | write synthetic sessions in LOBSTER file layout |
| `amm data validate --config <data cfg>` | data-quality checks; non-zero exit on critical failures |
| `amm data prepare --config <data cfg> [--force]` | validate, normalize to Parquet, write manifests |
| `amm features build --config <cfg>` | causal features per session → `data/interim/` |
| `amm labels build --config <cfg>` | future targets per session → `data/interim/` |
| `amm model train --config <cfg>` | fit candidates, calibrate, select on validation → `reports/models/` |
| `amm model evaluate --config <cfg>` | test-period metrics of the saved models |
| `amm backtest --strategy {fixed_spread,inventory_aware,adaptive,regime} --config <cfg> [--role test]` | one strategy, configured (not selected) parameters |
| `amm experiment run --config <exp cfg> [--jobs N] [--overwrite]` | full protocol + report |
| `amm experiment ablation --config configs/experiments/ablation.yaml` | protocol with the ablation phase only |
| `amm experiment sensitivity --config configs/experiments/sensitivity.yaml` | protocol with the sensitivity phase only |
| `amm report --experiment <id>` | build `report.md` and figures from saved results |
| `amm registry` | list every recorded experiment run |

Commands that take `--config` read section defaults from `configs/` (`data`, `features`, `models`,
`execution`, `strategies`) and overlay the given file; e.g. `amm features build --config configs/data_synthetic.yaml`
works on the synthetic dataset.

## Configuration

All material assumptions are in validated YAML (unknown keys are errors):

* `configs/data*.yaml`: source, paths, levels, tick size, session times, synthetic-market parameters
* `configs/features.yaml`: depth levels and weights, event/time windows, volatility window, warm-up
* `configs/models.yaml`: label kind and horizon, splits (mode, fractions, embargo, exclusions), model hyperparameters, fill model
* `configs/execution.yaml`: fill preset, post-only, latency per hop, fees/rebates; backtest settings
* `configs/strategies.yaml`: parameters for A–D and hard risk limits
* `configs/experiments/*.yaml`: which sections to include, validation grids, sensitivity grids, ablations

Values labelled illustrative (latency, fees, synthetic-market rates) are not calibrated to any venue.

## Repository layout

```
src/adaptive_market_maker/   data · orderbook · features · labels · models · strategies · execution
                             accounting · backtest · evaluation · experiments · reporting · cli
configs/                     section defaults + experiment definitions
docs/                        implementation plan (incl. spec review), architecture, maths, execution
                             assumptions, data dictionary, research protocol, limitations
tests/                       unit · accounting · execution (incl. property tests) · integration · regression
reports/                     generated experiment directories (curated results are committed separately)
```

## Documentation

* [Implementation plan, checklist and review of the original spec](docs/implementation_plan.md)
* [Architecture and event ordering](docs/architecture.md)
* [Mathematical model: every formula and unit](docs/mathematical_model.md)
* [Execution assumptions](docs/execution_assumptions.md)
* [Data dictionary](docs/data_dictionary.md)
* [Research protocol](docs/research_protocol.md)
* [Limitations](docs/limitations.md)

## Known limitations (short version)

Synthetic results only so far; one free LOBSTER day per ticker; no market impact of simulated orders; hidden
liquidity ahead of us is unobservable; constant latency; illustrative fees; within-session bootstrap. See
[docs/limitations.md](docs/limitations.md).
