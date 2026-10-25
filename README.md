# Adaptive Market Maker (AMM)

A reproducible research platform that asks one question:

> **Does a market maker that adapts its quotes to short-horizon price forecasts, execution conditions and
> inventory risk outperform simpler market makers once realistic execution, latency, fees and adverse
> selection are accounted for?**

This is not a price-prediction toy or a bot that claims to beat the market. It is an event-driven
historical-replay simulator with a queue-aware execution model, causal features, leakage-controlled models,
exact accounting, and a pre-declared evaluation protocol whose verdicts are allowed to come out negative.

> **Status.** Everything below is implemented and tested (≈85 tests). The protocol has been run on one real
> trading day (LOBSTER MSFT, 2012-06-21) and on the project's synthetic market. One day of real data is a
> first look, not evidence of a durable edge.

## Real data: MSFT, 21 June 2012 (one session)

Validation: all 665,148 book-changing messages reconciled against LOBSTER's snapshots with zero discrepancies.
Models and parameters were fit on 09:45–12:45 and 12:45–13:57; the held-out test is 13:57–15:45. The day was a
sharp sell-off (mid −83.5 ticks, about −2.8%).

| strategy (net P&L after fees, inventory liquidated) | net P&L | 95% CI | fills | mean abs inventory |
|---|---|---|---|---|
| A fixed spread: validation chose one tick *behind* the touch, so it never filled | $0.00 | — | 0 | 0 sh |
| B inventory-aware: joins the touch | −$62.66 | [−$137, $5] | 648 | 157 sh |
| C alpha-aware (B + model alpha) | **+$47.10** | [−$13, $107] | 310 | 176 sh |
| D regime rules | +$44.38 | [−$20, $111] | 395 | 164 sh |

* **Joining the touch on MSFT loses money.** B's fills realize −0.32 ticks after 30 s: adverse selection
  exceeds the half-spread, and the $0.002 rebate doesn't cover it. On validation, quoting at the touch lost for A
  as well, which is why the protocol picked A's do-nothing variant.
* **The signal is real out of sample:** log-loss gain over a constant forecast 0.21 nats [0.20, 0.23],
  directional AUC 0.88. Unlike in the synthetic market, order flow adds information beyond the book (dropping
  all flow features worsens test log loss by 0.008).
* **Its value is as an adverse-selection filter.** C − B = +$1.02 per minute [$0.32, $1.76]: C takes half as
  many fills and its realized spread improves from −0.32 to −0.02 ticks. The placebo alpha loses more than no
  alpha (−$78 vs −$63), so the gain comes from information.
* **But it does not beat not trading.** C vs A is +$0.44 per minute [−$0.12, $1.00]: the main hypothesis (H1)
  is *not established* on this day. Without the rebate, C loses money (−$12).
* **Fragile.** Changes in latency, fees or requote throttling move P&L by roughly the width of the confidence
  interval, so one day cannot separate these effects from noise.

## Synthetic market (4 held-out 27.5-minute sessions)

Full reports: [`reports/results/synthetic_informed_baseline/report.md`](reports/results/synthetic_informed_baseline/report.md)
(informed order flow) and [`reports/results/synthetic_null_baseline/report.md`](reports/results/synthetic_null_baseline/report.md)
(same protocol, no informed traders). 498 + 118 backtests, every one logged in `trials.csv`.

| strategy (net P&L after fees, inventory liquidated) | informed market | 95% CI | mean abs inventory | null market |
|---|---|---|---|---|
| A fixed spread | $1,883 | [$1,685, $2,067] | 248 sh | $2,533 |
| B inventory-aware (Avellaneda–Stoikov skew) | $1,970 | [$1,844, $2,100] | 99 sh | $2,586 |
| C alpha-aware (B + model alpha) | **$2,281** | [$2,163, $2,388] | 95 sh | $2,556 |
| D regime rules (C + stop in high vol) | $1,902 | [$1,771, $2,018] | 121 sh | $2,403 |

What the evidence says, mechanically scored (supported = 95% paired block-bootstrap interval excludes zero):

* **The alpha component earns its keep when there is information, and only then.** C − B (identical except
  alpha) is +$2.78 per minute [$1.94, $3.60] in the informed market and −$0.26 [−$0.80, $0.28] in the null
  market. A **placebo** alpha (the model's own signal shifted by half a session) earns *less* than no alpha
  ($1,857 vs $1,970), so the engine is not rewarding alpha-shaped noise.
* **Prediction is real but mostly comes from the book, not the flow.** Out-of-sample log-loss gain over a
  constant forecast: 0.24 nats [0.23, 0.25] (directional AUC 0.89, IC 0.45); half that in the null market (mechanical
  queue depletion). Dropping *all* order-flow, trade-flow and activity features worsens test log loss by only
  0.004, so order flow adds little beyond static book features here.
* **Inventory control buys risk, not P&L.** B cuts mean inventory 248 → 99 shares and improves risk-adjusted P&L
  (mean/std of 60 s P&L 1.77 → 2.93, difference CI [0.74, 1.75]); the raw P&L difference is not significant.
  Removing the skew from C *raises* P&L (+$126) at 2.6× the inventory.
* **The economics are rebate-driven and fragile.** Without the (illustrative) $0.002 maker rebate, every
  strategy's net P&L falls by 55–68%; at a +$0.001 maker fee B loses money and C's lead over A flips. Requoting at
  most once per second costs 23–32%. Higher risk aversion (γ ≥ 0.1) puts C below A. Fill assumptions move results
  from −31…−34% (conservative queue) to ×3.4–4.1 (optimistic "fill on touch"), which is why the optimistic preset is
  never used for conclusions.
* **Where the edge comes from.** Fills taken while the prevailing spread was wider than one tick (12% of the
  fixed-spread maker's fills) earn a full tick of effective spread and keep 0.39 ticks after 30 s. Fills at a
  one-tick spread keep only 0.14: most of their half-tick is lost to adverse selection. *(An earlier version of
  this bullet claimed the opposite. That came from a bug, since fixed, that labelled each fill with the book
  state its own sweep had created.)*
* **Negative results:** regime rules (D) and GLFT quoting with estimated fill intensity (H5) both did worse
  than simpler strategies; volatility adaptation slightly *lowered* P&L.

These numbers describe the project's simulated market, whose structure includes a planted signal. They show
the machinery separates information from noise and that conclusions hinge on fees and fill assumptions; they
say nothing yet about AAPL or MSFT. That requires the LOBSTER run below.

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

Python 3.11 or newer is required (`python3 --version`; macOS's built-in Python is usually too old, so install
3.11+ from python.org or Homebrew). On Windows, activate with `.venv\Scripts\activate`; experiments still run
there but backtests execute serially, because parallel workers rely on `fork`.

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
