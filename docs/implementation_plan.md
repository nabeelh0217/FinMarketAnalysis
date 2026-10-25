# Implementation Plan — Adaptive Market Maker (AMM)

Living document. The checklist at the bottom is the source of truth for progress.

## 0. Environment snapshot (Phase 0)

| Item | Finding |
|---|---|
| Repository | Fresh: one commit, a 2-line README. Nothing to preserve beyond it. |
| Python | 3.11.17, 3.12.3, 3.13.16 available. Project venv uses **3.11** (the stated minimum). |
| Compute | 4 vCPU, 15 GB RAM, no GPU. Everything is CPU/local. |
| Network | PyPI reachable. `lobsterdata.com`, `data.lobsterdata.com`, `emi.nasdaq.com`, `databento.com` are **blocked by the sandbox network policy**. |
| Local market data | None. |

**Consequence:** the historical-data path is implemented and tested against LOBSTER-format files, but the
real LOBSTER sample cannot be downloaded from this sandbox. All end-to-end runs here use the synthetic
market generator (Mode 2), which writes files in LOBSTER format so the *same* ingestion → validation →
features → models → backtest → report path is exercised. Synthetic results are labelled as such everywhere and
are evidence about the *machinery*, never about real-market profitability. See §3 for how to plug in the
real files.

## 1. Review of the original specification — changes made and why

The spec is strong on research discipline. The points below are where I think it was wrong, ambiguous, or
would lead to a weaker project, and what this implementation does instead.

1. **Instrument choice: prefer a large-tick stock (MSFT/INTC) as primary, AAPL as contrast.** In the free
   LOBSTER sample (2012-06-21) AAPL traded around $580 with a spread of many ticks; queue position barely
   matters and top-of-book imbalance is a weak predictor. MSFT (~$30) and INTC (~$27) are *large-tick*: the spread is
   almost always one tick, queues are long, and queue position and queue imbalance dominate market-making
   economics. That is the setting where H2/H5 are most interesting. Configs exist for both; MSFT is the
   default once real data is present, and AAPL is the small-tick contrast for H6.
2. **One free day ≠ walk-forward.** The free LOBSTER samples are a single session per ticker. Cross-session
   walk-forward is therefore impossible on the free data; the protocol uses an intraday
   train → validation → test split with purge + embargo, and states plainly that one session cannot establish a
   durable edge. Multi-session walk-forward is implemented and exercised on synthetic sessions.
3. **The time-horizon label rule was subtly wrong.** "First observation at or after t+Δ" returns the state
   produced by an event *after* the horizon. The book is piecewise constant, so the correct value is the
   **as-of** state: the last event with timestamp ≤ t+Δ. Implemented as-of; the spec's variant is not offered.
4. **Same-timestamp bursts.** A single marketable order that sweeps several resting orders appears in LOBSTER as
   several messages sharing one timestamp. A row in the middle of such a burst is a book state nobody could
   observe. Decision points (features, labels, strategy wake-ups) are therefore placed **only on the last
   message of each timestamp group**. This is the concrete meaning of "data from the same event before that
   event could be observed".
5. **LOBSTER is better than "L2" for queue modelling.** LOBSTER messages carry order IDs. Any order whose ID was
   never seen being submitted was already resting before the file starts, so it was necessarily *ahead* of any
   simulated order placed later; any order submitted after the simulated order is *behind* it. Under price-time
   priority this yields exact visible-queue-ahead tracking (cancellations ahead are identifiable). Hidden and
   reserve liquidity remain unobservable. The "central" fill preset uses this; "conservative" ignores IDs and
   decrements the queue pro-rata; "optimistic" assumes front-of-queue.
6. **Avellaneda–Stoikov with τ = time to close is numerically absurd intraday.** With σ² per second and
   τ ≈ 23,400 s, the reservation-price skew is tens of dollars for a 100-share position. Three horizon
   conventions are implemented and documented: `session_end` (literal AS, for reference), `rolling` (τ = a
   fixed inventory-holding horizon, an honest heuristic, not "optimal"), and **GLFT** (Guéant–Lehalle–
   Fernandez-Tapia 2013 stationary approximation, which removes τ entirely). Units: q is measured in
   *lots*, prices in *ticks*, so γ has units 1/tick and k has units 1/tick.
7. **"Realized spread" vs "markout" were listed as different things; for a maker they are the same number at
   the same horizon.** The report uses the standard decomposition:
   effective half-spread `s·(M_t − P)` = realized spread `s·(M_{t+h} − P)` + adverse selection
   `−s·(M_{t+h} − M_t)`, with `s = +1` for a buy fill. The spec's `Markout_h` equals the realized spread.
8. **Primary metric must charge for leftover inventory.** Net P&L marked at mid flatters strategies that end
   the session holding inventory. The primary metric is **net P&L after fees with terminal inventory liquidated
   at the far touch (walking visible depth) plus a taker fee**. Mid-marked P&L is reported as secondary.
9. **Fees.** Real Nasdaq maker rebates (~$0.002–0.003/share) are 20–30% of a one-cent spread, so they can flip the
   sign of a large-tick market maker's P&L. Fee levels are illustrative and swept in the sensitivity analysis.
10. **No market impact of simulated orders.** Historical replay cannot know how other participants would have
    reacted to our quotes, and liquidity our order consumes is not removed from history. This is the largest
    unfixable limitation of replay. It is stated in `docs/limitations.md`.
11. **Added: planted-signal validation.** The synthetic generator has an `informed` scenario (latent fair
    value + informed order flow that creates real adverse selection and a real, known predictive signal) and a
    `null` scenario (no information). The pipeline must *find* the signal in the first and *not* find
    profitable alpha in the second. This is the most convincing evidence that the backtester is not biased.
12. **Scope control.** ITCH parsing, Databento, HMM regimes, and deep learning are deliberately deferred.
    Strategy D is implemented as simple causal regime rules only.
13. **CLI framework:** `argparse` (stdlib) instead of Typer, to keep the dependency set small.
14. **Calibration:** temperature scaling implemented directly (one parameter, fit on validation) rather than
    relying on scikit-learn's calibration API, which changed across recent versions.
15. **H1 and H4 are risk-adjusted hypotheses, so they get a risk-adjusted test.** The spec makes net P&L
    the primary metric but phrases H1/H4 as "risk-adjusted". Both are reported: raw net P&L differences (primary)
    and a paired bootstrap of the difference in mean/std of interval P&L, which decides H1/H4.
16. **Bootstrap edge bias (found and fixed during the work).** A moving-block bootstrap under-samples each
    session's first and last interval; the last one is the flatten window and systematically different, which
    pushed bootstrap means above the sample mean. Replaced by a circular block bootstrap.
17. **Placebo test added.** Beyond the null market, strategy C is re-run with its own alpha shifted by half a
    session. A placebo that beat the no-alpha strategy would reveal an engine that rewards asymmetric quoting
    rather than information.
18. **Avoid confounded ablations.** "No volatility adaptation" must hold σ at a *typical* value; a far-too-low
    constant silently removes the inventory skew as well (caught in the first full run).

## 2. Architecture (implemented)

```
src/adaptive_market_maker/
  config.py            pydantic-validated config sections; YAML loader + overlays
  data/                LOBSTER parser, synthetic generator, normalization, manifest, validation
  orderbook/           book snapshot helpers, transition-integrity checks
  features/            causal vectorized features (book, flow, volatility) + pipeline
  labels/              event- and time-horizon labels (labels only; never fed to strategies)
  models/              baseline, logistic, HGB, temperature calibration, alpha mapping, fill model
  execution/           orders + lifecycle, latency, queue tracking, simulated exchange
  accounting/          integer micro-dollar ledger, P&L, reconciliation
  strategies/          fixed-spread, inventory-aware (AS/GLFT), alpha-aware, regime, risk manager
  backtest/            event-driven engine, splits (purge/embargo), results
  evaluation/          metrics, markouts, block bootstrap, ablation, sensitivity
  experiments/         registry (JSONL), runner
  reporting/           matplotlib figures, markdown report
  cli.py               `amm` command
```

## 3. Getting the real data in

1. Download from <https://data.lobsterdata.com/info/DataSamples.php> (free, registration-free sample):
   `LOBSTER_SampleFile_MSFT_2012-06-21_10.zip` and `LOBSTER_SampleFile_AAPL_2012-06-21_10.zip`.
2. Unzip into `data/raw/lobster/` so that e.g.
   `data/raw/lobster/MSFT_2012-06-21_34200000_57600000_message_10.csv` exists.
3. `amm data validate --config configs/data.yaml` then `amm experiment run --config configs/experiments/lobster_msft.yaml`.

## 4. Risks

| Risk | Mitigation |
|---|---|
| Real data unavailable in sandbox | LOBSTER-format synthetic files exercise the exact parser; instructions above. |
| Python event loop too slow | Features/predictions precomputed vectorized; loop only does matching + quoting. |
| Backtest optimism | Three fill presets, post-only rejects, latency, liquidation-based primary metric, null-scenario test. |
| Overfitting on validation | Small pre-declared grids, registry of every run, test window touched once per experiment. |

## 5. Checklist

- [x] Phase 0 — environment inspected, plan written
- [x] Phase 1 — skeleton, config (pydantic, strict), argparse CLI, structured logging, tests
- [x] Phase 2 — synthetic order-level market, order lifecycle, queue tracker, exchange, integer ledger, reconciliation
- [x] Phase 3 — LOBSTER adapter (exact time parsing, row-count check), normalization, manifest (sha256), validation,
      event-by-event book-transition reconciliation. **Real files not obtainable in the sandbox**; exercised on
      LOBSTER-layout synthetic files; `pytest -m needs_lobster` runs on the real sample once present.
- [x] Phase 4 — causal features on observable states, as-of labels, future-perturbation tests
- [x] Phase 5 — prior / logistic / HGB, temperature calibration, alpha mapping, purged chronological evaluation
- [x] Phase 6 — fixed-spread strategy, latency-aware engine, integration test
- [x] Phase 7 — inventory-aware (AS rolling / session-end, GLFT), risk limits (worst-case inventory)
- [x] Phase 8 — alpha-aware strategy; C inherits B's parameters so C − B isolates alpha (tested: C without alpha ≡ B)
- [x] Phase 9 — shadow-order fill model (logistic + Kaplan–Meier + intensity fit), three fill presets
- [x] Phase 10 — out-of-sample comparison, paired block bootstrap (raw and risk-adjusted), ablations, sensitivity
- [x] Phase 11 — report generator, docs, reproduction instructions
- [x] Real LOBSTER MSFT 2012-06-21 (10 levels) validated on the user's machine: 668,765 messages, all 665,148
      book-changing messages reconciled against LOBSTER's snapshots with zero discrepancies; 241 identical
      consecutive messages and 3 timestamps with >9 decimals flagged (see note below)
- [x] `lobster_msft.yaml` run locally (results summarized in README; H3 supported, H1 not established)
- [x] Fixed two analysis-only measurement bugs found while reading the MSFT report: fill-time mid taken
      mid-sweep, and fills classified by the post-fill book state (P&L, fills and realized spreads unaffected)
- [ ] `lobster_aapl.yaml` (small-tick contrast)

## 6. Phase notes

**Phases 1–3.** Added `config.py`, `types.py`, `data/` (base, lobster, normalize, synthetic, validate, manifest,
prepare), `orderbook/validation.py`. Gate: package installs, `amm --help` works; every synthetic message
reconciles against its book snapshot; corrupted books raise `DataIntegrityError`.

**Phase 2 finding.** The first synthetic calibration had almost no adverse selection (fixed-spread maker's
30 s realized spread 0.36 of a 0.46-tick effective half-spread), which made naive market making look
unrealistically easy. Informed-flow intensity and latent-value volatility were raised until adverse selection
consumed ~80% of the half-spread at 30 s, closer to what is documented for large-tick equities. The `null`
scenario keeps total aggressive flow equal (3/s) but uninformed.

**Phases 4–9.** Features, labels, models, fill model, strategies, risk, engine. Gates: hand-calculated feature,
OFI, microprice, AS/GLFT, ledger and markout tests; perturbing future events leaves earlier features, fills and
decisions unchanged; no fill precedes order arrival; inventory never exceeds the limit; replay is
deterministic; conservative ≤ central ≤ optimistic fill counts.

**Real-data validation.** The duplicate-message warning is informative, not a defect: a duplicated
visible message (types 1–4) would break the book reconciliation, which passed for every message, so those
rows are separate real events. The validator now reports the duplicates by message type. The 3 truncated
timestamps lose less than a nanosecond and timestamps remain monotonic.

**Phases 10–11.** Experiment runner, registry, statistics, plots, report, docs. Results: see README and
`reports/`.
