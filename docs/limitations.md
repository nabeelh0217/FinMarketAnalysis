# Limitations

A historical backtest is never proof of future profitability. These are the specific ways this one can mislead.

## Data

* **No real market data was available in the development sandbox.** Its network policy blocked
  lobsterdata.com, Nasdaq's ITCH samples and Databento. Every result produced so far is **synthetic**. The
  LOBSTER path is implemented and tested against files in LOBSTER's exact layout, but has not yet been run on
  the real samples.
* **Synthetic ≠ real.** The synthetic market (Poisson order flow, a latent fair value, informed market orders)
  is the project's own model. Its parameters were tuned so that the economics look plausible (one-tick
  spreads ~90% of the time, adverse selection eating most of the half-spread), not calibrated to any instrument.
  A strategy that exploits the generator's structure would not necessarily work on real data, and the presence
  of a planted signal guarantees that *some* signal exists there.
* **The free LOBSTER sample is one day per ticker (2012-06-21).** One session cannot establish a durable edge,
  supports only an intraday split, and gives few independent 60-second intervals. Market structure, fees and
  tick-size regimes have changed since 2012.
* **Partial visibility.** Messages beyond the requested depth are absent; hidden and reserve liquidity are
  invisible until executed; orders resting before the file starts have unknown submission times.
* **Timestamps** are exchange event times as published by LOBSTER, not receive times at a trading firm.

## Execution

* **No market impact.** Our simulated orders change neither other participants' behaviour nor the historical
  record. Liquidity our fills consume is still available to the historical flow. This is the largest
  unfixable limitation of historical replay and biases results upward for any strategy that trades often.
* **Queue uncertainty.** The central preset is exact for *visible* queue ahead, but hidden size ahead of us is
  unknown. The fill-preset sensitivity shows how much this matters.
* **Constant latency.** No jitter, no queueing, no exchange-side processing delays, and no co-location
  advantage of other participants.
* **Fees are illustrative.** Rebates are a large share of a one-tick spread; real tiered schedules depend on
  volume and change over time.
* **Flattening** walks visible depth at one instant and ignores the impact of doing so.

## Modelling

* **Model misspecification.** Avellaneda–Stoikov assumes Brownian prices, Poisson fills with exponential
  intensity and CARA utility; none hold exactly. The rolling-τ variant is a heuristic, and on a one-tick book
  the continuous formulas act only through rounding.
* **Probability-to-alpha mapping** uses class-conditional mean moves from the training period, which may drift.
* **Regimes** are defined by simple training-period thresholds; regime-conditional results are descriptive.

## Statistics

* **Dependence.** The block bootstrap captures dependence up to the block length (3 minutes) within a session;
  it does not capture session-level or regime-level dependence. With few sessions, intervals understate
  uncertainty about *future* sessions.
* **Selection bias.** Validation grids, sensitivity sweeps and ablations multiply the number of configurations
  examined (all recorded in `trials.csv`). Only the pre-declared final comparison is confirmatory; picking the
  best sensitivity point would overstate performance.
* **Generalization** beyond the tested instrument, dates and market (synthetic or real) is not established.
