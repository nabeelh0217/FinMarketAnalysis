# Mathematical model

Every formula below is the one implemented; the file and function are given so each can be checked.

## Units and conventions

| quantity | unit | notes |
|---|---|---|
| time `t` | integer ns after midnight | exchange timestamps, parsed exactly |
| price (storage) | integer *price units* = $ × 10 000 | LOBSTER convention; tick = 100 units = $0.01 |
| price (models, strategies) | ticks (float) | converted back to the tick grid by `round_quotes` |
| quantity `Q`, inventory `q` (accounting) | shares | |
| inventory `q` (quoting formulas) | lots of 100 shares | the AS/GLFT models assume unit order size |
| cash | integer micro-dollars | fills and per-share fees are exact |
| volatility `σ` | ticks / √s | realized, causal (see below) |
| alpha `α` | ticks over the label horizon | expected mid change, not a probability |

Side `s = +1` for a buy, `−1` for a sell. LOBSTER's `Direction` for an execution is the side of the
*resting* order, so the aggressor sign is `−Direction`.

## Observable states

A message group sharing one timestamp (e.g. a marketable order sweeping several resting orders) produces
several rows. Only the book after the *last* row of a group could ever be observed; features, labels and
decisions are defined on these observable states only (`SessionData.last_in_group`).

## Book features (`features/pipeline.py`)

For best bid `b`, best ask `a`, displayed sizes `Q^b`, `Q^a`:

* mid `m = (a + b)/2`, spread `s = a − b` (ticks)
* top-of-book imbalance `I = (Q^b − Q^a)/(Q^b + Q^a)`, undefined (NaN) when the book is one-sided
* microprice `μ = (a·Q^b + b·Q^a)/(Q^b + Q^a)`; feature `μ − m`. Weighting the ask by *bid* size moves μ toward
  the side that is more likely to be hit next. A candidate signal, not proof of alpha.
* multi-level imbalance over the first K levels with weights `w_i` (uniform / linear decay / exponential):
  `I^(K) = (Σ w_i Q^b_i − Σ w_i Q^a_i) / (Σ w_i Q^b_i + Σ w_i Q^a_i)`
* log depth ratio `log(1 + Σ Q^b_i) − log(1 + Σ Q^a_i)`

## Order-flow imbalance (Cont, Kukanov & Stoikov, 2014)

Per message n, with best prices/sizes after (n) and before (n−1):

```
e_n = 1{P^b_n ≥ P^b_{n−1}} Q^b_n − 1{P^b_n ≤ P^b_{n−1}} Q^b_{n−1}
    − 1{P^a_n ≤ P^a_{n−1}} Q^a_n + 1{P^a_n ≥ P^a_{n−1}} Q^a_{n−1}
```

summed over the last n observable states (`ofi_e{n}`) or over messages in `(t − W, t]` (`ofi_t{W}`), in lots.
A decrease of displayed size is **not** assumed to be a cancellation: OFI is agnostic to why depth changed,
while the separate activity features count LOBSTER message types (2/3 cancels, 1 submissions, 4/5 executions).

## Trade flow and activity

* signed volume `Σ −Direction · size` over executions (types 4 and 5) in the window, in lots
* trade imbalance = signed volume / total executed volume
* market-order rate: number of timestamp groups containing executions, per second
* cancel / submit / book-update rates per second; signed volume submitted *at the touch*
* time since the last mid change and since the last execution; fraction of the session elapsed

## Volatility

`σ_t = sqrt( Σ_{states in (t−W, t]} (Δm)² / min(t − t_0, W) )` in ticks/√s, floored at
`vol_floor_ticks_per_sqrt_s`. Purely backward-looking; never uses realized future volatility.

## Labels (`labels/targets.py`) — training/evaluation only

* time horizon h: `Δm = m(as-of t + h) − m(t)` where *as-of* means the state after the last message with
  timestamp ≤ t + h. (The book is piecewise constant; the "first observation at or after t + h" would use
  a state created after the horizon.)
* event horizon H: `Δm = m_{k+H} − m_k` over observable states.
* ternary class `sign(Δm)` with a flat band `|Δm| ≤ flat_threshold_ticks`; binary "up" optional.
* `label_end_ns` = latest timestamp the label depends on; used to purge samples at split boundaries.

## Direction models and alpha (`models/price_direction.py`, `models/calibration.py`)

Candidates: constant class prior, standardized multinomial logistic regression, histogram gradient
boosting. Probabilities are calibrated with a single temperature `T` fit on validation:
`p̃_c ∝ p_c^{1/T}`.

A probability is not an expected return. Alpha is the conditional expectation

```
α = Σ_c p̃_c · E_train[Δm | class c]          (ticks over the label horizon)
```

with the class-conditional means estimated on training rows only.

## Fill model (`models/fill_probability.py`)

Shadow 1-lot orders placed at the best price or `d` ticks behind it are run through the exchange's own queue
rules for `horizon_s` seconds, giving (filled?, time to fill, censored?). Then:

* logistic `P(fill within horizon | queue ahead, imbalance on our side, spread, σ, MO rate, aggression toward us, …)`
* Kaplan–Meier `S(t) = Π_{t_i ≤ t} (1 − d_i/n_i)` for time to fill with right-censoring at the horizon
* exponential intensities `λ(δ) = fills / exposure time` at distance `δ = s/2 + d` from the mid, and a
  log-linear fit `λ(δ) = A e^{−kδ}`, giving data-driven `A` (1/s) and `k` (1/tick).

## Strategy A — fixed spread (`strategies/fixed_spread.py`)

`p^bid = m − h`, `p^ask = m + h`, snapped to the grid. With `h = ½` tick on a one-tick book this joins the touch.

## Strategy B — inventory-aware (`strategies/inventory_aware.py`)

Avellaneda & Stoikov (2008), CARA utility, arithmetic Brownian mid, fill intensity `A e^{−kδ}`:

```
reservation   r = S − q γ σ² τ
half-spread   δ = (1/γ) ln(1 + γ/k) + ½ γ σ² τ
quotes        p^bid = r − δ,  p^ask = r + δ
```

Units: with `q` in lots, `σ` in ticks/√s and `τ` in s, `q γ σ² τ` is in ticks when γ is in 1/tick, i.e.
risk aversion per unit lot of P&L measured in ticks. Doubling the lot size at fixed γ doubles the skew per share
held, so γ must be re-chosen if `lot_size` changes.

The horizon is a modelling choice, made explicit:

* `session_end`: τ = time to the end of the trading window. Literal AS. Intraday, σ²τ is so large early in the
  day that the skew per lot is many ticks: shown for reference only.
* `rolling` (default): τ = `tau_s`, interpreted as the horizon over which inventory is expected to be held.
  A heuristic, **not** the AS optimum.
* `glft`: Guéant, Lehalle & Fernandez-Tapia (2013) asymptotic (stationary) approximation, no τ:
  ```
  ω       = sqrt( σ² γ / (2 k A) · (1 + γ/k)^(1 + k/γ) )
  δ^bid   = (1/γ) ln(1 + γ/k) + (2q + 1)/2 · ω
  δ^ask   = (1/γ) ln(1 + γ/k) − (2q − 1)/2 · ω
  ```
  i.e. skew `q·ω` per lot and half-spread `c + ω/2`.

`spread_mode = fixed` keeps the model's skew but uses A's half-spread, so B − A isolates inventory control.

**Discrete ticks.** On a one-tick book a continuous skew can only act through rounding. Quotes are rounded
to the *nearest* tick with exact halves resolved toward the passive side, so the skew moves a quote once it
exceeds half a tick (γ sets that inventory threshold). The alternative (`conservative`: bid down, ask up) moves
a quote a full tick for any non-zero skew, which makes γ irrelevant.

## Strategy C — alpha-aware (`strategies/adaptive.py`)

```
r = S + w · clip(α, ±α_max) − q · skew_per_lot
```

plus an *adverse-selection guard*: if `α > g` the ask is withdrawn, if `α < −g` the bid is withdrawn. Rationale:
a passive fill on the side alpha points against has expected markout ≈ −|α| ticks, which exceeds the
half-spread plus rebate it would earn once |α| is large enough. `g` is selected on validation data.

## Strategy D — regime rules

C plus causal rules: if σ ≥ `high_vol_ticks_per_sqrt_s` the quote size is multiplied by `high_vol_size_mult`
(default 0: stop quoting); if the market spread ≥ `wide_spread_ticks`, add `wide_spread_extra_ticks` to the
half-spread. Thresholds are configuration, not fit on test data.

## Accounting (`accounting/ledger.py`)

For a fill of signed quantity Δq at price P with per-share fee f (maker rebates are negative fees):

```
C_t = C_{t−1} − Δq·P − |Δq|·f        q_t = q_{t−1} + Δq        E_t = C_t + q_t·M_t
```

Realized P&L uses average cost and is gross of fees, so `E_t = realized + unrealized(M_t) − fees` exactly
(`Ledger.reconcile`, called at the end of every backtest). Fees are counted once, in the fee line.

**Primary metric:** net P&L with residual inventory liquidated at the far touch, walking visible depth, plus a
taker fee. Mid-marked P&L is reported alongside.

## Markouts and the spread decomposition (`evaluation/markouts.py`)

With `M_0` the mid of the quote prevailing *before the aggressive order arrived* (the observable state before
the fill's timestamp group: rows inside a sweep are already partly consumed by the same order) and as-of
mid `M_h` at fill time + h:

```
effective half-spread   e    = s (M_0 − P)
adverse selection       AS_h = −s (M_h − M_0)
realized spread         RS_h = e − AS_h = s (M_h − P)
```

`RS_h` equals the spec's `Markout_h`. Regime breakdowns of fills likewise use the state *before* the
fill's timestamp, never the post-fill book. Fees are not inside these numbers; they are added once when computing
net dollars per share. Fills whose horizon runs past the data are excluded and counted.

## Statistics (`evaluation/statistics.py`)

P&L is aggregated into fixed clock intervals (default 60 s) per session. A circular block bootstrap (blocks of
3 consecutive intervals that wrap around within a session and never span sessions) gives intervals for the
mean. Wrapping matters: a plain moving-block bootstrap under-samples each session's first and last interval,
and the last one (the flatten window) is systematically different, which biased an earlier version upward; strategy comparisons resample
the *paired* differences on the same intervals, and the risk-adjusted comparison resamples blocks jointly and
recomputes `mean/std` of interval P&L for each strategy. Nothing is annualized.
