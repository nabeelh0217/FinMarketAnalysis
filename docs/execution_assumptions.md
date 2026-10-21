# Execution assumptions

What the historical feed reveals, what the simulator must assume, and how each assumption is varied.

## What LOBSTER reveals

* Every message that changes the requested top-K levels: submissions, partial cancels, deletions, visible
  executions; plus hidden executions, cross trades and halt markers. Each carries an order id.
* The aggregated top-K book after each message.

## What it does not reveal

* Orders resting **before** the file starts (their ids first appear when they are cancelled or executed).
* Messages at prices beyond the K-th level.
* Hidden or reserve size resting at our price, and its priority relative to us.
* How other participants would have reacted to *our* quotes.

## Queue position (`execution/queue_model.py`)

A simulated passive order is never inserted into history. For each historical message we ask whether it would
have filled our order under price-time priority.

| preset | queue ahead on arrival | cancellations | executions at our price | crossing submissions | hidden executions |
|---|---|---|---|---|---|
| conservative | displayed size × (1 + extra) | never credited | consume queue first, excess fills us | no fill | no fill |
| **central** (default) | displayed size | credited only if the cancelled order rested before us (by order id) | an execution of an order that arrived *after* us fills us; one that arrived before reduces the queue | fill | fill only if it trades *through* our price |
| optimistic | 0 (front of queue) | — | every execution at our price fills us ("fill on touch") | fill | fill |

All presets fill on a **trade-through**: an execution on our side at a price worse than our limit means an
aggressor reached past our price, so our better-priced order would have traded first.

**Why order ids make "central" nearly exact for visible liquidity.** An order id never seen being submitted
belongs to an order that rested before the file began (or was entered beyond the recorded depth), so it was
ahead of any order we place during the file. An order submitted after our arrival is behind us. Price-time
priority then determines exactly which cancellations and executions are ahead of us. Hidden size ahead of us
remains unknowable, which is what the conservative preset's `conservative_extra_queue_frac` is for.

An additional consistency cap: queue ahead can never exceed the displayed size at our price.

## Latency (`execution/latency.py`, `backtest/engine.py`)

Constant one-way delays (defaults: 250 µs market data, 20 µs decision, 250 µs order entry, 250 µs reports).

1. An event with exchange timestamp t is observable from t + market_data.
2. A wake-up decides on the newest observable decision state.
3. Orders and cancels take effect at the venue order_entry after they are sent.
4. Fills and acknowledgements reach the strategy report later; until then the strategy's known position is
   stale. Risk limits use the *worst case* (known position + every open order), so the true position never
   exceeds the limit.

Ties: an agent event scheduled at exactly the timestamp of a historical message is processed **after** it.
Latency is swept from 0 to 20 ms in the sensitivity analysis. Production latency distributions (jitter, gateway
queues, matching-engine delays) are not reproduced.

## Order lifecycle (`execution/orders.py`)

`pending_new → active → partially_filled → filled`, `→ pending_cancel → cancelled`, `→ rejected`.
Impossible transitions raise. A cancel never undoes fills that happened before it reached the venue; a cancel
arriving after the order filled is a no-op ("too late to cancel").

* Quotes are **post-only**: an order that would cross the book on arrival is rejected, never converted into a
  taker order.
* Quote management keeps a resting order whenever its price still matches the intent (preserving queue
  priority) and cancel-replaces otherwise.
* On a halt message the venue cancels resting orders; new orders are rejected until trading resumes.
* At `flatten_before_end_s` before the segment end, quoting stops and inventory is closed with a marketable IOC
  order that walks visible depth and pays the taker fee. Anything left is valued at the liquidation price.

## Fees and rebates

Per share, maker and taker separately (`FeeConfig`). Defaults (−$0.0020 maker rebate, $0.0030 taker fee) are
**illustrative**, in the range of public exchange schedules, not a verified schedule for any venue or date.
They are swept from −$0.003 to +$0.001 per share. Regulatory fees on sales are not modelled.

## Not modelled

* Market impact of our orders; removal of the liquidity our fills consume from history.
* Odd-lot and mixed-lot display rules; midpoint-peg interactions beyond the hidden-execution rule.
* Auction participation (type-6 cross trades never fill simulated orders).
* Partial-fill reporting delays beyond the constant report latency.
