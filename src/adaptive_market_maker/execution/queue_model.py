"""Queue-position tracking and fill rules for simulated passive orders.

The simulated order is *not* inserted into history; we ask, for every historical message, "would this
message have filled our order?" under price-time priority. Three presets bracket the uncertainty:

conservative
    Queue ahead = displayed size at our price on arrival (x (1 + extra_queue_frac)). Only executions
    deplete it: cancellations are never credited as being ahead of us. No fills from crossing
    submissions or hidden executions.
central (default)
    Exact visible-queue tracking with order ids. Orders whose submission we never saw (origin -1)
    rested before us, so they are ahead; orders submitted after our arrival are behind. A cancellation
    of an order ahead reduces queue ahead; an execution of an order *behind* us means we would have
    been filled first. A historical limit order that would have crossed our price fills us.
    Hidden executions fill us only when they trade *through* our price.
optimistic
    Front of the queue on arrival ("fill on touch"). Shown only to quantify how much a naive fill
    assumption flatters results.

All presets fill on *trade-through*: an execution on our side at a price worse than our limit implies
an aggressor who would have hit our better-priced order first.

Not modelled (see docs/execution_assumptions.md): market impact of our orders on other participants,
removal of the liquidity we consume from history, hidden/reserve size ahead of us, odd-lot rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_market_maker.types import BUY, EventType

UNKNOWN_QUEUE = 1e18

_EXEC_VISIBLE = int(EventType.EXEC_VISIBLE)
_EXEC_HIDDEN = int(EventType.EXEC_HIDDEN)
_CANCEL = int(EventType.CANCEL)
_DELETE = int(EventType.DELETE)
_SUBMIT = int(EventType.SUBMIT)


@dataclass
class QueuePosition:
    side: int
    price: int
    arrival_seq: int  # first historical row processed after our arrival
    queue_ahead: float


@dataclass(frozen=True)
class FillRules:
    preset: str
    id_tracking: bool
    cancel_credit: bool
    crossing_add_fills: bool
    hidden_through_fills: bool
    hidden_same_price_fills: bool
    front_of_queue: bool
    extra_queue_frac: float = 0.0

    @classmethod
    def from_preset(cls, preset: str, extra_queue_frac: float = 0.0) -> FillRules:
        if preset == "conservative":
            return cls(preset, False, False, False, False, False, False, extra_queue_frac)
        if preset == "central":
            return cls(preset, True, True, True, True, False, False, 0.0)
        if preset == "optimistic":
            return cls(preset, False, False, True, True, True, True, 0.0)
        raise ValueError(f"unknown fill preset {preset!r}")

    def initial_queue(self, visible_at_price: int | None) -> float:
        if self.front_of_queue:
            return 0.0
        if visible_at_price is None:  # our price is deeper than the observable window
            return UNKNOWN_QUEUE
        return float(visible_at_price) * (1.0 + self.extra_queue_frac)

    def on_event(
        self,
        pos: QueuePosition,
        remaining: int,
        etype: int,
        ev_side: int,
        ev_price: int,
        ev_size: int,
        ev_origin: int,
    ) -> int:
        """Update ``pos`` for one historical message and return the quantity it fills (0 if none)."""
        s, p = pos.side, pos.price
        if etype in (_EXEC_VISIBLE, _EXEC_HIDDEN):
            if ev_side != s:
                return 0
            through = ev_price < p if s == BUY else ev_price > p
            if through:
                if etype == _EXEC_HIDDEN and not self.hidden_through_fills:
                    return 0
                return min(ev_size, remaining)
            if ev_price != p:
                return 0  # a better-priced resting order traded; it had price priority
            if etype == _EXEC_HIDDEN:
                # displayed orders have priority over hidden ones at the same price
                if self.front_of_queue or (self.hidden_same_price_fills and pos.queue_ahead <= 0):
                    return min(ev_size, remaining)
                return 0
            if self.front_of_queue:
                return min(ev_size, remaining)
            if self.id_tracking:
                if ev_origin >= pos.arrival_seq:  # the executed order was behind us
                    pos.queue_ahead = 0.0
                    return min(ev_size, remaining)
                pos.queue_ahead = max(0.0, pos.queue_ahead - ev_size)
                return 0
            consume = min(float(ev_size), pos.queue_ahead)
            pos.queue_ahead -= consume
            excess = ev_size - consume
            return min(int(excess), remaining) if excess >= 1 else 0
        if etype in (_CANCEL, _DELETE):
            if ev_side == s and ev_price == p and self.cancel_credit and ev_origin < pos.arrival_seq:
                pos.queue_ahead = max(0.0, pos.queue_ahead - ev_size)
            return 0
        if etype == _SUBMIT and self.crossing_add_fills and ev_side != s:
            crosses = ev_price <= p if s == BUY else ev_price >= p
            if crosses:
                return min(ev_size, remaining)
        return 0
