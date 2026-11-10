"""Simulated order objects and their lifecycle state machine.

A cancel request never retroactively undoes fills: an order stays fillable at the venue until the cancel
*arrives* there. Impossible transitions raise :class:`InvalidTransition`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class OrderState(StrEnum):
    PENDING_NEW = "pending_new"  # sent, not yet at the venue
    ACTIVE = "active"  # resting, nothing filled
    PARTIALLY_FILLED = "partially_filled"  # resting, some quantity filled
    PENDING_CANCEL = "pending_cancel"  # cancel sent but not yet effective; still fillable at the venue
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


TERMINAL = frozenset({OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED})

_ALLOWED: dict[OrderState, frozenset[OrderState]] = {
    OrderState.PENDING_NEW: frozenset(
        {
            OrderState.ACTIVE,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.PENDING_CANCEL,
            OrderState.CANCELLED,
        }
    ),
    OrderState.ACTIVE: frozenset(
        {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.PENDING_CANCEL, OrderState.CANCELLED}
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.PENDING_CANCEL, OrderState.CANCELLED}
    ),
    OrderState.PENDING_CANCEL: frozenset(
        {OrderState.PENDING_CANCEL, OrderState.CANCELLED, OrderState.FILLED, OrderState.REJECTED}
    ),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
}


class InvalidTransition(RuntimeError):
    pass


@dataclass
class Order:
    order_id: int
    side: int  # +1 buy, -1 sell
    price: int  # price units
    qty: int
    sent_ns: int
    post_only: bool = True
    ioc: bool = False
    tag: str = "quote"
    state: OrderState = OrderState.PENDING_NEW
    filled: int = 0
    at_venue: bool = False
    arrival_ns: int | None = None
    cancel_sent_ns: int | None = None
    done_ns: int | None = None
    reject_reason: str | None = None
    # what the strategy has been told so far (reports travel with latency)
    known_filled: int = 0
    known_terminal: bool = False
    history: list[tuple[int, str]] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.qty - self.filled

    @property
    def known_remaining(self) -> int:
        return 0 if self.known_terminal else self.qty - self.known_filled

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL

    @property
    def is_resting(self) -> bool:
        return self.at_venue and not self.is_terminal

    def transition(self, new: OrderState, t_ns: int) -> None:
        if new not in _ALLOWED[self.state]:
            raise InvalidTransition(f"order {self.order_id}: {self.state.value} -> {new.value}")
        self.state = new
        self.history.append((t_ns, new.value))
        if new in TERMINAL:
            self.done_ns = t_ns

    def apply_fill(self, qty: int, t_ns: int) -> None:
        if qty <= 0:
            raise ValueError("fill quantity must be positive")
        if not self.is_resting and not (self.ioc and self.state == OrderState.PENDING_NEW):
            raise InvalidTransition(f"order {self.order_id} cannot fill in state {self.state.value}")
        if qty > self.remaining:
            raise InvalidTransition(f"order {self.order_id}: fill {qty} exceeds remaining {self.remaining}")
        self.filled += qty
        if self.remaining == 0:
            self.transition(OrderState.FILLED, t_ns)
        elif self.state != OrderState.PENDING_CANCEL:
            self.transition(OrderState.PARTIALLY_FILLED, t_ns)
