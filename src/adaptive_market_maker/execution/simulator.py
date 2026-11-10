"""Simulated venue: matches *our* orders against the historical message stream.

The exchange is separate from the strategy: strategies can only send orders and cancels through the
engine; only this module creates fills. It sees the historical book exactly as of the last processed
message, so an order can never trade against a book state from before its arrival.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_market_maker.config import ExecutionConfig
from adaptive_market_maker.data.base import SessionData
from adaptive_market_maker.execution.orders import Order, OrderState
from adaptive_market_maker.execution.queue_model import FillRules, QueuePosition
from adaptive_market_maker.types import BUY, HALT_START, HALT_TRADE_RESUME, EventType

_HALT = int(EventType.HALT)
_CROSS = int(EventType.CROSS)


@dataclass
class Fill:
    ts_ns: int
    order_id: int
    side: int
    price: int
    qty: int
    liquidity: str  # "maker" | "taker"
    row: int  # historical row that caused it (-1 for taker fills at arrival)
    reason: str  # queue | through | crossing_add | hidden | taker


def visible_size(session: SessionData, row: int, side: int, price: int) -> int | None:
    """Displayed size at ``price`` after ``row``; None if ``price`` is outside the observable window."""
    px = session.bid_px[row] if side == BUY else session.ask_px[row]
    sz = session.bid_sz[row] if side == BUY else session.ask_sz[row]
    hit = np.nonzero((px == price) & (sz > 0))[0]
    if hit.size:
        return int(sz[hit[0]])
    live = sz > 0
    if live.all():  # full window: prices beyond the deepest level are unobservable
        deepest = int(px[-1])
        if (side == BUY and price < deepest) or (side != BUY and price > deepest):
            return None
    return 0


class SimulatedExchange:
    def __init__(self, session: SessionData, cfg: ExecutionConfig):
        self.s = session
        self.rules = FillRules.from_preset(cfg.fill_preset, cfg.conservative_extra_queue_frac)
        self.last_row = -1
        self.halted = False
        self.resting: list[tuple[Order, QueuePosition]] = []
        self.n_rejects = 0
        # local refs for the hot loop
        self._et = session.event_type
        self._side = session.side
        self._price = session.price
        self._size = session.size
        self._origin = session.origin_seq

    # ------------------------------------------------------------------ queries
    def best_bid(self) -> int:
        return int(self.s.bid_px[self.last_row, 0])

    def best_ask(self) -> int:
        return int(self.s.ask_px[self.last_row, 0])

    def book_valid(self) -> bool:
        return self.last_row >= 0 and bool(self.s.valid_book[self.last_row])

    # ------------------------------------------------------------------ agent actions
    def on_order_arrival(self, order: Order, t_ns: int) -> list[Fill]:
        order.arrival_ns = t_ns
        if order.state == OrderState.CANCELLED or order.is_terminal:
            return []
        if self.halted or not self.book_valid():
            self._reject(order, t_ns, "halted" if self.halted else "no_valid_book")
            return []
        bb, ba = self.best_bid(), self.best_ask()
        crosses = order.price >= ba if order.side == BUY else order.price <= bb
        if order.ioc:
            order.at_venue = True
            fills = self._take(order, t_ns)
            if not order.is_terminal:
                order.transition(OrderState.CANCELLED, t_ns)  # IOC remainder
            return fills
        if crosses and order.post_only:
            self._reject(order, t_ns, "post_only_would_cross")
            return []
        if crosses:
            raise NotImplementedError("non-post-only crossing limit orders are not supported; use ioc")
        order.at_venue = True
        if order.state == OrderState.PENDING_NEW:
            order.transition(OrderState.ACTIVE, t_ns)
        vis = visible_size(self.s, self.last_row, order.side, order.price)
        pos = QueuePosition(order.side, order.price, self.last_row + 1, self.rules.initial_queue(vis))
        self.resting.append((order, pos))
        return []

    def on_cancel_arrival(self, order: Order, t_ns: int) -> None:
        if order.is_terminal:
            return  # too late: already filled (or rejected); the fill stands
        order.transition(OrderState.CANCELLED, t_ns)
        self._drop(order)

    def _reject(self, order: Order, t_ns: int, reason: str) -> None:
        order.reject_reason = reason
        order.transition(OrderState.REJECTED, t_ns)
        self.n_rejects += 1

    def _drop(self, order: Order) -> None:
        self.resting = [(o, p) for (o, p) in self.resting if o is not order]

    def _take(self, order: Order, t_ns: int) -> list[Fill]:
        """Marketable IOC against the visible book (walks levels up to the limit). No impact on history."""
        row = self.last_row
        fills: list[Fill] = []
        if order.side == BUY:
            px, sz = self.s.ask_px[row], self.s.ask_sz[row]
            ok = lambda p: p <= order.price  # noqa: E731
        else:
            px, sz = self.s.bid_px[row], self.s.bid_sz[row]
            ok = lambda p: p >= order.price  # noqa: E731
        for lvl in range(self.s.levels):
            if order.remaining == 0 or sz[lvl] <= 0 or not ok(int(px[lvl])):
                break
            q = min(order.remaining, int(sz[lvl]))
            order.apply_fill(q, t_ns)
            fills.append(Fill(t_ns, order.order_id, order.side, int(px[lvl]), q, "taker", -1, "taker"))
        return fills

    # ------------------------------------------------------------------ historical events
    def on_market_event(self, i: int, t_ns: int) -> list[Fill]:
        self.last_row = i
        et = int(self._et[i])
        if et == _HALT:
            code = int(self._price[i])
            if code == HALT_START:
                self.halted = True
                for o, _ in list(self.resting):
                    o.transition(OrderState.CANCELLED, t_ns)
                self.resting = []
            elif code == HALT_TRADE_RESUME:
                self.halted = False
            return []
        if et == _CROSS or not self.resting:
            return []
        fills: list[Fill] = []
        done = False
        for order, pos in self.resting:
            q = queue_step(self.s, self.rules, pos, order.remaining, i)
            if q > 0:
                order.apply_fill(q, t_ns)
                fills.append(
                    Fill(
                        t_ns,
                        order.order_id,
                        order.side,
                        order.price,
                        q,
                        "maker",
                        i,
                        _reason(et, int(self._side[i]), int(self._price[i]), pos),
                    )
                )
                done = done or order.is_terminal
        if done:
            self.resting = [(o, p) for (o, p) in self.resting if not o.is_terminal]
        return fills


def queue_step(session: SessionData, rules: FillRules, pos: QueuePosition, remaining: int, i: int) -> int:
    """Apply historical message ``i`` to one passive order; return the quantity filled.

    Shared by the exchange and by the shadow-order fill-probability labels so both use identical rules.
    """
    ev_side, ev_price = int(session.side[i]), int(session.price[i])
    q = rules.on_event(
        pos,
        remaining,
        int(session.event_type[i]),
        ev_side,
        ev_price,
        int(session.size[i]),
        int(session.origin_seq[i]),
    )
    if q == 0 and ev_side == pos.side and ev_price == pos.price and pos.queue_ahead > 0:
        # nobody can be ahead of us beyond what is displayed at our price
        vis = visible_size(session, i, pos.side, pos.price)
        if vis is not None and vis < pos.queue_ahead:
            pos.queue_ahead = float(vis)
    return q


def _reason(et: int, ev_side: int, ev_price: int, pos: QueuePosition) -> str:
    if et == int(EventType.SUBMIT):
        return "crossing_add"
    if et == int(EventType.EXEC_HIDDEN):
        return "hidden"
    if ev_price != pos.price:
        return "through"
    return "queue"
