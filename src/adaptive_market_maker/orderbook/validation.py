"""Event-by-event book-transition integrity check.

For every message ``i`` the visible book after message ``i-1`` plus the message's effect must equal the
source-provided book after message ``i`` wherever both snapshots are fully observable.

Effect of a message on the visible size at (side, price):
    type 1 submit: +size; types 2/3/4 cancel/delete/visible execution: -size; types 5/6/7: none.

The comparable region excludes prices beyond the deepest visible level of a *full* side, because a level can
slide into or out of a K-level window without any message touching it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from adaptive_market_maker.data.base import SessionData
from adaptive_market_maker.types import BUY, EventType

_DELTA_SIGN = {
    int(EventType.SUBMIT): 1,
    int(EventType.CANCEL): -1,
    int(EventType.DELETE): -1,
    int(EventType.EXEC_VISIBLE): -1,
}


class DataIntegrityError(RuntimeError):
    pass


@dataclass
class TransitionDiscrepancy:
    row: int
    message: dict
    side: str
    previous: dict[int, int]
    expected: dict[int, int]
    actual: dict[int, int]

    def describe(self) -> str:
        return (
            f"row {self.row} {self.message}: {self.side} book prev={self.previous} "
            f"expected={self.expected} actual={self.actual}"
        )


@dataclass
class TransitionReport:
    n_checked: int = 0
    n_discrepancies: int = 0
    other_side_changed: int = 0
    examples: list[TransitionDiscrepancy] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.n_discrepancies == 0 and self.other_side_changed == 0


def _levels(px: list[int], sz: list[int]) -> dict[int, int]:
    return {p: s for p, s in zip(px, sz) if s > 0}


def check_transitions(session: SessionData, max_examples: int = 10) -> TransitionReport:
    rep = TransitionReport()
    k = session.levels
    et, side, price, size = session.event_type, session.side, session.price, session.size
    # vectorised pre-check: the side NOT touched by a message must be byte-identical across the row pair
    bid_same = np.all(session.bid_px[1:] == session.bid_px[:-1], axis=1) & np.all(
        session.bid_sz[1:] == session.bid_sz[:-1], axis=1
    )
    ask_same = np.all(session.ask_px[1:] == session.ask_px[:-1], axis=1) & np.all(
        session.ask_sz[1:] == session.ask_sz[:-1], axis=1
    )
    touches_book = np.isin(et[1:], list(_DELTA_SIGN))
    msg_is_bid = side[1:] == BUY
    other_same = np.where(msg_is_bid, ask_same, bid_same)
    other_same_or_no_touch = np.where(touches_book, other_same, bid_same & ask_same)
    rep.other_side_changed = int((~other_same_or_no_touch).sum())

    for i in range(1, len(session)):
        t = int(et[i])
        sign = _DELTA_SIGN.get(t)
        is_bid = int(side[i]) == BUY
        if sign is None:
            continue  # untouched-book rows were verified by the vectorised check
        rep.n_checked += 1
        if is_bid:
            pp, ps = session.bid_px[i - 1].tolist(), session.bid_sz[i - 1].tolist()
            np_, ns = session.bid_px[i].tolist(), session.bid_sz[i].tolist()
        else:
            pp, ps = session.ask_px[i - 1].tolist(), session.ask_sz[i - 1].tolist()
            np_, ns = session.ask_px[i].tolist(), session.ask_sz[i].tolist()
        prev, new = _levels(pp, ps), _levels(np_, ns)
        expected = dict(prev)
        p = int(price[i])
        expected[p] = expected.get(p, 0) + sign * int(size[i])
        if expected[p] == 0:
            del expected[p]
        # comparable region: inside the deepest visible level of each *full* snapshot
        full_prev, full_new = len(prev) >= k, len(new) >= k
        if is_bid:
            bound = max(
                min(prev) if full_prev and prev else -np.inf, min(new) if full_new and new else -np.inf
            )
            comparable = lambda q, b=bound: q >= b  # noqa: E731
        else:
            bound = min(max(prev) if full_prev and prev else np.inf, max(new) if full_new and new else np.inf)
            comparable = lambda q, b=bound: q <= b  # noqa: E731
        keys = {q for q in set(expected) | set(new) if comparable(q)}
        bad = any(expected.get(q, 0) != new.get(q, 0) for q in keys) or any(v < 0 for v in expected.values())
        if bad:
            rep.n_discrepancies += 1
            if len(rep.examples) < max_examples:
                rep.examples.append(
                    TransitionDiscrepancy(
                        row=i,
                        message={
                            "type": t,
                            "side": int(side[i]),
                            "price": p,
                            "size": int(size[i]),
                            "order_id": int(session.order_id[i]),
                        },
                        side="bid" if is_bid else "ask",
                        previous=prev,
                        expected={q: v for q, v in expected.items() if comparable(q)},
                        actual={q: v for q, v in new.items() if comparable(q)},
                    )
                )
    return rep
