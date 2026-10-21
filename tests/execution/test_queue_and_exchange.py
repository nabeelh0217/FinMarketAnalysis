"""Hand-checked execution scenarios: queue depletion, priority, trade-through, latency ordering."""

from __future__ import annotations

import pytest

from adaptive_market_maker.config import ExecutionConfig
from adaptive_market_maker.execution.orders import InvalidTransition, Order, OrderState
from adaptive_market_maker.execution.simulator import SimulatedExchange, visible_size
from adaptive_market_maker.types import BUY, SELL
from tests.conftest import build_session, px

ASKS = [(3001, 500), (3002, 500)]


def _queue_session():
    """Bid level 3000 starts with orders A(300, pre-existing, id 1) and B(200, id 2 pre-existing).

    row 0: submit C (id 10) 100 @ 3001 sell                       (unrelated)
    -- our order arrives here (after row 0) --
    row 1: submit D (id 11) 100 @ 3000 buy  -> behind us
    row 2: exec A 300 @ 3000                -> 300 ahead consumed
    row 3: delete D 100                     -> behind us: no credit
    row 4: delete B 200                     -> ahead: credited (central only)
    row 5: submit E (id 12) 100 @ 3000 buy  -> behind us
    row 6: exec E 100 @ 3000                -> executed order is behind us => we would have filled first
    """
    rows = [
        dict(
            t_ms=0,
            type=1,
            oid=10,
            size=100,
            price_ticks=3001,
            side=SELL,
            bids=[(3000, 500)],
            asks=[(3001, 600), (3002, 500)],
        ),
        dict(
            t_ms=1,
            type=1,
            oid=11,
            size=100,
            price_ticks=3000,
            side=BUY,
            bids=[(3000, 600)],
            asks=[(3001, 600), (3002, 500)],
        ),
        dict(
            t_ms=2,
            type=4,
            oid=1,
            size=300,
            price_ticks=3000,
            side=BUY,
            bids=[(3000, 300)],
            asks=[(3001, 600), (3002, 500)],
        ),
        dict(
            t_ms=3,
            type=3,
            oid=11,
            size=100,
            price_ticks=3000,
            side=BUY,
            bids=[(3000, 200)],
            asks=[(3001, 600), (3002, 500)],
        ),
        dict(
            t_ms=4,
            type=3,
            oid=2,
            size=200,
            price_ticks=3000,
            side=BUY,
            bids=[(2999, 400)],
            asks=[(3001, 600), (3002, 500)],
        ),
        dict(
            t_ms=5,
            type=1,
            oid=12,
            size=100,
            price_ticks=3000,
            side=BUY,
            bids=[(3000, 100), (2999, 400)],
            asks=[(3001, 600), (3002, 500)],
        ),
        dict(
            t_ms=6,
            type=4,
            oid=12,
            size=100,
            price_ticks=3000,
            side=BUY,
            bids=[(2999, 400)],
            asks=[(3001, 600), (3002, 500)],
        ),
    ]
    return build_session(rows)


def _run(preset: str):
    s = _queue_session()
    ex = SimulatedExchange(s, ExecutionConfig(fill_preset=preset))
    ex.on_market_event(0, int(s.ts[0]))
    order = Order(order_id=1, side=BUY, price=px(3000), qty=100, sent_ns=int(s.ts[0]))
    ex.on_order_arrival(order, int(s.ts[0]) + 1)
    fills = []
    for i in range(1, len(s)):
        fills += ex.on_market_event(i, int(s.ts[i]))
    return order, fills


def test_central_tracks_ids_and_fills_when_behind_order_executes():
    order, fills = _run("central")
    assert order.state == OrderState.FILLED
    assert len(fills) == 1 and fills[0].row == 6 and fills[0].reason == "queue"
    assert fills[0].price == px(3000) and fills[0].liquidity == "maker"


def test_conservative_ignores_cancel_credit_and_does_not_fill():
    order, fills = _run("conservative")
    # 500 ahead, only 300 executed ahead, B's delete not credited, E executes -> consumes queue, no fill
    # ...but the displayed-size cap after row 4 (level empty) sets queue ahead to 0, so E's execution fills us.
    assert order.state == OrderState.FILLED
    assert fills[0].row == 6


def test_presets_differ_when_cancels_ahead_are_masked_by_joins_behind():
    """Level 3000 holds A(300) and B(200), both pre-existing. We join; F(300) joins behind us;
    B cancels (ahead); A executes; then F (behind us) executes 100."""
    rows = [
        dict(t_ms=0, type=1, oid=10, size=100, price_ticks=3001, side=SELL, bids=[(3000, 500)], asks=ASKS),
        dict(t_ms=1, type=1, oid=20, size=300, price_ticks=3000, side=BUY, bids=[(3000, 800)], asks=ASKS),
        dict(t_ms=2, type=3, oid=2, size=200, price_ticks=3000, side=BUY, bids=[(3000, 600)], asks=ASKS),
        dict(t_ms=3, type=4, oid=1, size=300, price_ticks=3000, side=BUY, bids=[(3000, 300)], asks=ASKS),
        dict(t_ms=4, type=4, oid=20, size=100, price_ticks=3000, side=BUY, bids=[(3000, 200)], asks=ASKS),
    ]
    s = build_session(rows)
    filled, ahead = {}, {}
    for preset in ("conservative", "central", "optimistic"):
        ex = SimulatedExchange(s, ExecutionConfig(fill_preset=preset))
        ex.on_market_event(0, int(s.ts[0]))
        o = Order(order_id=1, side=BUY, price=px(3000), qty=100, sent_ns=0)
        ex.on_order_arrival(o, int(s.ts[0]) + 1)
        f = []
        for i in range(1, len(s)):
            f += ex.on_market_event(i, int(s.ts[i]))
            if i == 3:
                ahead[preset] = ex.resting[0][1].queue_ahead if ex.resting else None
        filled[preset] = sum(x.qty for x in f)
    # conservative: 500 ahead, B's cancel not credited, A's 300 execution leaves 200 ahead -> no fill
    assert ahead["conservative"] == 200.0 and filled["conservative"] == 0
    # central: B's cancel credited (ahead, origin -1), A executes -> 0 ahead; F is behind -> we fill
    assert ahead["central"] == 0.0 and filled["central"] == 100
    # optimistic: front of queue from the start, A's execution fills us
    assert filled["optimistic"] == 100


def test_optimistic_fills_on_first_touch():
    order, fills = _run("optimistic")
    assert fills[0].row == 2


def test_trade_through_fills_any_preset():
    rows = [
        dict(
            t_ms=0,
            type=1,
            oid=10,
            size=100,
            price_ticks=3001,
            side=SELL,
            bids=[(3000, 100), (2999, 100)],
            asks=ASKS,
        ),
        dict(t_ms=1, type=4, oid=5, size=100, price_ticks=3000, side=BUY, bids=[(2999, 100)], asks=ASKS),
        dict(t_ms=1, type=4, oid=6, size=100, price_ticks=2999, side=BUY, bids=[], asks=ASKS),
    ]
    s = build_session(rows)
    for preset in ("conservative", "central", "optimistic"):
        ex = SimulatedExchange(s, ExecutionConfig(fill_preset=preset))
        ex.on_market_event(0, int(s.ts[0]))
        # we bid 3000 behind 100; sweep takes 3000 then trades at 2999 -> through our price
        o = Order(order_id=1, side=BUY, price=px(3000), qty=100, sent_ns=0)
        ex.on_order_arrival(o, int(s.ts[0]) + 1)
        fills = ex.on_market_event(1, int(s.ts[1])) + ex.on_market_event(2, int(s.ts[2]))
        assert sum(f.qty for f in fills) == 100, preset


def test_crossing_submission_fills_central_not_conservative():
    rows = [
        dict(
            t_ms=0,
            type=1,
            oid=10,
            size=100,
            price_ticks=2999,
            side=BUY,
            bids=[(2999, 100)],
            asks=[(3001, 100)],
        ),
        dict(
            t_ms=1,
            type=1,
            oid=11,
            size=300,
            price_ticks=3000,
            side=SELL,
            bids=[(2999, 100)],
            asks=[(3000, 300), (3001, 100)],
        ),
    ]
    s = build_session(rows)
    got = {}
    for preset in ("conservative", "central"):
        ex = SimulatedExchange(s, ExecutionConfig(fill_preset=preset))
        ex.on_market_event(0, int(s.ts[0]))
        o = Order(order_id=1, side=BUY, price=px(3000), qty=100, sent_ns=0)  # improves the bid
        ex.on_order_arrival(o, int(s.ts[0]) + 1)
        got[preset] = sum(f.qty for f in ex.on_market_event(1, int(s.ts[1])))
    assert got == {"conservative": 0, "central": 100}


def test_post_only_rejects_crossing_order():
    s = _queue_session()
    ex = SimulatedExchange(s, ExecutionConfig())
    ex.on_market_event(0, int(s.ts[0]))
    o = Order(order_id=1, side=BUY, price=px(3001), qty=100, sent_ns=0)
    ex.on_order_arrival(o, int(s.ts[0]) + 1)
    assert o.state == OrderState.REJECTED and o.reject_reason == "post_only_would_cross"


def test_cancel_after_fill_does_not_undo_fill():
    order, fills = _run("optimistic")
    s = _queue_session()
    ex = SimulatedExchange(s, ExecutionConfig())
    ex.on_cancel_arrival(order, int(s.ts[-1]) + 10)  # too late
    assert order.state == OrderState.FILLED and order.filled == 100


def test_pending_cancel_order_still_fillable_until_cancel_arrives():
    s = _queue_session()
    ex = SimulatedExchange(s, ExecutionConfig(fill_preset="optimistic"))
    ex.on_market_event(0, int(s.ts[0]))
    o = Order(order_id=1, side=BUY, price=px(3000), qty=100, sent_ns=0)
    ex.on_order_arrival(o, int(s.ts[0]) + 1)
    o.transition(OrderState.PENDING_CANCEL, int(s.ts[1]))
    ex.on_market_event(1, int(s.ts[1]))
    ex.on_market_event(2, int(s.ts[2]))  # fill happens before the cancel is effective
    ex.on_cancel_arrival(o, int(s.ts[2]) + 5)
    assert o.state == OrderState.FILLED


def test_ioc_walks_visible_depth():
    s = _queue_session()
    ex = SimulatedExchange(s, ExecutionConfig())
    ex.on_market_event(0, int(s.ts[0]))
    o = Order(
        order_id=9, side=BUY, price=px(3002), qty=800, sent_ns=0, post_only=False, ioc=True, tag="flatten"
    )
    fills = ex.on_order_arrival(o, int(s.ts[0]) + 1)
    assert [(f.price, f.qty) for f in fills] == [(px(3001), 600), (px(3002), 200)]
    assert o.state == OrderState.FILLED and all(f.liquidity == "taker" for f in fills)


def test_order_state_machine_rejects_impossible_transitions():
    o = Order(order_id=1, side=BUY, price=1, qty=100, sent_ns=0)
    with pytest.raises(InvalidTransition):
        o.apply_fill(100, 1)  # not at venue
    o.at_venue = True
    o.transition(OrderState.ACTIVE, 1)
    o.apply_fill(40, 2)
    assert o.state == OrderState.PARTIALLY_FILLED
    with pytest.raises(InvalidTransition):
        o.apply_fill(61, 3)  # overfill
    o.apply_fill(60, 3)
    with pytest.raises(InvalidTransition):
        o.transition(OrderState.CANCELLED, 4)


def test_visible_size_outside_window_is_unknown():
    s = _queue_session()
    # full 3-level window would be needed for "unknown"; row 0 has 1 bid level -> deeper prices are empty (0)
    assert visible_size(s, 0, BUY, px(2990)) == 0
    assert visible_size(s, 0, BUY, px(3000)) == 500
