"""Property-based tests of the queue/fill rules."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from adaptive_market_maker.execution.queue_model import FillRules, QueuePosition
from adaptive_market_maker.types import BUY, SELL

P = 300_000
events = st.lists(
    st.tuples(
        st.sampled_from([1, 2, 3, 4, 5]),  # type
        st.sampled_from([BUY, SELL]),  # side of the resting order the message refers to
        st.sampled_from([P - 100, P, P + 100]),  # price
        st.integers(1, 5).map(lambda x: x * 100),  # size
        st.integers(-1, 30),  # origin row
    ),
    max_size=60,
)


@settings(max_examples=300, deadline=None)
@given(events, st.sampled_from(["conservative", "central", "optimistic"]), st.integers(0, 2000))
def test_fills_never_exceed_order_size(evs, preset, visible):
    rules = FillRules.from_preset(preset)
    pos = QueuePosition(BUY, P, arrival_seq=10, queue_ahead=rules.initial_queue(visible))
    remaining = 300
    for et, side, price, size, origin in evs:
        q = rules.on_event(pos, remaining, et, side, price, size, origin)
        assert 0 <= q <= remaining
        remaining -= q
        assert pos.queue_ahead >= 0
    assert remaining >= 0


@settings(max_examples=300, deadline=None)
@given(events, st.integers(100, 2000))
def test_conservative_queue_fill_requires_volume_ahead_to_trade_first(evs, visible):
    """At our price, the conservative preset only fills after executions at our price have consumed the
    entire initial queue ahead (cancellations never help)."""
    rules = FillRules.from_preset("conservative")
    pos = QueuePosition(BUY, P, arrival_seq=10, queue_ahead=rules.initial_queue(visible))
    executed_at_price = 0
    for et, side, price, size, origin in evs:
        q = rules.on_event(pos, 100, et, side, price, size, origin)
        if et == 4 and side == BUY and price == P:
            executed_at_price += size
            if q:
                assert executed_at_price > visible
        if q and not (et == 4 and side == BUY and price == P):
            assert et == 4 and side == BUY and price < P  # only a trade-through can fill otherwise
        if q:
            break


@settings(max_examples=200, deadline=None)
@given(events)
def test_better_priced_or_opposite_side_executions_never_fill(evs):
    rules = FillRules.from_preset("central")
    pos = QueuePosition(SELL, P, arrival_seq=10, queue_ahead=500.0)
    for et, side, price, size, origin in evs:
        q = rules.on_event(pos, 100, et, side, price, size, origin)
        if et in (4, 5) and (side != SELL or price < P):
            assert q == 0  # buy-side executions, or sells resting at a better (lower) price, are not ours
