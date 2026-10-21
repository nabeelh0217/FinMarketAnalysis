"""Hand-calculated and property-based accounting tests."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from adaptive_market_maker.accounting.ledger import Ledger, ReconciliationError, dollars_to_micro
from adaptive_market_maker.config import FeeConfig
from adaptive_market_maker.execution.simulator import Fill
from adaptive_market_maker.types import BUY, SELL


def _fill(side, price, qty, liq="maker"):
    return Fill(0, 1, side, price, qty, liq, 0, "queue")


def test_round_trip_with_rebates_hand_calculated():
    led = Ledger(10_000, FeeConfig(maker_per_share=-0.002, taker_per_share=0.003))
    led.apply_fill(_fill(BUY, 300_000, 100))  # buy 100 @ $30.00
    led.apply_fill(_fill(SELL, 300_100, 100))  # sell 100 @ $30.01
    # spread capture 100 x $0.01 = $1.00 plus rebates 2 x 100 x $0.002 = $0.40
    assert led.cash_micro == 1_400_000
    assert led.position == 0
    assert led.fees_micro == -400_000 and led.rebates_micro == 400_000
    assert led.realized_gross_micro == pytest.approx(1_000_000)
    led.reconcile(mark_units=300_050)


def test_taker_fee_and_mark_to_market():
    led = Ledger(10_000, FeeConfig(maker_per_share=-0.002, taker_per_share=0.003))
    led.apply_fill(_fill(BUY, 300_000, 200, "taker"))
    assert led.cash_micro == -(200 * 300_000 * 100) - 200 * 3000
    # marked at $30.02: unrealized = 200 x $0.02 = $4.00, net equity = 4.00 - 0.60
    assert led.equity_micro(300_200) == pytest.approx(4_000_000 - 600_000)
    led.reconcile(300_200)


def test_position_flip_resets_average_cost():
    led = Ledger(10_000, FeeConfig(maker_per_share=0.0, taker_per_share=0.0))
    led.apply_fill(_fill(BUY, 100_000, 100))
    led.apply_fill(_fill(SELL, 100_500, 300))  # close 100 (+$5) and open short 200 @ 10.05
    assert led.position == -200
    assert led.avg_cost == 100_500
    assert led.realized_gross_micro == pytest.approx(100 * 500 * 100)
    led.reconcile(100_000)


def test_unrepresentable_fee_rejected():
    with pytest.raises(ValueError):
        dollars_to_micro(0.0000005, "fee")


def test_tampered_ledger_fails_reconciliation():
    led = Ledger(10_000, FeeConfig())
    led.apply_fill(_fill(BUY, 300_000, 100))
    led.cash_micro += 1
    with pytest.raises(ReconciliationError):
        led.reconcile(300_000)


fills_strategy = st.lists(
    st.tuples(
        st.sampled_from([BUY, SELL]),
        st.integers(250_000, 350_000),
        st.integers(1, 10).map(lambda x: x * 100),
        st.sampled_from(["maker", "taker"]),
    ),
    min_size=1,
    max_size=60,
)


@settings(max_examples=150, deadline=None)
@given(fills_strategy, st.integers(250_000, 350_000))
def test_invariants_hold_for_random_fill_sequences(fills, mark):
    led = Ledger(10_000, FeeConfig(maker_per_share=-0.0025, taker_per_share=0.003))
    for side, price, qty, liq in fills:
        led.apply_fill(_fill(side, price, qty, liq))
    assert led.position == sum(s * q for s, _, q, _ in fills)
    led.reconcile(mark)
    # gross P&L = net P&L + fees
    gross = led.equity_micro(mark) + led.fees_micro
    assert gross == pytest.approx(led.realized_gross_micro + led.unrealized_micro(mark), abs=1e-3)
