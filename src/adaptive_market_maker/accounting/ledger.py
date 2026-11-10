"""Cash/inventory ledger in exact integer micro-dollars.

For a fill of signed quantity dq (buy > 0) at price P with fee f (negative for a rebate):

    C_t = C_{t-1} - dq * P - f            q_t = q_{t-1} + dq            E_t = C_t + q_t * M_t

Fees are per share (maker and taker rates from ``FeeConfig``); rebates are negative fees and are counted
once, in ``fees_micro``. Realized P&L uses average-cost accounting and is reported *gross* of fees, so

    E_t(M) = realized_gross + unrealized(M) - fees            (checked by :meth:`reconcile`)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adaptive_market_maker.config import FeeConfig
from adaptive_market_maker.execution.simulator import Fill

MICRO = 1_000_000


def dollars_to_micro(value: float, what: str) -> int:
    micro = round(value * MICRO)
    if abs(micro - value * MICRO) > 1e-6:
        raise ValueError(f"{what}={value} is not representable in whole micro-dollars")
    return int(micro)


class ReconciliationError(AssertionError):
    pass


@dataclass
class Ledger:
    price_scale: int
    fees: FeeConfig
    cash_micro: int = 0
    position: int = 0
    fees_micro: int = 0  # net of rebates; positive = paid
    rebates_micro: int = 0  # gross rebates received (informational; already inside fees_micro)
    realized_gross_micro: float = 0.0
    avg_cost: float = 0.0  # price units per share of the open position
    shares_traded: int = 0
    notional_micro: int = 0
    fills: list[Fill] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.micro_per_unit = MICRO // self.price_scale
        if self.micro_per_unit * self.price_scale != MICRO:
            raise ValueError("price_scale must divide 1e6")
        self.maker_fee_micro = dollars_to_micro(self.fees.maker_per_share, "maker_per_share")
        self.taker_fee_micro = dollars_to_micro(self.fees.taker_per_share, "taker_per_share")

    def fee_for(self, fill: Fill) -> int:
        rate = self.maker_fee_micro if fill.liquidity == "maker" else self.taker_fee_micro
        return fill.qty * rate

    def apply_fill(self, fill: Fill) -> None:
        if fill.qty <= 0:
            raise ValueError("fill qty must be positive")
        dq = fill.qty * fill.side
        notional = fill.qty * fill.price * self.micro_per_unit
        fee = self.fee_for(fill)
        self.cash_micro += -fill.side * notional - fee
        self.fees_micro += fee
        if fee < 0:
            self.rebates_micro += -fee
        self.shares_traded += fill.qty
        self.notional_micro += notional
        # average-cost realized P&L
        q0 = self.position
        if q0 == 0 or (q0 > 0) == (dq > 0):
            new_q = q0 + dq
            self.avg_cost = (self.avg_cost * abs(q0) + fill.price * fill.qty) / abs(new_q)
        else:
            closing = min(abs(q0), fill.qty)
            direction = 1 if q0 > 0 else -1
            self.realized_gross_micro += (
                closing * (fill.price - self.avg_cost) * direction * self.micro_per_unit
            )
            new_q = q0 + dq
            if new_q == 0:
                self.avg_cost = 0.0
            elif (new_q > 0) != (q0 > 0):
                self.avg_cost = float(fill.price)  # flipped: remainder opened at this fill's price
        self.position = q0 + dq
        self.fills.append(fill)

    # ------------------------------------------------------------------ valuation
    def equity_micro(self, mark_units: float) -> float:
        return self.cash_micro + self.position * mark_units * self.micro_per_unit

    def unrealized_micro(self, mark_units: float) -> float:
        return self.position * (mark_units - self.avg_cost) * self.micro_per_unit

    def liquidation_value_micro(self, bid_px, bid_sz, ask_px, ask_sz) -> tuple[float, int]:
        """Value of closing the position now with a marketable order walking visible depth, net of
        taker fees. Shares beyond visible depth are valued at the deepest visible price (a floor).
        Returns (cash change from closing, shares beyond visible depth)."""
        q = self.position
        if q == 0:
            return 0.0, 0
        px, sz = (bid_px, bid_sz) if q > 0 else (ask_px, ask_sz)
        left, value, last = abs(q), 0.0, None
        for p, s in zip(px, sz):
            if s <= 0 or left == 0:
                break
            take = min(left, int(s))
            value += take * int(p)
            left -= take
            last = int(p)
        beyond = left
        if left and last is not None:
            value += left * last
        sign = 1 if q > 0 else -1
        cash = sign * value * self.micro_per_unit - abs(q) * self.taker_fee_micro
        return cash, beyond

    # ------------------------------------------------------------------ audit
    def reconcile(self, mark_units: float, tol_micro: float = 1e-3) -> None:
        exp_cash = -sum(f.side * f.qty * f.price * self.micro_per_unit for f in self.fills) - sum(
            self.fee_for(f) for f in self.fills
        )
        exp_pos = sum(f.side * f.qty for f in self.fills)
        if exp_cash != self.cash_micro:
            raise ReconciliationError(f"cash {self.cash_micro} != fills-implied {exp_cash}")
        if exp_pos != self.position:
            raise ReconciliationError(f"position {self.position} != fills-implied {exp_pos}")
        if sum(self.fee_for(f) for f in self.fills) != self.fees_micro:
            raise ReconciliationError("fee total does not match fills")
        lhs = self.equity_micro(mark_units)
        rhs = self.realized_gross_micro + self.unrealized_micro(mark_units) - self.fees_micro
        if abs(lhs - rhs) > max(tol_micro, 1e-9 * abs(lhs)):
            raise ReconciliationError(f"equity {lhs} != realized+unrealized-fees {rhs}")
