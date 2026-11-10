"""Hard pre-trade risk limits, applied by the engine to every quote intent.

Inventory limits are enforced on the *worst case*: known position plus every open (or pending) order on
that side as if it all filled. Because fills can only come from open orders, the true position can never
exceed ``max_inventory`` even though fill reports arrive with latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adaptive_market_maker.config import RiskConfig
from adaptive_market_maker.strategies.base import MarketView, QuoteIntent


@dataclass
class RiskManager:
    cfg: RiskConfig
    price_scale: int
    halted_reason: str | None = None
    action_times: list[int] = field(default_factory=list)
    blocks: dict[str, int] = field(default_factory=dict)

    def _block(self, why: str) -> None:
        self.blocks[why] = self.blocks.get(why, 0) + 1

    def kill(self, reason: str) -> None:
        if self.halted_reason is None:
            self.halted_reason = reason

    def check(
        self,
        intent: QuoteIntent,
        view: MarketView,
        position: int,
        other_open_buy: int,
        other_open_sell: int,
        data_age_ns: int,
    ) -> QuoteIntent:
        """Clip/remove quotes. ``other_open_*`` = open quantity that will remain if this intent is applied."""
        c = self.cfg
        if self.halted_reason is not None:
            self._block("kill_switch")
            intent.bid_px = intent.ask_px = None
            return intent
        if data_age_ns > c.stale_data_s * 1e9:
            self._block("stale_data")
            intent.bid_px = intent.ask_px = None
            return intent
        if view.spread_ticks > c.max_market_spread_ticks or view.spread_ticks <= 0:
            self._block("market_spread")
            intent.bid_px = intent.ask_px = None
            return intent
        mid_units = view.mid_ticks * view.tick
        max_dev = c.max_price_deviation_ticks * view.tick
        for side in ("bid", "ask"):
            px = getattr(intent, f"{side}_px")
            if px is None:
                continue
            if abs(px - mid_units) > max_dev:
                self._block("price_deviation")
                setattr(intent, f"{side}_px", None)
        intent.bid_qty = min(
            intent.bid_qty, c.max_order_size, max(c.max_outstanding_per_side - other_open_buy, 0)
        )
        intent.ask_qty = min(
            intent.ask_qty, c.max_order_size, max(c.max_outstanding_per_side - other_open_sell, 0)
        )
        # worst-case inventory and notional
        long_room = c.max_inventory - (position + other_open_buy)
        short_room = c.max_inventory + (position - other_open_sell)
        notional_cap = int(c.max_notional * self.price_scale / max(mid_units, 1))
        long_room = min(long_room, notional_cap - (position + other_open_buy))
        short_room = min(short_room, notional_cap + (position - other_open_sell))
        if intent.bid_qty > long_room:
            self._block("inventory_long")
            intent.bid_qty = max(long_room, 0) // view.lot * view.lot
        if intent.ask_qty > short_room:
            self._block("inventory_short")
            intent.ask_qty = max(short_room, 0) // view.lot * view.lot
        if intent.bid_qty <= 0:
            intent.bid_px = None
        if intent.ask_qty <= 0:
            intent.ask_px = None
        return intent

    def allow_actions(self, t_ns: int, n: int) -> bool:
        """Sliding one-second window message-rate limit."""
        window_start = t_ns - 1_000_000_000
        self.action_times = [t for t in self.action_times if t > window_start]
        if len(self.action_times) + n > self.cfg.max_actions_per_second:
            self._block("rate_limit")
            return False
        self.action_times.extend([t_ns] * n)
        return True
