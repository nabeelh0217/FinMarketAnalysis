"""Strategy C (alpha-aware) and Strategy D (causal regime rules on top of C).

C shifts the reservation price by the model's expected mid change:

    r = S + w * clip(alpha, +-alpha_clip) - q * skew_per_lot

so a positive alpha makes the bid more and the ask less competitive. The *adverse-selection guard*
withdraws the quote on the side that the signal says is about to be picked off (alpha > guard: no ask;
alpha < -guard: no bid). Economic rationale: a passive fill on that side has negative expected markout
of roughly |alpha| ticks, which exceeds the half-spread plus rebate it earns when |alpha| is large.

D applies interpretable, causal regime rules: in high realized volatility it scales size (to zero by
default, i.e. stops quoting); when the market spread is wide it adds extra half-spread.
"""

from __future__ import annotations

import math

from adaptive_market_maker.config import AdaptiveParams, RegimeParams
from adaptive_market_maker.strategies.base import MarketView, QuoteIntent, StrategyState
from adaptive_market_maker.strategies.inventory_aware import InventoryAwareStrategy


class AdaptiveStrategy(InventoryAwareStrategy):
    name = "adaptive"
    p: AdaptiveParams

    def __init__(self, params: AdaptiveParams):
        super().__init__(params)

    def center(self, view: MarketView) -> tuple[float, float]:
        a = view.alpha
        if not self.p.use_alpha or not math.isfinite(a):
            return view.mid_ticks, a
        a_used = max(-self.p.alpha_clip_ticks, min(self.p.alpha_clip_ticks, a))
        return view.mid_ticks + self.p.alpha_weight * a_used, a

    def quote(self, view: MarketView, state: StrategyState) -> QuoteIntent:
        intent = super().quote(view, state)
        g = self.p.alpha_guard_ticks
        a = view.alpha
        if self.p.use_alpha and g is not None and math.isfinite(a):
            if a > g:
                intent.ask_px, intent.note = None, "guard_ask"
            elif a < -g:
                intent.bid_px, intent.note = None, "guard_bid"
        return intent


class RegimeStrategy(AdaptiveStrategy):
    name = "regime"
    p: RegimeParams

    def __init__(self, params: RegimeParams):
        super().__init__(params)

    def skew_and_half(self, view: MarketView) -> tuple[float, float]:
        skew, half = super().skew_and_half(view)
        if view.spread_ticks >= self.p.wide_spread_ticks:
            half += self.p.wide_spread_extra_ticks
        return skew, half

    def quote(self, view: MarketView, state: StrategyState) -> QuoteIntent:
        intent = super().quote(view, state)
        if view.vol >= self.p.high_vol_ticks_per_sqrt_s:
            size = int(self.p.quote_size * self.p.high_vol_size_mult) // view.lot * view.lot
            intent.bid_qty = intent.ask_qty = size
            if size <= 0:
                intent.bid_px = intent.ask_px = None
            intent.note = (intent.note + " high_vol").strip()
        return intent


def make_strategy(name: str, strategies_cfg) -> InventoryAwareStrategy:
    from adaptive_market_maker.strategies.fixed_spread import FixedSpreadStrategy

    if name == "fixed_spread":
        return FixedSpreadStrategy(strategies_cfg.fixed_spread)
    if name == "inventory_aware":
        return InventoryAwareStrategy(strategies_cfg.inventory_aware)
    if name == "adaptive":
        return AdaptiveStrategy(strategies_cfg.adaptive)
    if name == "regime":
        return RegimeStrategy(strategies_cfg.regime)
    raise ValueError(f"unknown strategy {name!r}")
