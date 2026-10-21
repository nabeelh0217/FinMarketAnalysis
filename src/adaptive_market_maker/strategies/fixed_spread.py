"""Strategy A: fixed-spread reference market maker (the benchmark, not the expected winner)."""

from __future__ import annotations

from adaptive_market_maker.config import FixedSpreadParams
from adaptive_market_maker.strategies.base import (
    MarketView,
    QuoteIntent,
    Strategy,
    StrategyState,
    round_quotes,
)


class FixedSpreadStrategy(Strategy):
    name = "fixed_spread"

    def __init__(self, params: FixedSpreadParams):
        self.p = params

    def quote(self, view: MarketView, state: StrategyState) -> QuoteIntent:
        h = self.p.half_spread_ticks
        bid, ask = round_quotes(view, view.mid_ticks - h, view.mid_ticks + h, self.p.rounding)
        return QuoteIntent(
            bid,
            self.p.quote_size,
            ask,
            self.p.quote_size,
            reservation_ticks=view.mid_ticks,
            half_spread_ticks=h,
        )

    def describe(self) -> dict:
        return {"name": self.name, **self.p.model_dump()}
