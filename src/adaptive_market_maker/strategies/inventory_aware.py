"""Strategy B: inventory-aware quoting (Avellaneda-Stoikov 2008 / Gueant-Lehalle-Fernandez-Tapia 2013).

Units: prices in ticks, sigma in ticks/sqrt(s), time in seconds, inventory q in *lots* (the models assume
unit order size), so gamma is in 1/tick and k in 1/tick; A is fills per second.

Reservation price and half-spread (AS, CARA utility, arithmetic Brownian mid, Poisson fills with intensity
A exp(-k delta)):

    r = S - q gamma sigma^2 tau
    delta = (1/gamma) ln(1 + gamma/k) + 0.5 gamma sigma^2 tau

``horizon_mode`` decides tau:
  * ``session_end``: tau = time to the end of the trading window (literal AS; huge skews early intraday);
  * ``rolling``:     tau = ``tau_s``, a fixed inventory-holding horizon (heuristic, NOT the AS optimum);
  * ``glft``:        no tau. GLFT's asymptotic approximation for the stationary problem:
        omega   = sqrt( sigma^2 gamma / (2 k A) * (1 + gamma/k)^(1 + k/gamma) )
        delta_b = (1/gamma) ln(1 + gamma/k) + (2q + 1)/2 * omega
        delta_a = (1/gamma) ln(1 + gamma/k) - (2q - 1)/2 * omega
    i.e. skew = q * omega and half-spread = c + omega / 2.

``spread_mode = fixed`` keeps the inventory skew but replaces the model half-spread with
``half_spread_ticks`` (so B differs from A only by the skew - a clean ablation of inventory control).
"""

from __future__ import annotations

import math

from adaptive_market_maker.config import InventoryAwareParams
from adaptive_market_maker.strategies.base import (
    MarketView,
    QuoteIntent,
    Strategy,
    StrategyState,
    round_quotes,
)


def as_terms(gamma: float, k: float, sigma: float, tau: float) -> tuple[float, float]:
    """(skew per lot, model half-spread) under Avellaneda-Stoikov."""
    var_term = gamma * sigma * sigma * tau
    return var_term, (1.0 / gamma) * math.log1p(gamma / k) + 0.5 * var_term


def glft_terms(gamma: float, k: float, A: float, sigma: float) -> tuple[float, float]:
    """(skew per lot, model half-spread) under the GLFT asymptotic approximation."""
    c = (1.0 / gamma) * math.log1p(gamma / k)
    omega = math.sqrt(sigma * sigma * gamma / (2.0 * k * A) * (1.0 + gamma / k) ** (1.0 + k / gamma))
    return omega, c + 0.5 * omega


class InventoryAwareStrategy(Strategy):
    name = "inventory_aware"

    def __init__(self, params: InventoryAwareParams):
        self.p = params

    def sigma(self, view: MarketView) -> float:
        return view.vol if self.p.use_vol_adaptation else self.p.static_vol_ticks_per_sqrt_s

    def skew_and_half(self, view: MarketView) -> tuple[float, float]:
        p, sigma = self.p, self.sigma(view)
        if p.horizon_mode == "glft":
            skew_per_lot, model_half = glft_terms(p.gamma, p.k, p.A, sigma)
        else:
            tau = p.tau_s if p.horizon_mode == "rolling" else view.secs_to_end
            skew_per_lot, model_half = as_terms(p.gamma, p.k, sigma, tau)
        half = model_half if p.spread_mode == "model" else p.half_spread_ticks
        return skew_per_lot, max(half, p.min_half_spread_ticks)

    def center(self, view: MarketView) -> tuple[float, float]:
        return view.mid_ticks, math.nan

    def quote(self, view: MarketView, state: StrategyState) -> QuoteIntent:
        skew_per_lot, half = self.skew_and_half(view)
        q_lots = state.position / view.lot
        skew = q_lots * skew_per_lot if self.p.use_inventory_skew else 0.0
        center, alpha = self.center(view)
        r = center - skew
        bid, ask = round_quotes(view, r - half, r + half, self.p.rounding)
        return QuoteIntent(
            bid,
            self.p.quote_size,
            ask,
            self.p.quote_size,
            reservation_ticks=r,
            half_spread_ticks=half,
            skew_ticks=skew,
            alpha_ticks=alpha,
        )

    def describe(self) -> dict:
        return {"name": self.name, **self.p.model_dump()}
