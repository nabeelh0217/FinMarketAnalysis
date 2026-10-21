"""Strategy interface.

A strategy sees only a :class:`MarketView` (the latest *observed* book state plus causal signals) and a
:class:`StrategyState` (what it has been told about its own orders and position). It returns a
:class:`QuoteIntent`; it cannot touch the exchange or the ledger.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from adaptive_market_maker.data.base import SessionData
from adaptive_market_maker.types import NS_PER_S


@dataclass
class Signals:
    """Per-observable-state causal inputs, aligned with ``FeatureTable.df`` rows."""

    ts_ns: np.ndarray
    rows: np.ndarray
    vol: np.ndarray  # ticks / sqrt(s)
    alpha: np.ndarray | None = None  # expected mid change, ticks, over ``alpha_horizon``
    alpha_horizon: str = ""
    eligible: np.ndarray | None = None  # warm & valid book


@dataclass
class MarketView:
    t_ns: int  # decision time
    k: int  # observable-state index
    row: int  # session row of that state
    best_bid: int
    best_ask: int
    bid_sz: int
    ask_sz: int
    tick: int
    lot: int
    mid_ticks: float
    spread_ticks: float
    vol: float
    alpha: float  # NaN if unavailable
    secs_to_end: float

    @classmethod
    def build(cls, s: SessionData, sig: Signals, k: int, t_ns: int, end_ns: int) -> MarketView:
        row = int(sig.rows[k])
        bb, ba = int(s.bid_px[row, 0]), int(s.ask_px[row, 0])
        a = float(sig.alpha[k]) if sig.alpha is not None else math.nan
        return cls(
            t_ns=t_ns,
            k=k,
            row=row,
            best_bid=bb,
            best_ask=ba,
            bid_sz=int(s.bid_sz[row, 0]),
            ask_sz=int(s.ask_sz[row, 0]),
            tick=s.tick_size,
            lot=s.lot_size,
            mid_ticks=(bb + ba) / (2.0 * s.tick_size),
            spread_ticks=(ba - bb) / s.tick_size,
            vol=float(sig.vol[k]),
            alpha=a,
            secs_to_end=max((end_ns - t_ns) / NS_PER_S, 0.0),
        )


@dataclass
class StrategyState:
    position: int  # shares, as known to the strategy (reports arrive with latency)
    open_bid_qty: int
    open_ask_qty: int


@dataclass
class QuoteIntent:
    bid_px: int | None
    bid_qty: int
    ask_px: int | None
    ask_qty: int
    reservation_ticks: float = math.nan
    half_spread_ticks: float = math.nan
    skew_ticks: float = 0.0
    alpha_ticks: float = math.nan
    note: str = ""


def round_quotes(
    view: MarketView, bid_ticks: float, ask_ticks: float, mode: str = "nearest"
) -> tuple[int, int]:
    """Snap model prices onto the tick grid, then keep both passive (post-only) and non-crossing.

    ``nearest``: nearest tick, exact half-ticks resolved toward the *less* aggressive price. On a 1-tick
    book this makes the model's continuous skew act through a threshold (e.g. back off the bid once the
    skew exceeds half a tick), so risk aversion still matters.
    ``conservative``: bid rounded down, ask rounded up; any skew at all moves a quote a full tick.
    """
    tick = view.tick
    if mode == "nearest":
        bid = int(math.ceil(bid_ticks - 0.5 - 1e-9)) * tick
        ask = int(math.floor(ask_ticks + 0.5 + 1e-9)) * tick
    else:
        bid = int(math.floor(bid_ticks + 1e-9)) * tick
        ask = int(math.ceil(ask_ticks - 1e-9)) * tick
    bid = min(bid, view.best_ask - tick)
    ask = max(ask, view.best_bid + tick)
    if bid >= ask:  # degenerate after clamping: fall back to the touch
        bid, ask = view.best_bid, view.best_ask
    return bid, ask


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def quote(self, view: MarketView, state: StrategyState) -> QuoteIntent: ...

    def describe(self) -> dict:
        return {"name": self.name}
