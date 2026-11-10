"""Shared enums and constants for the normalized market-event representation.

Units used throughout the package
---------------------------------
* time:      int64 nanoseconds after midnight (exchange timestamp as published in the source).
* price:     int64 *price units* = dollars x ``price_scale`` (LOBSTER: 10_000). Tick size is in the same units.
* quantity:  int64 shares.
* cash:      int64 micro-dollars (1e-6 $) so fills, fees and rebates are exact.
* strategy maths is done in *ticks* (float) and rounded back to price units on the tick grid.
"""

from __future__ import annotations

from enum import IntEnum


class EventType(IntEnum):
    """LOBSTER message types (event_type column)."""

    SUBMIT = 1  # submission of a new limit order
    CANCEL = 2  # partial cancellation (size reduced, priority kept)
    DELETE = 3  # full deletion of a limit order
    EXEC_VISIBLE = 4  # execution of a visible limit order
    EXEC_HIDDEN = 5  # execution of a hidden limit order
    CROSS = 6  # cross trade (e.g. auction); not continuous-trading liquidity
    HALT = 7  # trading-halt indicator (price field encodes the state)


BUY = 1
SELL = -1

# Normalized sentinel values for empty book levels (LOBSTER publishes +/-9999999999 with size 0).
EMPTY_ASK_PRICE = 9_999_999_999
EMPTY_BID_PRICE = -9_999_999_999

# LOBSTER halt-message price codes: -1 halt, 0 quoting resumes, 1 trading resumes.
HALT_START = -1
HALT_QUOTE_RESUME = 0
HALT_TRADE_RESUME = 1

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
NS_PER_US = 1_000

MESSAGE_COLUMNS = ["ts_ns", "event_type", "order_id", "size", "price", "side"]


def book_columns(levels: int) -> list[str]:
    cols: list[str] = []
    for i in range(1, levels + 1):
        cols += [f"ask_px_{i}", f"ask_sz_{i}", f"bid_px_{i}", f"bid_sz_{i}"]
    return cols
