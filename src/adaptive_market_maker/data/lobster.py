"""LOBSTER message/orderbook adapter.

Format (LOBSTER documentation, https://lobsterdata.com/info/DataStructure.php):

message file, no header, 6 columns
    Time       seconds after midnight, decimal string (nanosecond precision in the samples)
    Type       1 submit, 2 partial cancel, 3 delete, 4 visible execution, 5 hidden execution,
               6 cross trade, 7 trading halt indicator
    Order ID   order reference number
    Size       shares
    Price      dollar price x 10000
    Direction  -1 sell limit order, +1 buy limit order. For executions this is the side of the *resting*
               order: execution of a sell (buy) limit order = buyer- (seller-) initiated trade.

orderbook file, no header, 4 x levels columns, row i = book *after* message i
    Ask Price 1, Ask Size 1, Bid Price 1, Bid Size 1, Ask Price 2, ...
    empty levels: ask price 9999999999, bid price -9999999999, size 0

Only messages that affect the requested number of levels appear in the message file. Row counts of the
two files must be equal; this is checked, not assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_market_maker.config import DataConfig, SessionSpec
from adaptive_market_maker.data.base import SessionData
from adaptive_market_maker.data.normalize import (
    compute_origin_seq,
    format_decimal_seconds,
    normalize_book,
    parse_decimal_seconds,
)
from adaptive_market_maker.types import EMPTY_ASK_PRICE, EMPTY_BID_PRICE

MESSAGE_SOURCE_COLUMNS = ["time", "type", "order_id", "size", "price", "direction"]


class LobsterFormatError(ValueError):
    pass


def read_message_file(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(f"LOBSTER message file not found: {path}")
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if raw.shape[1] != 6:
        raise LobsterFormatError(f"{path}: expected 6 columns, found {raw.shape[1]}")
    raw.columns = MESSAGE_SOURCE_COLUMNS
    blank = (raw == "").sum()
    if int(blank.sum()):
        raise LobsterFormatError(f"{path}: blank values per column: {blank[blank > 0].to_dict()}")
    ts, truncated = parse_decimal_seconds(raw["time"])
    out = pd.DataFrame(
        {
            "ts_ns": ts,
            "event_type": pd.to_numeric(raw["type"], errors="raise").astype(np.int8),
            "order_id": pd.to_numeric(raw["order_id"], errors="raise").astype(np.int64),
            "size": pd.to_numeric(raw["size"], errors="raise").astype(np.int64),
            "price": pd.to_numeric(raw["price"], errors="raise").astype(np.int64),
            "side": pd.to_numeric(raw["direction"], errors="raise").astype(np.int8),
        }
    )
    return out, {"time_digits_truncated": truncated}


def read_orderbook_file(path: Path, levels: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"LOBSTER orderbook file not found: {path}")
    arr = pd.read_csv(path, header=None, dtype=np.int64).to_numpy()
    if arr.shape[1] != 4 * levels:
        raise LobsterFormatError(
            f"{path}: expected {4 * levels} columns for {levels} levels, got {arr.shape[1]}"
        )
    return arr


def load_session(spec: SessionSpec, cfg: DataConfig, source: str = "lobster") -> SessionData:
    msg_path = spec.message_path(cfg.raw_dir, cfg.levels)
    ob_path = spec.orderbook_path(cfg.raw_dir, cfg.levels)
    msgs, parse_stats = read_message_file(msg_path)
    book = read_orderbook_file(ob_path, cfg.levels)
    if len(msgs) != book.shape[0]:
        raise LobsterFormatError(
            f"row count mismatch: {msg_path.name} has {len(msgs)} rows, {ob_path.name} has {book.shape[0]}"
        )
    ask_px, ask_sz, bid_px, bid_sz, book_stats = normalize_book(
        book[:, 0::4], book[:, 1::4], book[:, 2::4], book[:, 3::4]
    )
    et = msgs["event_type"].to_numpy(np.int8)
    oid = msgs["order_id"].to_numpy(np.int64)
    origin, origin_stats = compute_origin_seq(et, oid)
    return SessionData(
        session_id=spec.session_id,
        ticker=spec.ticker,
        date=spec.date,
        source=source,
        levels=cfg.levels,
        tick_size=cfg.tick_size,
        price_scale=cfg.price_scale,
        lot_size=cfg.lot_size,
        open_ns=cfg.open_ns,
        close_ns=cfg.close_ns,
        ts=msgs["ts_ns"].to_numpy(np.int64),
        event_type=et,
        order_id=oid,
        size=msgs["size"].to_numpy(np.int64),
        price=msgs["price"].to_numpy(np.int64),
        side=msgs["side"].to_numpy(np.int8),
        origin_seq=origin,
        ask_px=ask_px,
        ask_sz=ask_sz,
        bid_px=bid_px,
        bid_sz=bid_sz,
        meta={
            "message_file": str(msg_path),
            "orderbook_file": str(ob_path),
            **parse_stats,
            **book_stats,
            **origin_stats,
        },
    )


def write_session(session: SessionData, raw_dir: Path, spec: SessionSpec) -> tuple[Path, Path]:
    """Write a session in LOBSTER's exact file layout (used for synthetic data and round-trip tests)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    msg_path = spec.message_path(raw_dir, session.levels)
    ob_path = spec.orderbook_path(raw_dir, session.levels)
    msgs = pd.DataFrame(
        {
            "time": format_decimal_seconds(session.ts),
            "type": session.event_type.astype(np.int64),
            "order_id": session.order_id,
            "size": session.size,
            "price": session.price,
            "direction": session.side.astype(np.int64),
        }
    )
    msgs.to_csv(msg_path, header=False, index=False)
    k = session.levels
    book = np.empty((len(session), 4 * k), dtype=np.int64)
    book[:, 0::4] = np.where(session.ask_sz > 0, session.ask_px, EMPTY_ASK_PRICE)
    book[:, 1::4] = session.ask_sz
    book[:, 2::4] = np.where(session.bid_sz > 0, session.bid_px, EMPTY_BID_PRICE)
    book[:, 3::4] = session.bid_sz
    pd.DataFrame(book).to_csv(ob_path, header=False, index=False)
    return msg_path, ob_path
