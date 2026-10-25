"""Source-independent in-memory representation of one trading session.

Strategies, features and the simulator only ever see :class:`SessionData`; nothing downstream depends
on LOBSTER column names or ITCH message types. Row ``i`` holds message ``i`` *and* the visible book
state immediately after that message was applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_market_maker.types import EMPTY_ASK_PRICE, EMPTY_BID_PRICE, book_columns

SCHEMA_VERSION = "1.0"


@dataclass
class SessionData:
    session_id: str
    ticker: str
    date: str
    source: str
    levels: int
    tick_size: int
    price_scale: int
    lot_size: int
    open_ns: int
    close_ns: int
    ts: np.ndarray
    event_type: np.ndarray
    order_id: np.ndarray
    size: np.ndarray
    price: np.ndarray
    side: np.ndarray
    origin_seq: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    bid_px: np.ndarray
    bid_sz: np.ndarray
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basic views
    def __len__(self) -> int:
        return int(self.ts.shape[0])

    @property
    def best_bid(self) -> np.ndarray:
        return self.bid_px[:, 0]

    @property
    def best_ask(self) -> np.ndarray:
        return self.ask_px[:, 0]

    @cached_property
    def valid_book(self) -> np.ndarray:
        """Both sides non-empty and not crossed/locked."""
        bid_ok = (self.bid_sz[:, 0] > 0) & (self.bid_px[:, 0] != EMPTY_BID_PRICE)
        ask_ok = (self.ask_sz[:, 0] > 0) & (self.ask_px[:, 0] != EMPTY_ASK_PRICE)
        return bid_ok & ask_ok & (self.ask_px[:, 0] > self.bid_px[:, 0])

    @cached_property
    def mid_ticks(self) -> np.ndarray:
        """Mid-price in ticks (float); NaN where the book is invalid."""
        mid = (self.ask_px[:, 0].astype(np.float64) + self.bid_px[:, 0].astype(np.float64)) / (
            2.0 * self.tick_size
        )
        return np.where(self.valid_book, mid, np.nan)

    @cached_property
    def spread_ticks(self) -> np.ndarray:
        s = (self.ask_px[:, 0] - self.bid_px[:, 0]).astype(np.float64) / self.tick_size
        return np.where(self.valid_book, s, np.nan)

    @cached_property
    def last_in_group(self) -> np.ndarray:
        """True for the last message of each same-timestamp group: the only observable book states."""
        out = np.ones(len(self), dtype=bool)
        if len(self) > 1:
            out[:-1] = self.ts[1:] != self.ts[:-1]
        return out

    @cached_property
    def state_before_group(self) -> np.ndarray:
        """For each row, the last row of the *previous* timestamp group (-1 if none): the book prevailing
        before the event that row belongs to. A sweep's intermediate rows are already partly consumed by
        the same aggressive order, so prices "before a fill" must be read here, not at row - 1."""
        n = len(self)
        starts = np.ones(n, dtype=bool)
        if n > 1:
            starts[1:] = self.ts[1:] != self.ts[:-1]
        group_start = np.maximum.accumulate(np.where(starts, np.arange(n), 0))
        return group_start - 1

    def ticks_to_price(self, ticks: float) -> int:
        return int(round(ticks * self.tick_size))

    def dollars(self, price_units: float) -> float:
        return price_units / self.price_scale

    def visible_size_at(self, row: int, side: int, price: int) -> int:
        px = self.bid_px[row] if side > 0 else self.ask_px[row]
        sz = self.bid_sz[row] if side > 0 else self.ask_sz[row]
        hit = np.nonzero(px == price)[0]
        return int(sz[hit[0]]) if hit.size else 0

    def slice_rows(self, start: int, stop: int) -> SessionData:
        sl = slice(start, stop)
        return SessionData(
            session_id=self.session_id,
            ticker=self.ticker,
            date=self.date,
            source=self.source,
            levels=self.levels,
            tick_size=self.tick_size,
            price_scale=self.price_scale,
            lot_size=self.lot_size,
            open_ns=self.open_ns,
            close_ns=self.close_ns,
            ts=self.ts[sl].copy(),
            event_type=self.event_type[sl].copy(),
            order_id=self.order_id[sl].copy(),
            size=self.size[sl].copy(),
            price=self.price[sl].copy(),
            side=self.side[sl].copy(),
            origin_seq=self.origin_seq[sl].copy(),
            ask_px=self.ask_px[sl].copy(),
            ask_sz=self.ask_sz[sl].copy(),
            bid_px=self.bid_px[sl].copy(),
            bid_sz=self.bid_sz[sl].copy(),
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------ persistence
    def to_frame(self) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {
            "seq": np.arange(len(self), dtype=np.int64),
            "ts_ns": self.ts,
            "event_type": self.event_type,
            "order_id": self.order_id,
            "size": self.size,
            "price": self.price,
            "side": self.side,
            "origin_seq": self.origin_seq,
        }
        for i in range(self.levels):
            data[f"ask_px_{i + 1}"] = self.ask_px[:, i]
            data[f"ask_sz_{i + 1}"] = self.ask_sz[:, i]
            data[f"bid_px_{i + 1}"] = self.bid_px[:, i]
            data[f"bid_sz_{i + 1}"] = self.bid_sz[:, i]
        return pd.DataFrame(data)

    def header(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "ticker": self.ticker,
            "date": self.date,
            "source": self.source,
            "levels": self.levels,
            "tick_size": self.tick_size,
            "price_scale": self.price_scale,
            "lot_size": self.lot_size,
            "open_ns": self.open_ns,
            "close_ns": self.close_ns,
            **{f"meta_{k}": v for k, v in self.meta.items()},
        }

    def save(self, path: Path) -> None:
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(self.to_frame(), preserve_index=False)
        table = table.replace_schema_metadata({b"amm_header": json.dumps(self.header()).encode()})
        pq.write_table(table, path)

    @classmethod
    def load(cls, path: Path) -> SessionData:
        import json

        import pyarrow.parquet as pq

        table = pq.read_table(path)
        header = json.loads(table.schema.metadata[b"amm_header"].decode())
        if header.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema {header.get('schema_version')} != {SCHEMA_VERSION}")
        df = table.to_pandas()
        meta = {k[5:]: v for k, v in header.items() if k.startswith("meta_")}
        return cls.from_frame(
            df,
            meta=meta,
            **{k: v for k, v in header.items() if not k.startswith("meta_") and k != "schema_version"},
        )

    @classmethod
    def from_frame(cls, df: pd.DataFrame, *, levels: int, meta: dict | None = None, **header) -> SessionData:
        def stack(prefix: str) -> np.ndarray:
            return np.column_stack([df[f"{prefix}_{i + 1}"].to_numpy(np.int64) for i in range(levels)])

        missing = [c for c in book_columns(levels) if c not in df.columns]
        if missing:
            raise ValueError(f"missing book columns: {missing[:4]}...")
        return cls(
            levels=levels,
            ts=df["ts_ns"].to_numpy(np.int64),
            event_type=df["event_type"].to_numpy(np.int8),
            order_id=df["order_id"].to_numpy(np.int64),
            size=df["size"].to_numpy(np.int64),
            price=df["price"].to_numpy(np.int64),
            side=df["side"].to_numpy(np.int8),
            origin_seq=df["origin_seq"].to_numpy(np.int64),
            ask_px=stack("ask_px"),
            ask_sz=stack("ask_sz"),
            bid_px=stack("bid_px"),
            bid_sz=stack("bid_sz"),
            meta=meta or {},
            **header,
        )


@dataclass
class Dataset:
    """An ordered collection of sessions for one instrument."""

    dataset_id: str
    sessions: list[SessionData]

    def __iter__(self):
        return iter(self.sessions)

    def __len__(self) -> int:
        return len(self.sessions)

    def by_id(self, session_id: str) -> SessionData:
        for s in self.sessions:
            if s.session_id == session_id:
                return s
        raise KeyError(session_id)
