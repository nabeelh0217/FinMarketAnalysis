"""Shared fixtures: tiny hand-built sessions and small deterministic synthetic sessions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adaptive_market_maker.config import DataConfig, ProjectConfig
from adaptive_market_maker.data.base import SessionData
from adaptive_market_maker.data.normalize import compute_origin_seq
from adaptive_market_maker.types import EMPTY_ASK_PRICE, EMPTY_BID_PRICE

TICK = 100
OPEN_NS = 34_200 * 1_000_000_000


def px(ticks: int) -> int:
    return ticks * TICK


def build_session(rows: list[dict], levels: int = 3, close_ns: int | None = None) -> SessionData:
    """rows: dict(t_ms, type, oid, size, price_ticks, side, bids=[(ticks, size)], asks=[(ticks, size)])."""
    n = len(rows)
    ask_px = np.full((n, levels), EMPTY_ASK_PRICE, dtype=np.int64)
    bid_px = np.full((n, levels), EMPTY_BID_PRICE, dtype=np.int64)
    ask_sz = np.zeros((n, levels), dtype=np.int64)
    bid_sz = np.zeros((n, levels), dtype=np.int64)
    for i, r in enumerate(rows):
        for j, (p, s) in enumerate(sorted(r["bids"], reverse=True)[:levels]):
            bid_px[i, j], bid_sz[i, j] = px(p), s
        for j, (p, s) in enumerate(sorted(r["asks"])[:levels]):
            ask_px[i, j], ask_sz[i, j] = px(p), s
    et = np.array([r["type"] for r in rows], dtype=np.int8)
    oid = np.array([r["oid"] for r in rows], dtype=np.int64)
    origin, _ = compute_origin_seq(et, oid)
    ts = np.array([OPEN_NS + int(r["t_ms"] * 1_000_000) for r in rows], dtype=np.int64)
    return SessionData(
        session_id="TEST_2000-01-03",
        ticker="TEST",
        date="2000-01-03",
        source="fixture",
        levels=levels,
        tick_size=TICK,
        price_scale=10_000,
        lot_size=100,
        open_ns=OPEN_NS,
        close_ns=close_ns or int(ts[-1] + 1_000_000_000),
        ts=ts,
        event_type=et,
        order_id=oid,
        size=np.array([r["size"] for r in rows], dtype=np.int64),
        price=np.array([px(r["price_ticks"]) for r in rows], dtype=np.int64),
        side=np.array([r["side"] for r in rows], dtype=np.int8),
        origin_seq=origin,
        ask_px=ask_px,
        ask_sz=ask_sz,
        bid_px=bid_px,
        bid_sz=bid_sz,
    )


@pytest.fixture
def synthetic_cfg(tmp_path: Path) -> ProjectConfig:
    data = DataConfig(
        source="synthetic",
        dataset_id="pytest_synth",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        manifest_dir=tmp_path / "manifests",
        synthetic={"n_sessions": 3, "duration_s": 240.0, "seed": 3},
    )
    return ProjectConfig(data=data)


@pytest.fixture(scope="session")
def small_synthetic_session() -> SessionData:
    from adaptive_market_maker.data.synthetic import generate_session, synthetic_session_specs

    cfg = DataConfig(
        source="synthetic",
        dataset_id="fx",
        raw_dir=Path("unused"),
        synthetic={"n_sessions": 1, "duration_s": 300.0, "seed": 5},
    )
    return generate_session(cfg, synthetic_session_specs(cfg)[0], 0)
