"""Chronological train / validation / test segments with purging and embargo.

A *segment* is a time interval inside one session. Samples belong to a segment if their decision time is
inside it AND their label window ends inside it (purging), so no training label peeks into validation or
test time. An embargo gap separates consecutive roles inside a session.

Modes
-----
intraday  one session (e.g. the free LOBSTER sample) split by time: [train | gap | val | gap | test]
sessions  whole sessions per role, in date order (walk-forward-style; requires several sessions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from adaptive_market_maker.config import SplitConfig
from adaptive_market_maker.data.base import Dataset
from adaptive_market_maker.types import NS_PER_S

Role = Literal["train", "val", "test"]


@dataclass(frozen=True)
class Segment:
    session_id: str
    role: Role
    start_ns: int
    end_ns: int

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / NS_PER_S

    def contains(self, ts_ns: np.ndarray, label_end_ns: np.ndarray | None = None) -> np.ndarray:
        m = (ts_ns >= self.start_ns) & (ts_ns < self.end_ns)
        if label_end_ns is not None:
            m &= label_end_ns <= self.end_ns
        return m


def tradable_window(session, cfg: SplitConfig) -> tuple[int, int]:
    first, last = int(session.ts[0]), int(session.ts[-1])
    start = max(first, session.open_ns) + int(cfg.exclude_open_s * NS_PER_S)
    end = min(last, session.close_ns) - int(cfg.exclude_close_s * NS_PER_S)
    if end <= start:
        raise ValueError(f"{session.session_id}: no tradable time left after open/close exclusions")
    return start, end


def make_segments(dataset: Dataset, cfg: SplitConfig) -> list[Segment]:
    emb = int(cfg.embargo_s * NS_PER_S)
    segs: list[Segment] = []
    if cfg.mode == "intraday":
        for s in dataset:
            a, b = tradable_window(s, cfg)
            span = b - a
            t1 = a + int(span * cfg.train_frac)
            t2 = t1 + int(span * cfg.val_frac)
            segs += [
                Segment(s.session_id, "train", a, t1),
                Segment(s.session_id, "val", t1 + emb, t2),
                Segment(s.session_id, "test", t2 + emb, b),
            ]
    else:
        n = len(dataset)
        if n < cfg.train_sessions + cfg.val_sessions + 1:
            raise ValueError(
                f"sessions mode needs >= {cfg.train_sessions + cfg.val_sessions + 1} sessions, have {n}"
            )
        for idx, s in enumerate(dataset):
            role: Role = (
                "train"
                if idx < cfg.train_sessions
                else ("val" if idx < cfg.train_sessions + cfg.val_sessions else "test")
            )
            a, b = tradable_window(s, cfg)
            segs.append(Segment(s.session_id, role, a, b))
    for seg in segs:
        if seg.end_ns <= seg.start_ns:
            raise ValueError(f"empty segment {seg}")
    return segs


def by_role(segments: list[Segment], role: Role) -> list[Segment]:
    return [s for s in segments if s.role == role]
