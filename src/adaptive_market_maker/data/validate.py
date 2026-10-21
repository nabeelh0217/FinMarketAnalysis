"""Data-quality checks run before any modelling.

Suspicious observations are flagged and counted, never silently dropped. ``critical`` failures make
:func:`validate_session` raise :class:`DataIntegrityError` when ``raise_on_critical`` is set, because a
corrupted book must not flow into features or the simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from adaptive_market_maker.data.base import SessionData
from adaptive_market_maker.orderbook.validation import DataIntegrityError, check_transitions
from adaptive_market_maker.types import (
    EMPTY_ASK_PRICE,
    EMPTY_BID_PRICE,
    HALT_START,
    HALT_TRADE_RESUME,
    NS_PER_S,
    EventType,
)

Severity = Literal["info", "warning", "critical"]


@dataclass
class Check:
    name: str
    severity: Severity
    passed: bool
    count: int = 0
    detail: str = ""


@dataclass
class ValidationReport:
    session_id: str
    n_rows: int
    checks: list[Check] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add(self, name: str, severity: Severity, passed: bool, count: int = 0, detail: str = "") -> None:
        self.checks.append(Check(name, severity, bool(passed), int(count), detail))

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "critical")

    @property
    def status(self) -> str:
        if not self.passed:
            return "FAILED"
        if any(not c.passed for c in self.checks if c.severity == "warning"):
            return "PASSED_WITH_WARNINGS"
        return "PASSED"

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "n_rows": self.n_rows,
            "status": self.status,
            "checks": [asdict(c) for c in self.checks],
            "summary": self.summary,
        }

    def format(self) -> str:
        lines = [f"[{self.status}] {self.session_id}: {self.n_rows:,} rows"]
        for c in self.checks:
            mark = "ok " if c.passed else ("!! " if c.severity == "critical" else "-- ")
            lines.append(f"  {mark}{c.severity:<8} {c.name:<34} count={c.count:<8} {c.detail}")
        return "\n".join(lines)


def halt_intervals(session: SessionData) -> list[tuple[int, int]]:
    """[(halt_start_ns, trade_resume_ns)] from type-7 messages; open-ended halts end at close."""
    rows = np.nonzero(session.event_type == EventType.HALT)[0]
    out, start = [], None
    for r in rows:
        code = int(session.price[r])
        if code == HALT_START and start is None:
            start = int(session.ts[r])
        elif code == HALT_TRADE_RESUME and start is not None:
            out.append((start, int(session.ts[r])))
            start = None
    if start is not None:
        out.append((start, session.close_ns))
    return out


def validate_session(
    session: SessionData, max_gap_warning_s: float = 30.0, raise_on_critical: bool = False
) -> ValidationReport:
    n = len(session)
    rep = ValidationReport(session.session_id, n)
    rep.add("non_empty", "critical", n > 0, n)
    if n == 0:
        if raise_on_critical:
            raise DataIntegrityError(rep.format())
        return rep

    et, side, size, price, ts = session.event_type, session.side, session.size, session.price, session.ts
    tick = session.tick_size

    bad_type = ~np.isin(et, [int(e) for e in EventType])
    rep.add("event_type_known", "critical", not bad_type.any(), bad_type.sum())
    order_msgs = np.isin(et, [1, 2, 3, 4, 5])
    rep.add(
        "side_valid",
        "critical",
        np.isin(side[order_msgs], [-1, 1]).all(),
        (~np.isin(side[order_msgs], [-1, 1])).sum(),
    )
    rep.add("size_positive", "critical", (size[order_msgs] > 0).all(), (size[order_msgs] <= 0).sum())
    vis = np.isin(et, [1, 2, 3, 4])
    rep.add("price_positive", "critical", (price[vis] > 0).all(), (price[vis] <= 0).sum())
    off_grid = (price[vis] % tick) != 0
    rep.add(
        "price_on_tick_grid",
        "warning",
        not off_grid.any(),
        off_grid.sum(),
        "visible-order prices not on the tick grid",
    )
    hidden_off_grid = (price[et == 5] % tick) != 0
    rep.add(
        "hidden_exec_sub_tick",
        "info",
        True,
        hidden_off_grid.sum(),
        "hidden executions at sub-tick prices (e.g. midpoint) are legitimate",
    )

    dts = np.diff(ts)
    rep.add("timestamps_monotonic", "critical", (dts >= 0).all(), (dts < 0).sum())
    same_ts = int((dts == 0).sum())
    rep.add(
        "same_timestamp_groups",
        "info",
        True,
        same_ts,
        "messages sharing a timestamp with the previous one (sweeps); only group ends are observable",
    )
    msg = np.column_stack([ts, et, session.order_id, size, price, side])
    dup = int((np.all(msg[1:] == msg[:-1], axis=1)).sum())
    rep.add("duplicate_messages", "warning", dup == 0, dup, "identical consecutive messages")

    outside = (ts < session.open_ns) | (ts > session.close_ns)
    rep.add(
        "within_session",
        "warning",
        not outside.any(),
        outside.sum(),
        f"session {session.open_ns / NS_PER_S:.0f}s-{session.close_ns / NS_PER_S:.0f}s",
    )
    big_gaps = np.nonzero(dts > max_gap_warning_s * NS_PER_S)[0]
    rep.add(
        "timestamp_gaps",
        "warning",
        big_gaps.size == 0,
        big_gaps.size,
        f"gaps > {max_gap_warning_s}s; largest {dts.max() / NS_PER_S:.2f}s" if dts.size else "",
    )

    halts = halt_intervals(session)
    rep.add("trading_halts", "info", True, len(halts), str(halts[:3]))
    rep.add(
        "cross_trades",
        "info",
        True,
        int((et == EventType.CROSS).sum()),
        "type-6 cross/auction trades excluded from continuous-trading fills",
    )

    # book shape
    bid_ok = session.bid_sz[:, 0] > 0
    ask_ok = session.ask_sz[:, 0] > 0
    rep.add("bid_side_non_empty", "warning", bid_ok.all(), (~bid_ok).sum())
    rep.add("ask_side_non_empty", "warning", ask_ok.all(), (~ask_ok).sum())
    both = bid_ok & ask_ok
    crossed = both & (session.ask_px[:, 0] < session.bid_px[:, 0])
    locked = both & (session.ask_px[:, 0] == session.bid_px[:, 0])
    rep.add("crossed_book", "warning", not crossed.any(), crossed.sum())
    rep.add("locked_book", "warning", not locked.any(), locked.sum())

    def monotone(px: np.ndarray, sz: np.ndarray, increasing: bool) -> int:
        live = (sz[:, 1:] > 0) & (sz[:, :-1] > 0)
        step = np.diff(px, axis=1)
        wrong = (step <= 0) if increasing else (step >= 0)
        return int((wrong & live).sum())

    # a gap inside the visible levels (empty level followed by a non-empty one) is malformed
    def holes(sz: np.ndarray) -> int:
        return int(((sz[:, :-1] == 0) & (sz[:, 1:] > 0)).sum())

    nm_a, nm_b = (
        monotone(session.ask_px, session.ask_sz, True),
        monotone(session.bid_px, session.bid_sz, False),
    )
    rep.add("ask_levels_increasing", "critical", nm_a == 0, nm_a)
    rep.add("bid_levels_decreasing", "critical", nm_b == 0, nm_b)
    h = holes(session.ask_sz) + holes(session.bid_sz)
    rep.add("no_interior_empty_levels", "critical", h == 0, h)
    live_a = session.ask_sz > 0
    live_b = session.bid_sz > 0
    neg = int((session.ask_sz < 0).sum() + (session.bid_sz < 0).sum())
    rep.add("book_sizes_non_negative", "critical", neg == 0, neg)
    grid = int(((session.ask_px[live_a] % tick) != 0).sum() + ((session.bid_px[live_b] % tick) != 0).sum())
    rep.add("book_prices_on_grid", "warning", grid == 0, grid)
    sentinel_bad = int(
        (session.ask_px[live_a] == EMPTY_ASK_PRICE).sum() + (session.bid_px[live_b] == EMPTY_BID_PRICE).sum()
    )
    rep.add("sentinel_with_size", "critical", sentinel_bad == 0, sentinel_bad)

    spread = session.spread_ticks
    valid_spread = spread[np.isfinite(spread)]
    wide = int((valid_spread > 50).sum())
    rep.add("anomalous_wide_spread", "warning", wide == 0, wide, "> 50 ticks")

    # normalisation / id bookkeeping recorded by the loader
    meta = session.meta
    for key in (
        "ask_zero_size_real_price",
        "bid_zero_size_real_price",
        "ask_dummy_price_positive_size",
        "bid_dummy_price_positive_size",
        "duplicate_submission_ids",
        "referenced_before_submit",
        "time_digits_truncated",
    ):
        if key in meta:
            rep.add(key, "warning", int(meta[key]) == 0, int(meta[key]))

    tr = check_transitions(session)
    detail = "; ".join(d.describe() for d in tr.examples[:2])
    rep.add(
        "book_transitions_consistent",
        "critical",
        tr.n_discrepancies == 0,
        tr.n_discrepancies,
        detail or f"{tr.n_checked:,} book-changing messages reconciled",
    )
    rep.add("untouched_side_unchanged", "critical", tr.other_side_changed == 0, tr.other_side_changed)

    rep.summary = {
        "first_ts_s": ts[0] / NS_PER_S,
        "last_ts_s": ts[-1] / NS_PER_S,
        "event_counts": {
            EventType(int(t)).name: int((et == t).sum()) for t in np.unique(et) if not bad_type[et == t].any()
        },
        "median_spread_ticks": float(np.median(valid_spread)) if valid_spread.size else None,
        "frac_one_tick_spread": float((valid_spread == 1).mean()) if valid_spread.size else None,
        "mean_l1_depth_shares": float(0.5 * (session.bid_sz[:, 0].mean() + session.ask_sz[:, 0].mean())),
        "observable_states": int(session.last_in_group.sum()),
        "mid_changes": int((np.diff(session.mid_ticks[session.last_in_group]) != 0).sum()),
        "halts": halts,
    }
    if raise_on_critical and not rep.passed:
        raise DataIntegrityError(rep.format())
    return rep
