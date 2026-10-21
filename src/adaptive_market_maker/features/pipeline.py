"""Causal, vectorized feature generation on *observable book states*.

A row of the feature table corresponds to the last message of a same-timestamp group (an observable
state, see ``SessionData.last_in_group``). Every feature at state k uses only messages with index <= the
state's row. Two clocks are kept distinct:

* event windows ``n``: the last n observable states (``*_e{n}`` columns),
* time windows ``W``: messages with timestamp in (t - W, t] (``*_t{W}`` columns).

Windows at the start of a session use whatever history exists; rows earlier than ``warmup_s`` after the
first message are flagged ``warm = False`` and excluded from modelling and trading. Missing values (an
invalid one-sided book) propagate as NaN; the mid used for returns is forward-filled causally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from adaptive_market_maker.config import FeatureConfig
from adaptive_market_maker.data.base import SessionData
from adaptive_market_maker.types import BUY, NS_PER_S, EventType


@dataclass
class FeatureTable:
    session_id: str
    df: pd.DataFrame  # one row per observable state
    groups: dict[str, list[str]] = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[str]:
        return [c for g in self.groups.values() for c in g]

    def columns_excluding(self, drop_groups: list[str]) -> list[str]:
        return [c for g, cols in self.groups.items() if g not in drop_groups for c in cols]


def _depth_weights(kind: str, k: int) -> np.ndarray:
    i = np.arange(1, k + 1, dtype=float)
    if kind == "uniform":
        return np.ones(k)
    if kind == "linear_decay":
        return (k + 1 - i) / k
    return np.exp(-(i - 1) / 2.0)


def _ffill(a: np.ndarray) -> np.ndarray:
    idx = np.where(np.isfinite(a), np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    return out


def _lag(a: np.ndarray, n: int) -> np.ndarray:
    """a[k - n] with the first available value for k < n (partial window at session start)."""
    idx = np.maximum(np.arange(a.size) - n, 0)
    return a[idx]


def _time_window_sum(cum: np.ndarray, ts: np.ndarray, rows: np.ndarray, window_ns: int) -> np.ndarray:
    """Sum of per-message x over messages with ts in (t - W, t], evaluated at observable rows."""
    t = ts[rows]
    j = np.searchsorted(ts, t - window_ns, side="right")
    before = np.where(j > 0, cum[np.maximum(j - 1, 0)], 0.0)
    return cum[rows] - before


def _asof_lag_time(values: np.ndarray, ts_obs: np.ndarray, window_ns: int) -> np.ndarray:
    """Observable-state value as of t - W (last state with ts <= t - W; first state if none)."""
    j = np.searchsorted(ts_obs, ts_obs - window_ns, side="right") - 1
    return values[np.maximum(j, 0)]


def build_features(session: SessionData, cfg: FeatureConfig) -> FeatureTable:
    rows = np.nonzero(session.last_in_group)[0]
    ts_all = session.ts
    ts = ts_all[rows]
    tick = session.tick_size
    groups: dict[str, list[str]] = {}
    out: dict[str, np.ndarray] = {}

    def add(group: str, name: str, values: np.ndarray) -> None:
        out[name] = np.asarray(values, dtype=np.float64)
        groups.setdefault(group, []).append(name)

    valid = session.valid_book[rows]
    bp = session.bid_px[rows].astype(np.float64) / tick
    ap = session.ask_px[rows].astype(np.float64) / tick
    bs = session.bid_sz[rows].astype(np.float64)
    as_ = session.ask_sz[rows].astype(np.float64)
    nan = np.nan

    mid = np.where(valid, 0.5 * (ap[:, 0] + bp[:, 0]), nan)
    spread = np.where(valid, ap[:, 0] - bp[:, 0], nan)
    denom = bs[:, 0] + as_[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        imb1 = np.where(valid & (denom > 0), (bs[:, 0] - as_[:, 0]) / denom, nan)
        micro = np.where(valid & (denom > 0), (ap[:, 0] * bs[:, 0] + bp[:, 0] * as_[:, 0]) / denom, nan)

    # ------------------------------------------------------------------ book (top of book)
    add("book_l1", "spread_ticks", spread)
    add("book_l1", "imb_l1", imb1)
    add("book_l1", "log_bid_sz1", np.log1p(bs[:, 0]))
    add("book_l1", "log_ask_sz1", np.log1p(as_[:, 0]))
    add("microprice", "micro_minus_mid", micro - mid)

    # ------------------------------------------------------------------ multi-level depth
    for k in cfg.depth_levels:
        if k == 1 or k > session.levels:
            continue
        w = _depth_weights(cfg.depth_weights, k)
        wb = (bs[:, :k] * w).sum(axis=1)
        wa = (as_[:, :k] * w).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            add("multi_level", f"imb_l{k}", np.where(valid & (wb + wa > 0), (wb - wa) / (wb + wa), nan))
        add(
            "multi_level",
            f"log_depth_ratio_l{k}",
            np.log1p(bs[:, :k].sum(axis=1)) - np.log1p(as_[:, :k].sum(axis=1)),
        )

    # ------------------------------------------------------------------ price history
    mid_ff = _ffill(mid)
    bb_ff, ba_ff = _ffill(np.where(valid, bp[:, 0], nan)), _ffill(np.where(valid, ap[:, 0], nan))
    add("price_history", "d_best_bid_e1", bb_ff - _lag(bb_ff, 1))
    add("price_history", "d_best_ask_e1", ba_ff - _lag(ba_ff, 1))
    for n in cfg.event_windows:
        add("price_history", f"ret_e{n}", mid_ff - _lag(mid_ff, n))
    for w_s in cfg.time_windows_s:
        add("price_history", f"ret_t{w_s:g}", mid_ff - _asof_lag_time(mid_ff, ts, int(w_s * NS_PER_S)))

    # ------------------------------------------------------------------ volatility (ticks / sqrt(s))
    dm = np.diff(mid_ff, prepend=mid_ff[0])
    dm = np.where(np.isfinite(dm), dm, 0.0)
    cum_sq = np.cumsum(dm * dm)
    vol_ns = int(cfg.vol_window_s * NS_PER_S)
    j = np.searchsorted(ts, ts - vol_ns, side="right")
    sq_sum = cum_sq - np.where(j > 0, cum_sq[np.maximum(j - 1, 0)], 0.0)
    elapsed = np.minimum((ts - ts[0]) / NS_PER_S, cfg.vol_window_s)
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = np.sqrt(np.where(elapsed > 0, sq_sum / np.maximum(elapsed, 1e-9), 0.0))
    add("volatility", f"rv_t{cfg.vol_window_s:g}", rv)

    # ------------------------------------------------------------------ order-flow imbalance (Cont-Kukanov-Stoikov 2014)
    # per message, using best bid/ask price and size after vs before the message
    pb, qb = session.bid_px[:, 0].astype(np.float64), session.bid_sz[:, 0].astype(np.float64)
    pa, qa = session.ask_px[:, 0].astype(np.float64), session.ask_sz[:, 0].astype(np.float64)
    pb0, qb0, pa0, qa0 = (np.roll(x, 1) for x in (pb, qb, pa, qa))
    e = (pb >= pb0) * qb - (pb <= pb0) * qb0 - (pa <= pa0) * qa + (pa >= pa0) * qa0
    e[0] = 0.0
    e = np.where(np.abs(e) > 1e12, 0.0, e)  # transitions involving an empty side carry no information
    lot = float(session.lot_size)
    cum_e = np.cumsum(e / lot)
    cum_e_obs = cum_e[rows]
    for n in cfg.event_windows:
        add("order_flow", f"ofi_e{n}", cum_e_obs - _lag(cum_e_obs, n))
    for w_s in cfg.time_windows_s:
        add("order_flow", f"ofi_t{w_s:g}", _time_window_sum(cum_e, ts_all, rows, int(w_s * NS_PER_S)))

    # ------------------------------------------------------------------ trade flow
    et = session.event_type
    is_exec = (et == EventType.EXEC_VISIBLE) | (et == EventType.EXEC_HIDDEN)
    # execution of a sell (buy) limit order = buyer (seller) initiated
    aggressor = np.where(is_exec, -session.side.astype(np.float64), 0.0)
    signed_vol = aggressor * session.size / lot
    cum_sv = np.cumsum(signed_vol)
    cum_tv = np.cumsum(np.where(is_exec, session.size / lot, 0.0))
    # market-order arrivals: one per timestamp group that contains executions
    exec_group_start = is_exec & np.concatenate([[True], (ts_all[1:] != ts_all[:-1]) | ~is_exec[:-1]])
    cum_mo = np.cumsum(exec_group_start.astype(np.float64))
    for w_s in cfg.time_windows_s:
        wn = int(w_s * NS_PER_S)
        sv = _time_window_sum(cum_sv, ts_all, rows, wn)
        tv = _time_window_sum(cum_tv, ts_all, rows, wn)
        add("trade_flow", f"signed_vol_t{w_s:g}", sv)
        with np.errstate(invalid="ignore", divide="ignore"):
            add("trade_flow", f"trade_imb_t{w_s:g}", np.where(tv > 0, sv / tv, 0.0))
        add("trade_flow", f"mo_rate_t{w_s:g}", _time_window_sum(cum_mo, ts_all, rows, wn) / w_s)

    # ------------------------------------------------------------------ activity / intensity
    is_cancel = (et == EventType.CANCEL) | (et == EventType.DELETE)
    is_submit = et == EventType.SUBMIT
    cum_c = np.cumsum(is_cancel.astype(np.float64))
    cum_s = np.cumsum(is_submit.astype(np.float64))
    # liquidity added at the touch, signed (+ bid side, - ask side), in lots
    best_b_after = session.bid_px[:, 0]
    best_a_after = session.ask_px[:, 0]
    at_touch = is_submit & (
        ((session.side == BUY) & (session.price == best_b_after))
        | ((session.side != BUY) & (session.price == best_a_after))
    )
    cum_rep = np.cumsum(np.where(at_touch, session.side * session.size / lot, 0.0))
    is_obs = session.last_in_group.astype(np.float64)
    cum_obs = np.cumsum(is_obs)
    w_mid = cfg.time_windows_s[len(cfg.time_windows_s) // 2]
    wn = int(w_mid * NS_PER_S)
    add("activity", f"cancel_rate_t{w_mid:g}", _time_window_sum(cum_c, ts_all, rows, wn) / w_mid)
    add("activity", f"submit_rate_t{w_mid:g}", _time_window_sum(cum_s, ts_all, rows, wn) / w_mid)
    add("activity", f"update_rate_t{w_mid:g}", _time_window_sum(cum_obs, ts_all, rows, wn) / w_mid)
    add("activity", f"touch_replenish_t{w_mid:g}", _time_window_sum(cum_rep, ts_all, rows, wn))

    # ------------------------------------------------------------------ timing
    changed = np.concatenate([[True], np.diff(mid_ff) != 0])
    last_change_ts = ts[np.maximum.accumulate(np.where(changed, np.arange(ts.size), 0))]
    add("timing", "secs_since_mid_change", (ts - last_change_ts) / NS_PER_S)
    exec_rows = np.where(is_exec, np.arange(ts_all.size), -1)
    last_exec = np.maximum.accumulate(exec_rows)[rows]
    since_trade = np.where(last_exec >= 0, (ts - ts_all[np.maximum(last_exec, 0)]) / NS_PER_S, np.nan)
    add("timing", "secs_since_trade", np.minimum(np.nan_to_num(since_trade, nan=600.0), 600.0))
    span = max(session.close_ns - session.open_ns, 1)
    add("timing", "session_frac", (ts - session.open_ns) / span)

    df = pd.DataFrame(out)
    df.insert(0, "row", rows)
    df.insert(1, "ts_ns", ts)
    df.insert(2, "feature_ts_ns", ts)  # latest event timestamp used to build the row
    df["mid_ticks"] = mid
    df["mid_ff_ticks"] = mid_ff
    df["valid_book"] = valid
    df["warm"] = (ts - ts_all[0]) >= int(cfg.warmup_s * NS_PER_S)
    df["vol_ticks_sqrt_s"] = np.maximum(rv, cfg.vol_floor_ticks_per_sqrt_s)
    return FeatureTable(session.session_id, df, groups)
