"""Engine invariants: causality, latency ordering, tie-breaking, risk limits, determinism, reconciliation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_market_maker.backtest.engine import _WAKE, BacktestEngine, run_backtest
from adaptive_market_maker.config import DataConfig, FeatureConfig, ProjectConfig, with_overrides
from adaptive_market_maker.execution.orders import OrderState
from adaptive_market_maker.features.pipeline import build_features
from adaptive_market_maker.strategies.adaptive import make_strategy
from adaptive_market_maker.strategies.base import Signals
from adaptive_market_maker.types import NS_PER_S


def _cfg(**over) -> ProjectConfig:
    cfg = ProjectConfig(data=DataConfig(source="synthetic", dataset_id="t", raw_dir="unused"))
    return with_overrides(cfg, over) if over else cfg


def _signals(session, alpha=None) -> Signals:
    df = build_features(session, FeatureConfig(warmup_s=20)).df
    return Signals(
        df["ts_ns"].to_numpy(),
        df["row"].to_numpy(),
        df["vol_ticks_sqrt_s"].to_numpy(),
        alpha,
        "",
        (df["warm"] & df["valid_book"]).to_numpy(),
    )


def _window(session):
    return int(session.ts[0] + 30 * NS_PER_S), int(session.ts[-1] - 5 * NS_PER_S)


@pytest.fixture(scope="module")
def runs(small_synthetic_session):
    s = small_synthetic_session
    cfg = _cfg()
    a, b = _window(s)
    eng = BacktestEngine(s, make_strategy("fixed_spread", cfg.strategies), cfg, _signals(s), a, b)
    res = eng.run()
    return s, cfg, eng, res


def test_run_produces_fills_and_reconciles(runs):
    _, _, eng, res = runs
    assert res.summary["n_fills"] > 20
    eng.ledger.reconcile(0.0)  # mark irrelevant for cash/position checks; equity identity holds for any mark


def test_no_fill_before_order_arrival_and_arrival_respects_latency(runs):
    _, cfg, eng, res = runs
    lat = cfg.execution.latency
    min_gap = int((lat.decision_us + lat.order_entry_us) * 1000)
    for o in eng.orders.values():
        if o.arrival_ns is not None:
            assert o.arrival_ns == o.sent_ns + int(lat.order_entry_us * 1000)
    by_id = {o.order_id: o for o in eng.orders.values()}
    for _, f in res.fills.iterrows():
        o = by_id[f["order_id"]]
        assert f["ts_ns"] >= o.arrival_ns
        if o.cancel_sent_ns is not None and o.state == OrderState.CANCELLED:
            assert f["ts_ns"] <= o.done_ns  # nothing fills after a cancel took effect
    decisions = res.decisions["ts_ns"].to_numpy()
    assert (np.diff(decisions) >= 0).all() and min_gap > 0


def test_inventory_never_exceeds_limit(small_synthetic_session):
    s = small_synthetic_session
    cfg = _cfg(**{"strategies.risk.max_inventory": 200})
    a, b = _window(s)
    res = run_backtest(s, make_strategy("fixed_spread", cfg.strategies), cfg, _signals(s), a, b)
    path = np.cumsum(res.fills["side"] * res.fills["qty"])
    assert np.abs(path).max() <= 200
    assert (
        res.summary["risk_blocks"].get("inventory_long", 0)
        + res.summary["risk_blocks"].get("inventory_short", 0)
        > 0
    )


def test_replay_is_deterministic(runs):
    s, cfg, _, res = runs
    a, b = _window(s)
    again = run_backtest(s, make_strategy("fixed_spread", cfg.strategies), cfg, _signals(s), a, b)
    pd.testing.assert_frame_equal(res.fills, again.fills)
    assert res.summary == again.summary


def test_future_data_cannot_change_earlier_decisions_or_fills(runs):
    s, cfg, _, res = runs
    a, b = _window(s)
    cut_row = len(s) * 2 // 3
    t_cut = int(s.ts[cut_row])
    s2 = s.slice_rows(0, len(s))
    s2.size[cut_row:] *= 2
    s2.bid_sz[cut_row:] *= 3
    s2.ask_sz[cut_row:] += 700
    res2 = run_backtest(s2, make_strategy("fixed_spread", cfg.strategies), cfg, _signals(s2), a, b)
    before = res.fills[res.fills["ts_ns"] < t_cut].reset_index(drop=True)
    before2 = res2.fills[res2.fills["ts_ns"] < t_cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, before2)
    d1 = res.decisions[res.decisions["ts_ns"] < t_cut].reset_index(drop=True)
    d2 = res2.decisions[res2.decisions["ts_ns"] < t_cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(d1, d2)


def test_agent_event_at_same_timestamp_waits_for_historical_message(runs):
    s, cfg, _, _ = runs
    a, b = _window(s)
    eng = BacktestEngine(s, make_strategy("fixed_spread", cfg.strategies), cfg, _signals(s), a, b)
    t = int(s.ts[1000])
    eng._push(t, _WAKE, None)
    eng._process_until(t)  # strictly-before rule: an event AT t is not processed before message t
    assert len(eng.heap) == 1
    eng._process_until(t + 1)
    assert len(eng.heap) == 0


def test_flatten_closes_inventory_with_taker_orders(small_synthetic_session):
    s = small_synthetic_session
    cfg = _cfg(**{"backtest.flatten_before_end_s": 20})
    a, b = _window(s)
    res = run_backtest(s, make_strategy("fixed_spread", cfg.strategies), cfg, _signals(s), a, b)
    flat = res.fills[res.fills["tag"] == "flatten"]
    assert res.summary["final_position"] == 0
    assert (flat["liquidity"] == "taker").all()
    assert res.summary["net_pnl_liquidation"] == pytest.approx(res.summary["net_pnl_mid"])


def test_adaptive_without_alpha_matches_inventory_aware(small_synthetic_session):
    s = small_synthetic_session
    cfg = _cfg(**{"strategies.adaptive.use_alpha": False})
    a, b = _window(s)
    sig = _signals(s, alpha=np.full(int(s.last_in_group.sum()), 0.9))
    r1 = run_backtest(s, make_strategy("adaptive", cfg.strategies), cfg, sig, a, b)
    r2 = run_backtest(s, make_strategy("inventory_aware", cfg.strategies), cfg, sig, a, b)
    pd.testing.assert_frame_equal(r1.fills, r2.fills)


def test_optimistic_fill_preset_fills_more_than_conservative(small_synthetic_session):
    s = small_synthetic_session
    a, b = _window(s)
    n = {}
    for preset in ("conservative", "central", "optimistic"):
        cfg = _cfg(**{"execution.fill_preset": preset})
        n[preset] = run_backtest(
            s, make_strategy("fixed_spread", cfg.strategies), cfg, _signals(s), a, b
        ).summary["n_fills"]
    assert n["conservative"] <= n["central"] <= n["optimistic"]
