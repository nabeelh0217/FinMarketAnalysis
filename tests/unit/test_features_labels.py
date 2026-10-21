"""Hand-calculated features, causality (future-perturbation), labels and chronological splits."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_market_maker.backtest.splits import make_segments
from adaptive_market_maker.config import FeatureConfig, LabelConfig, SplitConfig
from adaptive_market_maker.data.base import Dataset
from adaptive_market_maker.features.pipeline import build_features
from adaptive_market_maker.labels.targets import build_labels, non_overlapping_mask
from adaptive_market_maker.types import BUY, SELL
from tests.conftest import build_session

BIDS = [(3000, 300), (2999, 200), (2998, 100)]
ASKS = [(3001, 100), (3002, 200), (3003, 300)]
FCFG = FeatureConfig(
    depth_levels=[1, 3], event_windows=[1, 2], time_windows_s=[1.0, 5.0], vol_window_s=5.0, warmup_s=0.0
)


def _fixture():
    rows = [
        dict(t_ms=0, type=1, oid=1, size=100, price_ticks=2998, side=BUY, bids=BIDS, asks=ASKS),
        # +100 at the best bid: OFI = +100 shares
        dict(
            t_ms=200,
            type=1,
            oid=2,
            size=100,
            price_ticks=3000,
            side=BUY,
            bids=[(3000, 400), (2999, 200), (2998, 100)],
            asks=ASKS,
        ),
        # best ask (100) executed -> ask moves up: OFI = +100 (lost ask depth); buyer-initiated 100
        dict(
            t_ms=400,
            type=4,
            oid=3,
            size=100,
            price_ticks=3001,
            side=SELL,
            bids=[(3000, 400), (2999, 200), (2998, 100)],
            asks=[(3002, 200), (3003, 300)],
        ),
        dict(
            t_ms=1500,
            type=1,
            oid=4,
            size=100,
            price_ticks=3001,
            side=SELL,
            bids=[(3000, 400), (2999, 200), (2998, 100)],
            asks=[(3001, 100), (3002, 200), (3003, 300)],
        ),
    ]
    return build_session(rows)


def test_hand_calculated_book_features():
    df = build_features(_fixture(), FCFG).df
    r0 = df.iloc[0]
    assert r0["imb_l1"] == pytest.approx((300 - 100) / 400)
    # microprice = (a Qb + b Qa) / (Qb + Qa) = (3001*300 + 3000*100) / 400 = 3000.75 ; mid 3000.5
    assert r0["micro_minus_mid"] == pytest.approx(0.25)
    assert r0["imb_l3"] == pytest.approx(0.0)  # 600 vs 600 over three levels
    assert r0["spread_ticks"] == 1.0
    r2 = df.iloc[2]
    assert r2["spread_ticks"] == 2.0 and r2["mid_ticks"] == pytest.approx(3001.0)


def test_hand_calculated_order_flow_and_trade_flow():
    df = build_features(_fixture(), FCFG).df
    assert df["ofi_e1"].tolist()[:3] == pytest.approx([0.0, 1.0, 1.0])  # lots
    assert df["ofi_t1"].iloc[2] == pytest.approx(2.0)  # rows 1 and 2 within the last second
    assert df["ofi_t1"].iloc[3] == pytest.approx(
        -1.0
    )  # only row 3 (ask refilled at 3001: -100) in (0.5s, 1.5s]
    assert df["signed_vol_t1"].iloc[2] == pytest.approx(1.0)  # buyer-initiated 1 lot
    assert df["ret_e2"].iloc[2] == pytest.approx(0.5)  # mid 3000.5 -> 3001.0


def test_features_only_on_observable_states_and_carry_timestamps(small_synthetic_session):
    s = small_synthetic_session
    df = build_features(s, FeatureConfig()).df
    assert len(df) == int(s.last_in_group.sum())
    assert (df["feature_ts_ns"] == s.ts[df["row"]]).all()
    assert df["ts_ns"].is_monotonic_increasing


def test_features_do_not_change_when_only_future_events_change(small_synthetic_session):
    s = small_synthetic_session
    cut = len(s) // 2
    s2 = s.slice_rows(0, len(s))
    s2.size[cut:] *= 3
    s2.bid_sz[cut:] *= 2
    s2.ask_px[cut:] += s.tick_size
    a = build_features(s, FeatureConfig()).df
    b = build_features(s2, FeatureConfig()).df
    before = a["row"] < cut
    cols = [c for c in a.columns if c != "session_frac"]
    np.testing.assert_allclose(
        a.loc[before, cols].to_numpy(float), b.loc[before, cols].to_numpy(float), equal_nan=True
    )
    assert not np.allclose(
        a.loc[~before, "imb_l1"].to_numpy(), b.loc[~before, "imb_l1"].to_numpy(), equal_nan=True
    )


def test_time_label_uses_asof_state_not_first_after():
    df = build_features(_fixture(), FCFG).df  # observable states at 0, 0.2, 0.4, 1.5 s
    lab = build_labels(df, LabelConfig(kind="time", horizon_ms=1000))
    mid = df["mid_ff_ticks"].to_numpy()
    # t=0.2 -> target 1.2 s: as-of state is the one at 0.4 s, not the one at 1.5 s
    assert lab["dmid_ticks"].iloc[1] == pytest.approx(mid[2] - mid[1])
    assert lab["y"].iloc[1] == 1
    assert np.isnan(lab["y"].iloc[3])  # horizon beyond the data
    assert lab["label_end_ns"].iloc[1] == df["ts_ns"].iloc[1] + 1_000_000_000


def test_event_label_and_purge_metadata():
    df = build_features(_fixture(), FCFG).df
    lab = build_labels(df, LabelConfig(kind="event", horizon_events=2))
    mid = df["mid_ff_ticks"].to_numpy()
    assert lab["dmid_ticks"].iloc[0] == pytest.approx(mid[2] - mid[0])
    assert lab["label_end_ns"].iloc[1] == df["ts_ns"].iloc[3]
    assert lab["y"].isna().sum() == 2


def test_non_overlapping_mask():
    ts = np.array([0, 1, 2, 3, 4, 5])
    end = ts + 2
    assert non_overlapping_mask(ts, end).tolist() == [True, False, True, False, True, False]


def test_intraday_segments_are_ordered_with_embargo(small_synthetic_session):
    ds = Dataset("x", [small_synthetic_session])
    cfg = SplitConfig(
        mode="intraday", train_frac=0.5, val_frac=0.2, embargo_s=5, exclude_open_s=10, exclude_close_s=10
    )
    tr, va, te = make_segments(ds, cfg)
    assert (tr.role, va.role, te.role) == ("train", "val", "test")
    assert tr.end_ns + 5 * 10**9 == va.start_ns and va.end_ns + 5 * 10**9 == te.start_ns
    ts = np.array([tr.end_ns - 1, tr.end_ns - 1])
    ends = np.array([tr.end_ns - 1, tr.end_ns + 1])  # second label peeks past the train boundary
    assert tr.contains(ts, ends).tolist() == [True, False]


def test_sessions_mode_requires_enough_sessions(small_synthetic_session):
    with pytest.raises(ValueError):
        make_segments(Dataset("x", [small_synthetic_session]), SplitConfig(mode="sessions"))
