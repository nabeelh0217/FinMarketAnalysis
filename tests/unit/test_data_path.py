"""LOBSTER round trip, synthetic generator properties and data validation."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_market_maker.config import SessionSpec
from adaptive_market_maker.data.lobster import LobsterFormatError, load_session, write_session
from adaptive_market_maker.data.prepare import load_dataset, prepare_dataset
from adaptive_market_maker.data.validate import validate_session
from adaptive_market_maker.orderbook.validation import DataIntegrityError, check_transitions


def test_lobster_round_trip_is_exact(tmp_path, small_synthetic_session, synthetic_cfg):
    s = small_synthetic_session
    spec = SessionSpec(ticker="SYNTH", date="2099-01-05", start_ms=34_200_000, end_ms=34_500_000)
    write_session(s, tmp_path, spec)
    cfg = synthetic_cfg.data.model_copy(update={"raw_dir": tmp_path})
    back = load_session(spec, cfg, source="synthetic")
    for name in (
        "ts",
        "event_type",
        "order_id",
        "size",
        "price",
        "side",
        "origin_seq",
        "ask_px",
        "ask_sz",
        "bid_px",
        "bid_sz",
    ):
        np.testing.assert_array_equal(getattr(back, name), getattr(s, name), err_msg=name)


def test_row_count_mismatch_fails_loudly(tmp_path, small_synthetic_session, synthetic_cfg):
    spec = SessionSpec(ticker="SYNTH", date="2099-01-05")
    msg, ob = write_session(small_synthetic_session, tmp_path, spec)
    lines = ob.read_text().splitlines()
    ob.write_text("\n".join(lines[:-1]) + "\n")
    cfg = synthetic_cfg.data.model_copy(update={"raw_dir": tmp_path})
    with pytest.raises(LobsterFormatError, match="row count mismatch"):
        load_session(spec, cfg)


def test_missing_file_gives_clear_error(tmp_path, synthetic_cfg):
    cfg = synthetic_cfg.data.model_copy(update={"raw_dir": tmp_path})
    with pytest.raises(FileNotFoundError, match="LOBSTER message file not found"):
        load_session(SessionSpec(ticker="MSFT", date="2012-06-21"), cfg)


def test_synthetic_session_passes_every_critical_check(small_synthetic_session):
    rep = validate_session(small_synthetic_session)
    assert rep.passed, rep.format()
    tr = check_transitions(small_synthetic_session)
    assert tr.passed and tr.n_checked > 1000


def test_corrupted_book_is_detected_and_raises(small_synthetic_session):
    s = small_synthetic_session.slice_rows(0, 2000)
    row = 1500
    s.bid_sz[row:, 0] += 100  # a size change that no message explains
    rep = validate_session(s)
    assert not rep.passed
    bad = {c.name for c in rep.checks if not c.passed}
    assert "book_transitions_consistent" in bad or "untouched_side_unchanged" in bad
    with pytest.raises(DataIntegrityError):
        validate_session(s, raise_on_critical=True)


def test_non_monotonic_time_and_crossed_book_flagged(small_synthetic_session):
    s = small_synthetic_session.slice_rows(0, 500)
    s.ts[100] = s.ts[99] - 1
    s.ask_px[200, 0] = s.bid_px[200, 0] - s.tick_size
    rep = validate_session(s)
    by = {c.name: c for c in rep.checks}
    assert not by["timestamps_monotonic"].passed and by["timestamps_monotonic"].severity == "critical"
    assert not by["crossed_book"].passed


def test_synthetic_generation_is_deterministic(synthetic_cfg):
    from adaptive_market_maker.data.synthetic import generate_session, synthetic_session_specs

    spec = synthetic_session_specs(synthetic_cfg.data)[0]
    a = generate_session(synthetic_cfg.data, spec, 0)
    b = generate_session(synthetic_cfg.data, spec, 0)
    np.testing.assert_array_equal(a.ts, b.ts)
    np.testing.assert_array_equal(a.bid_sz, b.bid_sz)


def test_prepare_writes_parquet_manifest_and_validation(synthetic_cfg):
    paths = prepare_dataset(synthetic_cfg.data)
    assert len(paths) == synthetic_cfg.data.synthetic.n_sessions
    ds = load_dataset(synthetic_cfg.data, prepare_if_missing=False)
    from adaptive_market_maker.data.manifest import read_manifests

    mans = read_manifests(synthetic_cfg.data)
    assert len(mans) == len(ds)
    m = mans[0]
    assert m["synthetic"] is True and len(m["raw_files"][0]["sha256"]) == 64
    assert m["row_count"] == len(ds.sessions[0])
    assert m["validation_status"].startswith("PASSED")


def test_synthetic_generator_has_no_directional_bias():
    from adaptive_market_maker.config import DataConfig
    from adaptive_market_maker.data.synthetic import SyntheticMarket

    cfg = DataConfig(source="synthetic", dataset_id="t", raw_dir="x", synthetic={"duration_s": 240})
    buyer_share = []
    for seed in range(12):
        m = SyntheticMarket(cfg.synthetic, 10, 100, seed=500 + seed)
        m.initialize(3000)
        m.run(cfg.open_ns, 240)
        et, side = np.array(m.out_type), np.array(m.out_side)
        buyer_share.append(np.mean(side[et == 4] == -1))  # execution of a sell order = buyer-initiated
    assert abs(np.mean(buyer_share) - 0.5) < 0.02


def test_duplicate_messages_are_counted_by_type(small_synthetic_session):
    s = small_synthetic_session.slice_rows(0, 300)
    row = int(np.nonzero(s.event_type[1:] == 4)[0][0]) + 1  # duplicate one execution message
    for name in ("ts", "event_type", "order_id", "size", "price", "side", "origin_seq"):
        a = getattr(s, name)
        setattr(s, name, np.insert(a, row + 1, a[row]))
    for name in ("ask_px", "ask_sz", "bid_px", "bid_sz"):
        a = getattr(s, name)
        setattr(s, name, np.insert(a, row + 1, a[row], axis=0))
    check = next(c for c in validate_session(s).checks if c.name == "duplicate_messages")
    assert check.count == 1 and "EXEC_VISIBLE': 1" in check.detail
