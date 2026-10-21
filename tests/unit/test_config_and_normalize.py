"""Configuration loading/validation and exact source normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from adaptive_market_maker.config import (
    DataConfig,
    ProjectConfig,
    load_config,
    parse_clock,
    with_overrides,
)
from adaptive_market_maker.data.normalize import (
    compute_origin_seq,
    format_decimal_seconds,
    normalize_book,
    parse_decimal_seconds,
)
from adaptive_market_maker.types import EMPTY_ASK_PRICE, EMPTY_BID_PRICE


def test_parse_clock():
    assert parse_clock("09:30:00") == 34_200 * 10**9
    assert parse_clock("16:00:00.5") == 57_600 * 10**9 + 500_000_000
    with pytest.raises(ValueError):
        parse_clock("25:00:00")


def test_decimal_seconds_parsed_exactly_without_float_round_trip():
    ns, trunc = parse_decimal_seconds(pd.Series(["34200.004241176", "34200.5", "57599", "34713.6851552431"]))
    assert ns.tolist() == [34_200_004_241_176, 34_200_500_000_000, 57_599_000_000_000, 34_713_685_155_243]
    assert trunc == 1  # the 10th decimal was truncated and counted
    assert format_decimal_seconds(ns[:1]).tolist() == ["34200.004241176"]


def test_normalize_book_maps_lobster_dummies_and_counts_anomalies():
    ask_px = np.array([[9999999999, 300100], [300100, 300200]])
    ask_sz = np.array([[0, 100], [100, 0]])
    bid_px = np.array([[-9999999999, 300000], [300000, 299900]])
    bid_sz = np.array([[0, 5], [100, 200]])
    a, asz, b, bsz, stats = normalize_book(ask_px, ask_sz, bid_px, bid_sz)
    assert a[0, 0] == EMPTY_ASK_PRICE and asz[0, 0] == 0
    assert b[0, 0] == EMPTY_BID_PRICE and bsz[0, 0] == 0
    assert a[1, 1] == EMPTY_ASK_PRICE  # real price with zero size -> empty, and flagged
    assert stats["ask_zero_size_real_price"] == 1


def test_origin_seq_marks_preexisting_orders_and_reuse():
    et = np.array([1, 4, 3, 1, 2, 4])
    oid = np.array([10, 99, 10, 11, 12, 11])
    origin, stats = compute_origin_seq(et, oid)
    assert origin.tolist() == [0, -1, 0, 3, -1, 3]  # 99 and 12 rested before the file started
    _, stats2 = compute_origin_seq(np.array([3, 1]), np.array([5, 5]))
    assert stats2["referenced_before_submit"] == 1


def test_unknown_config_keys_are_errors():
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(
            {"data": {"source": "synthetic", "dataset_id": "x", "raw_dir": "r", "bogus": 1}}
        )


def test_lobster_config_requires_sessions():
    with pytest.raises(ValidationError):
        DataConfig(source="lobster", dataset_id="x", raw_dir="r")


def test_synthetic_close_time_follows_duration():
    d = DataConfig(source="synthetic", dataset_id="x", raw_dir="r", synthetic={"duration_s": 600})
    assert d.close_ns - d.open_ns == 600 * 10**9


def test_null_scenario_removes_information_but_keeps_activity():
    d = DataConfig(source="synthetic", dataset_id="x", raw_dir="r", synthetic={"scenario": "null"})
    assert d.synthetic.informed_rate == 0 and d.synthetic.limit_tilt == 0
    assert d.synthetic.market_rate == d.synthetic.null_market_rate


def test_load_config_include_replaces_section_and_overlay_merges(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    root = tmp_path / "configs"
    root.mkdir()
    (root / "data.yaml").write_text(
        "data:\n  source: lobster\n  dataset_id: real\n  raw_dir: r\n  sessions: [{ticker: X, date: '2020-01-02'}]\n"
    )
    (root / "execution.yaml").write_text(
        "execution:\n  fill_preset: central\n  fees: {maker_per_share: -0.002}\n"
    )
    (root / "syn.yaml").write_text("data:\n  source: synthetic\n  dataset_id: syn\n  raw_dir: s\n")
    exp = root / "exp.yaml"
    exp.write_text(f"include: [{root / 'syn.yaml'}]\nexecution:\n  fees: {{taker_per_share: 0.004}}\n")
    monkeypatch.setenv("AMM_CONFIG_DIR", str(root))
    cfg = load_config(exp)
    assert cfg.data.source == "synthetic" and cfg.data.sessions == []  # replaced wholesale
    assert (
        cfg.execution.fees.maker_per_share == -0.002 and cfg.execution.fees.taker_per_share == 0.004
    )  # merged


def test_with_overrides_and_fingerprint():
    cfg = ProjectConfig(data=DataConfig(source="synthetic", dataset_id="x", raw_dir="r"))
    cfg2 = with_overrides(cfg, {"execution.fill_preset": "optimistic"})
    assert cfg2.execution.fill_preset == "optimistic" and cfg.execution.fill_preset == "central"
    assert cfg.fingerprint() != cfg2.fingerprint()
    assert cfg.fingerprint() == ProjectConfig.model_validate(cfg.snapshot()).fingerprint()
