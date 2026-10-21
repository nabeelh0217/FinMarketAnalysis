"""Deterministic pipeline outputs pinned by the regression test.

If behaviour changes *intentionally*, regenerate with ``python scripts/update_regression_fixtures.py``
and explain the change in the commit message.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from adaptive_market_maker.backtest.engine import run_backtest
from adaptive_market_maker.config import DataConfig, FeatureConfig, LabelConfig, ProjectConfig
from adaptive_market_maker.data.synthetic import generate_session, synthetic_session_specs
from adaptive_market_maker.evaluation.metrics import trading_metrics
from adaptive_market_maker.features.pipeline import build_features
from adaptive_market_maker.labels.targets import build_labels
from adaptive_market_maker.strategies.adaptive import make_strategy
from adaptive_market_maker.strategies.base import Signals

EXPECTED = Path(__file__).with_name("expected_outputs.json")


def _sha(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:20]


def compute_outputs() -> dict:
    data = DataConfig(
        source="synthetic",
        dataset_id="regression",
        raw_dir="unused",
        synthetic={"n_sessions": 1, "duration_s": 180.0, "seed": 99},
    )
    s = generate_session(data, synthetic_session_specs(data)[0], 0)
    cfg = ProjectConfig(data=data)
    ft = build_features(s, FeatureConfig(warmup_s=20))
    df = ft.df
    lab = build_labels(df, LabelConfig(kind="time", horizon_ms=1000))
    sig = Signals(
        df["ts_ns"].to_numpy(),
        df["row"].to_numpy(),
        df["vol_ticks_sqrt_s"].to_numpy(),
        None,
        "",
        (df["warm"] & df["valid_book"]).to_numpy(),
    )
    out: dict = {
        "n_messages": len(s),
        "book_sha": _sha(s.ts, s.event_type, s.price, s.size, s.ask_px, s.ask_sz, s.bid_px, s.bid_sz),
        "n_observable_states": len(df),
        "feature_sample": {
            c: [round(float(v), 10) for v in df[c].iloc[[500, 1500, 2500]]]
            for c in ("imb_l1", "micro_minus_mid", "imb_l5", "ofi_t5", "signed_vol_t5", "rv_t30", "ret_t5")
        },
        "label_counts": {str(int(k)): int(v) for k, v in lab["y"].value_counts().sort_index().items()},
    }
    a, b = int(s.ts[0] + 30e9), int(s.ts[-1] - 5e9)
    for name in ("fixed_spread", "inventory_aware"):
        res = run_backtest(s, make_strategy(name, cfg.strategies), cfg, sig, a, b)
        m = trading_metrics(res)
        out[name] = {
            "n_fills": int(len(res.fills)),
            "fills_sha": _sha(res.fills[["ts_ns", "side", "price", "qty"]].to_numpy(np.int64)),
            "net_pnl": round(m["net_pnl"], 6),
            "fees_net": round(m["fees_net"], 6),
            "mean_abs_inventory": round(m["mean_abs_inventory"], 6),
            "max_drawdown": round(m["max_drawdown"], 6),
        }
    return out
