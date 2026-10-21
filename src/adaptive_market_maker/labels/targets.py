"""Future targets. Used ONLY to train and evaluate models; never passed to a strategy.

Targets are defined on observable states (rows of the feature table).

* event horizon H: the mid H observable states later.
* time horizon h:  the *as-of* mid at t + h, i.e. the state after the last message with timestamp <= t + h.
  (The book is piecewise constant; "first observation at or after t + h" would use a state produced by a
  message after the horizon.)

``label_end_ns`` is the latest timestamp a label depends on; it drives purging at split boundaries.
Labels whose horizon runs past the last message of the session are NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_market_maker.config import LabelConfig
from adaptive_market_maker.types import NS_PER_MS

CLASSES_TERNARY = np.array([-1, 0, 1])
CLASSES_BINARY = np.array([0, 1])


def build_labels(features: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    ts = features["ts_ns"].to_numpy(np.int64)
    mid = features["mid_ff_ticks"].to_numpy(np.float64)
    n = ts.size
    if cfg.kind == "event":
        j = np.arange(n) + cfg.horizon_events
        ok = j < n
        j = np.minimum(j, n - 1)
        end_ns = np.where(ok, ts[j], np.iinfo(np.int64).max)
    else:
        h = int(round(cfg.horizon_ms * NS_PER_MS))
        target = ts + h
        j = np.searchsorted(ts, target, side="right") - 1
        ok = target <= ts[-1]
        end_ns = np.where(ok, target, np.iinfo(np.int64).max)
    dmid = np.where(ok, mid[j] - mid, np.nan)
    thr = cfg.flat_threshold_ticks
    if cfg.classes == "ternary":
        y = np.where(dmid > thr, 1, np.where(dmid < -thr, -1, 0)).astype(float)
    else:
        y = (dmid > thr).astype(float)
    y = np.where(np.isfinite(dmid), y, np.nan)
    return pd.DataFrame({"ts_ns": ts, "label_end_ns": end_ns, "dmid_ticks": dmid, "y": y})


def classes_for(cfg: LabelConfig) -> np.ndarray:
    return CLASSES_TERNARY if cfg.classes == "ternary" else CLASSES_BINARY


def non_overlapping_mask(ts_ns: np.ndarray, label_end_ns: np.ndarray) -> np.ndarray:
    """Greedy subsample in time order so that no two selected label windows overlap."""
    keep = np.zeros(ts_ns.size, dtype=bool)
    free_from = np.iinfo(np.int64).min
    for i in range(ts_ns.size):
        if ts_ns[i] >= free_from:
            keep[i] = True
            free_from = label_end_ns[i]
    return keep
