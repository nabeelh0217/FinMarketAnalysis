"""Transformations from source columns to the normalized representation.

Every lossy step is listed here and in docs/data_dictionary.md:

* LOBSTER dummy prices (+/-9999999999) and zero-size levels become the internal empty-level
  sentinels with size 0 (no information lost: an empty level carries none).
* Timestamps are parsed from the decimal string *exactly* into integer nanoseconds (no float round-trip).
  Digits beyond the 9th decimal, if any, are truncated and counted.
* ``origin_seq`` is *derived*: the row of the submission message for the same order id, or -1 if the
  order was never seen being submitted in this file (it rested before the file starts, or was submitted
  outside the recorded depth window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_market_maker.types import EMPTY_ASK_PRICE, EMPTY_BID_PRICE, EventType


def parse_decimal_seconds(values: pd.Series) -> tuple[np.ndarray, int]:
    """'34200.004241176' -> 34200004241176 ns. Returns (ns array, number of truncated values)."""
    s = values.astype(str).str.strip()
    parts = s.str.split(".", n=1, expand=True)
    whole = parts[0].astype(np.int64).to_numpy()
    if parts.shape[1] == 1:
        return whole * 1_000_000_000, 0
    frac_str = parts[1].fillna("")
    truncated = int((frac_str.str.len() > 9).sum())
    frac = frac_str.str.slice(0, 9).str.pad(9, side="right", fillchar="0").astype(np.int64).to_numpy()
    return whole * 1_000_000_000 + frac, truncated


def format_decimal_seconds(ts_ns: np.ndarray) -> np.ndarray:
    whole = ts_ns // 1_000_000_000
    frac = ts_ns % 1_000_000_000
    return np.array([f"{w}.{f:09d}" for w, f in zip(whole.tolist(), frac.tolist())], dtype=object)


def normalize_book(
    ask_px: np.ndarray, ask_sz: np.ndarray, bid_px: np.ndarray, bid_sz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Map source dummy values for empty levels onto the internal sentinels."""
    ask_empty = (ask_px >= EMPTY_ASK_PRICE) | (ask_sz <= 0)
    bid_empty = (bid_px <= EMPTY_BID_PRICE) | (bid_sz <= 0)
    stats = {
        # a real price with zero size (or a dummy price with positive size) is not the documented
        # convention: count it so the validator can flag it
        "ask_zero_size_real_price": int(((ask_sz <= 0) & (ask_px < EMPTY_ASK_PRICE)).sum()),
        "bid_zero_size_real_price": int(((bid_sz <= 0) & (bid_px > EMPTY_BID_PRICE)).sum()),
        "ask_dummy_price_positive_size": int(((ask_px >= EMPTY_ASK_PRICE) & (ask_sz > 0)).sum()),
        "bid_dummy_price_positive_size": int(((bid_px <= EMPTY_BID_PRICE) & (bid_sz > 0)).sum()),
    }
    ask_px = np.where(ask_empty, EMPTY_ASK_PRICE, ask_px).astype(np.int64)
    ask_sz = np.where(ask_empty, 0, ask_sz).astype(np.int64)
    bid_px = np.where(bid_empty, EMPTY_BID_PRICE, bid_px).astype(np.int64)
    bid_sz = np.where(bid_empty, 0, bid_sz).astype(np.int64)
    return ask_px, ask_sz, bid_px, bid_sz, stats


def compute_origin_seq(event_type: np.ndarray, order_id: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Row index of each order's submission message, or -1 when unknown.

    Under price-time priority, an order with origin -1 rested *before* the file started (or was entered
    outside the recorded depth), so it was ahead of any simulated order placed during the file.
    """
    n = order_id.shape[0]
    sub_rows = np.nonzero(event_type == EventType.SUBMIT)[0]
    sub_ids = order_id[sub_rows]
    first = pd.Series(sub_rows, index=sub_ids)
    dup_submissions = int(first.index.duplicated().sum())
    first = first[~first.index.duplicated(keep="first")]
    origin = pd.Series(order_id).map(first).to_numpy(dtype=np.float64, na_value=np.nan)
    origin = np.where(np.isnan(origin), -1, origin).astype(np.int64)
    rows = np.arange(n, dtype=np.int64)
    # a message referencing an order *before* its submission indicates id reuse or mis-ordering
    referenced_before_submit = origin > rows
    n_bad = int(referenced_before_submit.sum())
    origin[referenced_before_submit] = -1
    return origin, {"duplicate_submission_ids": dup_submissions, "referenced_before_submit": n_bad}
