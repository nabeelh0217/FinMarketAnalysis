"""Dependence-aware uncertainty: circular block bootstrap for means and paired differences.

P&L is aggregated into fixed clock-time intervals (``bootstrap_block_s``) per session; resampling draws
blocks of ``blocks_per_draw`` consecutive intervals to keep short-range autocorrelation (inventory carried
between intervals) inside each draw. Blocks wrap around *within* a session (Politis & Romano's circular
block bootstrap) and never span two sessions. Wrapping gives every interval the same inclusion probability;
a plain moving-block bootstrap under-samples the first and last intervals of each session, which here are
systematically different (the last one is the flatten window), and biases the bootstrap mean.

This captures dependence up to the block length only; regime-level dependence across sessions is not
captured, and with few intervals the interval estimate itself is noisy, reported via ``n_intervals``.
"""

from __future__ import annotations

import numpy as np


def _blocks(groups: list[np.ndarray], b: int) -> list[np.ndarray]:
    """All circular blocks of length ``b`` within each group (a short group is one whole block)."""
    out = []
    for g in groups:
        n = g.size
        if n == 0:
            continue
        if n <= b:
            out.append(g)
            continue
        offsets = np.arange(b)
        for s in range(n):
            out.append(g[(s + offsets) % n])
    return out


def block_bootstrap_mean(
    groups: list[np.ndarray], blocks_per_draw: int, n_boot: int, seed: int, alpha: float = 0.05
) -> dict:
    """Bootstrap distribution of the total-sample mean of per-interval values."""
    flat = np.concatenate([g for g in groups if g.size]) if groups else np.array([])
    n = flat.size
    if n < 4:
        return {
            "n_intervals": int(n),
            "mean": float(flat.mean()) if n else None,
            "ci_low": None,
            "ci_high": None,
            "p_le_zero": None,
            "note": "too few intervals for a bootstrap",
        }
    blocks = _blocks(groups, blocks_per_draw)
    rng = np.random.default_rng(seed)
    draws_per = int(np.ceil(n / blocks_per_draw))
    means = np.empty(n_boot)
    for r in range(n_boot):
        pick = rng.integers(0, len(blocks), draws_per)
        sample = np.concatenate([blocks[i] for i in pick])[:n]
        means[r] = sample.mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "n_intervals": int(n),
        "mean": float(flat.mean()),
        "se": float(means.std(ddof=1)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_le_zero": float(np.mean(means <= 0)),
        "blocks_per_draw": blocks_per_draw,
    }


def paired_ratio_difference(
    a_groups: list[np.ndarray],
    b_groups: list[np.ndarray],
    blocks_per_draw: int,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap of [mean/std](a) - [mean/std](b): a risk-adjusted comparison on aligned intervals.

    Blocks are drawn jointly for both strategies so the comparison stays paired.
    """
    pairs = []
    for a, b in zip(a_groups, b_groups):
        m = min(a.size, b.size)
        pairs.append(np.column_stack([a[:m], b[:m]]))
    flat = np.vstack([p for p in pairs if p.size]) if pairs else np.empty((0, 2))
    n = flat.shape[0]

    def ratio(x: np.ndarray) -> float:
        s = x.std(ddof=1)
        return float(x.mean() / s) if s > 0 else 0.0

    if n < 8:
        return {
            "n_intervals": int(n),
            "diff": None,
            "ci_low": None,
            "ci_high": None,
            "note": "too few intervals for a bootstrap",
        }
    idx_groups = []
    start = 0
    for p in pairs:
        idx_groups.append(np.arange(start, start + p.shape[0]))
        start += p.shape[0]
    blocks = _blocks(idx_groups, blocks_per_draw)
    rng = np.random.default_rng(seed)
    draws_per = int(np.ceil(n / blocks_per_draw))
    diffs = np.empty(n_boot)
    for r in range(n_boot):
        pick = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), draws_per)])[:n]
        diffs[r] = ratio(flat[pick, 0]) - ratio(flat[pick, 1])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "n_intervals": int(n),
        "ratio_a": ratio(flat[:, 0]),
        "ratio_b": ratio(flat[:, 1]),
        "diff": ratio(flat[:, 0]) - ratio(flat[:, 1]),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_le_zero": float(np.mean(diffs <= 0)),
        "definition": "mean / std of per-interval net P&L (not annualised)",
    }


def paired_difference(
    a_groups: list[np.ndarray], b_groups: list[np.ndarray], blocks_per_draw: int, n_boot: int, seed: int
) -> dict:
    """Bootstrap of mean(a - b) over aligned intervals (same sessions, same clock intervals)."""
    diffs = []
    for a, b in zip(a_groups, b_groups):
        m = min(a.size, b.size)
        diffs.append(a[:m] - b[:m])
    res = block_bootstrap_mean(diffs, blocks_per_draw, n_boot, seed)
    res["interpretation"] = (
        "positive mean favours the first strategy; p_le_zero is the bootstrap share of draws with mean <= 0 "
        "(not a p-value corrected for multiple comparisons)"
    )
    return res
