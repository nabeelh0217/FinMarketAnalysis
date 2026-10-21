"""Probability calibration (temperature scaling) and reliability curves."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-12


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    logp = np.log(np.clip(proba, EPS, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def log_loss(y_idx: np.ndarray, proba: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(proba[np.arange(y_idx.size), y_idx], EPS, 1.0))))


def fit_temperature(proba: np.ndarray, y_idx: np.ndarray) -> float:
    """Single scalar T > 0 minimising validation log loss; T > 1 softens over-confident models."""
    if y_idx.size < 50:
        return 1.0
    res = minimize_scalar(
        lambda t: log_loss(y_idx, apply_temperature(proba, t)), bounds=(0.05, 20.0), method="bounded"
    )
    return float(res.x) if res.success else 1.0


def reliability_curve(p: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> dict[str, list[float]]:
    """Mean predicted vs observed frequency in equal-count bins of the predicted probability."""
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    return {
        "predicted": [float(p[b].mean()) for b in bins if b.size],
        "observed": [float(outcome[b].mean()) for b in bins if b.size],
        "count": [int(b.size) for b in bins if b.size],
    }
