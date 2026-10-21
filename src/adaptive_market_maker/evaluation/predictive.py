"""Predictive-model evaluation metrics (classification quality and signal quality)."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score

from adaptive_market_maker.models.calibration import log_loss, reliability_curve
from adaptive_market_maker.models.price_direction import class_index


def classification_metrics(y: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> dict:
    yi = class_index(y, classes)
    onehot = np.eye(classes.size)[yi]
    pred = proba.argmax(axis=1)
    out: dict = {
        "n": int(y.size),
        "class_balance": {int(c): float(np.mean(y == c)) for c in classes},
        "accuracy": float(np.mean(pred == yi)),
        "balanced_accuracy": float(balanced_accuracy_score(yi, pred)) if np.unique(yi).size > 1 else None,
        "log_loss": log_loss(yi, proba),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
    }
    prec, rec, _, _ = precision_recall_fscore_support(
        yi, pred, labels=np.arange(classes.size), zero_division=0
    )
    out["precision"] = {int(c): float(p) for c, p in zip(classes, prec)}
    out["recall"] = {int(c): float(r) for c, r in zip(classes, rec)}
    up = int(np.nonzero(classes == classes.max())[0][0])
    is_up = (yi == up).astype(int)
    out["auc_up_vs_rest"] = (
        float(roc_auc_score(is_up, proba[:, up])) if 0 < is_up.sum() < is_up.size else None
    )
    if classes.size == 3:
        dn = int(np.nonzero(classes == classes.min())[0][0])
        is_dn = (yi == dn).astype(int)
        out["auc_down_vs_rest"] = (
            float(roc_auc_score(is_dn, proba[:, dn])) if 0 < is_dn.sum() < is_dn.size else None
        )
        # directional AUC among rows that actually moved: P(up) - P(down) as a score
        moved = yi != int(np.nonzero(classes == 0)[0][0])
        if moved.sum() > 10 and 0 < is_up[moved].sum() < moved.sum():
            out["auc_direction_given_move"] = float(
                roc_auc_score(is_up[moved], (proba[:, up] - proba[:, dn])[moved])
            )
    out["reliability_up"] = reliability_curve(proba[:, up], is_up.astype(float))
    return out


def signal_quality(alpha: np.ndarray, dmid: np.ndarray) -> dict:
    ok = np.isfinite(alpha) & np.isfinite(dmid)
    a, d = alpha[ok], dmid[ok]
    if a.size < 10 or np.std(a) < 1e-12:  # e.g. the constant-probability baseline
        return {"n": int(a.size), "ic_spearman": None, "ic_pearson": None}
    q = np.quantile(a, [0.2, 0.8])
    return {
        "n": int(a.size),
        "ic_spearman": float(spearmanr(a, d).statistic),
        "ic_pearson": float(np.corrcoef(a, d)[0, 1]),
        "mean_dmid_top_quintile": float(d[a >= q[1]].mean()),
        "mean_dmid_bottom_quintile": float(d[a <= q[0]].mean()),
        "alpha_std_ticks": float(a.std()),
        "regression_slope": float(np.polyfit(a, d, 1)[0]),
    }
