"""Short-horizon mid-price direction models and the probability -> alpha conversion.

Candidates: ``prior`` (constant class frequencies from training), ``logistic`` (standardized multinomial
logistic regression), ``hgb`` (histogram gradient-boosted trees). Every preprocessing step lives inside a
scikit-learn ``Pipeline`` fitted on training rows only.

Alpha: a class probability is not an expected return. The model output is converted to an expected
mid change in *ticks over the label horizon*:

    alpha = sum_c p_c * E_train[dmid | class c]

with the conditional means estimated on training data only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_market_maker.config import ModelConfig
from adaptive_market_maker.models.calibration import apply_temperature, fit_temperature


def build_estimator(name: str, cfg: ModelConfig):
    if name == "prior":
        return DummyClassifier(strategy="prior")
    if name == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=cfg.logistic_C, max_iter=cfg.logistic_max_iter)),
            ]
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=cfg.hgb_max_iter,
            learning_rate=cfg.hgb_learning_rate,
            max_leaf_nodes=cfg.hgb_max_leaf_nodes,
            min_samples_leaf=cfg.hgb_min_samples_leaf,
            l2_regularization=cfg.hgb_l2,
            early_stopping=False,
            random_state=cfg.seed,
        )
    raise ValueError(f"unknown model {name!r}")


@dataclass
class DirectionModel:
    name: str
    estimator: object
    feature_columns: list[str]
    classes: np.ndarray
    temperature: float = 1.0
    class_mean_dmid: dict[int, float] = field(default_factory=dict)
    horizon_desc: str = ""

    def raw_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.estimator.predict_proba(X)
        # align estimator class order with self.classes (a class absent from training gets 0)
        out = np.zeros((X.shape[0], self.classes.size))
        for j, c in enumerate(self.estimator.classes_):
            out[:, int(np.nonzero(self.classes == c)[0][0])] = p[:, j]
        out = np.clip(out, 1e-9, None)
        return out / out.sum(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return apply_temperature(self.raw_proba(X), self.temperature)

    def alpha_ticks(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        mu = np.array([self.class_mean_dmid.get(int(c), 0.0) for c in self.classes])
        return p @ mu


def class_index(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {int(c): i for i, c in enumerate(classes)}
    return np.array([lookup[int(v)] for v in y], dtype=np.int64)


def fit_direction_model(
    name: str,
    cfg: ModelConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    dmid_train: np.ndarray,
    classes: np.ndarray,
    feature_columns: list[str],
    X_cal: np.ndarray | None = None,
    y_cal: np.ndarray | None = None,
) -> DirectionModel:
    est = build_estimator(name, cfg)
    if X_train.shape[0] > cfg.max_train_samples:
        rng = np.random.default_rng(cfg.seed)
        idx = np.sort(rng.choice(X_train.shape[0], cfg.max_train_samples, replace=False))
        X_fit, y_fit = X_train[idx], y_train[idx]
    else:
        X_fit, y_fit = X_train, y_train
    est.fit(X_fit, y_fit.astype(int))
    means = {int(c): float(dmid_train[y_train == c].mean()) if np.any(y_train == c) else 0.0 for c in classes}
    model = DirectionModel(name, est, feature_columns, classes, 1.0, means)
    if cfg.calibrate and name != "prior" and X_cal is not None and y_cal is not None and y_cal.size:
        model.temperature = fit_temperature(model.raw_proba(X_cal), class_index(y_cal, classes))
    return model
