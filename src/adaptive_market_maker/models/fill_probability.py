"""Fill-probability and time-to-fill models from *shadow orders*.

A shadow order is a hypothetical 1-lot passive order placed at an observable state, at the best price
(offset 0) or ``offset`` ticks behind it, then run forward through the historical messages with exactly
the exchange's queue rules (``queue_step``) until it fills or ``horizon_s`` elapses. That yields labelled
data (filled within horizon?, time to fill, censored?) without assuming that touching a price means a fill.

Outputs:
* a logistic model of P(fill within horizon | state at placement), evaluated out of time;
* Kaplan-Meier time-to-fill curves (right-censored at the horizon);
* exponential fill intensities per distance from mid, giving data-driven (A, k) for
  lambda(delta) = A exp(-k delta) used by the GLFT / Avellaneda-Stoikov quote formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_market_maker.data.base import SessionData
from adaptive_market_maker.execution.queue_model import FillRules, QueuePosition
from adaptive_market_maker.execution.simulator import queue_step, visible_size
from adaptive_market_maker.features.pipeline import FeatureTable
from adaptive_market_maker.models.calibration import reliability_curve
from adaptive_market_maker.types import BUY, NS_PER_S, SELL

FILL_FEATURES = [
    "offset_ticks",
    "log_queue_ahead",
    "own_side_imbalance",
    "spread_ticks",
    "vol",
    "mo_rate",
    "aggression_toward",
    "log_opposite_sz1",
    "session_frac",
]


def simulate_shadow_orders(
    session: SessionData,
    ft: FeatureTable,
    start_ns: int,
    end_ns: int,
    rules: FillRules,
    horizon_s: float,
    n_states: int,
    seed: int,
    offsets: tuple[int, ...] = (0, 1),
    qty: int | None = None,
) -> pd.DataFrame:
    df = ft.df
    qty = qty or session.lot_size
    ts_obs = df["ts_ns"].to_numpy()
    eligible = np.nonzero(
        df["warm"].to_numpy()
        & df["valid_book"].to_numpy()
        & (ts_obs >= start_ns)
        & (ts_obs + int(horizon_s * NS_PER_S) <= end_ns)
    )[0]
    if eligible.size == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(eligible, min(n_states, eligible.size), replace=False))
    tick = session.tick_size
    hz = int(horizon_s * NS_PER_S)
    mo_col = next((c for c in df.columns if c.startswith("mo_rate_t")), None)
    tf_col = next((c for c in df.columns if c.startswith("trade_imb_t")), None)
    vol = df["vol_ticks_sqrt_s"].to_numpy()
    recs = []
    for k in pick:
        row = int(df["row"].iat[k])
        t0 = int(ts_obs[k])
        for side in (BUY, SELL):
            best = int(session.bid_px[row, 0] if side == BUY else session.ask_px[row, 0])
            for off in offsets:
                price = best - off * tick if side == BUY else best + off * tick
                vis = visible_size(session, row, side, price)
                pos = QueuePosition(side, price, row + 1, rules.initial_queue(vis))
                q0 = pos.queue_ahead
                remaining, i, t_fill = qty, row + 1, None
                n = len(session)
                while i < n and session.ts[i] <= t0 + hz:
                    got = queue_step(session, rules, pos, remaining, i)
                    if got:
                        remaining -= got
                        if remaining <= 0:
                            t_fill = int(session.ts[i])
                            break
                    i += 1
                imb = float(df["imb_l1"].iat[k])
                opp_sz = float(session.ask_sz[row, 0] if side == BUY else session.bid_sz[row, 0])
                recs.append(
                    {
                        "k": int(k),
                        "row": row,
                        "ts_ns": t0,
                        "side": side,
                        "offset_ticks": off,
                        "log_queue_ahead": float(np.log1p(min(q0, 1e9))),
                        "own_side_imbalance": side * imb,
                        "spread_ticks": float(df["spread_ticks"].iat[k]),
                        "vol": float(vol[k]),
                        "mo_rate": float(df[mo_col].iat[k]) if mo_col else np.nan,
                        "aggression_toward": float(-side * df[tf_col].iat[k]) if tf_col else np.nan,
                        "log_opposite_sz1": float(np.log1p(opp_sz)),
                        "session_frac": float(df["session_frac"].iat[k]),
                        "filled": t_fill is not None,
                        "time_to_fill_s": (t_fill - t0) / NS_PER_S if t_fill is not None else horizon_s,
                        "censored": t_fill is None,
                    }
                )
    return pd.DataFrame(recs)


def kaplan_meier(durations: np.ndarray, observed: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Survival S(t) = P(not yet filled by t), right-censored product-limit estimator, on ``grid``."""
    order = np.argsort(durations, kind="stable")
    d, e = durations[order], observed[order].astype(bool)
    at_risk = d.size
    surv, s = [], 1.0
    times, steps = [], []
    for t in np.unique(d):
        m = d == t
        events = int(e[m].sum())
        if events:
            s *= 1.0 - events / at_risk
        times.append(t)
        steps.append(s)
        at_risk -= int(m.sum())
    times, steps = np.array(times), np.array(steps)
    for g in grid:
        j = np.searchsorted(times, g, side="right") - 1
        surv.append(1.0 if j < 0 else float(steps[j]))
    return np.array(surv)


def fill_intensities(shadow: pd.DataFrame) -> dict:
    """Exponential fill rate per offset: fills / exposure time, and a log-linear fit lambda = A exp(-k delta)."""
    out = {}
    deltas, lams = [], []
    for off, g in shadow.groupby("offset_ticks"):
        exposure = float(g["time_to_fill_s"].sum())
        fills = int(g["filled"].sum())
        lam = fills / exposure if exposure > 0 else np.nan
        delta = 0.5 * float(g["spread_ticks"].mean()) + float(off)
        out[int(off)] = {"fill_rate_per_s": lam, "delta_ticks": delta, "n": int(len(g)), "fills": fills}
        if lam and lam > 0:
            deltas.append(delta)
            lams.append(lam)
    if len(deltas) >= 2:
        slope, intercept = np.polyfit(deltas, np.log(lams), 1)
        out["k_per_tick"] = float(max(-slope, 1e-3))
        out["A_per_s"] = float(np.exp(intercept))
    return out


@dataclass
class FillModel:
    pipeline: Pipeline
    horizon_s: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[FILL_FEATURES].to_numpy())[:, 1]


def fit_fill_model(train: pd.DataFrame, horizon_s: float) -> FillModel | None:
    if train.empty or train["filled"].nunique() < 2:
        return None
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(train[FILL_FEATURES].to_numpy(), train["filled"].astype(int).to_numpy())
    return FillModel(pipe, horizon_s)


def evaluate_fill_model(model: FillModel | None, test: pd.DataFrame) -> dict:
    if test.empty:
        return {"n": 0}
    y = test["filled"].astype(float).to_numpy()
    out = {
        "n": int(len(test)),
        "observed_fill_rate": float(y.mean()),
        "by_offset": {int(o): float(g["filled"].mean()) for o, g in test.groupby("offset_ticks")},
    }
    base = np.full_like(y, y.mean())
    out["brier_base_rate"] = float(np.mean((base - y) ** 2))
    if model is None:
        return out
    p = model.predict(test)
    eps = 1e-9
    out["brier"] = float(np.mean((p - y) ** 2))
    out["log_loss"] = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    out["mean_predicted"] = float(p.mean())
    out["reliability"] = reliability_curve(p, y)
    try:
        from sklearn.metrics import roc_auc_score

        out["auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < y.size else None
    except ValueError:
        out["auc"] = None
    return out
