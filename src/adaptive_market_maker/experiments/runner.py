"""End-to-end research experiment.

Protocol (pre-declared, see docs/research_protocol.md):

1. Load + validate data; build causal features and labels per session.
2. Chronological segments (train / val / test) with purge + embargo.
3. Direction models: fit on train, temperature-calibrate on the first half of val, select by log loss on
   the second half of val. Test metrics are reported for *every* candidate.
4. Fill model from shadow orders: fit on train, evaluate on test; estimate (A, k) fill intensities.
5. Strategy parameters: small pre-declared grids evaluated on val only; the best net P&L is frozen.
   The alpha strategy inherits the inventory strategy's chosen parameters, so the only difference
   between B and C is the alpha component (isolating its contribution).
6. Test: every strategy once on identical test segments; paired block-bootstrap comparisons.
7. Robustness on test (reported, never used to re-select): sensitivity grids and ablations.

Every backtest is logged in ``trials.csv``; the run is appended to the global experiment registry.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_market_maker.backtest.engine import BacktestResult, run_backtest
from adaptive_market_maker.backtest.splits import Segment, by_role, make_segments
from adaptive_market_maker.config import ProjectConfig, save_snapshot, with_overrides
from adaptive_market_maker.data.base import Dataset, SessionData
from adaptive_market_maker.data.manifest import read_manifests
from adaptive_market_maker.data.prepare import load_dataset
from adaptive_market_maker.evaluation.markouts import markout_summary, markout_table
from adaptive_market_maker.evaluation.metrics import interval_pnl, trading_metrics
from adaptive_market_maker.evaluation.predictive import classification_metrics, signal_quality
from adaptive_market_maker.evaluation.statistics import (
    block_bootstrap_mean,
    paired_difference,
    paired_ratio_difference,
)
from adaptive_market_maker.execution.queue_model import FillRules
from adaptive_market_maker.experiments.registry import append_entry
from adaptive_market_maker.features.pipeline import FeatureTable, build_features
from adaptive_market_maker.labels.targets import build_labels, classes_for, non_overlapping_mask
from adaptive_market_maker.models.fill_probability import (
    evaluate_fill_model,
    fill_intensities,
    fit_fill_model,
    kaplan_meier,
    simulate_shadow_orders,
)
from adaptive_market_maker.models.price_direction import DirectionModel, fit_direction_model
from adaptive_market_maker.strategies.adaptive import make_strategy
from adaptive_market_maker.strategies.base import Signals
from adaptive_market_maker.types import NS_PER_S

log = logging.getLogger(__name__)

SHARED_INVENTORY_FIELDS = (
    "gamma",
    "k",
    "A",
    "horizon_mode",
    "tau_s",
    "spread_mode",
    "half_spread_ticks",
    "min_half_spread_ticks",
    "use_vol_adaptation",
    "static_vol_ticks_per_sqrt_s",
    "use_inventory_skew",
    "quote_size",
    "requote_threshold_ticks",
    "rounding",
    "estimate_fill_intensity",
)


@dataclass
class Bundle:
    session: SessionData
    features: FeatureTable
    labels: pd.DataFrame


@dataclass
class Context:
    cfg: ProjectConfig
    dataset: Dataset
    bundles: dict[str, Bundle]
    segments: list[Segment]
    signals: dict[tuple[str, str], Signals] = field(default_factory=dict)

    def base_signals(
        self, sid: str, alpha: np.ndarray | None, horizon: str = "", vol: np.ndarray | None = None
    ) -> Signals:
        df = self.bundles[sid].features.df
        return Signals(
            ts_ns=df["ts_ns"].to_numpy(np.int64),
            rows=df["row"].to_numpy(np.int64),
            vol=df["vol_ticks_sqrt_s"].to_numpy(np.float64) if vol is None else vol,
            alpha=alpha,
            alpha_horizon=horizon,
            eligible=(df["warm"] & df["valid_book"]).to_numpy(),
        )


_CTX: Context | None = None  # read by forked worker processes


def _run_job(job: dict) -> dict:
    ctx = _CTX
    assert ctx is not None
    cfg: ProjectConfig = job["cfg"]
    seg: Segment = job["segment"]
    b = ctx.bundles[seg.session_id]
    sig = ctx.signals[(job.get("signal_key", "default"), seg.session_id)]
    res = run_backtest(
        b.session, make_strategy(job["strategy"], cfg.strategies), cfg, sig, seg.start_ns, seg.end_ns
    )
    out = {
        "metrics": trading_metrics(res, cfg.backtest.bootstrap_block_s),
        "interval_pnl": interval_pnl(res, cfg.backtest.bootstrap_block_s),
        "segment": seg,
    }
    if job.get("keep"):
        out["result"] = res
    return out


def run_jobs(ctx: Context, jobs: list[dict], n_jobs: int) -> list[dict]:
    global _CTX
    _CTX = ctx
    if n_jobs > 1 and "fork" not in mp.get_all_start_methods():
        # workers read the (large) context by inheriting it through fork; without fork (Windows),
        # shipping it to every worker would cost more than it saves, so run serially
        log.warning("process fork unavailable on this platform; running %d backtests serially", len(jobs))
        n_jobs = 1
    if n_jobs <= 1 or len(jobs) <= 1:
        return [_run_job(j) for j in jobs]
    with mp.get_context("fork").Pool(n_jobs) as pool:
        return pool.map(_run_job, jobs, chunksize=1)


def state_index_before(state_ts: np.ndarray, event_ts: np.ndarray) -> np.ndarray:
    """Index of the last observable state strictly before each event (-1 if none).

    A fill shares its timestamp with the end of the burst that caused it; that end state already reflects
    the fill, so classifying a fill by it (side="right") would label fills by their own consequences.
    """
    return np.searchsorted(state_ts, event_ts, side="left") - 1


# ---------------------------------------------------------------------------------------- data/features
def build_context(cfg: ProjectConfig) -> Context:
    dataset = load_dataset(cfg.data)
    bundles = {}
    for s in dataset:
        ft = build_features(s, cfg.features)
        bundles[s.session_id] = Bundle(s, ft, build_labels(ft.df, cfg.models.labels))
    return Context(cfg, dataset, bundles, make_segments(dataset, cfg.models.splits))


def _split_half(seg: Segment) -> tuple[Segment, Segment]:
    mid = seg.start_ns + (seg.end_ns - seg.start_ns) // 2
    return Segment(seg.session_id, seg.role, seg.start_ns, mid), Segment(
        seg.session_id, seg.role, mid, seg.end_ns
    )


def assemble(ctx: Context, segs: list[Segment], cols: list[str], non_overlap: bool) -> dict:
    Xs, ys, ds, ts, sids, ks = [], [], [], [], [], []
    for seg in segs:
        b = ctx.bundles[seg.session_id]
        df, lab = b.features.df, b.labels
        m = seg.contains(df["ts_ns"].to_numpy(), lab["label_end_ns"].to_numpy())
        m &= (df["warm"] & df["valid_book"]).to_numpy() & lab["y"].notna().to_numpy()
        if non_overlap:
            idx = np.nonzero(m)[0]
            keep = non_overlapping_mask(lab["ts_ns"].to_numpy()[idx], lab["label_end_ns"].to_numpy()[idx])
            m = np.zeros_like(m)
            m[idx[keep]] = True
        Xs.append(df.loc[m, cols].to_numpy(np.float64))
        ys.append(lab.loc[m, "y"].to_numpy())
        ds.append(lab.loc[m, "dmid_ticks"].to_numpy())
        ts.append(df.loc[m, "ts_ns"].to_numpy())
        sids += [seg.session_id] * int(m.sum())
        ks.append(np.nonzero(m)[0])
    cat = lambda a: np.concatenate(a) if a else np.array([])  # noqa: E731
    return {
        "X": np.vstack(Xs) if Xs else np.empty((0, len(cols))),
        "y": cat(ys),
        "dmid": cat(ds),
        "ts": cat(ts),
        "sid": np.array(sids),
        "k": cat(ks),
    }


# ------------------------------------------------------------------------------------------ prediction
def train_models(
    ctx: Context, drop_groups: list[str] | None = None, candidates: list[str] | None = None, label_cfg=None
) -> dict:
    cfg = ctx.cfg
    mcfg = cfg.models
    any_ft = next(iter(ctx.bundles.values())).features
    groups = mcfg.feature_groups or list(any_ft.groups)
    drop = set(drop_groups or [])
    cols = [c for g in groups if g not in drop for c in any_ft.groups[g]]
    classes = classes_for(label_cfg or mcfg.labels)
    train_segs = by_role(ctx.segments, "train")
    val_segs = by_role(ctx.segments, "val")
    cal_segs, sel_segs = zip(*[_split_half(s) for s in val_segs])
    tr = assemble(ctx, train_segs, cols, non_overlap=False)
    cal = assemble(ctx, list(cal_segs), cols, non_overlap=False)
    sel = assemble(ctx, list(sel_segs), cols, non_overlap=True)
    te = assemble(ctx, by_role(ctx.segments, "test"), cols, non_overlap=mcfg.eval_stride == "non_overlapping")
    out: dict = {
        "feature_columns": cols,
        "dropped_groups": sorted(drop),
        "n_train": int(tr["y"].size),
        "n_val_cal": int(cal["y"].size),
        "n_val_sel": int(sel["y"].size),
        "n_test": int(te["y"].size),
        "candidates": {},
    }
    models: dict[str, DirectionModel] = {}
    for name in candidates or mcfg.candidates:
        t0 = time.time()
        model = fit_direction_model(
            name, mcfg, tr["X"], tr["y"], tr["dmid"], classes, cols, cal["X"], cal["y"]
        )
        models[name] = model
        rec = {
            "temperature": model.temperature,
            "class_mean_dmid": model.class_mean_dmid,
            "fit_seconds": round(time.time() - t0, 2),
        }
        rec["val"] = (
            classification_metrics(sel["y"], model.predict_proba(sel["X"]), classes) if sel["y"].size else {}
        )
        if te["y"].size:
            p_te = model.predict_proba(te["X"])
            rec["test"] = classification_metrics(te["y"], p_te, classes)
            rec["test_signal"] = signal_quality(model.alpha_ticks(te["X"]), te["dmid"])
        out["candidates"][name] = rec
    # dependence-aware uncertainty on the test log-loss gain over the constant-probability baseline
    if "prior" in models and te["y"].size:
        from adaptive_market_maker.models.calibration import EPS
        from adaptive_market_maker.models.price_direction import class_index

        yi = class_index(te["y"], classes)

        def per_sample(m: DirectionModel) -> np.ndarray:
            p = m.predict_proba(te["X"])
            return -np.log(np.clip(p[np.arange(yi.size), yi], EPS, 1.0))

        base = per_sample(models["prior"])
        bt = cfg.backtest
        for name, m in models.items():
            if name == "prior":
                continue
            gain = base - per_sample(m)
            groups = [gain[te["sid"] == sid] for sid in dict.fromkeys(te["sid"].tolist())]
            res = block_bootstrap_mean(groups, 30, bt.n_bootstrap, bt.seed)
            res["note"] = (
                "per-sample log-loss reduction vs prior (nats); moving blocks of 30 non-overlapping samples"
            )
            out["candidates"][name]["test_logloss_gain_vs_prior"] = res
    ranked = sorted(out["candidates"], key=lambda n: out["candidates"][n]["val"].get("log_loss", np.inf))
    out["selected"] = ranked[0]
    out["selection_rule"] = "lowest log loss on the second half of validation (non-overlapping labels)"
    out["test_rows"] = {"ts": te["ts"], "sid": te["sid"], "k": te["k"], "y": te["y"], "dmid": te["dmid"]}
    out["models"] = models
    return out


def alpha_arrays(ctx: Context, model: DirectionModel) -> dict[str, np.ndarray]:
    out = {}
    for sid, b in ctx.bundles.items():
        df = b.features.df
        X = df[model.feature_columns].to_numpy(np.float64)
        a = model.alpha_ticks(X)
        a[~(df["warm"] & df["valid_book"]).to_numpy()] = np.nan
        out[sid] = a
    return out


def regime_labels(ctx: Context, thresholds: dict | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Causal regimes per observable state; thresholds come from TRAINING rows only."""
    vcol = next(c for c in next(iter(ctx.bundles.values())).features.df.columns if c.startswith("rv_t"))
    tcol = next(
        c for c in next(iter(ctx.bundles.values())).features.df.columns if c.startswith("trade_imb_t")
    )
    if thresholds is None:
        vols, depth = [], []
        for seg in by_role(ctx.segments, "train"):
            df = ctx.bundles[seg.session_id].features.df
            m = seg.contains(df["ts_ns"].to_numpy()) & df["valid_book"].to_numpy()
            vols.append(df.loc[m, vcol].to_numpy())
            depth.append((np.expm1(df.loc[m, "log_bid_sz1"]) + np.expm1(df.loc[m, "log_ask_sz1"])).to_numpy())
        v = np.concatenate(vols)
        thresholds = {
            "vol_low": float(np.quantile(v, 1 / 3)),
            "vol_high": float(np.quantile(v, 2 / 3)),
            "depth_median": float(np.median(np.concatenate(depth))),
            "flow_directional": 0.5,
            "vol_column": vcol,
            "flow_column": tcol,
        }
    out = {}
    for sid, b in ctx.bundles.items():
        df = b.features.df
        vol = df[vcol].to_numpy()
        depth = (np.expm1(df["log_bid_sz1"]) + np.expm1(df["log_ask_sz1"])).to_numpy()
        out[sid] = pd.DataFrame(
            {
                "ts_ns": df["ts_ns"].to_numpy(),
                "volatility": np.where(
                    vol < thresholds["vol_low"], "low", np.where(vol < thresholds["vol_high"], "mid", "high")
                ),
                "spread": np.where(df["spread_ticks"].to_numpy() <= 1, "1 tick", ">1 tick"),
                "depth": np.where(depth >= thresholds["depth_median"], "deep", "thin"),
                "flow": np.where(
                    np.abs(df[tcol].to_numpy()) >= thresholds["flow_directional"], "directional", "balanced"
                ),
                "time": np.where(df["session_frac"].to_numpy() < 0.5, "early", "late"),
            }
        )
    return out, thresholds


# -------------------------------------------------------------------------------------- the experiment
class ExperimentRunner:
    def __init__(self, cfg: ProjectConfig, overwrite: bool | None = None):
        self.cfg = cfg
        self.exp = cfg.experiment
        self.dir = self.exp.output_dir / self.exp.id
        if self.dir.exists() and any(self.dir.iterdir()) and not (overwrite or self.exp.overwrite):
            raise FileExistsError(f"{self.dir} exists; pass --overwrite to replace it")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.trials: list[dict] = []
        self.results: dict = {
            "experiment": self.exp.model_dump(mode="json"),
            "synthetic": cfg.data.source == "synthetic",
        }

    # ---------------------------------------------------------------- helpers
    def _log_trials(self, phase: str, jobs: list[dict], outs: list[dict], extra: list[dict]) -> None:
        for j, o, e in zip(jobs, outs, extra):
            m = o["metrics"]
            self.trials.append(
                {
                    "phase": phase,
                    "strategy": j["strategy"],
                    "session_id": o["segment"].session_id,
                    "role": o["segment"].role,
                    **e,
                    "net_pnl": m["net_pnl"],
                    "net_pnl_mid": m["net_pnl_mid"],
                    "n_fills": m["n_fills"],
                    "mean_abs_inventory": m["mean_abs_inventory"],
                    "max_drawdown": m["max_drawdown"],
                    "fees_net": m["fees_net"],
                }
            )

    def _strategy_cfg(self, base: ProjectConfig, strategy: str, params: dict) -> ProjectConfig:
        return with_overrides(base, {f"strategies.{strategy}.{k}": v for k, v in params.items()})

    def _apply_sensitivity(self, base: ProjectConfig, key: str, value) -> tuple[ProjectConfig, str]:
        """Return (config, signal_key) for one sensitivity point."""
        if key == "latency_us":
            return with_overrides(
                base,
                {
                    "execution.latency.market_data_us": value,
                    "execution.latency.order_entry_us": value,
                    "execution.latency.report_us": value,
                },
            ), "default"
        if key.startswith("strategies.*."):
            leaf = key.split(".", 2)[2]
            targets = [
                s
                for s in ("fixed_spread", "inventory_aware", "adaptive", "regime")
                if leaf in type(getattr(base.strategies, s)).model_fields
            ]
            if not targets:
                raise ValueError(f"no strategy has a parameter named {leaf!r}")
            return with_overrides(base, {f"strategies.{s}.{leaf}": value for s in targets}), "default"
        if key == "features.vol_window_s":
            sk = f"vol:{value}"
            if (sk, next(iter(self.ctx.bundles))) not in self.ctx.signals:
                fcfg = self.cfg.features.model_copy(update={"vol_window_s": float(value)})
                for sid, b in self.ctx.bundles.items():
                    vol = build_features(b.session, fcfg).df["vol_ticks_sqrt_s"].to_numpy()
                    default = self.ctx.signals[("default", sid)]
                    self.ctx.signals[(sk, sid)] = self.ctx.base_signals(
                        sid, default.alpha, default.alpha_horizon, vol
                    )
            return base, sk
        if key == "models.labels.horizon_ms":
            sk = f"horizon:{value}"
            if (sk, next(iter(self.ctx.bundles))) not in self.ctx.signals:
                lcfg = self.cfg.models.labels.model_copy(update={"horizon_ms": float(value)})
                saved = {sid: b.labels for sid, b in self.ctx.bundles.items()}
                for b in self.ctx.bundles.values():
                    b.labels = build_labels(b.features.df, lcfg)
                tm = train_models(self.ctx, candidates=[self.selected_model_name], label_cfg=lcfg)
                for sid, b in self.ctx.bundles.items():
                    b.labels = saved[sid]
                alphas = alpha_arrays(self.ctx, tm["models"][self.selected_model_name])
                for sid in self.ctx.bundles:
                    self.ctx.signals[(sk, sid)] = self.ctx.base_signals(sid, alphas[sid], f"{value}ms")
            return base, sk
        return with_overrides(base, {key: value}), "default"

    # ---------------------------------------------------------------- phases
    def run(self) -> dict:
        t_start = time.time()
        cfg = self.cfg
        save_snapshot(cfg, self.dir / "config_snapshot.yaml")
        log.info("loading data and building features")
        self.ctx = ctx = build_context(cfg)
        self.results["data"] = {
            "dataset_id": cfg.data.dataset_id,
            "source": cfg.data.source,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "rows": len(s),
                    "observable_states": int(s.last_in_group.sum()),
                    "mid_drift_ticks": float(s.mid_ticks[s.valid_book][-1] - s.mid_ticks[s.valid_book][0]),
                }
                for s in ctx.dataset
            ],
            "manifests": read_manifests(cfg.data),
            "segments": [
                {
                    "session_id": s.session_id,
                    "role": s.role,
                    "start_s": s.start_ns / NS_PER_S,
                    "end_s": s.end_ns / NS_PER_S,
                    "duration_s": s.duration_s,
                }
                for s in ctx.segments
            ],
        }
        (self.dir / "manifests.json").write_text(
            json.dumps(self.results["data"]["manifests"], indent=2, default=str)
        )

        log.info("training direction models")
        tm = train_models(ctx)
        self.selected_model_name = tm["selected"]
        model = tm["models"][tm["selected"]]
        horizon = (
            f"{cfg.models.labels.horizon_ms:g}ms"
            if cfg.models.labels.kind == "time"
            else f"{cfg.models.labels.horizon_events} states"
        )
        alphas = alpha_arrays(ctx, model)
        for sid in ctx.bundles:
            ctx.signals[("default", sid)] = ctx.base_signals(sid, alphas[sid], horizon)
            ctx.signals[("none", sid)] = ctx.base_signals(sid, None)
        self.results["prediction"] = {k: v for k, v in tm.items() if k not in ("models", "test_rows")}
        self.results["prediction"]["alpha_horizon"] = horizon
        self._prediction_by_regime(tm, model)
        self._save_test_predictions(tm, model)

        log.info("fill-probability model from shadow orders")
        self._fill_model()

        log.info("validation-period parameter selection")
        selected = self._select_parameters()
        final_cfg = cfg
        for strat, params in selected.items():
            final_cfg = self._strategy_cfg(final_cfg, strat, params)
        self.final_cfg = final_cfg
        save_snapshot(final_cfg, self.dir / "config_selected.yaml")

        log.info("test-period backtests")
        self._test_backtests()

        if self.exp.sensitivity:
            log.info("sensitivity analysis")
            self._sensitivity()
        if self.exp.ablations:
            log.info("ablations")
            self._ablations()

        pd.DataFrame(self.trials).to_csv(self.dir / "trials.csv", index=False)
        self.results["n_backtests"] = len(self.trials)
        self.results["runtime_s"] = round(time.time() - t_start, 1)
        (self.dir / "results.json").write_text(json.dumps(self.results, indent=2, default=_json_default))
        append_entry(
            self.exp.output_dir / "registry.jsonl",
            {
                "experiment_id": self.exp.id,
                "kind": self.exp.kind,
                "dataset_id": cfg.data.dataset_id,
                "synthetic": cfg.data.source == "synthetic",
                "config_fingerprint": cfg.fingerprint(),
                "n_backtests": len(self.trials),
                "selected_model": self.selected_model_name,
                "test_net_pnl": {
                    k: v["net_pnl_total"] for k, v in self.results["test"]["by_strategy"].items()
                },
            },
        )
        log.info(
            "experiment %s finished in %.0fs (%d backtests)",
            self.exp.id,
            time.time() - t_start,
            len(self.trials),
        )
        return self.results

    def _prediction_by_regime(self, tm: dict, model: DirectionModel) -> None:
        regimes, thr = regime_labels(self.ctx)
        self.regimes, self.regime_thresholds = regimes, thr
        rows = tm["test_rows"]
        if rows["y"].size == 0:
            return
        # rows are grouped by session in segment order, so per-session blocks concatenate in order
        p = np.vstack(
            [
                model.predict_proba(
                    self.ctx.bundles[sid]
                    .features.df.iloc[rows["k"][rows["sid"] == sid]][model.feature_columns]
                    .to_numpy()
                )
                for sid in dict.fromkeys(rows["sid"].tolist())
            ]
        )
        out = {}
        for dim in ("volatility", "spread"):
            labels = np.array([regimes[sid][dim].iat[k] for sid, k in zip(rows["sid"], rows["k"])])
            out[dim] = {}
            for lab in np.unique(labels):
                m = labels == lab
                if m.sum() >= 50:
                    met = classification_metrics(rows["y"][m], p[m], model.classes)
                    out[dim][str(lab)] = {k: met[k] for k in ("n", "log_loss", "accuracy", "auc_up_vs_rest")}
        by_session = {}
        for sid in np.unique(rows["sid"]):
            m = rows["sid"] == sid
            met = classification_metrics(rows["y"][m], p[m], model.classes)
            by_session[str(sid)] = {k: met[k] for k in ("n", "log_loss", "accuracy", "auc_up_vs_rest")}
        self.results["prediction"]["selected_by_regime"] = out
        self.results["prediction"]["selected_by_session"] = by_session
        self.results["prediction"]["regime_thresholds"] = thr

    def _save_test_predictions(self, tm: dict, model: DirectionModel) -> None:
        rows = tm["test_rows"]
        if rows["y"].size == 0:
            return
        recs = []
        for sid in np.unique(rows["sid"]):
            m = rows["sid"] == sid
            df = self.ctx.bundles[sid].features.df.iloc[rows["k"][m]]
            p = model.predict_proba(df[model.feature_columns].to_numpy())
            recs.append(
                pd.DataFrame(
                    {
                        "session_id": sid,
                        "ts_ns": rows["ts"][m],
                        "y": rows["y"][m],
                        "dmid_ticks": rows["dmid"][m],
                        "alpha_ticks": model.alpha_ticks(df[model.feature_columns].to_numpy()),
                        **{f"p_{int(c)}": p[:, j] for j, c in enumerate(model.classes)},
                    }
                )
            )
        pd.concat(recs).to_csv(self.dir / "test_predictions.csv", index=False)

    def _fill_model(self) -> None:
        cfg = self.cfg
        fcfg = cfg.models.fill_model
        rules = FillRules.from_preset(cfg.execution.fill_preset, cfg.execution.conservative_extra_queue_frac)
        parts = {"train": [], "test": []}
        for role in parts:
            segs = by_role(self.ctx.segments, role)
            per = max(fcfg.n_shadow_orders // (4 * max(len(segs), 1)), 25)
            for i, seg in enumerate(segs):
                b = self.ctx.bundles[seg.session_id]
                parts[role].append(
                    simulate_shadow_orders(
                        b.session,
                        b.features,
                        seg.start_ns,
                        seg.end_ns,
                        rules,
                        fcfg.horizon_s,
                        per,
                        fcfg.seed + i + (1000 if role == "test" else 0),
                    )
                )
        train = pd.concat(parts["train"], ignore_index=True) if parts["train"] else pd.DataFrame()
        test = pd.concat(parts["test"], ignore_index=True) if parts["test"] else pd.DataFrame()
        model = fit_fill_model(train, fcfg.horizon_s)
        res = {
            "horizon_s": fcfg.horizon_s,
            "fill_preset": cfg.execution.fill_preset,
            "n_train": int(len(train)),
            "test": evaluate_fill_model(model, test),
            "intensity_train": fill_intensities(train) if not train.empty else {},
        }
        grid = np.linspace(0, fcfg.horizon_s, 21)
        if not test.empty:
            res["km_grid_s"] = grid.tolist()
            res["km_survival_by_offset"] = {
                int(o): kaplan_meier(
                    g["time_to_fill_s"].to_numpy(), (~g["censored"]).to_numpy(), grid
                ).tolist()
                for o, g in test.groupby("offset_ticks")
            }
            test.assign(p_fill=model.predict(test) if model else np.nan).to_csv(
                self.dir / "shadow_orders_test.csv", index=False
            )
        self.fill_results = res
        self.results["fill_model"] = res

    def _select_parameters(self) -> dict[str, dict]:
        cfg, exp = self.cfg, self.exp
        val = by_role(self.ctx.segments, "val")
        selected: dict[str, dict] = {}
        intensity = self.fill_results.get("intensity_train", {})
        order = [s for s in ("fixed_spread", "inventory_aware", "adaptive", "regime") if s in exp.strategies]
        summary = {}
        base_cfg = cfg
        for strat in order:
            grid = exp.validation_grid.get(strat, {})
            inherit = {}
            if strat in ("adaptive", "regime") and "inventory_aware" in selected:
                inv = base_cfg.strategies.inventory_aware.model_dump()
                inherit = {f: inv[f] for f in SHARED_INVENTORY_FIELDS if f not in grid}
            params_list = (
                [dict(zip(grid, combo)) for combo in itertools.product(*grid.values())] if grid else [{}]
            )
            scored = []
            for params in params_list:
                p = {**inherit, **params}
                scfg = self._strategy_cfg(base_cfg, strat, p)
                sp = getattr(scfg.strategies, strat)
                if getattr(sp, "estimate_fill_intensity", False) and "k_per_tick" in intensity:
                    scfg = self._strategy_cfg(
                        scfg, strat, {"k": intensity["k_per_tick"], "A": intensity["A_per_s"]}
                    )
                    p = {**p, "k": intensity["k_per_tick"], "A": intensity["A_per_s"]}
                jobs = [
                    {"cfg": scfg, "segment": seg, "strategy": strat, "signal_key": "default"} for seg in val
                ]
                outs = run_jobs(self.ctx, jobs, exp.n_jobs)
                self._log_trials("validation", jobs, outs, [{"params": json.dumps(params)}] * len(jobs))
                total = float(sum(o["metrics"]["net_pnl"] for o in outs))
                scored.append((total, params, p))
            best = max(scored, key=lambda x: x[0])
            selected[strat] = best[2]
            base_cfg = self._strategy_cfg(base_cfg, strat, best[2])
            summary[strat] = {
                "grid": {k: list(v) for k, v in grid.items()},
                "n_candidates": len(params_list),
                "selected": best[1],
                "inherited_from_inventory_aware": inherit,
                "val_net_pnl_by_candidate": [{"params": s[1], "val_net_pnl": s[0]} for s in scored],
            }
        self.results["selection"] = summary
        return selected

    def _test_backtests(self) -> None:
        cfg, exp = self.final_cfg, self.exp
        test = by_role(self.ctx.segments, "test")
        jobs = [
            {"cfg": cfg, "segment": seg, "strategy": strat, "signal_key": "default", "keep": True}
            for strat in exp.strategies
            for seg in test
        ]
        outs = run_jobs(self.ctx, jobs, exp.n_jobs)
        self._log_trials("test", jobs, outs, [{"params": "selected"}] * len(jobs))
        bt = cfg.backtest
        by_strat: dict[str, dict] = {}
        self.test_results: dict[str, list[BacktestResult]] = {}
        self.test_intervals: dict[str, list[np.ndarray]] = {}
        art = self.dir / "backtests"
        art.mkdir(exist_ok=True)
        for strat in exp.strategies:
            mine = [o for j, o in zip(jobs, outs) if j["strategy"] == strat]
            self.test_results[strat] = [o["result"] for o in mine]
            self.test_intervals[strat] = [o["interval_pnl"] for o in mine]
            fills = []
            for o in mine:
                r: BacktestResult = o["result"]
                b = self.ctx.bundles[r.session_id]
                obs = b.features.df
                okm = obs["valid_book"].to_numpy()
                mk = markout_table(
                    r.fills,
                    obs["ts_ns"].to_numpy()[okm],
                    obs["mid_ff_ticks"].to_numpy()[okm],
                    b.session.tick_size,
                    bt.markout_horizons_ms,
                )
                if not mk.empty:
                    fills.append(mk.assign(session_id=r.session_id))
                r.equity.assign(session_id=r.session_id).to_csv(
                    art / f"equity_{strat}_{r.session_id}.csv", index=False
                )
                r.decisions.iloc[:: max(len(r.decisions) // 20000, 1)].to_csv(
                    art / f"decisions_{strat}_{r.session_id}.csv", index=False
                )
            mk_all = pd.concat(fills, ignore_index=True) if fills else pd.DataFrame()
            if not mk_all.empty:
                mk_all.to_csv(art / f"fills_{strat}.csv", index=False)
            tick_d = self.ctx.dataset.sessions[0].tick_size / self.ctx.dataset.sessions[0].price_scale
            per_seg = [o["metrics"] for o in mine]
            boot = block_bootstrap_mean(
                self.test_intervals[strat], bt.bootstrap_blocks_per_draw, bt.n_bootstrap, bt.seed
            )
            n_int = sum(x.size for x in self.test_intervals[strat])
            by_strat[strat] = {
                "net_pnl_total": float(sum(m["net_pnl"] for m in per_seg)),
                "net_pnl_mid_total": float(sum(m["net_pnl_mid"] for m in per_seg)),
                "gross_pnl_total": float(sum(m["gross_pnl"] for m in per_seg)),
                "fees_net_total": float(sum(m["fees_net"] for m in per_seg)),
                "n_fills_total": int(sum(m["n_fills"] for m in per_seg)),
                "shares_traded_total": int(sum(m["shares_traded"] for m in per_seg)),
                "mean_abs_inventory": float(np.mean([m["mean_abs_inventory"] for m in per_seg])),
                "max_abs_inventory": float(max(m["max_abs_inventory"] for m in per_seg)),
                "max_drawdown_worst": float(max(m["max_drawdown"] for m in per_seg)),
                "per_segment": per_seg,
                "interval_pnl_bootstrap": boot,
                "interval_pnl_total_ci": (
                    [boot["ci_low"] * n_int, boot["ci_high"] * n_int]
                    if boot.get("ci_low") is not None
                    else None
                ),
                "markouts": markout_summary(
                    mk_all,
                    bt.markout_horizons_ms,
                    tick_d,
                    cfg.execution.fees.maker_per_share,
                    cfg.execution.fees.taker_per_share,
                ),
                "by_regime": self._pnl_by_regime(self.test_results[strat]),
                "markouts_by_regime": self._markouts_by_regime(mk_all, bt.markout_horizons_ms[-1]),
            }
        pairs = [
            ("inventory_aware", "fixed_spread"),
            ("adaptive", "inventory_aware"),
            ("adaptive", "fixed_spread"),
            ("regime", "adaptive"),
        ]
        comparisons, risk_adjusted = {}, {}
        for a, b in pairs:
            if a in self.test_intervals and b in self.test_intervals:
                comparisons[f"{a} - {b}"] = paired_difference(
                    self.test_intervals[a],
                    self.test_intervals[b],
                    bt.bootstrap_blocks_per_draw,
                    bt.n_bootstrap,
                    bt.seed,
                )
                risk_adjusted[f"{a} - {b}"] = paired_ratio_difference(
                    self.test_intervals[a],
                    self.test_intervals[b],
                    bt.bootstrap_blocks_per_draw,
                    bt.n_bootstrap,
                    bt.seed,
                )
        self.results["test"] = {
            "by_strategy": by_strat,
            "paired_comparisons": comparisons,
            "risk_adjusted_comparisons": risk_adjusted,
            "interval_s": bt.bootstrap_block_s,
            "assumptions": self.test_results[exp.strategies[0]][0].assumptions,
        }

    def _markouts_by_regime(self, mk: pd.DataFrame, horizon_ms: float) -> dict:
        """Maker effective half-spread and realized spread by the causal regime prevailing at each fill."""
        if mk.empty:
            return {}
        mk = mk[mk["liquidity"] == "maker"]
        col = f"realized_spread_{horizon_ms:g}ms"
        out: dict = {}
        for sid, g in mk.groupby("session_id"):
            reg = self.regimes[sid]
            j = state_index_before(reg["ts_ns"].to_numpy(), g["ts_ns"].to_numpy())
            for dim in ("volatility", "spread", "depth", "flow"):
                lab = reg[dim].to_numpy()[np.maximum(j, 0)]
                for value in np.unique(lab):
                    m = lab == value
                    e = out.setdefault(dim, {}).setdefault(
                        str(value), {"n": 0, "eff_sum": 0.0, "rs_sum": 0.0, "rs_n": 0}
                    )
                    rs = g[col].to_numpy()[m]
                    e["n"] += int(m.sum())
                    e["eff_sum"] += float(g["eff_half_spread_ticks"].to_numpy()[m].sum())
                    e["rs_sum"] += float(np.nansum(rs))
                    e["rs_n"] += int(np.isfinite(rs).sum())
        for d in out.values():
            for k, e in d.items():
                d[k] = {
                    "n": e["n"],
                    "eff_half_spread_ticks": e["eff_sum"] / max(e["n"], 1),
                    f"realized_spread_{horizon_ms:g}ms_ticks": e["rs_sum"] / e["rs_n"] if e["rs_n"] else None,
                }
        return out

    def _pnl_by_regime(self, results: list[BacktestResult]) -> dict:
        out: dict = {}
        for r in results:
            reg = self.regimes[r.session_id]
            eq = r.equity
            if len(eq) < 2:
                continue
            t = eq["ts_ns"].to_numpy()[:-1]
            de = np.diff(eq["equity_mid"].ffill().fillna(0.0).to_numpy())
            j = np.searchsorted(reg["ts_ns"].to_numpy(), t, side="right") - 1
            ok = j >= 0
            fl_rows = (
                state_index_before(reg["ts_ns"].to_numpy(), r.fills["ts_ns"].to_numpy())
                if not r.fills.empty
                else np.array([], int)
            )
            for dim in ("volatility", "spread", "depth", "flow", "time"):
                lab = reg[dim].to_numpy()[j[ok]]
                d = out.setdefault(dim, {})
                for value in np.unique(lab):
                    e = d.setdefault(str(value), {"pnl_mid": 0.0, "seconds": 0, "fills": 0})
                    e["pnl_mid"] += float(de[ok][lab == value].sum())
                    e["seconds"] += int((lab == value).sum())
                if fl_rows.size:
                    fl = reg[dim].to_numpy()[np.maximum(fl_rows, 0)]
                    for value in np.unique(fl):
                        d.setdefault(str(value), {"pnl_mid": 0.0, "seconds": 0, "fills": 0})["fills"] += int(
                            (fl == value).sum()
                        )
        return out

    def _sensitivity(self) -> None:
        exp = self.exp
        test = by_role(self.ctx.segments, "test")
        out: dict = {}
        for key, values in exp.sensitivity.items():
            out[key] = {}
            for v in values:
                cfg_v, sk = self._apply_sensitivity(self.final_cfg, key, v)
                strategies = [s for s in exp.sensitivity_strategies if s in exp.strategies]
                jobs = [
                    {"cfg": cfg_v, "segment": seg, "strategy": s, "signal_key": sk}
                    for s in strategies
                    for seg in test
                ]
                outs = run_jobs(self.ctx, jobs, exp.n_jobs)
                self._log_trials("sensitivity", jobs, outs, [{"params": json.dumps({key: v})}] * len(jobs))
                out[key][str(v)] = {
                    s: {
                        "net_pnl": float(
                            sum(o["metrics"]["net_pnl"] for j, o in zip(jobs, outs) if j["strategy"] == s)
                        ),
                        "n_fills": int(
                            sum(o["metrics"]["n_fills"] for j, o in zip(jobs, outs) if j["strategy"] == s)
                        ),
                        "mean_abs_inventory": float(
                            np.mean(
                                [
                                    o["metrics"]["mean_abs_inventory"]
                                    for j, o in zip(jobs, outs)
                                    if j["strategy"] == s
                                ]
                            )
                        ),
                    }
                    for s in strategies
                }
        self.results["sensitivity"] = out

    def _train_mean_vol(self) -> float:
        vals = []
        for seg in by_role(self.ctx.segments, "train"):
            df = self.ctx.bundles[seg.session_id].features.df
            m = seg.contains(df["ts_ns"].to_numpy()) & (df["warm"] & df["valid_book"]).to_numpy()
            vals.append(df.loc[m, "vol_ticks_sqrt_s"].to_numpy())
        return float(np.mean(np.concatenate(vals))) if vals else 0.1

    def _ablations(self) -> None:
        exp = self.exp
        test = by_role(self.ctx.segments, "test")
        out: dict = {}
        adaptive_overrides = {
            "no_alpha": {"use_alpha": False},
            "no_alpha_guard": {"alpha_guard_ticks": None},
            "no_alpha_shift": {"alpha_weight": 0.0},
            "no_inventory_skew": {"use_inventory_skew": False},
            # constant sigma at the training-period average, so the skew keeps its typical size
            "no_vol_adaptation": {
                "use_vol_adaptation": False,
                "static_vol_ticks_per_sqrt_s": self._train_mean_vol(),
            },
        }
        jobs, labels = [], []
        for name in exp.ablations:
            if name in adaptive_overrides:
                c = self._strategy_cfg(self.final_cfg, "adaptive", adaptive_overrides[name])
                jobs += [
                    {"cfg": c, "segment": seg, "strategy": "adaptive", "signal_key": "default"}
                    for seg in test
                ]
                labels += [name] * len(test)
            elif name == "fill_aware_glft":
                fi = self.fill_results.get("intensity_train", {})
                if "k_per_tick" not in fi:
                    continue
                c = self._strategy_cfg(
                    self.final_cfg,
                    "inventory_aware",
                    {
                        "horizon_mode": "glft",
                        "spread_mode": "model",
                        "k": fi["k_per_tick"],
                        "A": fi["A_per_s"],
                    },
                )
                jobs += [
                    {"cfg": c, "segment": seg, "strategy": "inventory_aware", "signal_key": "default"}
                    for seg in test
                ]
                labels += [name] * len(test)
            elif name == "placebo_alpha":
                # the model's own alpha, circularly shifted by half a session: same distribution and
                # autocorrelation, no alignment with the market. It should not beat B; if it did, the
                # engine would be rewarding something other than information.
                for sid in self.ctx.bundles:
                    a = self.ctx.signals[("default", sid)].alpha
                    self.ctx.signals[("placebo", sid)] = self.ctx.base_signals(sid, np.roll(a, a.size // 2))
                jobs += [
                    {"cfg": self.final_cfg, "segment": seg, "strategy": "adaptive", "signal_key": "placebo"}
                    for seg in test
                ]
                labels += [name] * len(test)
            elif name.startswith("drop:"):
                groups = name.split(":", 1)[1].split("+")
                sk = name
                tm = train_models(self.ctx, drop_groups=groups, candidates=[self.selected_model_name])
                alphas = alpha_arrays(self.ctx, tm["models"][self.selected_model_name])
                for sid in self.ctx.bundles:
                    self.ctx.signals[(sk, sid)] = self.ctx.base_signals(sid, alphas[sid])
                cand = tm["candidates"][self.selected_model_name]
                out.setdefault("prediction", {})[name] = {
                    "val_log_loss": cand["val"].get("log_loss"),
                    "test_log_loss": cand.get("test", {}).get("log_loss"),
                    "test_auc_up": cand.get("test", {}).get("auc_up_vs_rest"),
                    "test_ic": cand.get("test_signal", {}).get("ic_spearman"),
                }
                jobs += [
                    {"cfg": self.final_cfg, "segment": seg, "strategy": "adaptive", "signal_key": sk}
                    for seg in test
                ]
                labels += [name] * len(test)
        outs = run_jobs(self.ctx, jobs, exp.n_jobs)
        self._log_trials("ablation", jobs, outs, [{"params": lab} for lab in labels])
        trading: dict = {}
        for lab, o in zip(labels, outs):
            e = trading.setdefault(lab, {"net_pnl": 0.0, "n_fills": 0, "mean_abs_inventory": []})
            e["net_pnl"] += o["metrics"]["net_pnl"]
            e["n_fills"] += o["metrics"]["n_fills"]
            e["mean_abs_inventory"].append(o["metrics"]["mean_abs_inventory"])
        for e in trading.values():
            e["mean_abs_inventory"] = float(np.mean(e["mean_abs_inventory"]))
        out["trading"] = trading
        # the full model, prediction-side, for reference
        sel = self.results["prediction"]["candidates"][self.selected_model_name]
        out.setdefault("prediction", {})["full_model"] = {
            "val_log_loss": sel["val"].get("log_loss"),
            "test_log_loss": sel.get("test", {}).get("log_loss"),
            "test_auc_up": sel.get("test", {}).get("auc_up_vs_rest"),
            "test_ic": sel.get("test_signal", {}).get("ic_spearman"),
        }
        self.results["ablations"] = out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, Segment):
        return o.__dict__
    return str(o)


def run_experiment(cfg: ProjectConfig, overwrite: bool = False) -> Path:
    runner = ExperimentRunner(copy.deepcopy(cfg), overwrite=overwrite)
    runner.run()
    return runner.dir
