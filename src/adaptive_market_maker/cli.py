"""`amm` command-line interface."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pickle
import sys
from pathlib import Path

log = logging.getLogger("amm")

MODEL_DIR = Path("reports/models")


def _cfg(args):
    from adaptive_market_maker.config import load_config

    overrides = {}
    if getattr(args, "jobs", None):
        overrides = {"experiment": {"n_jobs": args.jobs}}
    return load_config(args.config, overrides=overrides or None)


# ------------------------------------------------------------------------------------------------ data
def cmd_data_synth(args) -> int:
    from adaptive_market_maker.data.prepare import generate_synthetic_raw

    cfg = _cfg(args)
    if cfg.data.source != "synthetic":
        log.error("data.source must be 'synthetic' for `amm data synth` (got %s)", cfg.data.source)
        return 2
    paths = generate_synthetic_raw(cfg.data, force=args.force)
    print(f"wrote {len(paths)} LOBSTER-format files to {cfg.data.raw_dir} (SYNTHETIC DATA)")
    return 0


def cmd_data_validate(args) -> int:
    from adaptive_market_maker.data.prepare import generate_synthetic_raw, validate_raw

    cfg = _cfg(args)
    if cfg.data.source == "synthetic":
        generate_synthetic_raw(cfg.data)
    try:
        reports = validate_raw(cfg.data)
    except FileNotFoundError as e:
        print(
            f"ERROR: {e}\nDownload the LOBSTER sample files and place them as described in README.md "
            f"(Data acquisition).",
            file=sys.stderr,
        )
        return 2
    out = cfg.data.manifest_dir / cfg.data.dataset_id
    out.mkdir(parents=True, exist_ok=True)
    for r in reports:
        print(r.format())
        (out / f"{r.session_id}.validation.json").write_text(json.dumps(r.to_dict(), indent=2, default=str))
    return 0 if all(r.passed for r in reports) else 1


def cmd_data_prepare(args) -> int:
    from adaptive_market_maker.data.prepare import prepare_dataset

    cfg = _cfg(args)
    try:
        paths = prepare_dataset(cfg.data, force=args.force)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for p in paths:
        print(p)
    return 0


# ------------------------------------------------------------------------------------- features/labels
def cmd_features_build(args) -> int:
    from adaptive_market_maker.data.prepare import load_dataset
    from adaptive_market_maker.features.pipeline import build_features

    cfg = _cfg(args)
    out = Path("data/interim") / cfg.data.dataset_id
    out.mkdir(parents=True, exist_ok=True)
    for s in load_dataset(cfg.data):
        ft = build_features(s, cfg.features)
        path = out / f"features_{s.session_id}.parquet"
        ft.df.to_parquet(path, index=False)
        (out / "feature_groups.json").write_text(json.dumps(ft.groups, indent=2))
        nan_frac = ft.df[ft.feature_columns][ft.df["warm"] & ft.df["valid_book"]].isna().mean()
        print(
            f"{s.session_id}: {len(ft.df):,} observable states, {len(ft.feature_columns)} features -> {path}; "
            f"max NaN fraction after warm-up {nan_frac.max():.4f}"
        )
    return 0


def cmd_labels_build(args) -> int:
    from adaptive_market_maker.data.prepare import load_dataset
    from adaptive_market_maker.features.pipeline import build_features
    from adaptive_market_maker.labels.targets import build_labels

    cfg = _cfg(args)
    out = Path("data/interim") / cfg.data.dataset_id
    out.mkdir(parents=True, exist_ok=True)
    for s in load_dataset(cfg.data):
        lab = build_labels(build_features(s, cfg.features).df, cfg.models.labels)
        path = out / f"labels_{s.session_id}.parquet"
        lab.to_parquet(path, index=False)
        bal = lab["y"].value_counts(normalize=True).sort_index().round(3).to_dict()
        print(
            f"{s.session_id}: {lab['y'].notna().sum():,} labels ({cfg.models.labels.kind} horizon) "
            f"class balance {bal} -> {path}"
        )
    return 0


# ---------------------------------------------------------------------------------------------- models
def cmd_model_train(args) -> int:
    from adaptive_market_maker.experiments.runner import build_context, train_models

    cfg = _cfg(args)
    ctx = build_context(cfg)
    tm = train_models(ctx)
    d = MODEL_DIR / cfg.data.dataset_id
    d.mkdir(parents=True, exist_ok=True)
    with (d / "direction_models.pkl").open("wb") as fh:
        pickle.dump(
            {"models": tm["models"], "selected": tm["selected"], "config_fingerprint": cfg.fingerprint()}, fh
        )
    summary = {
        n: {
            "val_log_loss": c["val"].get("log_loss"),
            "val_accuracy": c["val"].get("accuracy"),
            "temperature": c["temperature"],
        }
        for n, c in tm["candidates"].items()
    }
    (d / "train_summary.json").write_text(
        json.dumps({"selected": tm["selected"], "candidates": summary, "n_train": tm["n_train"]}, indent=2)
    )
    print(json.dumps({"selected": tm["selected"], "validation": summary}, indent=2))
    print(f"saved models to {d} (validation metrics only; run `amm model evaluate` for test metrics)")
    return 0


def cmd_model_evaluate(args) -> int:
    from adaptive_market_maker.backtest.splits import by_role
    from adaptive_market_maker.evaluation.predictive import classification_metrics, signal_quality
    from adaptive_market_maker.experiments.runner import assemble, build_context

    cfg = _cfg(args)
    path = MODEL_DIR / cfg.data.dataset_id / "direction_models.pkl"
    if not path.exists():
        print(f"ERROR: {path} not found; run `amm model train` first", file=sys.stderr)
        return 2
    with path.open("rb") as fh:
        saved = pickle.load(fh)
    ctx = build_context(cfg)
    out = {}
    for name, model in saved["models"].items():
        te = assemble(
            ctx,
            by_role(ctx.segments, "test"),
            model.feature_columns,
            non_overlap=cfg.models.eval_stride == "non_overlapping",
        )
        met = classification_metrics(te["y"], model.predict_proba(te["X"]), model.classes)
        sq = signal_quality(model.alpha_ticks(te["X"]), te["dmid"])
        out[name] = {
            k: met[k] for k in ("n", "accuracy", "balanced_accuracy", "log_loss", "brier", "auc_up_vs_rest")
        }
        out[name]["ic_spearman"] = sq["ic_spearman"]
    (path.parent / "test_evaluation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"selected_on_validation": saved["selected"], "test": out}, indent=2))
    return 0


# -------------------------------------------------------------------------------------------- backtest
def cmd_backtest(args) -> int:
    from adaptive_market_maker.backtest.engine import run_backtest
    from adaptive_market_maker.backtest.splits import by_role
    from adaptive_market_maker.evaluation.metrics import trading_metrics
    from adaptive_market_maker.experiments.runner import alpha_arrays, build_context
    from adaptive_market_maker.strategies.adaptive import make_strategy

    cfg = _cfg(args)
    ctx = build_context(cfg)
    alphas = None
    if args.strategy in ("adaptive", "regime"):
        path = MODEL_DIR / cfg.data.dataset_id / "direction_models.pkl"
        if not path.exists():
            print(
                f"ERROR: {args.strategy} needs a trained model; run `amm model train` first", file=sys.stderr
            )
            return 2
        with path.open("rb") as fh:
            saved = pickle.load(fh)
        alphas = alpha_arrays(ctx, saved["models"][saved["selected"]])
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("reports/backtests") / f"{args.strategy}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for seg in by_role(ctx.segments, args.role):
        sig = ctx.base_signals(seg.session_id, alphas[seg.session_id] if alphas else None)
        res = run_backtest(
            ctx.bundles[seg.session_id].session,
            make_strategy(args.strategy, cfg.strategies),
            cfg,
            sig,
            seg.start_ns,
            seg.end_ns,
        )
        m = trading_metrics(res, cfg.backtest.bootstrap_block_s)
        rows.append({"session_id": seg.session_id, **m})
        res.fills.to_csv(out_dir / f"fills_{seg.session_id}.csv", index=False)
        res.equity.to_csv(out_dir / f"equity_{seg.session_id}.csv", index=False)
        (out_dir / f"assumptions_{seg.session_id}.json").write_text(
            json.dumps(res.assumptions, indent=2, default=str)
        )
        print(
            f"{seg.session_id} [{args.role}] net P&L ${m['net_pnl']:.2f} (mid ${m['net_pnl_mid']:.2f}) "
            f"fills {m['n_fills']} mean|q| {m['mean_abs_inventory']:.0f} maxDD ${m['max_drawdown']:.2f}"
        )
    import pandas as pd

    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    print(f"artifacts in {out_dir}" + ("  (SYNTHETIC DATA)" if cfg.data.source == "synthetic" else ""))
    return 0


# ------------------------------------------------------------------------------------------ experiment
def cmd_experiment(args) -> int:
    from adaptive_market_maker.config import with_overrides
    from adaptive_market_maker.experiments.runner import run_experiment

    cfg = _cfg(args)
    if args.mode == "ablation":
        cfg = with_overrides(cfg, {"experiment.sensitivity": {}})
    elif args.mode == "sensitivity":
        cfg = with_overrides(cfg, {"experiment.ablations": []})
    if not cfg.experiment.ablations and args.mode == "ablation":
        print("ERROR: experiment.ablations is empty in this config", file=sys.stderr)
        return 2
    if not cfg.experiment.sensitivity and args.mode == "sensitivity":
        print("ERROR: experiment.sensitivity is empty in this config", file=sys.stderr)
        return 2
    out = run_experiment(cfg, overwrite=args.overwrite)
    print(f"experiment written to {out}; next: amm report --experiment {cfg.experiment.id}")
    if not args.no_report:
        from adaptive_market_maker.reporting.report import build_report

        print(f"report: {build_report(out)}")
    return 0


def cmd_report(args) -> int:
    from adaptive_market_maker.reporting.report import build_report

    d = Path(args.root) / args.experiment
    if not (d / "results.json").exists():
        print(f"ERROR: {d}/results.json not found", file=sys.stderr)
        return 2
    print(build_report(d))
    return 0


def cmd_registry(args) -> int:
    from adaptive_market_maker.experiments.registry import read_entries

    for e in read_entries(Path(args.root) / "registry.jsonl"):
        print(json.dumps(e))
    return 0


# ------------------------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="amm", description="Adaptive Market Maker research platform")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    def with_config(sp, default="configs/data.yaml"):
        sp.add_argument("--config", default=default, help="YAML config (section defaults come from configs/)")
        return sp

    data = sub.add_parser("data", help="data acquisition, validation, normalization")
    dsub = data.add_subparsers(dest="action", required=True)
    s = with_config(dsub.add_parser("synth", help="generate synthetic LOBSTER-format sessions"))
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_data_synth)
    with_config(dsub.add_parser("validate", help="run data-quality and book-integrity checks")).set_defaults(
        func=cmd_data_validate
    )
    s = with_config(dsub.add_parser("prepare", help="validate and normalize to Parquet with manifests"))
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_data_prepare)

    feats = sub.add_parser("features", help="causal feature generation")
    fsub = feats.add_subparsers(dest="action", required=True)
    with_config(fsub.add_parser("build"), "configs/features.yaml").set_defaults(func=cmd_features_build)

    labels = sub.add_parser("labels", help="future targets for training/evaluation only")
    lsub = labels.add_subparsers(dest="action", required=True)
    with_config(lsub.add_parser("build"), "configs/models.yaml").set_defaults(func=cmd_labels_build)

    model = sub.add_parser("model", help="direction models")
    msub = model.add_subparsers(dest="action", required=True)
    with_config(msub.add_parser("train"), "configs/models.yaml").set_defaults(func=cmd_model_train)
    with_config(msub.add_parser("evaluate"), "configs/models.yaml").set_defaults(func=cmd_model_evaluate)

    bt = with_config(
        sub.add_parser("backtest", help="run one strategy with configured (unselected) parameters"),
        "configs/strategies.yaml",
    )
    bt.add_argument(
        "--strategy", required=True, choices=["fixed_spread", "inventory_aware", "adaptive", "regime"]
    )
    bt.add_argument("--role", default="test", choices=["train", "val", "test"])
    bt.set_defaults(func=cmd_backtest)

    ex = sub.add_parser("experiment", help="full research protocol")
    esub = ex.add_subparsers(dest="mode", required=True)
    for mode, helptext in (
        ("run", "selection + test + sensitivity + ablations"),
        ("ablation", "protocol with ablations only"),
        ("sensitivity", "protocol with sensitivity grid only"),
    ):
        e = with_config(esub.add_parser(mode, help=helptext), "configs/experiments/baseline_comparison.yaml")
        e.add_argument("--overwrite", action="store_true", help="replace an existing experiment directory")
        e.add_argument("--jobs", type=int, default=None, help="parallel backtest processes")
        e.add_argument("--no-report", action="store_true")
        e.set_defaults(func=cmd_experiment)

    rp = sub.add_parser("report", help="build the markdown research report for an experiment")
    rp.add_argument("--experiment", required=True)
    rp.add_argument("--root", default="reports/experiments")
    rp.set_defaults(func=cmd_report)

    rg = sub.add_parser("registry", help="list every recorded experiment run")
    rg.add_argument("--root", default="reports/experiments")
    rg.set_defaults(func=cmd_registry)
    return p


def main(argv: list[str] | None = None) -> int:
    from adaptive_market_maker.logging_config import configure_logging

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
