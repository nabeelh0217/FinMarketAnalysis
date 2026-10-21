"""Programmatic figures for the research report (static PNG, light surface).

Colour follows the entity: each strategy keeps one categorical slot in every figure (validated palette,
see the dataviz reference: adjacent CVD dE >= 9.1, normal-vision dE >= 22.9). Two slots sit below 3:1
contrast on the light surface, so every multi-series figure carries a legend and every number is also in
the report's tables. One y-axis per panel, recessive hairline grid, text in neutral inks.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STRATEGY_COLOR = {
    "fixed_spread": SERIES[0],
    "inventory_aware": SERIES[1],
    "adaptive": SERIES[2],
    "regime": SERIES[3],
}
REGIME_ORDER = [
    "low",
    "mid",
    "high",
    "1 tick",
    ">1 tick",
    "balanced",
    "directional",
    "thin",
    "deep",
    "early",
    "late",
]
STRATEGY_LABEL = {
    "fixed_spread": "A fixed spread",
    "inventory_aware": "B inventory-aware",
    "adaptive": "C alpha-aware",
    "regime": "D regime rules",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "figure.dpi": 130,
        }
    )


def _save(fig, path: Path, synthetic: bool) -> str:
    if synthetic:
        fig.text(0.995, 0.005, "SYNTHETIC DATA", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _strategies(results: dict) -> list[str]:
    return list(results["test"]["by_strategy"])


# ------------------------------------------------------------------------------------- market figures
def market_figures(
    session, features: pd.DataFrame, out: Path, synthetic: bool, start_ns: int, minutes: float = 2.0
) -> dict:
    files = {}
    end_ns = start_ns + int(minutes * 60e9)
    df = features[(features["ts_ns"] >= start_ns) & (features["ts_ns"] < end_ns) & features["valid_book"]]
    if df.empty:
        return files
    t = (df["ts_ns"].to_numpy() - start_ns) / 1e9
    tick_d = session.tick_size / session.price_scale
    rows = df["row"].to_numpy()
    bid = session.bid_px[rows, 0] / session.price_scale
    ask = session.ask_px[rows, 0] / session.price_scale
    clock = pd.to_datetime(start_ns, unit="ns").strftime("%H:%M:%S")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.step(t, ask, where="post", color=SERIES[7], label="best ask")
    ax.step(t, bid, where="post", color=SERIES[0], label="best bid")
    ax.set(
        title=f"Best bid and ask, {session.session_id}, {minutes:g} min from {clock}",
        xlabel="seconds into window",
        ylabel="price ($)",
    )
    ax.legend(loc="upper left", ncols=2)
    files["book"] = _save(fig, out / "01_best_bid_ask.png", synthetic)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    mid = df["mid_ticks"].to_numpy() * tick_d
    micro = mid + df["micro_minus_mid"].to_numpy() * tick_d
    ax.step(t, mid, where="post", color=SERIES[0], label="mid-price")
    ax.step(t, micro, where="post", color=SERIES[1], label="microprice", linewidth=1.0)
    ax.set(title="Mid-price and microprice", xlabel="seconds into window", ylabel="price ($)")
    ax.legend(loc="upper left", ncols=2)
    files["micro"] = _save(fig, out / "02_mid_microprice.png", synthetic)

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.step(t, df["imb_l1"].to_numpy(), where="post", color=SERIES[0], label="top of book")
    if "imb_l5" in df:
        ax.step(t, df["imb_l5"].to_numpy(), where="post", color=SERIES[1], label="5 levels", linewidth=1.0)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set(
        title="Order-book imbalance (Qb - Qa) / (Qb + Qa)",
        xlabel="seconds into window",
        ylabel="imbalance",
        ylim=(-1.05, 1.05),
    )
    ax.legend(loc="upper left", ncols=2)
    files["imbalance"] = _save(fig, out / "03_imbalance.png", synthetic)
    return files


# --------------------------------------------------------------------------------- prediction figures
def prediction_figures(results: dict, exp_dir: Path, out: Path, synthetic: bool) -> dict:
    files = {}
    pred_path = exp_dir / "test_predictions.csv"
    if pred_path.exists():
        tp = pd.read_csv(pred_path)
        tp["decile"] = pd.qcut(tp["alpha_ticks"].rank(method="first"), 10, labels=False) + 1
        g = tp.groupby("decile").agg(
            alpha=("alpha_ticks", "mean"),
            realized=("dmid_ticks", "mean"),
            se=("dmid_ticks", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        )
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.bar(g.index, g["realized"], width=0.6, color=SERIES[0], label="realized mean mid change")
        ax.grid(axis="x", visible=False)
        ax.errorbar(
            g.index, g["realized"], yerr=1.96 * g["se"], fmt="none", ecolor=INK2, elinewidth=1, capsize=2
        )
        ax.plot(
            g.index, g["alpha"], color=SERIES[1], marker="o", markersize=4, label="predicted alpha (mean)"
        )
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set(
            title=f"Predicted alpha vs realized mid change (test, {results['prediction']['alpha_horizon']})",
            xlabel="alpha decile (1 = most bearish)",
            ylabel="mid change (ticks)",
        )
        ax.set_xticks(range(1, 11))
        ax.legend(loc="upper left")
        files["alpha_deciles"] = _save(fig, out / "04_alpha_vs_realized.png", synthetic)

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, label="perfect calibration")
    for i, (name, c) in enumerate(results["prediction"]["candidates"].items()):
        rel = c.get("test", {}).get("reliability_up")
        if rel:
            ax.plot(rel["predicted"], rel["observed"], marker="o", markersize=4, color=SERIES[i], label=name)
    hi = max(
        [
            max(
                c.get("test", {}).get("reliability_up", {}).get("predicted", [0])
                + c.get("test", {}).get("reliability_up", {}).get("observed", [0])
            )
            for c in results["prediction"]["candidates"].values()
        ]
        + [0.1]
    )
    lim = min(1.0, hi * 1.1)
    ax.set(
        xlim=(0, lim),
        ylim=(0, lim),
        title="Reliability of P(up), test period",
        xlabel="predicted probability",
        ylabel="observed frequency",
    )
    ax.legend(loc="upper left")
    files["reliability"] = _save(fig, out / "05_reliability.png", synthetic)
    return files


# ------------------------------------------------------------------------------------ trading figures
def _equity_frames(exp_dir: Path, strat: str) -> list[pd.DataFrame]:
    return [pd.read_csv(p) for p in sorted((exp_dir / "backtests").glob(f"equity_{strat}_*.csv"))]


def trading_figures(results: dict, exp_dir: Path, out: Path, synthetic: bool) -> dict:
    files = {}
    strats = _strategies(results)
    # 6 / 7: equity and drawdown over concatenated test segments
    fig_e, ax_e = plt.subplots(figsize=(8, 3.6))
    fig_d, ax_d = plt.subplots(figsize=(8, 3.0))
    boundaries = []
    for s in strats:
        offset_t, offset_e, xs, ys = 0.0, 0.0, [], []
        bnd = []
        for eq in _equity_frames(exp_dir, s):
            t = (eq["ts_ns"] - eq["ts_ns"].iloc[0]) / 60e9
            e = eq["equity_mid"].ffill().fillna(0.0).to_numpy()
            xs.append(t.to_numpy() + offset_t)
            ys.append(e + offset_e)
            offset_t += float(t.iloc[-1])
            offset_e += float(e[-1])
            bnd.append(offset_t)
        if not xs:
            continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        ax_e.plot(x, y, color=STRATEGY_COLOR.get(s, INK), label=STRATEGY_LABEL.get(s, s))
        ax_d.plot(
            x,
            -(np.maximum.accumulate(y) - y),
            color=STRATEGY_COLOR.get(s, INK),
            label=STRATEGY_LABEL.get(s, s),
        )
        boundaries = bnd[:-1]
    for ax in (ax_e, ax_d):
        for b in boundaries:
            ax.axvline(b, color=AXIS, linewidth=0.8)
    ax_e.set(
        title="Cumulative mark-to-mid equity over held-out test sessions",
        xlabel="test time (minutes; vertical lines = session boundaries)",
        ylabel="P&L ($)",
    )
    ax_e.legend(loc="upper left", ncols=2)
    ax_d.set(title="Drawdown from running peak", xlabel="test time (minutes)", ylabel="drawdown ($)")
    ax_d.legend(loc="lower left", ncols=2)
    files["equity"] = _save(fig_e, out / "06_equity_curves.png", synthetic)
    files["drawdown"] = _save(fig_d, out / "07_drawdown.png", synthetic)

    # 8: inventory small multiples (first test session)
    fig, axes = plt.subplots(len(strats), 1, figsize=(8, 1.5 * len(strats) + 0.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, strats):
        frames = _equity_frames(exp_dir, s)
        if frames:
            eq = frames[0]
            t = (eq["ts_ns"] - eq["ts_ns"].iloc[0]) / 60e9
            ax.fill_between(
                t, eq["position"], step="post", color=STRATEGY_COLOR.get(s, INK), alpha=0.12, linewidth=0
            )
            ax.step(t, eq["position"], where="post", color=STRATEGY_COLOR.get(s, INK), linewidth=1.2)
            ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_title(STRATEGY_LABEL.get(s, s), loc="left", fontsize=9.5)
        ax.set_ylabel("shares")
    axes[-1].set_xlabel("minutes into first test session")
    fig.suptitle("Inventory over time", fontsize=11, fontweight="bold", x=0.01, ha="left")
    files["inventory"] = _save(fig, out / "08_inventory.png", synthetic)

    # 9: order and fill statistics
    keys = [
        ("orders_sent", "orders sent"),
        ("n_fills", "fills"),
        ("cancels_sent", "cancels"),
        ("exchange_rejects", "rejects"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.8 / len(strats)
    for i, s in enumerate(strats):
        segs = results["test"]["by_strategy"][s]["per_segment"]
        vals = [sum(m[k] for m in segs) for k, _ in keys]
        xs = np.arange(len(keys)) + (i - (len(strats) - 1) / 2) * width
        ax.bar(xs, vals, width=width * 0.9, color=STRATEGY_COLOR.get(s, INK), label=STRATEGY_LABEL.get(s, s))
    ax.set_xticks(range(len(keys)), [lbl for _, lbl in keys])
    ax.grid(axis="x", visible=False)
    ax.set(title="Order activity on test sessions", ylabel="count")
    ax.legend(ncols=2)
    files["orders"] = _save(fig, out / "09_order_stats.png", synthetic)

    # 10: markouts by horizon
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for s in strats:
        mk = results["test"]["by_strategy"][s]["markouts"].get("by_liquidity", {}).get("maker")
        if not mk:
            continue
        hs = sorted(
            int(float(k.split("_")[2][:-2]))
            for k in mk
            if k.startswith("realized_spread_") and mk[k] is not None
        )
        vals = [mk[f"realized_spread_{h:g}ms_ticks"] for h in hs]
        ax.plot(
            [h / 1000 for h in hs],
            vals,
            marker="o",
            markersize=4,
            color=STRATEGY_COLOR.get(s, INK),
            label=STRATEGY_LABEL.get(s, s),
        )
        ax.scatter([0.02], [mk["eff_half_spread_ticks"]], color=STRATEGY_COLOR.get(s, INK), s=18, marker="D")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xscale("log")
    ax.set(
        title="Maker realized spread by horizon (diamond = effective half-spread at fill)",
        xlabel="seconds after fill (log scale)",
        ylabel="ticks per share",
    )
    ax.legend(loc="lower left", ncols=2)
    files["markouts"] = _save(fig, out / "10_markouts.png", synthetic)

    # 11: net P&L with bootstrap CI
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, s in enumerate(strats):
        d = results["test"]["by_strategy"][s]
        ax.bar(i, d["net_pnl_total"], width=0.55, color=STRATEGY_COLOR.get(s, INK))
        ci = d.get("interval_pnl_total_ci")
        if ci:
            ax.errorbar(
                i,
                d["net_pnl_total"],
                yerr=[[d["net_pnl_total"] - ci[0]], [ci[1] - d["net_pnl_total"]]],
                fmt="none",
                ecolor=INK2,
                elinewidth=1,
                capsize=3,
            )
        ax.annotate(
            f"${d['net_pnl_total']:,.0f}",
            (i, d["net_pnl_total"]),
            textcoords="offset points",
            xytext=(10, 4),
            fontsize=8.5,
            color=INK2,
        )
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks(range(len(strats)), [STRATEGY_LABEL.get(s, s) for s in strats])
    ax.grid(axis="x", visible=False)
    ax.set(
        title="Net P&L on test sessions (liquidation-marked, after fees; 95% block-bootstrap CI)",
        ylabel="net P&L ($)",
    )
    files["net_pnl"] = _save(fig, out / "11_net_pnl.png", synthetic)

    # 12: P&L by regime
    dims = ["volatility", "spread", "flow", "depth"]
    fig, axes = plt.subplots(1, len(dims), figsize=(11, 3.2), sharey=True)
    for ax, dim in zip(axes, dims):
        present = {lv for s in strats for lv in results["test"]["by_strategy"][s]["by_regime"].get(dim, {})}
        levels = [lv for lv in REGIME_ORDER if lv in present] + sorted(present - set(REGIME_ORDER))
        ax.grid(axis="x", visible=False)
        width = 0.8 / len(strats)
        for i, s in enumerate(strats):
            reg = results["test"]["by_strategy"][s]["by_regime"].get(dim, {})
            vals = [reg.get(lv, {}).get("pnl_mid", 0.0) for lv in levels]
            xs = np.arange(len(levels)) + (i - (len(strats) - 1) / 2) * width
            ax.bar(
                xs, vals, width=width * 0.9, color=STRATEGY_COLOR.get(s, INK), label=STRATEGY_LABEL.get(s, s)
            )
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_xticks(range(len(levels)), levels)
        ax.set_title(dim, loc="left", fontsize=9.5)
    axes[0].set_ylabel("mark-to-mid P&L ($)")
    axes[-1].legend(fontsize=7.5)
    fig.suptitle("P&L by causal market regime (test)", fontsize=11, fontweight="bold", x=0.01, ha="left")
    files["regimes"] = _save(fig, out / "12_pnl_by_regime.png", synthetic)
    return files


def robustness_figures(results: dict, out: Path, synthetic: bool) -> dict:
    files = {}
    sens = results.get("sensitivity", {})

    def line_panel(ax, key, title, xlabel, log=False):
        pts = sens[key]
        xs_raw = list(pts)
        strats = list(next(iter(pts.values())))
        numeric = all(_is_number(x) for x in xs_raw)
        xs = [float(x) for x in xs_raw] if numeric else list(range(len(xs_raw)))
        for s in strats:
            ys = [pts[x][s]["net_pnl"] for x in xs_raw]
            ax.plot(
                xs if not log else [max(v, 1e-1) for v in xs],
                ys,
                marker="o",
                markersize=4,
                color=STRATEGY_COLOR.get(s, INK),
                label=STRATEGY_LABEL.get(s, s),
            )
        if not numeric:
            ax.set_xticks(xs, xs_raw)
        if log:
            ax.set_xscale("symlog", linthresh=50)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set(title=title, xlabel=xlabel, ylabel="test net P&L ($)")

    if "latency_us" in sens:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        line_panel(
            ax,
            "latency_us",
            "Sensitivity of net P&L to one-way latency",
            "latency per hop (microseconds, symlog)",
            log=True,
        )
        ax.legend()
        files["latency"] = _save(fig, out / "13_sensitivity_latency.png", synthetic)
    panels = [k for k in ("execution.fees.maker_per_share", "execution.fill_preset") if k in sens]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, k in zip(axes, panels):
            line_panel(
                ax,
                k,
                "Maker fee ($/share, negative = rebate)" if "fees" in k else "Fill assumption preset",
                "maker fee" if "fees" in k else "preset",
            )
        axes[0].legend()
        fig.suptitle(
            "Sensitivity to fees and fill assumptions", fontsize=11, fontweight="bold", x=0.01, ha="left"
        )
        files["fees_fills"] = _save(fig, out / "14_sensitivity_fees_fills.png", synthetic)
    others = [
        k for k in sens if k not in ("latency_us", "execution.fees.maker_per_share", "execution.fill_preset")
    ]
    if others:
        n = len(others)
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(11, 2.9 * rows), sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, k in zip(axes, others):
            line_panel(ax, k, k.replace("strategies.*.", "all strategies: "), "value")
            ax.title.set_fontsize(9)
        for ax in axes[n:]:
            ax.set_visible(False)
        axes[0].legend(fontsize=7.5)
        fig.suptitle(
            "Other sensitivity grids (net P&L on test)", fontsize=11, fontweight="bold", x=0.01, ha="left"
        )
        files["sensitivity_other"] = _save(fig, out / "14b_sensitivity_other.png", synthetic)

    abl = results.get("ablations", {}).get("trading")
    if abl:
        full = results["test"]["by_strategy"].get("adaptive", {}).get("net_pnl_total")
        names = list(abl)
        vals = [abl[n]["net_pnl"] for n in names]
        fig, ax = plt.subplots(figsize=(7.2, 0.38 * len(names) + 1.4))
        ax.barh(range(len(names)), vals, height=0.55, color=SERIES[2])
        if full is not None:
            ax.axvline(full, color=INK2, linewidth=1.2, label=f"full strategy C (${full:,.0f})")
            ax.legend(loc="lower right")
        ax.set_yticks(range(len(names)), names)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
        ax.axvline(0, color=AXIS, linewidth=0.8)
        ax.set(title="Ablations: test net P&L with one component removed", xlabel="net P&L ($)")
        files["ablations"] = _save(fig, out / "15_ablations.png", synthetic)

    fm = results.get("fill_model", {})
    if fm.get("km_survival_by_offset"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        grid = fm["km_grid_s"]
        for i, (off, surv) in enumerate(fm["km_survival_by_offset"].items()):
            axes[0].step(grid, surv, where="post", color=SERIES[i], label=f"{off} tick(s) behind best")
        axes[0].set(
            title="Kaplan-Meier: P(not yet filled), test shadow orders",
            xlabel="seconds after placement",
            ylabel="survival probability",
            ylim=(0, 1.02),
        )
        axes[0].legend()
        rel = fm.get("test", {}).get("reliability")
        if rel:
            axes[1].plot([0, 1], [0, 1], color=AXIS, linewidth=1, label="perfect calibration")
            axes[1].plot(
                rel["predicted"],
                rel["observed"],
                marker="o",
                markersize=4,
                color=SERIES[0],
                label="logistic fill model",
            )
            axes[1].set(
                title=f"Fill-probability calibration ({fm['horizon_s']:g}s horizon, test)",
                xlabel="predicted P(fill)",
                ylabel="observed fill rate",
                xlim=(0, 1),
                ylim=(0, 1),
            )
            axes[1].legend(loc="upper left")
        files["fill_model"] = _save(fig, out / "16_fill_model.png", synthetic)
    return files


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def make_all(exp_dir: Path) -> dict:
    _style()
    results = json.loads((exp_dir / "results.json").read_text())
    out = exp_dir / "figures"
    out.mkdir(exist_ok=True)
    synthetic = bool(results.get("synthetic"))
    files: dict = {}
    files.update(prediction_figures(results, exp_dir, out, synthetic))
    files.update(trading_figures(results, exp_dir, out, synthetic))
    files.update(robustness_figures(results, out, synthetic))
    try:
        import yaml

        from adaptive_market_maker.config import ProjectConfig
        from adaptive_market_maker.data.prepare import load_dataset
        from adaptive_market_maker.features.pipeline import build_features

        cfg = ProjectConfig.model_validate(yaml.safe_load((exp_dir / "config_snapshot.yaml").read_text()))
        seg = next(s for s in results["data"]["segments"] if s["role"] == "test")
        ds = load_dataset(cfg.data, prepare_if_missing=False)
        session = ds.by_id(seg["session_id"])
        ft = build_features(session, cfg.features)
        files.update(market_figures(session, ft.df, out, synthetic, int(seg["start_s"] * 1e9) + int(300e9)))
    except (FileNotFoundError, StopIteration, KeyError) as e:  # data may live on another machine
        files["market_figures_skipped"] = str(e)
    return files
