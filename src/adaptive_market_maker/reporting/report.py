"""Markdown research report generated from an experiment directory.

Every number comes from ``results.json`` / saved artifacts of an executed experiment. Hypothesis verdicts
are mechanical: "supported" only when a 95% block-bootstrap interval excludes zero in the hypothesised
direction, "contradicted" when it excludes zero the other way, otherwise "not established".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adaptive_market_maker.reporting.plots import REGIME_ORDER, STRATEGY_LABEL, make_all


def _f(x, nd=2, money=False, pct=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)) and not money:
        return f"{x:,}"
    if pct:
        return f"{100 * x:.1f}%"
    if money:
        return f"${x:,.{nd}f}" if x >= 0 else f"-${-x:,.{nd}f}"
    return f"{x:,.{nd}f}"


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _verdict(ci_low, ci_high, direction: int = 1) -> str:
    if ci_low is None or ci_high is None:
        return "not established (too little data for an interval)"
    if direction * ci_low > 0 and direction * ci_high > 0:
        return "supported"
    if direction * ci_high < 0 and direction * ci_low < 0:
        return "contradicted"
    return "not established (95% CI includes 0)"


def _name(s: str) -> str:
    return STRATEGY_LABEL.get(s, s)


def _config_path_for(exp_id: str) -> str:
    """The experiment config file declaring this id (falls back to the saved snapshot)."""
    import yaml

    for p in sorted(Path("configs/experiments").glob("*.yaml")):
        try:
            if (yaml.safe_load(p.read_text()) or {}).get("experiment", {}).get("id") == exp_id:
                return str(p)
        except yaml.YAMLError:
            continue
    return f"reports/experiments/{exp_id}/config_snapshot.yaml"


def build_report(exp_dir: Path) -> Path:
    exp_dir = Path(exp_dir)
    figs = make_all(exp_dir)
    R = json.loads((exp_dir / "results.json").read_text())
    syn = bool(R.get("synthetic"))
    exp = R["experiment"]
    test = R["test"]
    bys = test["by_strategy"]
    strats = list(bys)
    comps = test["paired_comparisons"]
    pred = R["prediction"]
    sel_model = pred["selected"]
    L: list[str] = []
    add = L.append

    def fig(key: str, caption: str) -> None:
        if key in figs:
            add(f"![{caption}](figures/{figs[key]})\n\n*{caption}*\n")

    banner = (
        "> **SYNTHETIC DATA (Mode 2).** These results come from the project's simulated order-level market, "
        "not from historical market data. They test whether the research machinery behaves correctly on a "
        "market with known properties; they are **not evidence that any strategy is profitable in a real "
        "market**.\n"
        if syn
        else "> **Historical replay (Mode 1).** Results are simulated fills against recorded market data under the "
        "stated execution assumptions; a backtest is not proof of future profitability.\n"
    )

    # ------------------------------------------------------------------ verdicts
    def comp(a, b):
        return comps.get(f"{a} - {b}", {})

    radj = test.get("risk_adjusted_comparisons", {})
    h = {}
    c = comp("adaptive", "fixed_spread")
    r1 = radj.get("adaptive - fixed_spread", {})
    h["H1"] = (_verdict(r1.get("ci_low"), r1.get("ci_high")), c)
    cands = pred["candidates"]
    best_np = min(
        (n for n in cands if n != "prior"), key=lambda n: cands[n]["val"].get("log_loss", 9), default=None
    )
    gain = cands.get(best_np, {}).get("test_logloss_gain_vs_prior", {}) if best_np else {}
    abl_pred = R.get("ablations", {}).get("prediction", {})
    full_ll = abl_pred.get("full_model", {}).get("test_log_loss")
    flow_deltas = {
        g: (abl_pred[g]["test_log_loss"] - full_ll)
        for g in abl_pred
        if g.startswith("drop:")
        and ("order_flow" in g or "trade_flow" in g)
        and abl_pred[g].get("test_log_loss") is not None
        and full_ll is not None
    }
    h["H2"] = (_verdict(gain.get("ci_low"), gain.get("ci_high")), gain)
    c3 = comp("adaptive", "inventory_aware")
    h["H3"] = (_verdict(c3.get("ci_low"), c3.get("ci_high")), c3)
    c4 = comp("inventory_aware", "fixed_spread")
    r4 = radj.get("inventory_aware - fixed_spread", {})
    h["H4"] = (_verdict(r4.get("ci_low"), r4.get("ci_high")), c4)

    # ------------------------------------------------------------------ 1. executive summary
    add(f"# Adaptive Market Maker — research report: `{exp['id']}`\n")
    add(banner)
    if exp.get("description", "").strip():
        add(f"_{exp['description'].strip()}_\n")
    add("## 1. Executive summary\n")
    rows = []
    for s in strats:
        d = bys[s]
        ci = d.get("interval_pnl_total_ci")
        rows.append(
            [
                _name(s),
                _f(d["net_pnl_total"], money=True),
                f"[{_f(ci[0], money=True)}, {_f(ci[1], money=True)}]" if ci else "n/a",
                _f(d["gross_pnl_total"], money=True),
                _f(d["fees_net_total"], money=True),
                _f(d["n_fills_total"]),
                _f(d["mean_abs_inventory"], 0),
                _f(d["max_drawdown_worst"], money=True),
            ]
        )
    add(
        _table(
            [
                "strategy",
                "net P&L (primary)",
                "95% CI",
                "gross P&L",
                "fees net of rebates",
                "fills",
                "mean abs inventory (sh)",
                "worst max drawdown",
            ],
            rows,
        )
    )
    add(
        "\nNet P&L is after modelled fees and rebates with any residual inventory liquidated at the far touch "
        "(walking visible depth, taker fee). Fees below zero are net rebates. Confidence intervals come from a "
        f"circular block bootstrap of {_f(test['interval_s'], 0)}-second P&L intervals.\n"
    )
    add("**Hypothesis verdicts (mechanical, 95% intervals):**\n")
    add(
        f"- **H1** (C beats A on risk-adjusted net P&L): {h['H1'][0]}; mean/std of interval P&L "
        f"{_f(r1.get('ratio_a'), 3)} vs {_f(r1.get('ratio_b'), 3)}, difference {_f(r1.get('diff'), 3)} "
        f"[{_f(r1.get('ci_low'), 3)}, {_f(r1.get('ci_high'), 3)}]. Raw net P&L difference per interval "
        f"{_f(c.get('mean'), 3, money=True)} [{_f(c.get('ci_low'), 3, money=True)}, {_f(c.get('ci_high'), 3, money=True)}] "
        f"({_verdict(c.get('ci_low'), c.get('ci_high'))})."
    )
    add(
        f"- **H2** (features predict out of sample): {h['H2'][0]}; test log-loss gain of `{best_np}` vs prior "
        f"{_f(gain.get('mean'), 4)} nats [{_f(gain.get('ci_low'), 4)}, {_f(gain.get('ci_high'), 4)}]."
    )
    add(
        f"- **H3** (alpha survives costs: C beats B): {h['H3'][0]}; {_f(c3.get('mean'), 3, money=True)} per interval "
        f"[{_f(c3.get('ci_low'), 3, money=True)}, {_f(c3.get('ci_high'), 3, money=True)}]."
    )
    add(
        f"- **H4** (inventory control improves risk-adjusted P&L: B vs A): {h['H4'][0]}; risk-adjusted difference "
        f"{_f(r4.get('diff'), 3)} [{_f(r4.get('ci_low'), 3)}, {_f(r4.get('ci_high'), 3)}]; raw net P&L "
        f"{_verdict(c4.get('ci_low'), c4.get('ci_high'))}; mean abs inventory "
        f"{_f(bys.get('fixed_spread', {}).get('mean_abs_inventory'), 0)} → "
        f"{_f(bys.get('inventory_aware', {}).get('mean_abs_inventory'), 0)} shares."
    )
    abl_tr = R.get("ablations", {}).get("trading", {})
    glft, b_tot = abl_tr.get("fill_aware_glft"), bys.get("inventory_aware", {}).get("net_pnl_total")
    if glft and b_tot is not None:
        add(
            f"- **H5** (fill-aware quoting): GLFT quotes with the estimated fill intensity earned "
            f"{_f(glft['net_pnl'], money=True)} vs {_f(b_tot, money=True)} for B "
            f"({glft['n_fills']:,} vs {bys['inventory_aware']['n_fills_total']:,} fills); point estimate only."
        )
    reg_c = bys.get("adaptive", {}).get("by_regime", {})
    reg_b = bys.get("inventory_aware", {}).get("by_regime", {})
    if reg_c and reg_b:
        parts = []
        for dim in ("volatility", "spread", "flow"):
            for lv, e in reg_c.get(dim, {}).items():
                if lv in reg_b.get(dim, {}):
                    parts.append(f"{dim}={lv}: {_f(e['pnl_mid'] - reg_b[dim][lv]['pnl_mid'], 0, money=True)}")
        add(
            "- **H6** (regime dependence), C − B mark-to-mid P&L by regime: "
            + "; ".join(parts)
            + " (descriptive)."
        )
    add("- **H7** (robustness): see section 13 for every sensitivity grid and where results change sign.\n")

    # ------------------------------------------------------------------ 2. questions
    add("## 2. Research question and hypotheses\n")
    add(
        "Does a market maker that adapts its quotes to short-horizon price forecasts, execution conditions and "
        "inventory risk outperform simpler market makers under realistic replay assumptions?\n"
    )
    add(
        _table(
            ["hypothesis", "experiment", "metric", "baseline", "verdict"],
            [
                [
                    "H1 adaptive > fixed, risk-adjusted",
                    "test backtests",
                    "paired mean/std of interval net P&L",
                    "A fixed spread",
                    h["H1"][0],
                ],
                [
                    "H2 incremental short-horizon information",
                    "direction models + feature ablations",
                    "test log loss vs prior; Δ when dropping flow groups",
                    "constant-probability prior",
                    h["H2"][0],
                ],
                [
                    "H3 signal survives costs",
                    "C vs B (identical except alpha)",
                    "paired interval net P&L",
                    "B inventory-aware",
                    h["H3"][0],
                ],
                [
                    "H4 inventory control improves risk-adjusted P&L",
                    "B vs A",
                    "paired mean/std of interval net P&L; inventory",
                    "A fixed spread",
                    h["H4"][0],
                ],
                [
                    "H5 fill-aware quoting",
                    "GLFT with estimated fill intensity vs B",
                    "net P&L (point estimate)",
                    "B",
                    "point estimate only",
                ],
                [
                    "H6 regime dependence",
                    "P&L by causal regime",
                    "P&L per regime",
                    "C vs B by regime",
                    "descriptive",
                ],
                [
                    "H7 robustness",
                    "sensitivity grids",
                    "sign changes of net P&L",
                    "selected configuration",
                    "descriptive",
                ],
            ],
        )
    )
    add("")

    # ------------------------------------------------------------------ 3. data
    add("## 3. Dataset and quality assessment\n")
    data = R["data"]
    add(f"Dataset `{data['dataset_id']}` (source: **{data['source']}**).\n")
    rows = []
    mans = {m["session_id"]: m for m in data.get("manifests", [])}
    for s in data["sessions"]:
        m = mans.get(s["session_id"], {})
        rows.append(
            [
                s["session_id"],
                _f(s["rows"]),
                _f(s["observable_states"]),
                _f(s.get("mid_drift_ticks"), 1),
                m.get("validation_status", "n/a"),
                ", ".join(m.get("validation_failed_checks", [])) or "none",
            ]
        )
    add(
        _table(
            ["session", "messages", "observable states", "mid drift (ticks)", "validation", "flagged checks"],
            rows,
        )
    )
    add(
        "\nMid drift (last − first mid) matters for reading side-specific markouts: in a session that drifts "
        "down, buy fills look adversely selected and sell fills look favourable, independent of the strategy.\n"
    )
    add(
        "\nEvery book-changing message was reconciled against the source book snapshot (check "
        "`book_transitions_consistent`); critical failures stop the pipeline. Messages sharing one timestamp are "
        "treated as one observable state (only the last book in a burst could be observed).\n"
    )
    add(
        _table(
            ["role", "session", "start (s after midnight)", "end", "duration (s)"],
            [
                [g["role"], g["session_id"], _f(g["start_s"], 1), _f(g["end_s"], 1), _f(g["duration_s"], 0)]
                for g in data["segments"]
            ],
        )
    )
    add("")
    fig("book", "Best bid and ask over a representative interval of the first test session.")
    fig("micro", "Mid-price and microprice.")
    fig("imbalance", "Top-of-book and 5-level imbalance.")

    # ------------------------------------------------------------------ 4. methodology
    add("## 4. Methodology\n")
    add(
        "1. Validate data; build causal features on observable states; labels = as-of mid change over the horizon.\n"
        "2. Chronological train / validation / test segments; labels purged at segment ends.\n"
        "3. Direction models fit on train, temperature-calibrated on the first half of validation, selected by "
        "log loss on the second half (non-overlapping labels).\n"
        "4. Strategy parameters chosen on validation only from small pre-declared grids; C inherits B's choice so "
        "C − B isolates the alpha component.\n"
        "5. Every strategy is run once on identical test segments; robustness grids and ablations are reported on test but never "
        "used to re-select anything.\n"
    )

    # ------------------------------------------------------------------ 5. features
    add("## 5. Feature definitions\n")
    add("See `docs/mathematical_model.md` for formulas. Feature groups used by the models:\n")
    cols = pred.get("feature_columns", [])
    add(f"{len(cols)} features: " + ", ".join(f"`{c}`" for c in cols) + "\n")

    # ------------------------------------------------------------------ 6. prediction
    add("## 6. Predictive-model evaluation\n")
    add(
        f"Target: ternary sign of the as-of mid change over **{pred['alpha_horizon']}**. Train/val-cal/val-select/test "
        f"samples: {_f(pred['n_train'])} / {_f(pred['n_val_cal'])} / {_f(pred['n_val_sel'])} / {_f(pred['n_test'])} "
        "(validation-select and test use non-overlapping labels).\n"
    )
    rows = []
    for n, cd in cands.items():
        v, t, sq = cd.get("val", {}), cd.get("test", {}), cd.get("test_signal", {})
        g = cd.get("test_logloss_gain_vs_prior", {})
        rows.append(
            [
                f"**{n}**" if n == sel_model else n,
                _f(cd.get("temperature"), 2),
                _f(v.get("log_loss"), 4),
                _f(t.get("log_loss"), 4),
                f"{_f(g.get('mean'), 4)} [{_f(g.get('ci_low'), 4)}, {_f(g.get('ci_high'), 4)}]" if g else "—",
                _f(t.get("accuracy"), 3),
                _f(t.get("balanced_accuracy"), 3),
                _f(t.get("brier"), 4),
                _f(t.get("auc_up_vs_rest"), 3),
                _f(t.get("auc_direction_given_move"), 3),
                _f(sq.get("ic_spearman"), 3),
            ]
        )
    add(
        _table(
            [
                "model",
                "T",
                "val log loss",
                "test log loss",
                "test gain vs prior (95% CI)",
                "accuracy",
                "balanced acc.",
                "Brier",
                "AUC up",
                "AUC dir. given move",
                "IC (Spearman)",
            ],
            rows,
        )
    )
    tb = cands.get(sel_model, {}).get("test", {}).get("class_balance", {})
    add(
        f"\nSelected (bold): `{sel_model}`. Test class balance (down/flat/up): "
        + ", ".join(f"{k}: {_f(v, pct=True)}" for k, v in tb.items())
        + ". Accuracy alone is misleading when the flat "
        "class dominates; log loss and the directional AUC are the informative measures.\n"
    )
    reg = pred.get("selected_by_regime", {})
    if reg:
        rows = [
            [dim, lab, _f(m["n"]), _f(m["log_loss"], 4), _f(m["accuracy"], 3), _f(m["auc_up_vs_rest"], 3)]
            for dim, d in reg.items()
            for lab, m in d.items()
        ]
        add(_table(["regime dimension", "regime", "n", "log loss", "accuracy", "AUC up"], rows))
        add("")
    bys_pred = pred.get("selected_by_session", {})
    if bys_pred:
        add("Selected model by test session:\n")
        add(
            _table(
                ["session", "n", "log loss", "accuracy", "AUC up"],
                [
                    [k, _f(m["n"]), _f(m["log_loss"], 4), _f(m["accuracy"], 3), _f(m["auc_up_vs_rest"], 3)]
                    for k, m in bys_pred.items()
                ],
            )
        )
        add("")
    fig("alpha_deciles", "Realized mid change by decile of the model's alpha (test); bars show ±1.96 SE.")
    fig("reliability", "Reliability diagram for P(up) on the test period.")

    # ------------------------------------------------------------------ 7. execution
    add("## 7. Execution simulator and assumptions\n")
    a = test.get("assumptions", {})
    add(
        _table(
            ["assumption", "value"],
            [
                ["execution mode", a.get("execution_mode")],
                ["fill preset", a.get("fill_preset")],
                ["latency (µs, per hop)", json.dumps(a.get("latency"))],
                ["fees ($/share)", json.dumps(a.get("fees"))],
                ["post-only quotes", a.get("post_only")],
                ["flatten before end (s)", a.get("flatten_before_end_s")],
                [
                    "risk limits",
                    json.dumps({k: v for k, v in (a.get("risk") or {}).items() if v is not None}),
                ],
            ],
        )
    )
    add(
        "\nThe central preset tracks the visible queue ahead of each simulated order exactly using order ids "
        "(orders whose submission was never seen rested earlier and are ahead; later orders are behind). "
        "Simulated orders do not change other participants' behaviour and do not remove liquidity from the "
        "historical record. See `docs/execution_assumptions.md`.\n"
    )
    fm = R.get("fill_model", {})
    ft = fm.get("test", {})
    if ft:
        add(
            f"**Fill-probability model** (shadow 1-lot orders, {fm.get('horizon_s')} s horizon, preset "
            f"`{fm.get('fill_preset')}`): test n = {_f(ft.get('n'))}, observed fill rate {_f(ft.get('observed_fill_rate'), pct=True)} "
            f"(by ticks behind best: "
            + ", ".join(f"{k}: {_f(v, pct=True)}" for k, v in ft.get("by_offset", {}).items())
            + f"); Brier {_f(ft.get('brier'), 4)} vs base-rate {_f(ft.get('brier_base_rate'), 4)}; AUC {_f(ft.get('auc'), 3)}."
        )
        it = fm.get("intensity_train", {})
        if "k_per_tick" in it:
            add(
                f" Estimated fill intensity λ(δ) = A·exp(−kδ): A = {_f(it['A_per_s'], 3)} /s, "
                f"k = {_f(it['k_per_tick'], 3)} /tick (training shadow orders).\n"
            )
        fig(
            "fill_model",
            "Time-to-fill survival curves and fill-probability calibration (test shadow orders).",
        )

    # ------------------------------------------------------------------ 8. strategies
    add("## 8. Strategy definitions and selected parameters\n")
    add(
        "A: fixed half-spread around the mid. B: Avellaneda–Stoikov-style reservation price "
        "r = S − q·γ·σ²·τ (q in lots, σ in ticks/√s, rolling τ) with the same half-spread as A. "
        "C: B plus r += clip(α) and an adverse-selection guard that withdraws the side α says will be picked off. "
        "D: C plus causal regime rules (stop quoting in high realized volatility, widen when the market spread is wide).\n"
    )
    rows = []
    for s, d in R.get("selection", {}).items():
        best = d["selected"]
        cand = "; ".join(
            f"{json.dumps(x['params'])} → {_f(x['val_net_pnl'], money=True)}"
            for x in d["val_net_pnl_by_candidate"]
        )
        rows.append([_name(s), json.dumps(best) if best else "defaults", d["n_candidates"], cand])
    add(_table(["strategy", "selected on validation", "candidates", "validation net P&L by candidate"], rows))
    add("")

    # ------------------------------------------------------------------ 9-10. backtest results
    add("## 9. Backtesting methodology\n")
    add(
        f"Event-driven replay with explicit latency; historical messages win timestamp ties. Each strategy ran on "
        f"the same {len(bys[strats[0]]['per_segment'])} test segment(s) with identical data, fees, latency, limits and "
        f"simulator. {R.get('n_backtests', 'n/a')} backtests were run in total (all logged in `trials.csv`).\n"
    )
    add("## 10. Benchmark results\n")
    rows = []
    for s in strats:
        segs = bys[s]["per_segment"]
        ratio = [m.get("interval_pnl_ratio") for m in segs if m.get("interval_pnl_ratio") is not None]
        rows.append(
            [
                _name(s),
                _f(bys[s]["net_pnl_total"], money=True),
                _f(bys[s]["net_pnl_mid_total"], money=True),
                _f(sum(m["orders_sent"] for m in segs)),
                _f(bys[s]["n_fills_total"]),
                _f(np.mean([m["fill_rate"] or 0 for m in segs]), pct=True),
                _f(bys[s]["shares_traded_total"]),
                _f(bys[s]["max_abs_inventory"], 0),
                _f(np.mean([m.get("pnl_per_share") or 0 for m in segs]), 4, money=True),
                _f(np.mean(ratio), 3) if ratio else "n/a",
            ]
        )
    add(
        _table(
            [
                "strategy",
                "net P&L",
                "net P&L (mid-marked)",
                "orders",
                "fills",
                "fill rate",
                "shares traded",
                "max abs inventory",
                "net P&L / share",
                f"mean/std of {_f(test['interval_s'], 0)}s P&L",
            ],
            rows,
        )
    )
    add("\nPer-segment results:\n")
    rows = []
    for s in strats:
        for seg_m, seg in zip(bys[s]["per_segment"], [g for g in data["segments"] if g["role"] == "test"]):
            rows.append(
                [
                    _name(s),
                    seg["session_id"],
                    _f(seg_m["net_pnl"], money=True),
                    _f(seg_m["n_fills"]),
                    _f(seg_m["mean_abs_inventory"], 0),
                    _f(seg_m["max_drawdown"], money=True),
                    seg_m.get("kill_switch") or "",
                ]
            )
    add(
        _table(
            ["strategy", "segment", "net P&L", "fills", "mean abs inv.", "max drawdown", "kill switch"], rows
        )
    )
    add("")
    fig("net_pnl", "Net P&L by strategy on the test segments.")
    fig("equity", "Cumulative equity across test segments.")
    fig("drawdown", "Drawdown from the running peak.")
    fig("inventory", "Inventory paths in the first test session.")
    fig("orders", "Order, fill, cancel and reject counts.")

    # ------------------------------------------------------------------ 11. costs
    add("## 11. Transaction costs and adverse selection\n")
    add(
        "Per maker fill (ticks per share, side-signed so positive = good for us): effective half-spread "
        "e = s(M₀ − P); adverse selection AS_h = −s(M_h − M₀); realized spread RS_h = e − AS_h = s(M_h − P). "
        "Fees are excluded here and added once in the last column.\n"
    )
    rows = []
    for s in strats:
        mk = bys[s]["markouts"]
        mm = mk.get("by_liquidity", {}).get("maker", {})
        if not mm:
            continue
        hz = [k for k in mm if k.startswith("realized_spread_")]
        rows.append(
            [_name(s), _f(mm.get("n")), _f(mm.get("eff_half_spread_ticks"), 3)]
            + [_f(mm[k], 3) for k in hz]
            + [_f(mm.get(k.replace("realized_spread", "adverse_sel")), 3) for k in hz[-1:]]
            + [_f(next((v for k, v in mk.items() if k.startswith("net_per_share")), None), 5, money=True)]
        )
        hz_names = [k.replace("realized_spread_", "RS ").replace("_ticks", "") for k in hz]
    if rows:
        add(
            _table(
                ["strategy", "maker fills", "e"]
                + hz_names
                + [f"AS {hz_names[-1][3:]}", "net $/share (longest h, incl. fees)"],
                rows,
            )
        )
        add("")
    rows = []
    for s in strats:
        br = bys[s]["markouts"].get("by_side", {})
        for side, d in br.items():
            last = [k for k in d if k.startswith("realized_spread_")][-1]
            rows.append([_name(s), side, _f(d["n"]), _f(d["eff_half_spread_ticks"], 3), _f(d[last], 3)])
    add(_table(["strategy", "side", "fills", "e (ticks)", "RS longest h (ticks)"], rows))
    add("")
    mbr = {st: bys[st].get("markouts_by_regime", {}) for st in strats}
    if any(mbr.values()):
        add(
            "\nMaker realized spread at the longest horizon by causal regime at the time of the fill (ticks):\n"
        )
        rows = []
        for dim in ("volatility", "spread", "flow", "depth"):
            present = {lv for d in mbr.values() for lv in d.get(dim, {})}
            levels = [lv for lv in REGIME_ORDER if lv in present] + sorted(present - set(REGIME_ORDER))
            for lv in levels:
                row = [dim, lv]
                for st in strats:
                    e = mbr[st].get(dim, {}).get(lv)
                    rk = next((k for k in (e or {}) if k.startswith("realized_spread_")), None)
                    row.append(f"{_f(e[rk], 3)} (n={_f(e['n'])})" if e and rk else "n/a")
                rows.append(row)
        add(_table(["dimension", "regime"] + [_name(st) for st in strats], rows))
        add("")
    fig("markouts", "Maker realized spread by horizon.")

    # ------------------------------------------------------------------ 12. ablations
    add("## 12. Ablation experiments\n")
    abl = R.get("ablations", {})
    if abl.get("trading"):
        full = bys.get("adaptive", {}).get("net_pnl_total")
        rows = [
            [
                n,
                _f(d["net_pnl"], money=True),
                _f(d["net_pnl"] - full, money=True) if full is not None else "n/a",
                _f(d["n_fills"]),
                _f(d["mean_abs_inventory"], 0),
            ]
            for n, d in abl["trading"].items()
        ]
        add(_table(["ablation", "test net P&L", "Δ vs full C", "fills", "mean abs inv."], rows))
        add(
            "\n`no_alpha` reduces C to B exactly (same parameters); `no_vol_adaptation` holds σ at its training-period "
            "mean; `fill_aware_glft` replaces B's fixed half-spread with GLFT quotes using the estimated fill intensity "
            "(H5); `drop:<groups>` retrains the selected model without those feature groups and re-runs C.\n"
        )
        plc, b_pnl = abl["trading"].get("placebo_alpha"), bys.get("inventory_aware", {}).get("net_pnl_total")
        if plc and full is not None and b_pnl is not None:
            ok = plc["net_pnl"] <= b_pnl
            add(
                f"**Placebo check.** C fed its own alpha shifted by half a session (same distribution, no "
                f"alignment with the market) earned {_f(plc['net_pnl'], money=True)}, versus "
                f"{_f(b_pnl, money=True)} for B (no alpha) and {_f(full, money=True)} for C with the real alpha. "
                + (
                    "The placebo does not beat B, so the engine is not rewarding alpha-shaped noise.\n"
                    if ok
                    else "**The placebo beats B: the alpha component's gains may not come from information. "
                    "Investigate before trusting H3.**\n"
                )
            )
        vacuous = [
            n for n, d in abl["trading"].items() if full is not None and abs(d["net_pnl"] - full) < 1e-9
        ]
        if vacuous:
            add(
                "Identical to the full strategy (the component is inactive in the validation-selected "
                "configuration, so the ablation removes nothing): "
                + ", ".join(f"`{n}`" for n in vacuous)
                + ".\n"
            )
    if abl.get("prediction"):
        rows = [
            [
                n,
                _f(d.get("val_log_loss"), 4),
                _f(d.get("test_log_loss"), 4),
                _f(d.get("test_auc_up"), 3),
                _f(d.get("test_ic"), 3),
            ]
            for n, d in abl["prediction"].items()
        ]
        add(_table(["model variant", "val log loss", "test log loss", "test AUC up", "test IC"], rows))
        if flow_deltas:
            add(
                "\nIncremental value of order-flow information (H2): test log-loss change when the group is removed: "
                + ", ".join(f"{k}: {_f(v, 5)}" for k, v in flow_deltas.items())
                + " (positive = the group helped).\n"
            )
    fig("ablations", "Ablations of strategy C.")

    # ------------------------------------------------------------------ 13. sensitivity
    add("## 13. Sensitivity and robustness\n")
    sens = R.get("sensitivity", {})
    fragile = []
    for key, pts in sens.items():
        ss = list(next(iter(pts.values())))
        add(f"**{key}**\n")
        add(
            _table(
                ["value"] + [_name(s) for s in ss],
                [[v] + [_f(pts[v][s]["net_pnl"], money=True) for s in ss] for v in pts],
            )
        )
        add("")
        for s in ss:
            vals = [pts[v][s]["net_pnl"] for v in pts]
            if min(vals) < 0 < max(vals):
                fragile.append(
                    f"{_name(s)} changes sign across `{key}` ({_f(min(vals), money=True)} to {_f(max(vals), money=True)})"
                )
        if "adaptive" in ss and "fixed_spread" in ss:
            diffs = [pts[v]["adaptive"]["net_pnl"] - pts[v]["fixed_spread"]["net_pnl"] for v in pts]
            if min(diffs) < 0 < max(diffs):
                fragile.append(f"the ranking of C vs A flips across `{key}`")
    fig("latency", "Net P&L vs latency.")
    fig("fees_fills", "Net P&L vs maker fee and fill-assumption preset.")
    fig("sensitivity_other", "Other sensitivity grids.")
    if fragile:
        add("**Where results change sign (H7):**\n")
        add("\n".join(f"- {x}" for x in fragile) + "\n")

    # ------------------------------------------------------------------ 14. uncertainty
    add("## 14. Statistical uncertainty\n")
    rows = [
        [
            k,
            _f(v.get("n_intervals")),
            _f(v.get("mean"), 3, money=True),
            _f(v.get("ci_low"), 3, money=True),
            _f(v.get("ci_high"), 3, money=True),
            _f(v.get("p_le_zero"), 3),
            _verdict(v.get("ci_low"), v.get("ci_high")),
        ]
        for k, v in comps.items()
    ]
    add(
        _table(
            [
                "comparison (first − second)",
                "paired intervals",
                "mean / interval",
                "95% low",
                "95% high",
                "share of draws ≤ 0",
                "verdict",
            ],
            rows,
        )
    )
    if radj:
        add(
            "\nRisk-adjusted (mean / std of interval net P&L, not annualised), paired circular block bootstrap:\n"
        )
        add(
            _table(
                ["comparison", "first", "second", "difference", "95% low", "95% high", "verdict"],
                [
                    [
                        k,
                        _f(v.get("ratio_a"), 3),
                        _f(v.get("ratio_b"), 3),
                        _f(v.get("diff"), 3),
                        _f(v.get("ci_low"), 3),
                        _f(v.get("ci_high"), 3),
                        _verdict(v.get("ci_low"), v.get("ci_high")),
                    ]
                    for k, v in radj.items()
                ],
            )
        )
    n_trials = R.get("n_backtests", 0)
    n_val = sum(d["n_candidates"] for d in R.get("selection", {}).values())
    add(
        f"\nIntervals are {_f(test['interval_s'], 0)} s of clock time within a session; circular blocks of consecutive "
        "intervals wrap within a session and never span sessions. Dependence longer than a block, and differences between sessions as "
        "a whole, are not captured.\n\n**Multiple testing.** "
        f"{n_val} parameter candidates were evaluated on validation and {n_trials} backtests were run overall. Only the "
        "pre-declared final comparisons above are confirmatory; sensitivity and ablation results are exploratory, and "
        "a best case picked from them would be biased upward by selection.\n"
    )

    # ------------------------------------------------------------------ 15-19
    add("## 15. Failure cases and negative results\n")
    neg = []
    for k, (v, _) in h.items():
        if not v.startswith("supported"):
            neg.append(f"{k}: {v}.")
    for s in strats:
        if bys[s]["net_pnl_total"] < 0:
            neg.append(
                f"{_name(s)} lost money on the test segments ({_f(bys[s]['net_pnl_total'], money=True)})."
            )
        mm = bys[s]["markouts"].get("by_liquidity", {}).get("maker", {})
        rs = [mm[k] for k in mm if k.startswith("realized_spread_") and mm[k] is not None]
        if rs and rs[-1] < 0:
            neg.append(
                f"{_name(s)}: maker fills are adversely selected beyond the spread at the longest horizon "
                f"(RS {_f(rs[-1], 3)} ticks)."
            )
    neg += fragile
    add(
        "\n".join(f"- {x}" for x in neg) + "\n"
        if neg
        else "No hypothesis was contradicted and no sign changes were observed.\n"
    )
    add("## 16. Limitations\n")
    add(
        (
            "- **Synthetic market.** The generating process (latent value, informed flow, Poisson order arrivals) is "
            "the project's own; real markets are non-stationary, have richer participant behaviour and queue dynamics.\n"
            if syn
            else "- **One instrument, few sessions.** Results do not generalise beyond the tested instrument and dates.\n"
        )
        + "- **No market impact.** Simulated orders do not change other participants' behaviour or remove liquidity from history.\n"
        "- **Queue uncertainty.** Hidden/reserve liquidity ahead of us is unobservable; messages beyond the recorded depth are absent.\n"
        "- **Latency is a constant.** No jitter, gateway queueing or exchange-side processing.\n"
        "- **Fees are illustrative.** Maker rebates matter a lot for one-tick markets (see sensitivity).\n"
        "- **Statistics.** Intra-session block bootstrap; few sessions; many backtests were run (selection risk).\n"
        "See `docs/limitations.md` for the full discussion.\n"
    )
    add("## 17. Conclusions\n")
    concl = [
        f"On this {'synthetic' if syn else 'historical'} sample, the alpha-aware strategy versus the fixed-spread "
        f"benchmark on risk-adjusted net P&L: {h['H1'][0]}.",
        f"The direction model's out-of-sample gain over a constant-probability forecast: {h['H2'][0]}.",
        f"The alpha component itself (C vs B, everything else identical): {h['H3'][0]}.",
        f"Inventory control (B vs A) changes mean absolute inventory from "
        f"{_f(bys.get('fixed_spread', {}).get('mean_abs_inventory'), 0)} to "
        f"{_f(bys.get('inventory_aware', {}).get('mean_abs_inventory'), 0)} shares; its risk-adjusted effect: {h['H4'][0]}.",
    ]
    add("\n".join(f"- {c}" for c in concl) + "\n")
    if syn:
        add(
            "Because the data are synthetic, these conclusions describe the simulator's market, not a real one. "
            "The decisive next step is running the identical protocol on historical LOBSTER data.\n"
        )
    add("## 18. Next research steps\n")
    add(
        "- Run `configs/experiments/lobster_msft.yaml` and `lobster_aapl.yaml` on the real samples (large-tick vs small-tick).\n"
        "- Use more sessions (paid LOBSTER history or Databento MBO, after a cost check) for cross-session walk-forward.\n"
        "- Queue-position-aware quoting: choose join/step-back from the fill model's P(fill) and expected markout.\n"
        "- Latency distributions instead of constants; an impact model for our own orders.\n"
    )
    add("## 19. Reproduction\n")
    add(
        '```bash\npip install -e ".[dev]"\npytest\n'
        f"amm experiment run --config {_config_path_for(exp['id'])} --overwrite\n"
        f"amm report --experiment {exp['id']}\n```\n\n"
        "The exact configuration is saved next to this report as `config_snapshot.yaml` (as run) and "
        "`config_selected.yaml` (with validation-selected parameters); data provenance is in `manifests.json`.\n"
    )
    out = exp_dir / "report.md"
    out.write_text("\n".join(L))
    return out
