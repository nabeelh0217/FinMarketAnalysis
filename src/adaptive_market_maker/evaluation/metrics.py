"""Trading performance metrics computed from a :class:`BacktestResult`.

Only metrics whose data requirements are met are reported. Nothing is annualized: with intraday samples
an annualized Sharpe ratio would be noise dressed up as evidence. The per-interval mean/std ratio is
reported with its sampling interval stated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_market_maker.backtest.engine import BacktestResult
from adaptive_market_maker.types import NS_PER_S


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.nan_to_num(equity, nan=-np.inf))
    dd = np.where(np.isfinite(peak), peak - equity, 0.0)
    return float(np.nanmax(dd)) if dd.size else 0.0


def interval_pnl(res: BacktestResult, interval_s: float) -> np.ndarray:
    """Mark-to-mid P&L per clock interval; the final interval includes the liquidation adjustment."""
    eq = res.equity
    if eq.empty:
        return np.array([])
    t = eq["ts_ns"].to_numpy()
    e = eq["equity_mid"].ffill().fillna(0.0).to_numpy()
    edges = np.arange(res.start_ns, res.end_ns + 1, int(interval_s * NS_PER_S))
    if edges[-1] < res.end_ns:
        edges = np.append(edges, res.end_ns)
    idx = np.searchsorted(t, edges, side="right") - 1
    vals = np.where(idx >= 0, e[np.maximum(idx, 0)], 0.0)
    pnl = np.diff(vals)
    if pnl.size:
        pnl[-1] += res.summary["net_pnl_liquidation"] - res.summary["net_pnl_mid"]
    return pnl


def trading_metrics(res: BacktestResult, interval_s: float = 300.0) -> dict:
    s = dict(res.summary)
    fills, eq = res.fills, res.equity
    dur = max(s["duration_s"], 1e-9)
    pos = eq["position"].to_numpy(np.float64) if not eq.empty else np.array([0.0])
    m = {
        "net_pnl": s["net_pnl_liquidation"],
        "net_pnl_mid": s["net_pnl_mid"],
        "gross_pnl": s["net_pnl_liquidation"] + s["fees_net"],
        "fees_net": s["fees_net"],
        "rebates": s["rebates"],
        "n_fills": s["n_fills"],
        "bid_fills": int((fills["side"] > 0).sum()) if not fills.empty else 0,
        "ask_fills": int((fills["side"] < 0).sum()) if not fills.empty else 0,
        "maker_fills": int((fills["liquidity"] == "maker").sum()) if not fills.empty else 0,
        "taker_fills": int((fills["liquidity"] == "taker").sum()) if not fills.empty else 0,
        "shares_traded": s["shares_traded"],
        "turnover_shares_per_hour": s["shares_traded"] / dur * 3600.0,
        "orders_sent": s["orders_sent"],
        "cancels_sent": s["cancels_sent"],
        "exchange_rejects": s["exchange_rejects"],
        "final_position": s["final_position"],
        "mean_abs_inventory": float(np.mean(np.abs(pos))),
        "max_abs_inventory": float(np.max(np.abs(pos))),
        "peak_long": float(pos.max()),
        "peak_short": float(pos.min()),
        "kill_switch": s["kill_switch"],
    }
    quote_orders = int(s["order_states"].get("filled", 0))
    m["fill_rate"] = quote_orders / s["orders_sent"] if s["orders_sent"] else None
    m["cancel_rate"] = s["cancels_sent"] / s["orders_sent"] if s["orders_sent"] else None
    m["quote_to_fill"] = s["orders_sent"] / s["n_fills"] if s["n_fills"] else None
    m["pnl_per_fill"] = m["net_pnl"] / s["n_fills"] if s["n_fills"] else None
    m["pnl_per_share"] = m["net_pnl"] / s["shares_traded"] if s["shares_traded"] else None
    eqv = eq["equity_mid"].to_numpy(np.float64) if not eq.empty else np.array([0.0])
    m["max_drawdown"] = max_drawdown(eqv)
    d1 = np.diff(eqv)
    m["equity_change_std_per_interval"] = float(np.nanstd(d1)) if d1.size > 1 else None
    m["equity_interval_s"] = float(np.median(np.diff(eq["ts_ns"])) / NS_PER_S) if len(eq) > 1 else None
    ip = interval_pnl(res, interval_s)
    m["interval_s"] = interval_s
    m["n_intervals"] = int(ip.size)
    m["mean_interval_pnl"] = float(ip.mean()) if ip.size else None
    m["interval_pnl_ratio"] = (
        float(ip.mean() / ip.std(ddof=1)) if ip.size > 2 and ip.std(ddof=1) > 0 else None
    )
    m["pnl_per_avg_share_inventory"] = (
        m["net_pnl"] / m["mean_abs_inventory"] if m["mean_abs_inventory"] > 0 else None
    )
    if s["shares_traded"]:
        # Little's law: average inventory / one-way throughput
        m["avg_holding_time_s"] = m["mean_abs_inventory"] / (s["shares_traded"] / 2.0 / dur)
    return m


def summarize_many(results: list[BacktestResult], interval_s: float) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({"strategy": r.strategy, "session_id": r.session_id, **trading_metrics(r, interval_s)})
    return pd.DataFrame(rows)
