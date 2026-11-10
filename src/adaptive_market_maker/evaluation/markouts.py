"""Post-fill markouts and the maker spread decomposition.

For a fill at price P with side s (+1 buy, -1 sell), mid M_0 just before the fill and as-of mid M_h at
fill time + h (all in ticks):

    effective half-spread   e    = s (M_0 - P)          what the quote earned against the prevailing mid
    adverse selection       AS_h = -s (M_h - M_0)       how far the mid moved against the new position
    realized spread         RS_h = e - AS_h = s (M_h - P)  ( = the spec's Markout_h )

Fees/rebates are NOT inside these numbers; they are added separately in ``net_per_share``, so nothing is
counted twice. Fills whose horizon runs past the end of the session data are excluded and counted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_market_maker.types import NS_PER_MS


def markout_table(
    fills: pd.DataFrame, obs_ts: np.ndarray, obs_mid: np.ndarray, tick: int, horizons_ms: list[float]
) -> pd.DataFrame:
    """One row per fill with e, AS_h and RS_h columns (ticks)."""
    if fills.empty:
        return fills.copy()
    out = fills.copy()
    s = out["side"].to_numpy(np.float64)
    p = out["price"].to_numpy(np.float64) / tick
    m0 = out["mid_before_ticks"].to_numpy(np.float64)
    t = out["ts_ns"].to_numpy(np.int64)
    out["eff_half_spread_ticks"] = s * (m0 - p)
    for h in horizons_ms:
        target = t + int(h * NS_PER_MS)
        j = np.searchsorted(obs_ts, target, side="right") - 1
        ok = (target <= obs_ts[-1]) & (j >= 0)
        mh = np.where(ok, obs_mid[np.maximum(j, 0)], np.nan)
        out[f"adverse_sel_{h:g}ms"] = -s * (mh - m0)
        out[f"realized_spread_{h:g}ms"] = s * (mh - p)
    return out


def markout_summary(
    mk: pd.DataFrame, horizons_ms: list[float], tick_dollars: float, maker_fee: float, taker_fee: float
) -> dict:
    if mk.empty:
        return {"n_fills": 0}
    res: dict = {"n_fills": int(len(mk))}

    def block(df: pd.DataFrame) -> dict:
        d = {"n": int(len(df)), "eff_half_spread_ticks": float(df["eff_half_spread_ticks"].mean())}
        for h in horizons_ms:
            rs = df[f"realized_spread_{h:g}ms"]
            d[f"realized_spread_{h:g}ms_ticks"] = float(rs.mean()) if rs.notna().any() else None
            d[f"adverse_sel_{h:g}ms_ticks"] = (
                float(df[f"adverse_sel_{h:g}ms"].mean()) if rs.notna().any() else None
            )
            d[f"excluded_{h:g}ms"] = int(rs.isna().sum())
        return d

    res["all"] = block(mk)
    res["by_side"] = {("buy" if k > 0 else "sell"): block(g) for k, g in mk.groupby("side")}
    res["by_liquidity"] = {k: block(g) for k, g in mk.groupby("liquidity")}
    res["by_reason"] = {k: block(g) for k, g in mk.groupby("reason")}
    # net economics per share at the longest horizon, in dollars, fees included once
    h = horizons_ms[-1]
    rs = mk[f"realized_spread_{h:g}ms"].to_numpy()
    fee = np.where(mk["liquidity"].to_numpy() == "maker", maker_fee, taker_fee)
    ok = np.isfinite(rs)
    w = mk["qty"].to_numpy(np.float64)
    if ok.any():
        res[f"net_per_share_{h:g}ms_dollars"] = float(
            np.average(rs[ok] * tick_dollars - fee[ok], weights=w[ok])
        )
    return res
