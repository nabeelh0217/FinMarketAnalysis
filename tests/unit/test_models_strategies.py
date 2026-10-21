"""Models, calibration, quoting formulas, risk limits, statistics and markouts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from adaptive_market_maker.config import (
    AdaptiveParams,
    InventoryAwareParams,
    ModelConfig,
    RegimeParams,
    RiskConfig,
)
from adaptive_market_maker.evaluation.markouts import markout_table
from adaptive_market_maker.evaluation.metrics import max_drawdown
from adaptive_market_maker.evaluation.statistics import (
    block_bootstrap_mean,
    paired_difference,
    paired_ratio_difference,
)
from adaptive_market_maker.models.calibration import apply_temperature, fit_temperature, log_loss
from adaptive_market_maker.models.fill_probability import kaplan_meier
from adaptive_market_maker.models.price_direction import fit_direction_model
from adaptive_market_maker.strategies.adaptive import AdaptiveStrategy, RegimeStrategy
from adaptive_market_maker.strategies.base import MarketView, QuoteIntent, StrategyState, round_quotes
from adaptive_market_maker.strategies.inventory_aware import InventoryAwareStrategy, as_terms, glft_terms
from adaptive_market_maker.strategies.risk import RiskManager

TICK = 100


def view(alpha=float("nan"), vol=0.5, bid=3000, ask=3001, secs=600.0) -> MarketView:
    return MarketView(
        t_ns=0,
        k=0,
        row=0,
        best_bid=bid * TICK,
        best_ask=ask * TICK,
        bid_sz=500,
        ask_sz=500,
        tick=TICK,
        lot=100,
        mid_ticks=(bid + ask) / 2,
        spread_ticks=ask - bid,
        vol=vol,
        alpha=alpha,
        secs_to_end=secs,
    )


# ------------------------------------------------------------------------------------- quoting maths
def test_round_quotes_nearest_resolves_ties_passively():
    v = view()
    assert round_quotes(v, 3000.0, 3001.0) == (3000 * TICK, 3001 * TICK)
    assert round_quotes(v, 2999.5, 3001.5) == (2999 * TICK, 3002 * TICK)  # exact halves -> less aggressive
    assert round_quotes(v, 2999.6, 3001.4) == (3000 * TICK, 3001 * TICK)
    assert round_quotes(v, 3005.0, 2990.0) == (3000 * TICK, 3001 * TICK)  # clamped to stay passive
    assert round_quotes(v, 2999.9, 3001.1, "conservative") == (2999 * TICK, 3002 * TICK)


def test_avellaneda_stoikov_terms_hand_calculated():
    skew, half = as_terms(gamma=0.1, k=1.5, sigma=0.5, tau=30.0)
    assert skew == pytest.approx(0.1 * 0.25 * 30)
    assert half == pytest.approx(10 * math.log(1 + 0.1 / 1.5) + 0.5 * 0.75)


def test_glft_terms_hand_calculated():
    omega, half = glft_terms(gamma=0.1, k=1.5, A=1.0, sigma=0.5)
    expected = math.sqrt(0.25 * 0.1 / (2 * 1.5 * 1.0) * (1 + 0.1 / 1.5) ** (1 + 15))
    assert omega == pytest.approx(expected)
    assert half == pytest.approx(10 * math.log(1 + 0.1 / 1.5) + expected / 2)


def test_inventory_skew_direction_and_magnitude():
    s = InventoryAwareStrategy(InventoryAwareParams(gamma=0.1, tau_s=30, half_spread_ticks=0.5))
    flat = s.quote(view(), StrategyState(0, 0, 0))
    long_ = s.quote(view(), StrategyState(100, 0, 0))  # +1 lot -> skew 0.75 tick
    short = s.quote(view(), StrategyState(-100, 0, 0))
    assert (flat.bid_px, flat.ask_px) == (3000 * TICK, 3001 * TICK)
    assert long_.reservation_ticks == pytest.approx(3000.5 - 0.75)
    assert (long_.bid_px, long_.ask_px) == (2999 * TICK, 3001 * TICK)  # backs off the bid
    assert (short.bid_px, short.ask_px) == (3000 * TICK, 3002 * TICK)  # backs off the ask


def test_model_half_spread_widens_with_volatility():
    s = InventoryAwareStrategy(InventoryAwareParams(spread_mode="model", gamma=0.1, k=1.5, tau_s=30))
    assert s.skew_and_half(view(vol=1.0))[1] > s.skew_and_half(view(vol=0.2))[1]


def test_session_end_horizon_shrinks_skew_near_close():
    s = InventoryAwareStrategy(InventoryAwareParams(horizon_mode="session_end", gamma=0.1))
    assert s.skew_and_half(view(secs=10))[0] < s.skew_and_half(view(secs=1000))[0]


def test_alpha_shifts_reservation_and_guard_withdraws_adverse_side():
    p = AdaptiveParams(gamma=0.1, tau_s=30, alpha_guard_ticks=0.4)
    s = AdaptiveStrategy(p)
    up = s.quote(view(alpha=0.3), StrategyState(0, 0, 0))
    assert up.reservation_ticks == pytest.approx(3000.8)
    assert up.ask_px is not None
    strong = s.quote(view(alpha=0.6), StrategyState(0, 0, 0))
    assert strong.ask_px is None and strong.bid_px == 3000 * TICK and strong.note == "guard_ask"
    down = s.quote(view(alpha=-0.6), StrategyState(0, 0, 0))
    assert down.bid_px is None
    no_alpha = AdaptiveStrategy(p.model_copy(update={"use_alpha": False})).quote(
        view(alpha=0.6), StrategyState(0, 0, 0)
    )
    base = InventoryAwareStrategy(InventoryAwareParams(gamma=0.1, tau_s=30)).quote(
        view(alpha=0.6), StrategyState(0, 0, 0)
    )
    assert (no_alpha.bid_px, no_alpha.ask_px) == (base.bid_px, base.ask_px)


def test_regime_rules_stop_quoting_in_high_volatility():
    s = RegimeStrategy(RegimeParams(high_vol_ticks_per_sqrt_s=1.0, high_vol_size_mult=0.0))
    q = s.quote(view(vol=2.0), StrategyState(0, 0, 0))
    assert q.bid_px is None and q.ask_px is None


# ------------------------------------------------------------------------------------------- risk
def test_risk_inventory_limit_is_worst_case():
    rm = RiskManager(RiskConfig(max_inventory=200, max_order_size=500, max_outstanding_per_side=500), 10_000)
    intent = QuoteIntent(3000 * TICK, 100, 3001 * TICK, 300, 3000.5, 0.5)
    out = rm.check(intent, view(), position=100, other_open_buy=100, other_open_sell=0, data_age_ns=0)
    assert out.bid_px is None  # 100 held + 100 pending buy + 100 new > 200
    assert out.ask_qty == 300  # short room = 200 + 100
    assert rm.blocks["inventory_long"] == 1


def test_risk_stale_data_kill_switch_and_price_deviation():
    rm = RiskManager(RiskConfig(stale_data_s=1, max_price_deviation_ticks=2), 10_000)
    assert (
        rm.check(QuoteIntent(3000 * TICK, 100, 3001 * TICK, 100), view(), 0, 0, 0, 2 * 10**9).bid_px is None
    )
    out = rm.check(QuoteIntent(2990 * TICK, 100, 3001 * TICK, 100), view(), 0, 0, 0, 0)
    assert out.bid_px is None and out.ask_px == 3001 * TICK
    rm.kill("max_session_loss")
    killed = rm.check(QuoteIntent(3000 * TICK, 100, 3001 * TICK, 100), view(), 0, 0, 0, 0)
    assert killed.bid_px is None and killed.ask_px is None


def test_rate_limit_window():
    rm = RiskManager(RiskConfig(max_actions_per_second=3), 10_000)
    assert rm.allow_actions(0, 2) and not rm.allow_actions(1, 2) and rm.allow_actions(2 * 10**9, 2)


# ------------------------------------------------------------------------------------------ models
def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 4000)
    logits = np.eye(3)[y] * 1.0 + rng.normal(0, 1.0, (4000, 3))
    p = np.exp(logits * 4)  # sharpened -> overconfident
    p /= p.sum(axis=1, keepdims=True)
    t = fit_temperature(p, y)
    assert t > 1.5
    assert log_loss(y, apply_temperature(p, t)) < log_loss(y, p)


def test_direction_model_learns_planted_signal_and_alpha_units():
    rng = np.random.default_rng(1)
    n = 6000
    x = rng.normal(size=(n, 2))
    dmid = np.where(
        x[:, 0] + 0.3 * rng.normal(size=n) > 0.5,
        1.0,
        np.where(x[:, 0] + 0.3 * rng.normal(size=n) < -0.5, -1.0, 0.0),
    )
    y = np.sign(dmid)
    cfg = ModelConfig(calibrate=False)
    classes = np.array([-1, 0, 1])
    m = fit_direction_model("logistic", cfg, x, y, dmid, classes, ["a", "b"])
    prior = fit_direction_model("prior", cfg, x, y, dmid, classes, ["a", "b"])
    assert m.class_mean_dmid == {-1: -1.0, 0: 0.0, 1: 1.0}
    a = m.alpha_ticks(np.array([[2.0, 0.0], [-2.0, 0.0]]))
    assert a[0] > 0.5 and a[1] < -0.5
    np.testing.assert_allclose(
        prior.predict_proba(x[:3]), np.tile([np.mean(y == c) for c in classes], (3, 1)), atol=1e-6
    )


def test_kaplan_meier_with_censoring():
    d = np.array([1.0, 2.0, 2.0, 5.0, 5.0])
    e = np.array([1, 1, 0, 0, 0])  # two fills, three censored
    s = kaplan_meier(d, e, np.array([0.5, 1.0, 2.0, 6.0]))
    assert s.tolist() == pytest.approx([1.0, 0.8, 0.8 * (1 - 1 / 4), 0.6])


# ------------------------------------------------------------------------------------- statistics
def test_block_bootstrap_covers_true_mean_and_flags_small_samples():
    rng = np.random.default_rng(2)
    res = block_bootstrap_mean([rng.normal(1.0, 1.0, 400)], 3, 1000, 0)
    assert res["ci_low"] < 1.0 < res["ci_high"] and res["p_le_zero"] == 0.0
    assert block_bootstrap_mean([np.array([1.0, 2.0])], 3, 100, 0)["ci_low"] is None


def test_paired_comparisons_detect_a_shift():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 300)
    better = base + 0.5
    d = paired_difference([better], [base], 3, 1000, 0)
    assert d["ci_low"] > 0.4  # paired: the common noise cancels
    r = paired_ratio_difference([better], [base], 3, 1000, 0)
    assert r["ci_low"] > 0


def test_max_drawdown():
    assert max_drawdown(np.array([0, 5, 3, 8, 2, 4])) == 6


def test_markout_decomposition_hand_calculated():
    fills = pd.DataFrame(
        {
            "ts_ns": [1_000_000_000],
            "side": [1],
            "price": [3000 * TICK],
            "qty": [100],
            "liquidity": ["maker"],
            "mid_before_ticks": [3000.5],
        }
    )
    obs_ts = np.array([0, 1_000_000_000, 1_500_000_000, 3_000_000_000])
    obs_mid = np.array([3000.5, 3000.5, 2999.5, 3001.0])
    mk = markout_table(fills, obs_ts, obs_mid, TICK, [1000.0, 5000.0])
    assert mk["eff_half_spread_ticks"].iloc[0] == pytest.approx(0.5)
    assert mk["realized_spread_1000ms"].iloc[0] == pytest.approx(-0.5)  # as-of mid at 2.0 s is 2999.5
    assert mk["adverse_sel_1000ms"].iloc[0] == pytest.approx(1.0)
    assert np.isnan(mk["realized_spread_5000ms"].iloc[0])  # beyond the data: excluded, not guessed
