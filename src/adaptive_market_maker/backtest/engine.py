"""Deterministic event-driven historical replay.

Event ordering (tie-breaking is explicit and tested):

1. Historical messages are processed in file order.
2. Agent events (order/cancel arrivals at the venue, fill/ack reports to the strategy, strategy wake-ups)
   live in a priority queue keyed by (time, priority, sequence). An agent event at time t is processed
   before historical message i iff t < ts_i: on exact ties the *historical* message goes first, the
   conservative choice for a simulated order racing real flow.
3. Among agent events at the same time: venue arrivals, then reports, then wake-ups.

Latency: a historical state with timestamp t is observable from t + market_data; the strategy decides on
the newest observable decision state, sends at wake + decision, and its messages take effect at the venue
order_entry later. Fills and acks reach the strategy report later.

Quoting stops ``flatten_before_end_s`` before the segment end; remaining inventory is then closed with a
marketable IOC order (taker fee). Whatever is still open at the end is valued at the liquidation price
(walking visible depth) for the primary metric.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from adaptive_market_maker.accounting.ledger import MICRO, Ledger
from adaptive_market_maker.config import ProjectConfig
from adaptive_market_maker.data.base import SessionData
from adaptive_market_maker.execution.latency import LatencyModel
from adaptive_market_maker.execution.orders import Order, OrderState
from adaptive_market_maker.execution.simulator import Fill, SimulatedExchange
from adaptive_market_maker.strategies.base import MarketView, Signals, Strategy, StrategyState
from adaptive_market_maker.strategies.risk import RiskManager
from adaptive_market_maker.types import BUY, NS_PER_MS, NS_PER_S, SELL, EventType

_ARRIVE, _CANCEL, _REPORT_FILL, _REPORT_DONE, _WAKE = range(5)
_PRIORITY = {_ARRIVE: 0, _CANCEL: 0, _REPORT_FILL: 1, _REPORT_DONE: 1, _WAKE: 2}


@dataclass
class BacktestResult:
    strategy: str
    session_id: str
    start_ns: int
    end_ns: int
    fills: pd.DataFrame
    equity: pd.DataFrame
    decisions: pd.DataFrame
    summary: dict
    assumptions: dict = field(default_factory=dict)


class BacktestEngine:
    def __init__(
        self,
        session: SessionData,
        strategy: Strategy,
        cfg: ProjectConfig,
        signals: Signals,
        start_ns: int,
        end_ns: int,
    ):
        self.s = session
        self.strategy = strategy
        self.cfg = cfg
        self.sig = signals
        self.start_ns, self.end_ns = int(start_ns), int(end_ns)
        self.quote_end_ns = self.end_ns - int(cfg.backtest.flatten_before_end_s * NS_PER_S)
        self.lat = LatencyModel.from_config(cfg.execution.latency)
        self.ex = SimulatedExchange(session, cfg.execution)
        self.ledger = Ledger(session.price_scale, cfg.execution.fees)
        self.risk = RiskManager(cfg.strategies.risk, session.price_scale)
        self.min_wake_gap = int(cfg.backtest.decision_min_interval_ms * NS_PER_MS)
        self.heap: list = []
        self._seq = 0
        self._next_oid = 1
        self.orders: dict[int, Order] = {}
        self.live: dict[int, list[Order]] = {BUY: [], SELL: []}  # strategy's view: not known terminal
        self.known_position = 0
        self.wake_pending = False
        self.last_wake = -(10**18)
        self.flatten_order: Order | None = None
        self.counts = {"orders_sent": 0, "cancels_sent": 0, "decisions": 0, "rate_limited": 0}
        self.fill_rows: list[tuple] = []
        self.eq_rows: list[tuple] = []
        self.dec_rows: list[tuple] = []
        elig = signals.eligible if signals.eligible is not None else np.ones(signals.ts_ns.size, bool)
        elig = elig & (signals.ts_ns >= self.start_ns) & (signals.ts_ns < self.end_ns)
        self.elig_k = np.nonzero(elig)[0]
        self.elig_ts = signals.ts_ns[self.elig_k]
        self.k_of_row = np.full(len(session), -1, dtype=np.int64)
        self.k_of_row[signals.rows[self.elig_k]] = self.elig_k

    # ------------------------------------------------------------------ scheduling
    def _push(self, t: int, kind: int, payload) -> None:
        self._seq += 1
        heapq.heappush(self.heap, (t, _PRIORITY[kind], self._seq, kind, payload))

    def _process_until(self, t_limit: int) -> None:
        """Process agent events with time strictly before ``t_limit``."""
        while self.heap and self.heap[0][0] < t_limit:
            t, _, _, kind, payload = heapq.heappop(self.heap)
            if kind == _ARRIVE:
                for f in self.ex.on_order_arrival(payload, t):
                    self._on_fill(f)
                if payload.is_terminal:
                    self._push(self.lat.report(t), _REPORT_DONE, payload)
            elif kind == _CANCEL:
                was_terminal = payload.is_terminal
                self.ex.on_cancel_arrival(payload, t)
                if not was_terminal:
                    self._push(self.lat.report(t), _REPORT_DONE, payload)
            elif kind == _REPORT_FILL:
                f: Fill = payload
                o = self.orders[f.order_id]
                o.known_filled += f.qty
                self.known_position += f.side * f.qty
                if o.known_filled >= o.qty:
                    self._mark_known_terminal(o)
            elif kind == _REPORT_DONE:
                self._mark_known_terminal(payload)
            elif kind == _WAKE:
                self._on_wake(t)

    def _mark_known_terminal(self, o: Order) -> None:
        if not o.known_terminal:
            o.known_terminal = True
            lst = self.live[o.side]
            if o in lst:
                lst.remove(o)
            if o is self.flatten_order:
                self.flatten_order = None

    # ------------------------------------------------------------------ fills
    def _on_fill(self, f: Fill) -> None:
        self.ledger.apply_fill(f)
        row = max(self.ex.last_row - (1 if f.row >= 0 else 0), 0)
        mid_before = float(self.s.mid_ticks[row]) if math.isfinite(self.s.mid_ticks[row]) else math.nan
        o = self.orders[f.order_id]
        self.fill_rows.append(
            (
                f.ts_ns,
                f.order_id,
                f.side,
                f.price,
                f.qty,
                f.liquidity,
                f.reason,
                f.row,
                mid_before,
                o.tag,
                (f.ts_ns - (o.arrival_ns or f.ts_ns)) / NS_PER_S,
            )
        )
        self._push(self.lat.report(f.ts_ns), _REPORT_FILL, f)

    # ------------------------------------------------------------------ order actions
    def _send_new(
        self, side: int, price: int, qty: int, t_send: int, ioc: bool = False, tag: str = "quote"
    ) -> Order:
        o = Order(self._next_oid, side, price, qty, t_send, post_only=not ioc, ioc=ioc, tag=tag)
        self._next_oid += 1
        self.orders[o.order_id] = o
        self.live[side].append(o)
        self._push(self.lat.arrive(t_send), _ARRIVE, o)
        self.counts["orders_sent"] += 1
        return o

    def _send_cancel(self, o: Order, t_send: int) -> None:
        if o.cancel_sent_ns is not None or o.known_terminal:
            return
        o.cancel_sent_ns = t_send
        if not o.is_terminal:
            o.transition(OrderState.PENDING_CANCEL, t_send)
        self._push(self.lat.arrive(t_send), _CANCEL, o)
        self.counts["cancels_sent"] += 1

    def _cancel_all(self, t_send: int) -> int:
        n = 0
        for side in (BUY, SELL):
            for o in list(self.live[side]):
                if o.cancel_sent_ns is None and o.tag == "quote":
                    self._send_cancel(o, t_send)
                    n += 1
        return n

    # ------------------------------------------------------------------ decisions
    def _on_wake(self, t_w: int) -> None:
        self.wake_pending = False
        self.last_wake = t_w
        j = int(np.searchsorted(self.elig_ts, t_w - self.lat.market_data_ns, side="right")) - 1
        if j < 0:
            return
        k = int(self.elig_k[j])
        t_send = self.lat.send(t_w)
        self.counts["decisions"] += 1
        if t_w >= self.quote_end_ns or self.risk.halted_reason is not None:
            self._cancel_all(t_send)
            if t_w >= self.quote_end_ns and self.known_position != 0 and self.flatten_order is None:
                self._send_flatten(k, t_send)
            return
        view = MarketView.build(self.s, self.sig, k, t_w, self.quote_end_ns)
        live_b = [o for o in self.live[BUY] if o.cancel_sent_ns is None and o.tag == "quote"]
        live_a = [o for o in self.live[SELL] if o.cancel_sent_ns is None and o.tag == "quote"]
        state = StrategyState(
            self.known_position,
            sum(o.known_remaining for o in live_b),
            sum(o.known_remaining for o in live_a),
        )
        intent = self.strategy.quote(view, state)
        thr = self.strategy.p.requote_threshold_ticks * view.tick if hasattr(self.strategy, "p") else 0
        max_age = self.cfg.strategies.risk.max_quote_age_s

        def matches(o: Order, px: int | None, qty: int) -> bool:
            if px is None or abs(o.price - px) > thr or o.known_remaining > qty:
                return False
            return not (max_age is not None and t_w - o.sent_ns > max_age * NS_PER_S)

        pending = {
            side: sum(o.known_remaining for o in self.live[side] if o.cancel_sent_ns is not None)
            for side in (BUY, SELL)
        }
        # pass 1: everything that does not match the pre-risk intent will be cancelled
        keep_b = [o for o in live_b if matches(o, intent.bid_px, intent.bid_qty)][:1]
        keep_a = [o for o in live_a if matches(o, intent.ask_px, intent.ask_qty)][:1]
        going_b = sum(o.known_remaining for o in live_b if o not in keep_b)
        going_a = sum(o.known_remaining for o in live_a if o not in keep_a)
        data_age = t_w - int(self.sig.ts_ns[k])
        intent = self.risk.check(
            intent, view, self.known_position, pending[BUY] + going_b, pending[SELL] + going_a, data_age
        )
        # pass 2: a kept order that no longer matches the risk-adjusted intent is cancelled, and no new order
        # is placed on that side this round (keeps the worst-case inventory bound exact)
        actions: list[tuple] = []
        for side, live, keep, px, qty in (
            (BUY, live_b, keep_b, intent.bid_px, intent.bid_qty),
            (SELL, live_a, keep_a, intent.ask_px, intent.ask_qty),
        ):
            still = [o for o in keep if matches(o, px, qty)]
            for o in live:
                if o not in still:
                    actions.append(("cancel", o))
            if not still and px is not None and qty > 0 and len(keep) == len(still):
                actions.append(("new", side, px, qty))
        n_new = sum(1 for a in actions if a[0] == "new")
        allowed = self.risk.allow_actions(t_send, len(actions)) if actions else True
        for a in actions:
            if a[0] == "cancel":
                self._send_cancel(a[1], t_send)  # cancels always go out: they reduce risk
            elif allowed:
                self._send_new(a[1], a[2], a[3], t_send)
        if actions and not allowed and n_new:
            self.counts["rate_limited"] += 1
        self.dec_rows.append(
            (
                t_w,
                k,
                view.mid_ticks,
                intent.alpha_ticks,
                intent.reservation_ticks,
                intent.half_spread_ticks,
                intent.skew_ticks,
                intent.bid_px or 0,
                intent.ask_px or 0,
                self.known_position,
                len(actions),
                intent.note,
            )
        )

    def _send_flatten(self, k: int, t_send: int) -> None:
        row = int(self.sig.rows[k])
        q = self.known_position
        dev = self.cfg.strategies.risk.max_price_deviation_ticks * self.s.tick_size
        if q > 0:
            px = int(self.s.bid_px[row, 0]) - dev
            self.flatten_order = self._send_new(SELL, px, q, t_send, ioc=True, tag="flatten")
        else:
            px = int(self.s.ask_px[row, 0]) + dev
            self.flatten_order = self._send_new(BUY, px, -q, t_send, ioc=True, tag="flatten")

    # ------------------------------------------------------------------ main loop
    def _snapshot(self, t: int, row: int) -> None:
        mid = self.s.mid_ticks[row]
        if not math.isfinite(mid):
            mid = self.eq_rows[-1][3] if self.eq_rows else math.nan
        mark_units = mid * self.s.tick_size
        eq = self.ledger.equity_micro(mark_units) / MICRO if math.isfinite(mark_units) else math.nan
        self.eq_rows.append(
            (
                t,
                self.ledger.cash_micro / MICRO,
                self.ledger.position,
                mid,
                eq,
                self.ledger.fees_micro / MICRO,
                self.ledger.shares_traded,
                len(self.ledger.fills),
            )
        )
        if (
            math.isfinite(eq)
            and eq < -self.cfg.strategies.risk.max_session_loss
            and self.risk.halted_reason is None
        ):
            self.risk.kill("max_session_loss")

    def run(self) -> BacktestResult:
        s = self.s
        ts = s.ts
        i0 = int(np.searchsorted(ts, self.start_ns, side="left"))
        i1 = int(np.searchsorted(ts, self.end_ns, side="left"))
        if i1 <= i0 + 1:
            raise ValueError("backtest window contains no events")
        self.ex.last_row = max(i0 - 1, 0)
        interval = int(self.cfg.backtest.equity_interval_s * NS_PER_S)
        next_snap = self.start_ns
        md = self.lat.market_data_ns
        k_of_row = self.k_of_row
        et = s.event_type
        for i in range(i0, i1):
            t = int(ts[i])
            if self.heap and self.heap[0][0] < t:
                self._process_until(t)
            while t >= next_snap:
                self._snapshot(next_snap, max(i - 1, 0))
                next_snap += interval
            was_halted = self.ex.halted
            for f in self.ex.on_market_event(i, t):
                self._on_fill(f)
            if et[i] == EventType.HALT and self.ex.halted and not was_halted:
                for side in (BUY, SELL):
                    for o in list(self.live[side]):
                        if o.is_terminal:
                            self._push(self.lat.report(t), _REPORT_DONE, o)
            if k_of_row[i] >= 0 and not self.wake_pending:
                self._push(max(t + md, self.last_wake + self.min_wake_gap), _WAKE, None)
                self.wake_pending = True
        self._process_until(self.end_ns)
        last = i1 - 1
        self._snapshot(self.end_ns, last)
        return self._result(last)

    def _result(self, last_row: int) -> BacktestResult:
        s, led = self.s, self.ledger
        mid = float(s.mid_ticks[last_row])
        mark_units = mid * s.tick_size
        led.reconcile(mark_units)  # accounting must reconcile or the run fails
        liq_cash, beyond = led.liquidation_value_micro(
            s.bid_px[last_row], s.bid_sz[last_row], s.ask_px[last_row], s.ask_sz[last_row]
        )
        fills = pd.DataFrame(
            self.fill_rows,
            columns=[
                "ts_ns",
                "order_id",
                "side",
                "price",
                "qty",
                "liquidity",
                "reason",
                "row",
                "mid_before_ticks",
                "tag",
                "age_s",
            ],
        )
        equity = pd.DataFrame(
            self.eq_rows,
            columns=[
                "ts_ns",
                "cash",
                "position",
                "mid_ticks",
                "equity_mid",
                "fees",
                "shares_traded",
                "n_fills",
            ],
        )
        decisions = pd.DataFrame(
            self.dec_rows,
            columns=[
                "ts_ns",
                "k",
                "mid_ticks",
                "alpha_ticks",
                "reservation_ticks",
                "half_spread_ticks",
                "skew_ticks",
                "bid_px",
                "ask_px",
                "known_position",
                "n_actions",
                "note",
            ],
        )
        states = pd.Series([o.state.value for o in self.orders.values()]).value_counts().to_dict()
        summary = {
            "net_pnl_liquidation": (led.cash_micro + liq_cash) / MICRO,
            "net_pnl_mid": led.equity_micro(mark_units) / MICRO,
            "fees_net": led.fees_micro / MICRO,
            "rebates": led.rebates_micro / MICRO,
            "final_position": led.position,
            "liquidation_shares_beyond_depth": beyond,
            "shares_traded": led.shares_traded,
            "notional_traded": led.notional_micro / MICRO,
            "n_fills": len(led.fills),
            "order_states": states,
            "exchange_rejects": self.ex.n_rejects,
            "risk_blocks": dict(self.risk.blocks),
            "kill_switch": self.risk.halted_reason,
            **self.counts,
            "duration_s": (self.end_ns - self.start_ns) / NS_PER_S,
        }
        assumptions = {
            "data_source": s.source,
            "session": s.session_id,
            "execution_mode": "historical_replay" if s.source != "synthetic" else "synthetic_market_replay",
            "fill_preset": self.cfg.execution.fill_preset,
            "latency": self.cfg.execution.latency.model_dump(),
            "fees": self.cfg.execution.fees.model_dump(),
            "post_only": self.cfg.execution.post_only,
            "tick_size_units": s.tick_size,
            "strategy": self.strategy.describe(),
            "risk": self.cfg.strategies.risk.model_dump(),
            "flatten_before_end_s": self.cfg.backtest.flatten_before_end_s,
        }
        return BacktestResult(
            self.strategy.name,
            s.session_id,
            self.start_ns,
            self.end_ns,
            fills,
            equity,
            decisions,
            summary,
            assumptions,
        )


def run_backtest(session, strategy, cfg, signals, start_ns, end_ns) -> BacktestResult:
    return BacktestEngine(session, strategy, cfg, signals, start_ns, end_ns).run()
