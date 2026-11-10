"""Mode-2 synthetic market: an order-level limit order book emitting LOBSTER-format output.

SYNTHETIC DATA. Useful for testing execution mechanics and for *planted-signal* validation of the
research pipeline. It is not evidence that any strategy is profitable in a real market.

Model (continuous-time competing Poisson processes, simulated exactly with Gillespie's method):

* Limit orders arrive at rate ``limit_rate``. Side is buy with probability ``0.5 + limit_tilt *
  tanh(g / informed_scale_ticks)`` where ``g = V - mid`` is the gap between a latent fair value V and the
  mid. Price is ``d`` ticks from the *opposite* best with ``P(d) ~ d^-limit_distance_alpha``, d = 1..max_distance.
* Noise market orders arrive at rate ``market_rate``, side 50/50, and sweep the opposite side FIFO.
* Informed market orders arrive at rate ``informed_rate * (1 - exp(-|g| / informed_scale_ticks))`` in the
  direction of ``sign(g)``. They are the source of adverse selection for liquidity providers.
* Each resting order is cancelled at rate ``cancel_rate_per_order`` (partially with prob ``partial_cancel_prob``).
* V follows a Brownian motion (``fair_value_vol`` ticks/sqrt(s)) with Poisson jumps.

In the ``null`` scenario informed flow and limit tilt are switched off, so the only predictability left
is the mechanical kind that any queue-based book has.

Orders resting at the start are *not* written as submissions, exactly like LOBSTER files that start
after the open: their ids first appear when they are cancelled or executed.
"""

from __future__ import annotations

import bisect
import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

from adaptive_market_maker.config import DataConfig, SessionSpec, SyntheticConfig
from adaptive_market_maker.data.base import SessionData
from adaptive_market_maker.data.normalize import compute_origin_seq
from adaptive_market_maker.types import BUY, EMPTY_ASK_PRICE, EMPTY_BID_PRICE, SELL, EventType


@dataclass
class _Order:
    oid: int
    side: int
    price: int  # ticks
    size: int


class SyntheticMarket:
    def __init__(self, cfg: SyntheticConfig, levels: int, lot_size: int, seed: int):
        self.cfg = cfg
        self.levels = levels
        self.lot = lot_size
        self.rng = random.Random(seed)
        self.next_oid = 10_000_000
        # price (ticks) -> deque of orders; sorted price lists for O(log n) best-price queries
        self.queues: dict[int, dict[int, deque[_Order]]] = {BUY: {}, SELL: {}}
        self.level_size: dict[int, dict[int, int]] = {BUY: {}, SELL: {}}
        self.prices: dict[int, list[int]] = {BUY: [], SELL: []}
        self.active: list[_Order] = []
        self.active_pos: dict[int, int] = {}
        # output buffers
        self.out_ts: list[int] = []
        self.out_type: list[int] = []
        self.out_oid: list[int] = []
        self.out_size: list[int] = []
        self.out_price: list[int] = []
        self.out_side: list[int] = []
        self.out_book: list[list[int]] = []
        self._last_ts = 0
        d = np.arange(1, cfg.max_distance + 1, dtype=float)
        w = d ** (-cfg.limit_distance_alpha)
        self._dist_cdf = np.cumsum(w / w.sum()).tolist()

    # ------------------------------------------------------------------ book primitives
    def best(self, side: int) -> int | None:
        p = self.prices[side]
        if not p:
            return None
        return p[-1] if side == BUY else p[0]

    def mid(self) -> float:
        b, a = self.best(BUY), self.best(SELL)
        if b is not None and a is not None:
            return 0.5 * (a + b)
        if b is not None:
            return b + 0.5
        if a is not None:
            return a - 0.5
        return 0.0

    def _add_level(self, side: int, price: int) -> None:
        self.queues[side][price] = deque()
        self.level_size[side][price] = 0
        bisect.insort(self.prices[side], price)

    def _remove_level(self, side: int, price: int) -> None:
        del self.queues[side][price]
        del self.level_size[side][price]
        lst = self.prices[side]
        lst.pop(bisect.bisect_left(lst, price))

    def _activate(self, o: _Order) -> None:
        self.active_pos[o.oid] = len(self.active)
        self.active.append(o)

    def _deactivate(self, o: _Order) -> None:
        idx = self.active_pos.pop(o.oid)
        last = self.active.pop()
        if idx < len(self.active):
            self.active[idx] = last
            self.active_pos[last.oid] = idx

    def _insert(self, o: _Order) -> None:
        if o.price not in self.queues[o.side]:
            self._add_level(o.side, o.price)
        self.queues[o.side][o.price].append(o)
        self.level_size[o.side][o.price] += o.size
        self._activate(o)

    def _reduce(self, o: _Order, qty: int) -> None:
        o.size -= qty
        self.level_size[o.side][o.price] -= qty
        if o.size == 0:
            q = self.queues[o.side][o.price]
            if q and q[0] is o:
                q.popleft()
            else:
                q.remove(o)
            self._deactivate(o)
            if not q:
                self._remove_level(o.side, o.price)

    def snapshot(self) -> list[int]:
        k = self.levels
        row = [0] * (4 * k)
        asks = self.prices[SELL][:k]
        bids = self.prices[BUY][::-1][:k]
        for i in range(k):
            if i < len(asks):
                row[4 * i] = asks[i]
                row[4 * i + 1] = self.level_size[SELL][asks[i]]
            else:
                row[4 * i] = EMPTY_ASK_PRICE
            if i < len(bids):
                row[4 * i + 2] = bids[i]
                row[4 * i + 3] = self.level_size[BUY][bids[i]]
            else:
                row[4 * i + 2] = EMPTY_BID_PRICE
        return row

    def _emit(self, ts: int, etype: int, oid: int, size: int, price: int, side: int) -> None:
        ts = max(ts, self._last_ts)  # timestamps are non-decreasing by construction
        self._last_ts = ts
        self.out_ts.append(ts)
        self.out_type.append(etype)
        self.out_oid.append(oid)
        self.out_size.append(size)
        self.out_price.append(price)
        self.out_side.append(side)
        self.out_book.append(self.snapshot())

    def _new_oid(self) -> int:
        self.next_oid += self.rng.randint(1, 7)
        return self.next_oid

    def _lots(self, mean: float) -> int:
        # geometric number of lots with the given mean (>= 1)
        p = 1.0 / max(mean, 1.0)
        u = self.rng.random()
        n = 1 + int(math.log(1.0 - u) / math.log(1.0 - p)) if p < 1.0 else 1
        return n * self.lot

    # ------------------------------------------------------------------ events
    def initialize(self, mid_ticks: int) -> None:
        c = self.cfg
        for d in range(1, c.init_levels + 1):
            for _ in range(c.init_orders_per_level):
                self._insert(_Order(self._new_oid(), BUY, mid_ticks - d, self._lots(c.mean_limit_lots)))
                self._insert(_Order(self._new_oid(), SELL, mid_ticks + d, self._lots(c.mean_limit_lots)))

    def submit_limit(self, ts: int, side: int, emit: bool = True, distance: int | None = None) -> None:
        if distance is None:
            u = self.rng.random()
            distance = 1 + bisect.bisect_left(self._dist_cdf, u)
        opp = self.best(-side)
        own = self.best(side)
        if opp is None:
            ref = own if own is not None else int(self.mid())
            price = ref - (distance - 1) if side == BUY else ref + (distance - 1)
        else:
            price = opp - distance if side == BUY else opp + distance
        o = _Order(self._new_oid(), side, price, self._lots(self.cfg.mean_limit_lots))
        self._insert(o)
        if emit:
            self._emit(ts, EventType.SUBMIT, o.oid, o.size, price, side)

    def market_order(self, ts: int, aggressor_side: int, qty: int) -> None:
        resting = -aggressor_side
        while qty > 0 and self.prices[resting]:
            price = self.best(resting)
            o = self.queues[resting][price][0]
            fill = min(qty, o.size)
            qty -= fill
            self._reduce(o, fill)
            self._emit(ts, EventType.EXEC_VISIBLE, o.oid, fill, price, resting)

    def cancel_random(self, ts: int) -> None:
        if not self.active:
            return
        o = self.active[self.rng.randrange(len(self.active))]
        oid, price, side = o.oid, o.price, o.side
        n_lots = o.size // self.lot
        if n_lots >= 2 and self.rng.random() < self.cfg.partial_cancel_prob:
            qty = self.lot * self.rng.randint(1, n_lots - 1)
            self._reduce(o, qty)
            self._emit(ts, EventType.CANCEL, oid, qty, price, side)
        else:
            qty = o.size
            self._reduce(o, qty)
            self._emit(ts, EventType.DELETE, oid, qty, price, side)

    def _repair(self, ts: int) -> None:
        """A liquidity provider of last resort refills an emptied side 1 microsecond later."""
        for side in (BUY, SELL):
            if not self.prices[side]:
                for d in (1, 2, 3):
                    self.submit_limit(ts + 1000, side, distance=d)

    # ------------------------------------------------------------------ driver
    def run(self, start_ns: int, duration_s: float) -> None:
        c = self.cfg
        rng = self.rng
        t = 0.0
        fv = self.mid()
        end = duration_s
        while True:
            m = self.mid()
            gap = fv - m
            inf_rate = c.informed_rate * (1.0 - math.exp(-abs(gap) / c.informed_scale_ticks))
            cancel_rate = c.cancel_rate_per_order * len(self.active)
            total = c.limit_rate + c.market_rate + inf_rate + cancel_rate
            dt = rng.expovariate(total)
            t += dt
            if t >= end:
                break
            # latent value evolves over the waiting time
            fv += c.fair_value_vol * math.sqrt(dt) * rng.gauss(0.0, 1.0)
            if c.jump_rate > 0 and rng.random() < 1.0 - math.exp(-c.jump_rate * dt):
                fv += c.jump_size_ticks * (1 if rng.random() < 0.5 else -1)
            ts = max(start_ns + int(t * 1e9), self._last_ts)
            u = rng.random() * total
            if u < c.limit_rate:
                p_buy = 0.5 + c.limit_tilt * math.tanh(gap / c.informed_scale_ticks)
                self.submit_limit(ts, BUY if rng.random() < p_buy else SELL)
            elif u < c.limit_rate + c.market_rate:
                self.market_order(ts, BUY if rng.random() < 0.5 else SELL, self._lots(c.mean_market_lots))
            elif u < c.limit_rate + c.market_rate + inf_rate:
                self.market_order(ts, BUY if gap > 0 else SELL, self._lots(c.mean_market_lots))
            else:
                self.cancel_random(ts)
            self._repair(ts)

    def to_session(self, spec: SessionSpec, data_cfg: DataConfig) -> SessionData:
        book = np.asarray(self.out_book, dtype=np.int64).reshape(-1, 4 * self.levels)
        tick = data_cfg.tick_size
        et_all = np.asarray(self.out_type, dtype=np.int8)
        # LOBSTER only publishes messages that affect the requested depth; mimic that so order-id
        # coverage (and therefore queue tracking) is no better here than with the real files.
        keep = np.ones(book.shape[0], dtype=bool)
        keep[1:] = np.any(book[1:] != book[:-1], axis=1) | np.isin(et_all[1:], [4, 5, 6, 7])
        book = book[keep]

        def px(a: np.ndarray, empty: int) -> np.ndarray:
            return np.where(a == empty, empty, a * tick)

        et = et_all[keep]
        oid = np.asarray(self.out_oid, dtype=np.int64)[keep]
        origin, _ = compute_origin_seq(et, oid)
        return SessionData(
            session_id=spec.session_id,
            ticker=spec.ticker,
            date=spec.date,
            source="synthetic",
            levels=self.levels,
            tick_size=tick,
            price_scale=data_cfg.price_scale,
            lot_size=data_cfg.lot_size,
            open_ns=data_cfg.open_ns,
            close_ns=data_cfg.close_ns,
            ts=np.asarray(self.out_ts, dtype=np.int64)[keep],
            event_type=et,
            order_id=oid,
            size=np.asarray(self.out_size, dtype=np.int64)[keep],
            price=np.asarray(self.out_price, dtype=np.int64)[keep] * tick,
            side=np.asarray(self.out_side, dtype=np.int8)[keep],
            origin_seq=origin,
            ask_px=px(book[:, 0::4], EMPTY_ASK_PRICE),
            ask_sz=book[:, 1::4].copy(),
            bid_px=px(book[:, 2::4], EMPTY_BID_PRICE),
            bid_sz=book[:, 3::4].copy(),
            meta={
                "synthetic_scenario": self.cfg.scenario,
                "messages_dropped_outside_depth": int((~keep).sum()),
            },
        )


def synthetic_session_specs(data_cfg: DataConfig) -> list[SessionSpec]:
    import datetime as dt

    syn = data_cfg.synthetic
    assert syn is not None
    day = dt.date.fromisoformat(syn.first_date)
    start_ms = data_cfg.open_ns // 1_000_000
    end_ms = start_ms + int(syn.duration_s * 1000)
    specs = []
    while len(specs) < syn.n_sessions:
        if day.weekday() < 5:
            specs.append(
                SessionSpec(ticker=syn.ticker, date=day.isoformat(), start_ms=start_ms, end_ms=end_ms)
            )
        day += dt.timedelta(days=1)
    return specs


def generate_session(data_cfg: DataConfig, spec: SessionSpec, index: int) -> SessionData:
    syn = data_cfg.synthetic
    assert syn is not None
    mkt = SyntheticMarket(syn, data_cfg.levels, data_cfg.lot_size, seed=syn.seed * 1000 + index)
    mkt.initialize(syn.initial_mid_ticks)
    mkt.run(start_ns=data_cfg.open_ns, duration_s=syn.duration_s)
    return mkt.to_session(spec, data_cfg)
