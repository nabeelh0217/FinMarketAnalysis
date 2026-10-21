"""Constant-latency model.

    event at venue (t) --market_data--> strategy observes (t + Lmd)
    observation --decision--> order leaves strategy
    order leaves --order_entry--> order/cancel effective at venue
    venue fill/ack --report--> strategy learns of it

This is a deliberately simple, documented approximation; it does not reproduce production latency
distributions (no jitter, no queueing in gateways).
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_market_maker.config import LatencyConfig
from adaptive_market_maker.types import NS_PER_US


@dataclass(frozen=True)
class LatencyModel:
    market_data_ns: int
    decision_ns: int
    order_entry_ns: int
    report_ns: int

    @classmethod
    def from_config(cls, cfg: LatencyConfig) -> LatencyModel:
        return cls(
            market_data_ns=int(round(cfg.market_data_us * NS_PER_US)),
            decision_ns=int(round(cfg.decision_us * NS_PER_US)),
            order_entry_ns=int(round(cfg.order_entry_us * NS_PER_US)),
            report_ns=int(round(cfg.report_us * NS_PER_US)),
        )

    def observe(self, event_ns: int) -> int:
        return event_ns + self.market_data_ns

    def send(self, wake_ns: int) -> int:
        return wake_ns + self.decision_ns

    def arrive(self, sent_ns: int) -> int:
        return sent_ns + self.order_entry_ns

    def report(self, venue_ns: int) -> int:
        return venue_ns + self.report_ns

    @property
    def round_trip_ns(self) -> int:
        return self.market_data_ns + self.decision_ns + self.order_entry_ns
