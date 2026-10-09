"""Effective network parameters returned by the selected evaluator backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkContext:
    backend: str
    bandwidth_source: str
    path_latency_source: str
    frequency_hz: float
    routes: dict[tuple[int, int], tuple[int, ...]]
    link_bandwidths_bits_per_cycle: dict[tuple[int, int], float]
    path_latencies_cycles: dict[tuple[int, int], float]

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "bandwidth_source": self.bandwidth_source,
            "path_latency_source": self.path_latency_source,
            "frequency_hz": self.frequency_hz,
            "links": [
                {"src": src, "dst": dst, "bandwidth_bits_per_cycle": bandwidth}
                for (src, dst), bandwidth in sorted(self.link_bandwidths_bits_per_cycle.items())
            ],
            "paths": [
                {"src": src, "dst": dst, "route": list(route),
                 "latency_cycles": self.path_latencies_cycles[(src, dst)]}
                for (src, dst), route in sorted(self.routes.items())
            ],
        }
