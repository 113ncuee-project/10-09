from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ChipletConfig, NetworkConfig
from .topology import Topology


@dataclass(frozen=True)
class LatencyResult:
    min_cycles: float
    avg_cycles: float
    max_cycles: float
    avg_ns: float


@dataclass(frozen=True)
class ThroughputResult:
    saturation_fps: float
    aggregate_throughput_bits_per_second: float
    bottleneck_link: tuple[int, int] | None
    bottleneck_cycles_per_inference: float


def compute_latency(
    topology: Topology,
    routes: dict[tuple[int, int], tuple[int, ...]],
    traffic_bits_per_inference: dict[tuple[int, int], float],
    chiplet: ChipletConfig,
    network: NetworkConfig,
) -> LatencyResult:
    if not traffic_bits_per_inference:
        return LatencyResult(0.0, 0.0, 0.0, 0.0)

    edge_lengths = topology.edge_lengths()
    node_latency = chiplet.internal_latency_cycles + chiplet.phy_latency_cycles
    relay_latency = chiplet.phy_latency_cycles + chiplet.internal_latency_cycles + chiplet.phy_latency_cycles
    weighted_sum = 0.0
    total_weight = 0.0
    observed: list[float] = []

    for (src, dst), amount in traffic_bits_per_inference.items():
        path = routes[(src, dst)]
        latency = 1.0 + node_latency
        for a, b in zip(path, path[1:]):
            length = edge_lengths[tuple(sorted((a, b)))]
            latency += network.link_latency_base_cycles
            latency += network.link_latency_cycles_per_mm * length
            if b != dst:
                latency += relay_latency
        latency += node_latency + 2.0
        observed.append(latency)
        weighted_sum += latency * amount
        total_weight += amount

    avg_cycles = weighted_sum / total_weight
    avg_ns = avg_cycles / chiplet.frequency_hz * 1e9
    return LatencyResult(
        min_cycles=min(observed),
        avg_cycles=avg_cycles,
        max_cycles=max(observed),
        avg_ns=avg_ns,
    )


def compute_throughput(
    topology: Topology,
    routes: dict[tuple[int, int], tuple[int, ...]],
    traffic_bits_per_inference: dict[tuple[int, int], float],
    chiplet: ChipletConfig,
    network: NetworkConfig,
) -> ThroughputResult:
    if not traffic_bits_per_inference:
        return ThroughputResult(
            saturation_fps=math.inf,
            aggregate_throughput_bits_per_second=math.inf,
            bottleneck_link=None,
            bottleneck_cycles_per_inference=0.0,
        )

    directed_loads = {}
    for edge in topology.edges:
        directed_loads[(edge.a, edge.b)] = 0.0
        directed_loads[(edge.b, edge.a)] = 0.0

    for (src, dst), amount in traffic_bits_per_inference.items():
        path = routes[(src, dst)]
        for a, b in zip(path, path[1:]):
            directed_loads[(a, b)] += amount

    bottleneck_link = None
    bottleneck_cycles = 0.0
    for link, load_bits in directed_loads.items():
        cycles = load_bits / network.link_bandwidth_bits_per_cycle
        if cycles > bottleneck_cycles:
            bottleneck_cycles = cycles
            bottleneck_link = link

    if bottleneck_cycles == 0:
        saturation_fps = math.inf
        aggregate_bps = math.inf
    else:
        saturation_fps = chiplet.frequency_hz / bottleneck_cycles
        aggregate_bps = saturation_fps * sum(traffic_bits_per_inference.values())

    return ThroughputResult(
        saturation_fps=saturation_fps,
        aggregate_throughput_bits_per_second=aggregate_bps,
        bottleneck_link=bottleneck_link,
        bottleneck_cycles_per_inference=bottleneck_cycles,
    )
