"""Serial Batch=1 compute + canonical-flow communication service.

Paths, per-flow latencies and bandwidths come from the selected network
backend. This aggregates serialization and path delay, not queues or overlap.
Compute uses one global hardware rate; no mapping-specific efficiency.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .config import EvalConfig
from .mapping import MappingPlan
from .model import ModelSpec
from .network_context import NetworkContext
from .topology import Topology
from .workload import WorkloadPartition


@dataclass(frozen=True)
class E2ELatencyResult:
    feasible: bool
    compute_latency_sum_s: float
    communication_serialization_sum_s: float
    communication_path_latency_sum_s: float
    communication_latency_sum_s: float
    estimated_e2e_latency_s: float
    estimated_e2e_latency_ns: float
    group_compute_times_s: tuple[float, ...]
    communication_event_times_s: tuple[float, ...]
    boundary_timings: tuple[dict[str, object], ...]
    path_latency_source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_batch1_e2e_latency(
    model: ModelSpec,
    workload: WorkloadPartition,
    mapping_plan: MappingPlan,
    topology: Topology,
    cfg: EvalConfig,
    network_context: NetworkContext,
) -> E2ELatencyResult:
    if not workload.group_specs:
        raise ValueError("E2E estimation requires explicit group_specs")
    if len(mapping_plan.groups) != len(workload.group_specs):
        raise ValueError("mapping plan and workload group count do not match")
    if not math.isfinite(network_context.frequency_hz) or network_context.frequency_hz <= 0:
        raise ValueError("network frequency must be finite and positive")
    compute_times = tuple(
        _group_compute_time_s(model, group, mapping_group, cfg)
        for group, mapping_group in zip(workload.group_specs, mapping_plan.groups)
    )
    events: dict[tuple[str, int, int, str], list[int]] = {}
    for index, flow in enumerate(workload.flows):
        if flow.src >= topology.node_count or flow.dst >= topology.node_count:
            raise ValueError("flow endpoint is outside topology")
        events.setdefault(flow.event_key, []).append(index)
    timings = tuple(
        _communication_event(key, indices, workload, network_context)
        for key, indices in sorted(events.items())
    )
    event_times = tuple(float(event["service_s"]) for event in timings)
    compute_sum = sum(compute_times)
    communication_sum = sum(event_times)
    total = compute_sum + communication_sum
    return E2ELatencyResult(
        feasible=mapping_plan.feasible,
        compute_latency_sum_s=compute_sum,
        communication_serialization_sum_s=sum(float(event["serialization_s"]) for event in timings),
        communication_path_latency_sum_s=sum(float(event["path_latency_s"]) for event in timings),
        communication_latency_sum_s=communication_sum,
        estimated_e2e_latency_s=total,
        estimated_e2e_latency_ns=total * 1e9,
        group_compute_times_s=compute_times,
        communication_event_times_s=event_times,
        boundary_timings=timings,
        path_latency_source=network_context.path_latency_source,
    )


def _group_compute_time_s(model, group, mapping_group, cfg: EvalConfig) -> float:
    start, end, parts = group
    group_ops = sum(model.blocks[index].macs_g for index in range(start, end)) * 1e9 * cfg.chiplet.op_per_mac
    base_rate = cfg.chiplet.peak_ops_per_second * cfg.chiplet.utilization
    return (group_ops / parts) / max(base_rate, 1e-12)


def _communication_event(key, flow_indices, workload, context: NetworkContext):
    loads: dict[tuple[int, int], float] = {}
    max_path_cycles = 0.0
    max_path: tuple[int, ...] = ()
    total_bytes = 0.0
    for index in flow_indices:
        flow = workload.flows[index]
        pair = flow.src, flow.dst
        route = context.routes[pair]
        path_cycles = context.path_latencies_cycles[pair]
        if path_cycles >= max_path_cycles:
            max_path_cycles, max_path = path_cycles, route
        for link in zip(route, route[1:]):
            loads[link] = loads.get(link, 0.0) + flow.bytes * 8
        total_bytes += flow.bytes
    cycles_by_link = {}
    for link, bits in loads.items():
        bandwidth = context.link_bandwidths_bits_per_cycle[link]
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("backend link bandwidth must be finite and positive")
        cycles_by_link[link] = bits / bandwidth
    bottleneck = max(sorted(cycles_by_link), key=cycles_by_link.get) if cycles_by_link else None
    serialization_s = cycles_by_link[bottleneck] / context.frequency_hz if bottleneck else 0.0
    path_latency_s = max_path_cycles / context.frequency_hz
    return {
        "kind": key[0],
        "event_id": key[3],
        "source_group_index": key[1],
        "destination_group_index": key[2],
        "flow_indices": list(flow_indices),
        "tensor_ids": sorted({workload.flows[index].tensor_id for index in flow_indices}),
        "traffic_bytes": total_bytes,
        "traffic_bits": total_bytes * 8,
        "routed_traffic_bits": total_bytes * 8,
        "pair_count": len({(workload.flows[index].src, workload.flows[index].dst) for index in flow_indices}),
        "bottleneck_link": list(bottleneck) if bottleneck else None,
        "bottleneck_link_bits": loads[bottleneck] if bottleneck else 0.0,
        "bottleneck_bandwidth_bits_per_cycle": context.link_bandwidths_bits_per_cycle[bottleneck] if bottleneck else None,
        "max_path": list(max_path),
        "max_path_cycles": max_path_cycles,
        "bandwidth_source": context.bandwidth_source,
        "path_latency_source": context.path_latency_source,
        "serialization_s": serialization_s,
        "path_latency_s": path_latency_s,
        "service_s": serialization_s + path_latency_s,
    }
