"""Actual K shards and canonical flows, one authoritative serial timeline."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace

from .e2e_latency import _communication_event
from .mapping import build_physical_mapping
from .network_diagnostics import link_diagnostics
from .rapidchiplet_engine import evaluate_with_rapidchiplet
from .topology import make_mesh
from .traffic_model import derive_operator_workload


ASSUMPTIONS = {
    "latency_policy": "serial: FX dependency order, communication before each operator; residual branches also serial",
    "compute_formula": "max(shard MACs * op_per_mac) / (PE rows * PE cols * frequency * ops/PE/cycle * global utilization)",
    "compute_coverage": "Conv2d/Linear MACs only; BN/ReLU fused; Add/pool/flatten MAC cost zero in this baseline",
    "communication_formula": "per consumer and kind: max(link bits / native BW) / frequency + max(native per-pair path cycles) / frequency",
    "communication_service": "store-and-forward-free analytical link bottleneck service; no queues, arbitration, overlap, or multicast replication",
    "tensor_residency": "retain received versioned slices until last graph consumer; no eviction; reject SRAM overflow",
    "sram_accounting": "all live logical activation versions + current operator weight shards, including output before input retirement; conservative fused-version allocation",
    "external_memory": "EXTERNAL image and static weights have byte bookkeeping; fetch latency and energy excluded, no fake C0 traffic",
    "power": "official fixed chiplet/packaging power includes all N_total chiplets and physical links; no idle gating or mapping-sensitive dynamic energy model",
    "assignment_order": "ordered chiplet IDs assign ascending integer K slices; permutations can change physical ownership",
    "precision": "activation FP32, weight FP32; configuration validates four bytes per element",
    "normalization": "fixed per-model serial single-chip compute and full configured mesh native area/power; hard limits replace respective soft scales",
    "RL_initialization": "first min(budget, max_chiplets) episodes cover uniform maximum legal active count at each N_total; terminal Q updates use the same L/A/P reward",
}


@dataclass(frozen=True)
class PhysicalEvaluation:
    metrics: dict
    derived: object
    network_context: object
    operator_timings: tuple[dict, ...]
    communication_events: tuple[dict, ...]
    mapping_plan: dict

    def to_dict(self):
        return {**self.metrics, "mapping_plan": self.mapping_plan,
                "canonical_flows": [flow.to_dict() for flow in self.derived.workload.flows],
                "operator_timings": self.operator_timings,
                "communication_events": self.communication_events,
                "ownership_trace": self.derived.ownership_trace,
                "peak_sram_bytes": self.derived.peak_sram_bytes,
                "final_residency": self.derived.final_residency,
                "network_context": self.network_context.to_dict()}


def evaluate_physical(graph, total_chiplets, assignments, cfg, *, prefix=False):
    if not 1 <= total_chiplets <= cfg.max_chiplets:
        raise ValueError("N_total outside configured hardware limit")
    if prefix:
        # Build exact full plan once, then evaluate only selected blocks. The
        # original graph supplies future consumers for residency/liveness.
        complete = tuple(assignments) + ((0,),) * (len(graph.blocks)-len(assignments))
        plan = build_physical_mapping(graph, total_chiplets, complete)
        plan = replace(plan, block_mappings=plan.block_mappings[:len(assignments)])
    else:
        plan = build_physical_mapping(graph, total_chiplets, assignments)
    derived = derive_operator_workload(graph, plan, cfg)
    topology = make_mesh(total_chiplets, cfg.chiplet.width_mm, cfg.chiplet.spacing_mm)
    rapid = evaluate_with_rapidchiplet(topology, derived.workload, cfg, f"{graph.model_name}_physical_{total_chiplets}")
    context = rapid.network_context
    events_by_op = defaultdict(dict)
    for index, flow in enumerate(derived.workload.flows):
        # Inputs from different producer blocks feed the same ready event.
        # Group by consumer and kind, not by producer: all same-kind loads
        # contend in one service interval before the operator can compute.
        key = flow.kind, -1, flow.destination_group_index, flow.consumer_operator
        events_by_op[flow.consumer_operator].setdefault(key, []).append(index)
    rate = cfg.chiplet.peak_ops_per_second * cfg.chiplet.utilization
    if rate <= 0:
        raise ValueError("global compute rate must be positive")
    time_ns = compute_ns = network_ns = transition_ns = serialization_ns = path_ns = 0.0
    timings, events = [], []
    for work in derived.operator_workloads:
        start = time_ns
        for key, indices in events_by_op[work["operator_id"]].items():
            event = _communication_event(key, indices, derived.workload, context)
            event["source_group_indices"] = sorted({derived.workload.flows[i].source_group_index for i in indices})
            event["start_ns"] = time_ns
            service = event["service_s"] * 1e9
            time_ns += service
            event["finish_ns"] = time_ns
            serialization_ns += event["serialization_s"] * 1e9
            path_ns += event["path_latency_s"] * 1e9
            if key[0] == "intra_block":
                network_ns += service
            else:
                transition_ns += service
            events.append(event)
        compute_start = time_ns
        duration = max(work["ops_per_chiplet"], default=0) / rate * 1e9
        compute_ns += duration
        time_ns += duration
        timings.append({**work, "start_ns":start, "compute_start_ns":compute_start,
                        "compute_latency_ns":duration, "communication_latency_ns":compute_start-start,
                        "finish_ns":time_ns})
    counts = [block.active_chiplet_count for block in plan.block_mappings]
    diagnostics = link_diagnostics(derived.workload.flows, context, time_ns)
    metrics = dict(model=graph.model_name, topology="mesh", N_total=total_chiplets,
                   active_chiplet_counts=counts, assignments=[list(block.active_chiplets) for block in plan.block_mappings],
                   compute_latency_ns=compute_ns, network_latency_ns=network_ns,
                   transition_latency_ns=transition_ns, total_latency_ns=time_ns,
                   communication_serialization_ns=serialization_ns, communication_path_latency_ns=path_ns,
                   achieved_fps=1e9/time_ns if time_ns > 0 else None,
                   total_area_mm2=rapid.total_area_mm2, total_power_w=rapid.total_power_w,
                   total_chiplet_power_w=rapid.total_chiplet_power_w, total_link_power_w=rapid.total_link_power_w,
                   total_traffic_bytes=sum(flow.bytes for flow in derived.workload.flows),
                   intra_block_traffic_bytes=sum(flow.bytes for flow in derived.workload.flows if flow.kind == "intra_block"),
                   transition_traffic_bytes=sum(flow.bytes for flow in derived.workload.flows if flow.kind == "transition"),
                   flow_count=len(derived.workload.flows), architecture_feasible=not derived.violations,
                   architecture_violations=derived.violations, latency_policy="serial", backend=context.backend,
                   rapid_avg_ici_latency_ns=rapid.avg_latency_ns, connection_graph=rapid.graph,
                   external_input_bytes=graph.tensor_map[graph.input_id].bytes, weight_bytes=graph.weight_bytes,
                   model_fingerprint=graph.fingerprint, **diagnostics)
    mapping_dict = plan.to_dict()
    mapping_dict.update(feasible=not derived.violations, violations=derived.violations)
    owner_lookup = {row["tensor_id"]:row["owners"] for row in derived.ownership_trace}
    for block, row in zip(graph.blocks, mapping_dict["blocks"]):
        row["input_sources"] = {name:("EXTERNAL" if graph.tensor_map[name].source == "EXTERNAL" else owner_lookup[name]) for name in block.inputs}
        row["output_owners"] = {name:owner_lookup[name] for name in block.outputs}
    return PhysicalEvaluation(metrics, derived, context, tuple(timings), tuple(events), mapping_dict)
