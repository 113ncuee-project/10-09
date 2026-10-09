from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

from .config import EvalConfig
from .dag_scheduler import schedule_block_dag
from .e2e_latency import estimate_batch1_e2e_latency
from .mapping import build_mapping_plan
from .model import ModelSpec
from .rapidchiplet_engine import evaluate_with_rapidchiplet
from .topology import Topology, make_topology
from .workload import DpCostMode, WorkloadPartition, brute_force_pipeline_workloads, pareto_dp_pipeline_workloads


@dataclass(frozen=True)
class PpaWeights:
    latency: float
    area: float
    power: float


@dataclass(frozen=True)
class PpaReference:
    max_latency_ns: float
    max_area_mm2: float
    max_power_w: float


PPA_WEIGHT_PRESETS: dict[str, PpaWeights] = {
    "balanced": PpaWeights(latency=1.0, area=1.0, power=1.0),
    "latency": PpaWeights(latency=0.8, area=0.1, power=0.1),
    "area": PpaWeights(latency=0.1, area=0.8, power=0.1),
    "power": PpaWeights(latency=0.1, area=0.1, power=0.8),
}


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    dataset: str
    topology: str
    available_chiplets: int
    selected_chiplets: int
    unused_chiplets: int
    peak_tops_per_chiplet: float
    achieved_fps: float
    avg_latency_ns: float
    min_latency_cycles: float
    avg_latency_cycles: float
    max_latency_cycles: float
    aggregate_throughput_bits_per_cycle: float
    total_area_mm2: float
    total_chiplet_area_mm2: float
    chip_width_mm: float
    chip_height_mm: float
    total_power_w: float
    total_chiplet_power_w: float
    total_interposer_power_w: float
    total_link_power_w: float
    performance_per_w: float
    ppa_goal: str
    ppa_score: float
    ppa_latency_weight: float
    ppa_area_weight: float
    ppa_power_weight: float
    total_traffic_mb_per_inference: float
    edge_count: int
    link_count: int
    min_link_length_mm: float
    avg_link_length_mm: float
    max_link_length_mm: float
    bottleneck_link: str
    workload_search: str
    workload_plan: str
    connection_graph: dict[str, object]
    block_graph: dict[str, object] = field(default_factory=dict)
    architecture_feasible: bool = True
    architecture_violations: tuple[str, ...] = ()
    mapping_plan: dict[str, object] = field(default_factory=dict)
    schedule: dict[str, object] = field(default_factory=dict)
    schedule_latency_ns: float = 0.0
    # ``avg_latency_ns`` is now the canonical Batch=1 E2E latency.  Rapid's
    # traffic-weighted ICI latency is kept separately for diagnostics.
    e2e_latency_ns: float = 0.0
    rapid_avg_latency_ns: float = 0.0
    rapid_avg_latency_cycles: float = 0.0
    e2e_compute_latency_ns: float = 0.0
    e2e_communication_serialization_ns: float = 0.0
    e2e_communication_path_latency_ns: float = 0.0
    e2e_communication_latency_ns: float = 0.0
    e2e_group_compute_times_ns: tuple[float, ...] = ()
    e2e_communication_event_times_ns: tuple[float, ...] = ()
    e2e_boundary_timings: tuple[dict[str, object], ...] = ()
    e2e_latency_model: str = "batch1_sum_group_compute_plus_boundary_service"
    e2e_path_latency_source: str = ""
    latency_policy: str = "serial"
    canonical_flows: tuple[dict[str, object], ...] = ()
    network_context: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_ppa_weights(goal: str, weights: tuple[float, float, float] | None = None) -> PpaWeights:
    if weights is None:
        if goal not in PPA_WEIGHT_PRESETS:
            known = ", ".join(sorted(PPA_WEIGHT_PRESETS))
            raise ValueError(f"Unknown PPA goal '{goal}'. Known goals: {known}")
        weights = (
            PPA_WEIGHT_PRESETS[goal].latency,
            PPA_WEIGHT_PRESETS[goal].area,
            PPA_WEIGHT_PRESETS[goal].power,
        )

    latency, area, power = weights
    total = latency + area + power
    if total <= 0:
        raise ValueError("PPA weights must sum to a positive value")
    return PpaWeights(latency=latency / total, area=area / total, power=power / total)


def evaluate_one(
    model: ModelSpec,
    topology_kind: str,
    chiplet_count: int,
    cfg: EvalConfig,
    workload: WorkloadPartition | None = None,
    ppa_goal: str = "balanced",
    ppa_weights: PpaWeights | None = None,
    workload_search: str = "pareto-dp",
    dp_top_k: int = 12,
    dp_cost_mode: DpCostMode = "both",
) -> EvaluationResult:
    topology = make_topology(
        kind=topology_kind,
        node_count=chiplet_count,
        chiplet_width_mm=cfg.chiplet.width_mm,
        spacing_mm=cfg.chiplet.spacing_mm,
    )
    if workload is None:
        workloads = _generate_workloads(
            model=model,
            chiplets=chiplet_count,
            cfg=cfg,
            workload_search=workload_search,
            dp_top_k=dp_top_k,
            dp_cost_mode=dp_cost_mode,
        )
        if not workloads:
            raise ValueError(f"{workload_search} did not generate a workload candidate")
        workload = workloads[0]
    if len(workload.ops_per_chiplet) != chiplet_count:
        raise ValueError("workload chiplet count must match topology chiplet count")
    return _evaluate_workload(
        model=model,
        topology_kind=topology_kind,
        topology=topology,
        workload=workload,
        cfg=cfg,
        ppa_goal=ppa_goal,
        ppa_weights=resolve_ppa_weights(ppa_goal) if ppa_weights is None else ppa_weights,
    )


def _evaluate_workload(
    model: ModelSpec,
    topology_kind: str,
    topology: Topology,
    workload: WorkloadPartition,
    cfg: EvalConfig,
    ppa_goal: str,
    ppa_weights: PpaWeights,
) -> EvaluationResult:
    mapping_plan = build_mapping_plan(
        model,
        workload.group_specs,
        workload.mapping_strategies or None,
        sram_mb=cfg.chiplet.sram_mb,
    )
    rapid_metrics = evaluate_with_rapidchiplet(
        topology, workload, cfg,
        design_name=f"{model.name}_{topology_kind}_{topology.node_count}",
    )
    schedule = schedule_block_dag(
        model,
        workload.group_specs,
        cfg,
        topology,
        workload.mapping_strategies or None,
        rapid_metrics.network_context,
    )
    e2e = estimate_batch1_e2e_latency(
        model=model, workload=workload, mapping_plan=mapping_plan,
        topology=topology, cfg=cfg, network_context=rapid_metrics.network_context,
    )

    # FPS is an inverse-E2E diagnostic only, never a constraint or objective.
    achieved_fps = 1.0 / e2e.estimated_e2e_latency_s if e2e.estimated_e2e_latency_s > 0 else math.inf
    power_w = rapid_metrics.total_power_w
    perf_per_w = achieved_fps / power_w if power_w > 0 else math.inf

    bottleneck = (
        "-"
        if rapid_metrics.bottleneck_link is None
        else f"C{rapid_metrics.bottleneck_link[0] + 1}->C{rapid_metrics.bottleneck_link[1] + 1}"
    )
    # Do not use Rapid's average ICI path latency as an inference latency and
    # do not take max(ICI, scheduler makespan).  The reference E2E method is
    # the explicit sum of group compute and boundary communication service.
    avg_latency_ns = e2e.estimated_e2e_latency_ns
    return EvaluationResult(
        model=model.name,
        dataset=model.dataset,
        topology=topology_kind,
        available_chiplets=cfg.max_chiplets,
        selected_chiplets=topology.node_count,
        unused_chiplets=max(0, cfg.max_chiplets - topology.node_count),
        peak_tops_per_chiplet=cfg.chiplet.peak_tops,
        achieved_fps=achieved_fps,
        avg_latency_ns=avg_latency_ns,
        min_latency_cycles=rapid_metrics.min_latency_cycles,
        avg_latency_cycles=rapid_metrics.avg_latency_cycles,
        max_latency_cycles=rapid_metrics.max_latency_cycles,
        aggregate_throughput_bits_per_cycle=rapid_metrics.aggregate_throughput_bits_per_cycle,
        total_area_mm2=rapid_metrics.total_area_mm2,
        total_chiplet_area_mm2=rapid_metrics.total_chiplet_area_mm2,
        chip_width_mm=rapid_metrics.chip_width_mm,
        chip_height_mm=rapid_metrics.chip_height_mm,
        total_power_w=power_w,
        total_chiplet_power_w=rapid_metrics.total_chiplet_power_w,
        total_interposer_power_w=rapid_metrics.total_interposer_power_w,
        total_link_power_w=rapid_metrics.total_link_power_w,
        performance_per_w=perf_per_w,
        ppa_goal=ppa_goal,
        ppa_score=math.inf,
        ppa_latency_weight=ppa_weights.latency,
        ppa_area_weight=ppa_weights.area,
        ppa_power_weight=ppa_weights.power,
        total_traffic_mb_per_inference=workload.total_traffic_bits_per_inference / 8 / 1024 / 1024,
        edge_count=len(topology.edges),
        link_count=rapid_metrics.link_count,
        min_link_length_mm=rapid_metrics.min_link_length_mm,
        avg_link_length_mm=rapid_metrics.avg_link_length_mm,
        max_link_length_mm=rapid_metrics.max_link_length_mm,
        bottleneck_link=bottleneck,
        workload_search=workload.search_method,
        workload_plan=workload.plan,
        connection_graph=rapid_metrics.graph,
        block_graph=model.block_graph(),
        architecture_feasible=schedule.feasible and mapping_plan.feasible,
        architecture_violations=tuple(
            dict.fromkeys((*mapping_plan.violations, *schedule.violations))
        ),
        mapping_plan=mapping_plan.to_dict(),
        schedule=schedule.to_dict(),
        schedule_latency_ns=schedule.makespan_ns,
        e2e_latency_ns=e2e.estimated_e2e_latency_ns,
        rapid_avg_latency_ns=rapid_metrics.avg_latency_ns,
        rapid_avg_latency_cycles=rapid_metrics.avg_latency_cycles,
        e2e_compute_latency_ns=e2e.compute_latency_sum_s * 1e9,
        e2e_communication_serialization_ns=(
            e2e.communication_serialization_sum_s * 1e9
        ),
        e2e_communication_path_latency_ns=(
            e2e.communication_path_latency_sum_s * 1e9
        ),
        e2e_communication_latency_ns=e2e.communication_latency_sum_s * 1e9,
        e2e_group_compute_times_ns=tuple(
            value * 1e9 for value in e2e.group_compute_times_s
        ),
        e2e_communication_event_times_ns=tuple(
            value * 1e9 for value in e2e.communication_event_times_s
        ),
        e2e_boundary_timings=e2e.boundary_timings,
        e2e_path_latency_source=e2e.path_latency_source,
        canonical_flows=tuple(flow.to_dict() for flow in workload.flows),
        network_context=rapid_metrics.network_context.to_dict(),
    )


def evaluate_models(
    models: list[ModelSpec],
    topology_kinds: list[str],
    cfg: EvalConfig,
    ppa_goal: str = "balanced",
    ppa_weights: tuple[float, float, float] | None = None,
    dp_top_k: int = 12,
    workload_search: str = "pareto-dp",
    ppa_reference: PpaReference | None = None,
    dp_cost_mode: DpCostMode = "both",
) -> tuple[list[EvaluationResult], list[EvaluationResult]]:
    weights = resolve_ppa_weights(ppa_goal, ppa_weights)
    summaries: list[EvaluationResult] = []
    sweeps: list[EvaluationResult] = []
    for model in models:
        model_candidates: list[EvaluationResult] = []
        for topology_kind in topology_kinds:
            for chiplets in range(1, cfg.max_chiplets + 1):
                topology = make_topology(
                    kind=topology_kind,
                    node_count=chiplets,
                    chiplet_width_mm=cfg.chiplet.width_mm,
                    spacing_mm=cfg.chiplet.spacing_mm,
                )
                for workload in _generate_workloads(
                    model=model,
                    chiplets=chiplets,
                    cfg=cfg,
                    workload_search=workload_search,
                    dp_top_k=dp_top_k,
                    dp_cost_mode=dp_cost_mode,
                ):
                    model_candidates.append(
                        _evaluate_workload(
                            model=model,
                            topology_kind=topology_kind,
                            topology=topology,
                            workload=workload,
                            cfg=cfg,
                            ppa_goal=ppa_goal,
                            ppa_weights=weights,
                        )
                    )
        scored = _score_results(model_candidates, weights, ppa_reference)
        sweeps.extend(scored)
        for topology_kind in topology_kinds:
            topology_scored = [row for row in scored if row.topology == topology_kind]
            best = _select_summary(topology_scored)
            if best is not None:
                summaries.append(best)
    return summaries, sweeps


def _generate_workloads(
    model: ModelSpec,
    chiplets: int,
    cfg: EvalConfig,
    workload_search: str,
    dp_top_k: int,
    dp_cost_mode: DpCostMode,
) -> list[WorkloadPartition]:
    if workload_search == "brute-force":
        return brute_force_pipeline_workloads(model, chiplets, cfg.chiplet.op_per_mac)
    if workload_search == "pareto-dp":
        return pareto_dp_pipeline_workloads(
            model,
            chiplets,
            cfg.chiplet.op_per_mac,
            top_k=dp_top_k,
            cost_mode=dp_cost_mode,
        )
    raise ValueError(f"Unknown workload search method: {workload_search}")


def make_ppa_reference(results: list[EvaluationResult]) -> PpaReference:
    return PpaReference(
        max_latency_ns=_finite_max(row.avg_latency_ns for row in results),
        max_area_mm2=_finite_max(row.total_area_mm2 for row in results),
        max_power_w=_finite_max(row.total_power_w for row in results),
    )


def _score_results(
    results: list[EvaluationResult],
    weights: PpaWeights,
    reference: PpaReference | None = None,
) -> list[EvaluationResult]:
    ref = make_ppa_reference(results) if reference is None else reference

    scored: list[EvaluationResult] = []
    for row in results:
        latency_score = _normalized_metric(row.avg_latency_ns, ref.max_latency_ns)
        area_score = _normalized_metric(row.total_area_mm2, ref.max_area_mm2)
        power_score = _normalized_metric(row.total_power_w, ref.max_power_w)
        ppa_score = (
            weights.latency * latency_score
            + weights.area * area_score
            + weights.power * power_score
        )
        scored.append(replace(row, ppa_score=ppa_score))
    return scored


def _select_summary(results: list[EvaluationResult]) -> EvaluationResult | None:
    if not results:
        return None
    return min(
        results,
        key=lambda row: (
            row.ppa_score,
            row.selected_chiplets,
            row.total_power_w,
            row.avg_latency_ns,
        ),
    )


def select_best_by_model(results: list[EvaluationResult]) -> list[EvaluationResult]:
    grouped: dict[str, list[EvaluationResult]] = {}
    for row in results:
        grouped.setdefault(row.model, []).append(row)

    best_rows: list[EvaluationResult] = []
    for model in sorted(grouped):
        best = _select_summary(grouped[model])
        if best is not None:
            best_rows.append(best)
    return best_rows


def _finite_max(values) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return max(finite, default=1.0)


def _normalized_metric(value: float, maximum: float) -> float:
    if not math.isfinite(value) or maximum <= 0:
        return 0.0
    return value / maximum
