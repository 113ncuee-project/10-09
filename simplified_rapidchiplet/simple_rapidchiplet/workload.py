from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .flows import Flow, traffic_bits_by_pair
from .mapping import build_mapping_plan, default_mapping_strategies
from .model import ModelSpec


@dataclass(frozen=True)
class WorkloadPartition:
    flows: tuple[Flow, ...]
    ops_per_chiplet: tuple[float, ...]
    stage_labels: tuple[str, ...]
    search_method: str
    plan: str
    group_specs: tuple[tuple[int, int, int], ...] = ()
    mapping_strategies: tuple[str, ...] = ()
    mapping_plan: dict[str, object] = field(default_factory=dict)

    @property
    def traffic_bits_per_inference(self) -> dict[tuple[int, int], float]:
        return traffic_bits_by_pair(self.flows)

    @property
    def total_traffic_bits_per_inference(self) -> float:
        return sum(flow.bytes for flow in self.flows) * 8

    @property
    def block_labels(self) -> tuple[str, ...]:
        """Compatibility alias for callers that use the new block wording."""

        return self.stage_labels


@dataclass(frozen=True)
class _DpState:
    bottleneck_ops: float
    traffic_bits: float
    groups: tuple[tuple[int, int, int], ...]


DpCostMode = Literal["both", "bottleneck", "traffic"]


def brute_force_pipeline_workloads(
    model: ModelSpec,
    chiplets: int,
    op_per_mac: float,
) -> list[WorkloadPartition]:
    if chiplets < 1:
        raise ValueError("chiplets must be >= 1")

    workloads: list[WorkloadPartition] = []
    seen: set[tuple[str, ...]] = set()
    stage_count = len(model.stages)
    for groups in _contiguous_stage_groupings(stage_count):
        if len(groups) > chiplets:
            continue

        # Every group can now be replicated over multiple chiplets.  The old
        # implementation restricted this list to single-stage groups, which
        # made a multi-block group permanently single-chiplet.
        expandable_groups = list(range(len(groups)))
        remaining_chiplets = chiplets - len(groups)

        for extras in _nonnegative_compositions(remaining_chiplets, len(expandable_groups)):
            allocation = [1] * len(groups)
            for group_index, extra_chiplets in zip(expandable_groups, extras):
                allocation[group_index] += extra_chiplets
            workload = _build_partition(
                model=model,
                op_per_mac=op_per_mac,
                groups=groups,
                allocation=tuple(allocation),
                search_method="brute-force",
            )
            key = workload.stage_labels
            if key in seen:
                continue
            seen.add(key)
            workloads.append(workload)

    return sorted(workloads, key=lambda workload: (workload.ops_per_chiplet, workload.stage_labels))


def pareto_dp_pipeline_workloads(
    model: ModelSpec,
    chiplets: int,
    op_per_mac: float,
    top_k: int | None = 0,
    cost_mode: DpCostMode = "both",
) -> list[WorkloadPartition]:
    if chiplets < 1:
        raise ValueError("chiplets must be >= 1")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be >= 0")
    if cost_mode not in {"both", "bottleneck", "traffic"}:
        raise ValueError("cost_mode must be 'both', 'bottleneck', or 'traffic'")
    if chiplets == 1:
        return [_single_chiplet_workload(model, op_per_mac)]

    stage_count = len(model.stages)
    cumulative_ops = _cumulative_stage_ops(model, op_per_mac)
    dp: dict[tuple[int, int], list[_DpState]] = {
        (0, 0): [_DpState(bottleneck_ops=0.0, traffic_bits=0.0, groups=())]
    }

    for used_chiplets in range(1, chiplets + 1):
        for end_stage in range(1, stage_count + 1):
            candidates: list[_DpState] = []
            for start_stage in range(end_stage):
                segment_ops = _range_ops(cumulative_ops, start_stage, end_stage)
                for prev_chiplets in range(used_chiplets):
                    group_chiplets = used_chiplets - prev_chiplets
                    for prev in dp.get((prev_chiplets, start_stage), ()):
                        segment_bottleneck = segment_ops / group_chiplets
                        segment_traffic = _inter_group_traffic_bits(model, start_stage)
                        segment_traffic += _internal_group_traffic_bits(
                            model,
                            cumulative_ops,
                            start_stage,
                            end_stage,
                            group_chiplets,
                        )
                        candidates.append(
                            _DpState(
                                bottleneck_ops=max(prev.bottleneck_ops, segment_bottleneck),
                                traffic_bits=prev.traffic_bits + segment_traffic,
                                groups=prev.groups + ((start_stage, end_stage, group_chiplets),),
                            )
                        )
            if candidates:
                dp[(used_chiplets, end_stage)] = _prune_dp_states(candidates, top_k, cost_mode)

    return [
        _build_partition(
            model=model,
            op_per_mac=op_per_mac,
            groups=tuple((start, end) for start, end, _chiplets in state.groups),
            allocation=tuple(group_chiplets for _start, _end, group_chiplets in state.groups),
            search_method="pareto-dp",
        )
        for state in dp.get((chiplets, stage_count), ())
    ]


def _build_partition(
    model: ModelSpec,
    op_per_mac: float,
    groups: tuple[tuple[int, int], ...],
    allocation: tuple[int, ...],
    search_method: str,
    mapping_strategies: tuple[str, ...] | None = None,
) -> WorkloadPartition:
    cumulative_ops = _cumulative_stage_ops(model, op_per_mac)
    flows: list[Flow] = []
    ops_per_chiplet: list[float] = []
    labels: list[str] = []
    group_chiplet_ids: list[tuple[int, ...]] = []

    for group_index, ((start, end), group_chiplets) in enumerate(zip(groups, allocation)):
        group_start_ops = 0.0 if start == 0 else cumulative_ops[start - 1]
        group_end_ops = cumulative_ops[end - 1]
        group_ops = group_end_ops - group_start_ops
        ops_per_part = group_ops / group_chiplets
        base_label = _stage_range_label(model, start, end)
        first_id = len(ops_per_chiplet)
        group_chiplet_ids.append(tuple(range(first_id, first_id + group_chiplets)))

        for part in range(group_chiplets):
            ops_per_chiplet.append(ops_per_part)
            labels.append(base_label if group_chiplets == 1 else f"{base_label}[{part + 1}/{group_chiplets}]")

    # One canonical construction replaces the former chain in Rapid and
    # independently reconstructed all-to-all events in E2E. This preserves
    # the legacy equal-pair boundary policy; it is not exact K ownership.
    for group_index in range(len(groups) - 1):
        producer = model.blocks[groups[group_index][1] - 1]
        consumer = model.blocks[groups[group_index + 1][0]]
        sources = group_chiplet_ids[group_index]
        destinations = group_chiplet_ids[group_index + 1]
        pair_bytes = producer.output_mb * 1024 * 1024 / (len(sources) * len(destinations))
        if pair_bytes > 0:
            flows.extend(
                Flow(f"{producer.name}:output", src, dst, pair_bytes,
                     "group_boundary", group_index, group_index + 1,
                     producer.name, consumer.name)
                for src in sources for dst in destinations if src != dst
            )

    group_specs = tuple(
        (start, end, group_chiplets)
        for (start, end), group_chiplets in zip(groups, allocation)
    )
    strategies = (
        default_mapping_strategies(group_specs)
        if mapping_strategies is None
        else tuple(mapping_strategies)
    )
    mapping_plan = build_mapping_plan(model, group_specs, strategies)
    for edge in mapping_plan.traffic_edges:
        source = int(edge["source"])
        target = int(edge["target"])
        amount = float(edge["mb"]) * 1024 * 1024
        group_index = int(edge["group"])
        start, end, _parts = group_specs[group_index]
        if start == 0 and edge["kind"] in ("input_broadcast", "spatial_halo_exchange"):
            # The initial activation source is EXTERNAL, not representative
            # compute chiplet 0. Do not invent a mesh endpoint for host IO.
            # Operator-level input/weight bookkeeping is added in Phase 3–5.
            continue
        is_reduction = edge["kind"] == "partial_output_reduction"
        tensor_block = model.blocks[end - 1] if is_reduction else model.blocks[start]
        tensor_role = "partial_output" if is_reduction else "input"
        if amount > 0 and source != target:
            flows.append(Flow(
                f"{tensor_block.name}:{tensor_role}", source, target, amount,
                "intra_group_mapping", group_index, group_index,
                tensor_block.name, tensor_block.name, str(edge["kind"]),
            ))

    return WorkloadPartition(
        flows=tuple(flows),
        ops_per_chiplet=tuple(ops_per_chiplet),
        stage_labels=tuple(labels),
        search_method=search_method,
        plan=" | ".join(labels),
        group_specs=group_specs,
        mapping_strategies=strategies,
        mapping_plan=mapping_plan.to_dict(),
    )


def build_pipeline_workload(
    model: ModelSpec,
    op_per_mac: float,
    groups: tuple[tuple[int, int, int], ...],
    search_method: str = "direct-action",
    mapping_strategies: tuple[str, ...] | None = None,
) -> WorkloadPartition:
    """Build one legal block-level pipeline design from explicit actions.

    ``groups`` contains ``(start_block, end_block, chiplet_count)`` records.
    Any contiguous group may use any positive number of chiplets.  This is the
    primitive used by the preference-aware RL agent.
    """

    if not groups:
        raise ValueError("at least one block group is required")
    cursor = 0
    used_chiplets = 0
    for start, end, parts in groups:
        if start != cursor or not start < end <= len(model.blocks):
            raise ValueError("groups must cover all blocks contiguously")
        if parts < 1:
            raise ValueError("each group must use at least one chiplet")
        cursor = end
        used_chiplets += parts
    if cursor != len(model.blocks):
        raise ValueError("groups must cover every model block")

    return _build_partition(
        model=model,
        op_per_mac=op_per_mac,
        groups=tuple((start, end) for start, end, _parts in groups),
        allocation=tuple(parts for _start, _end, parts in groups),
        search_method=search_method,
        mapping_strategies=mapping_strategies,
    )


def _single_chiplet_workload(model: ModelSpec, op_per_mac: float) -> WorkloadPartition:
    return WorkloadPartition(
        flows=(),
        ops_per_chiplet=(model.ops_per_inference(op_per_mac),),
        stage_labels=("all-stages",),
        search_method="pareto-dp",
        plan="single-chiplet",
        group_specs=((0, len(model.blocks), 1),),
    )


def _cumulative_stage_ops(model: ModelSpec, op_per_mac: float) -> list[float]:
    cumulative = []
    running = 0.0
    for stage in model.stages:
        running += stage.macs_g * 1e9 * op_per_mac
        cumulative.append(running)
    return cumulative


def _prune_dp_states(
    states: list[_DpState],
    top_k: int | None,
    cost_mode: DpCostMode = "both",
) -> list[_DpState]:
    frontier: list[_DpState] = []
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    ordered = sorted(states, key=lambda state: (state.bottleneck_ops, state.traffic_bits, state.groups))
    for state in ordered:
        if state.groups in seen:
            continue
        seen.add(state.groups)
        if any(_dominates(existing, state, cost_mode) for existing in frontier):
            continue
        frontier = [existing for existing in frontier if not _dominates(state, existing, cost_mode)]
        frontier.append(state)
    ordered_frontier = sorted(frontier, key=lambda state: (state.bottleneck_ops, state.traffic_bits, state.groups))
    if top_k is None or top_k == 0:
        return ordered_frontier
    return ordered_frontier[:top_k]


def _dominates(left: _DpState, right: _DpState, cost_mode: DpCostMode = "both") -> bool:
    if cost_mode == "bottleneck":
        return left.bottleneck_ops <= right.bottleneck_ops and left.bottleneck_ops < right.bottleneck_ops
    if cost_mode == "traffic":
        return left.traffic_bits <= right.traffic_bits and left.traffic_bits < right.traffic_bits
    return (
        left.bottleneck_ops <= right.bottleneck_ops
        and left.traffic_bits <= right.traffic_bits
        and (left.bottleneck_ops < right.bottleneck_ops or left.traffic_bits < right.traffic_bits)
    )


def _range_ops(cumulative: list[float], start: int, end: int) -> float:
    return cumulative[end - 1] - (0.0 if start == 0 else cumulative[start - 1])


def _inter_group_traffic_bits(model: ModelSpec, start: int) -> float:
    if start == 0:
        return 0.0
    return model.stages[start - 1].output_mb * 1024 * 1024 * 8


def _internal_group_traffic_bits(
    model: ModelSpec,
    cumulative_ops: list[float],
    start: int,
    end: int,
    group_chiplets: int,
) -> float:
    if group_chiplets <= 1:
        return 0.0

    group_start_ops = 0.0 if start == 0 else cumulative_ops[start - 1]
    group_ops = _range_ops(cumulative_ops, start, end)
    ops_per_part = group_ops / group_chiplets
    total = 0.0
    for part in range(1, group_chiplets):
        boundary_ops = group_start_ops + ops_per_part * part
        stage_index = _stage_at_boundary(boundary_ops, cumulative_ops)
        total += model.stages[stage_index].output_mb * 1024 * 1024 * 8
    return total


def _stage_range_label(model: ModelSpec, start: int, end: int) -> str:
    return " + ".join(stage.name for stage in model.stages[start:end])


def _stage_at_boundary(boundary_ops: float, cumulative: list[float]) -> int:
    for index, ops in enumerate(cumulative):
        if boundary_ops <= ops:
            return index
    return len(cumulative) - 1


def _contiguous_stage_groupings(stage_count: int) -> list[tuple[tuple[int, int], ...]]:
    def build(start: int) -> list[tuple[tuple[int, int], ...]]:
        if start == stage_count:
            return [()]
        groupings = []
        for end in range(start + 1, stage_count + 1):
            for tail in build(end):
                groupings.append(((start, end),) + tail)
        return groupings

    return build(0)


def _nonnegative_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 0:
        return [()] if total == 0 else []
    if parts == 1:
        return [(total,)]

    compositions = []
    for value in range(total + 1):
        for tail in _nonnegative_compositions(total - value, parts - 1):
            compositions.append((value,) + tail)
    return compositions
