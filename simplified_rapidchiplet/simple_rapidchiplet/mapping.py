"""Intra-block mapping and architecture-level feasibility checks.

The first block-level prototype treated ``assign(group, N)`` as if compute
simply became ``MACs / N``.  This module makes that action meaningful by
requiring a mapping strategy and by estimating the data movement that the
strategy creates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .model import ModelSpec


MappingStrategy = Literal["single", "output_channel", "input_channel", "spatial"]
PARALLEL_MAPPING_STRATEGIES: tuple[MappingStrategy, ...] = (
    "output_channel",
)
# Retained only for explicitly requested historical comparisons.
LEGACY_HEURISTIC_STRATEGIES: tuple[MappingStrategy, ...] = ("input_channel", "spatial")
LEGACY_DISABLED_BY_DEFAULT = True
ALL_MAPPING_STRATEGIES: tuple[MappingStrategy, ...] = (
    "single",
    *PARALLEL_MAPPING_STRATEGIES,
    *LEGACY_HEURISTIC_STRATEGIES,
)
BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class MappingGroupResult:
    group_index: int
    start_block: int
    end_block: int
    chiplets: int
    strategy: MappingStrategy
    feasible: bool
    violations: tuple[str, ...]
    input_mb: float
    output_mb: float
    weight_mb: float
    memory_per_chiplet_mb: float
    compute_efficiency: float
    extra_traffic_mb: float
    partition: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "group_index": self.group_index,
            "start_block": self.start_block,
            "end_block": self.end_block,
            "chiplets": self.chiplets,
            "strategy": self.strategy,
            "feasible": self.feasible,
            "violations": list(self.violations),
            "input_mb": self.input_mb,
            "output_mb": self.output_mb,
            "weight_mb": self.weight_mb,
            "memory_per_chiplet_mb": self.memory_per_chiplet_mb,
            "compute_efficiency": self.compute_efficiency,
            "extra_traffic_mb": self.extra_traffic_mb,
            "partition": list(self.partition),
        }


@dataclass(frozen=True)
class MappingPlan:
    feasible: bool
    violations: tuple[str, ...]
    groups: tuple[MappingGroupResult, ...]
    traffic_edges: tuple[dict[str, object], ...]
    total_chiplets: int = 0
    block_mappings: tuple["BlockMapping", ...] = ()

    @property
    def extra_traffic_mb(self) -> float:
        return sum(float(edge["mb"]) for edge in self.traffic_edges)

    def to_dict(self) -> dict[str, object]:
        return {
            "feasible": self.feasible,
            "violations": list(self.violations),
            "groups": [group.to_dict() for group in self.groups],
            "traffic_edges": list(self.traffic_edges),
            "extra_traffic_mb": self.extra_traffic_mb,
            "total_chiplets": self.total_chiplets,
            "blocks": [block.to_dict() for block in self.block_mappings],
        }


@dataclass(frozen=True)
class ChannelShard:
    chiplet: int
    start: int
    end: int
    macs: int
    weight_bytes: int

    @property
    def channels(self):
        return self.end - self.start


@dataclass(frozen=True)
class BlockMapping:
    block_id: str
    active_chiplets: tuple[int, ...]
    operator_shards: tuple[tuple[str, tuple[ChannelShard, ...]], ...]
    partition_axis: str = "K"

    @property
    def active_chiplet_count(self):
        return len(self.active_chiplets)

    def to_dict(self):
        from dataclasses import asdict
        result = asdict(self)
        result["active_chiplet_count"] = self.active_chiplet_count
        return result


def integer_channel_ranges(channels: int, parts: int):
    if channels < 1 or not 1 <= parts <= channels:
        raise ValueError("active count must be between one and output channels")
    quotient, remainder = divmod(channels, parts)
    start = 0
    result = []
    for index in range(parts):
        end = start + quotient + (index < remainder)
        result.append((start, end))
        start = end
    return tuple(result)


def build_physical_mapping(graph, total_chiplets, assignments) -> MappingPlan:
    """N_total describes hardware; every block can reuse any physical IDs."""
    if total_chiplets < 1 or len(assignments) != len(graph.blocks):
        raise ValueError("one assignment per semantic block and positive N_total required")
    operators = graph.operation_map
    tensors = graph.tensor_map
    mappings = []
    for block, raw_ids in zip(graph.blocks, assignments):
        ids = tuple(raw_ids)
        if not ids or len(ids) != len(set(ids)) or any(node < 0 or node >= total_chiplets for node in ids):
            raise ValueError(f"invalid active chiplet IDs for {block.id}")
        shards = []
        for op_id in block.operators:
            op = operators[op_id]
            channels = tensors[op.output].channels
            ranges = integer_channel_ranges(channels, len(ids))
            if op.macs % channels or op.parameter_bytes % channels:
                raise ValueError(f"non-integral per-channel work/weights for {op.id}")
            shards.append((op_id, tuple(ChannelShard(node, start, end,
                         (end-start)*(op.macs//channels), (end-start)*(op.parameter_bytes//channels))
                         for node, (start, end) in zip(ids, ranges))))
        mappings.append(BlockMapping(block.id, ids, tuple(shards)))
    return MappingPlan(True, (), (), (), total_chiplets, tuple(mappings))


def default_mapping_strategies(groups: tuple[tuple[int, int, int], ...]) -> tuple[MappingStrategy, ...]:
    return tuple("single" if parts == 1 else "output_channel" for _start, _end, parts in groups)


def build_mapping_plan(
    model: ModelSpec,
    groups: tuple[tuple[int, int, int], ...],
    mapping_strategies: tuple[str, ...] | None = None,
    *,
    sram_mb: float = float("inf"),
    require_full_coverage: bool = True,
) -> MappingPlan:
    """Build mapping metadata and reject physically impossible assignments.

    The traffic values are architecture-level estimates.  They are not a
    replacement for a cycle-accurate accelerator simulator, but unlike
    ``MACs / N`` they expose the broadcast/reduction cost that distinguishes
    output-channel, input-channel, and spatial parallelism.
    """

    if not groups:
        raise ValueError("at least one mapping group is required")
    strategies = (
        default_mapping_strategies(groups)
        if mapping_strategies is None
        else tuple(str(value) for value in mapping_strategies)
    )
    if len(strategies) != len(groups):
        raise ValueError("mapping_strategies must have one entry per group")
    if sram_mb <= 0.0:
        raise ValueError("sram_mb must be positive")

    cursor = 0 if require_full_coverage else groups[0][0]
    group_results: list[MappingGroupResult] = []
    traffic_edges: list[dict[str, object]] = []
    all_violations: list[str] = []
    for group_index, ((start, end, parts), requested_strategy) in enumerate(zip(groups, strategies)):
        if start != cursor or not start < end <= len(model.blocks):
            raise ValueError("mapping groups must cover blocks contiguously")
        if parts < 1:
            raise ValueError("mapping group chiplet count must be positive")
        if requested_strategy not in ALL_MAPPING_STRATEGIES:
            raise ValueError(
                f"unknown mapping strategy '{requested_strategy}'; "
                f"use {', '.join(ALL_MAPPING_STRATEGIES)}"
            )

        strategy = _normalise_strategy(requested_strategy, parts)
        blocks = model.blocks[start:end]
        input_mb = max(blocks[0].input_mb, blocks[0].output_mb)
        output_mb = blocks[-1].output_mb
        weight_mb = sum(block.weight_mb for block in blocks)
        memory_per_chiplet_mb = max(input_mb, output_mb) + weight_mb / parts
        violations: list[str] = []
        if memory_per_chiplet_mb > sram_mb:
            violations.append(
                f"group {group_index} needs {memory_per_chiplet_mb:.3f} MB/chiplet "
                f"but SRAM is {sram_mb:.3f} MB"
            )
        if parts > 1 and strategy == "single":
            violations.append(f"group {group_index} uses multiple chiplets with single mapping")
        if parts > 1 and any(block.block_type.lower() == "head" for block in blocks):
            violations.append(f"group {group_index} contains a non-parallelizable Head block")

        extra_traffic_mb = _extra_traffic_mb(strategy, parts, input_mb, output_mb)
        partition = _partition_description(strategy, parts, input_mb, output_mb)
        result = MappingGroupResult(
            group_index=group_index,
            start_block=start,
            end_block=end,
            chiplets=parts,
            strategy=strategy,
            feasible=not violations,
            violations=tuple(violations),
            input_mb=input_mb,
            output_mb=output_mb,
            weight_mb=weight_mb,
            memory_per_chiplet_mb=memory_per_chiplet_mb,
            compute_efficiency=1.0,  # Compatibility metadata; never scales compute.
            extra_traffic_mb=extra_traffic_mb,
            partition=partition,
        )
        group_results.append(result)
        all_violations.extend(violations)

        if parts > 1 and extra_traffic_mb > 0.0:
            chiplet_start = sum(previous.chiplets for previous in group_results[:-1])
            if strategy == "input_channel":
                representative = chiplet_start
                for part in range(1, parts):
                    traffic_edges.append(
                        {
                            "source": chiplet_start + part,
                            "target": representative,
                            "mb": output_mb,
                            "kind": "partial_output_reduction",
                            "group": group_index,
                        }
                    )
            else:
                representative = chiplet_start
                for part in range(1, parts):
                    traffic_edges.append(
                        {
                            "source": representative,
                            "target": chiplet_start + part,
                            "mb": extra_traffic_mb / (parts - 1),
                            "kind": "input_broadcast" if strategy == "output_channel" else "spatial_halo_exchange",
                            "group": group_index,
                        }
                    )
        cursor = end

    if require_full_coverage and cursor != len(model.blocks):
        raise ValueError("mapping groups must cover every model block")
    return MappingPlan(
        feasible=not all_violations,
        violations=tuple(all_violations),
        groups=tuple(group_results),
        traffic_edges=tuple(traffic_edges),
    )


def mapping_action_is_feasible(
    model: ModelSpec,
    start: int,
    end: int,
    chiplets: int,
    strategy: str,
    *,
    sram_mb: float = float("inf"),
) -> bool:
    """Check one prospective RL action without evaluating a full design."""

    if not 0 <= start < end <= len(model.blocks):
        return False
    plan = build_mapping_plan(
        model,
        ((start, end, chiplets),),
        (strategy,),
        sram_mb=sram_mb,
        require_full_coverage=False,
    )
    return plan.feasible


def _normalise_strategy(requested: str, parts: int) -> MappingStrategy:
    if parts == 1:
        return "single"
    if requested == "single":
        return "single"
    return requested  # type: ignore[return-value]


def _extra_traffic_mb(
    strategy: MappingStrategy,
    parts: int,
    input_mb: float,
    output_mb: float,
) -> float:
    if parts <= 1:
        return 0.0
    if strategy == "output_channel":
        return input_mb * (parts - 1)
    if strategy == "input_channel":
        return output_mb * (parts - 1)
    if strategy == "spatial":
        # LEGACY_HEURISTIC: disabled in all default action generators.
        return input_mb * 0.25 * (parts - 1)
    return 0.0


def _partition_description(
    strategy: MappingStrategy,
    parts: int,
    input_mb: float,
    output_mb: float,
) -> tuple[dict[str, object], ...]:
    if parts == 1 or strategy == "single":
        return ({"chiplet": 0, "fraction": 1.0, "input_mb": input_mb, "output_mb": output_mb},)
    return tuple(
        {
            "chiplet": part,
            "fraction": 1.0 / parts,
            "input_mb": input_mb if strategy == "output_channel" else input_mb / parts,
            "output_mb": output_mb / parts if strategy == "output_channel" else output_mb,
        }
        for part in range(parts)
    )
