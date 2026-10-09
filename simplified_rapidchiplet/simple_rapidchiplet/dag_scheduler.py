"""Dependency-aware schedule estimation for a block graph.

This is an architecture-level scheduler, not an RTL simulator.  It performs
the important dependency/resource checks that an ordered pipeline misses:
multiple ready blocks may start together, and a join waits for every
predecessor.  Communication delay is estimated from tensor bytes and link
bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import EvalConfig
from .mapping import MappingPlan, build_mapping_plan
from .model import ModelSpec
from .network_context import NetworkContext
from .rapid_proxy import compute_latency
from .routing import shortest_paths_for_pairs
from .topology import Topology

BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class ScheduledBlock:
    name: str
    block_index: int
    group_index: int
    chiplet_ids: tuple[int, ...]
    strategy: str
    predecessors: tuple[str, ...]
    start_cycles: float
    finish_cycles: float
    duration_cycles: float
    incoming_comm_cycles: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "block_index": self.block_index,
            "group_index": self.group_index,
            "chiplet_ids": list(self.chiplet_ids),
            "strategy": self.strategy,
            "predecessors": list(self.predecessors),
            "start_cycles": self.start_cycles,
            "finish_cycles": self.finish_cycles,
            "duration_cycles": self.duration_cycles,
            "incoming_comm_cycles": self.incoming_comm_cycles,
        }


@dataclass(frozen=True)
class ScheduleResult:
    feasible: bool
    violations: tuple[str, ...]
    makespan_cycles: float
    makespan_ns: float
    tasks: tuple[ScheduledBlock, ...]
    critical_path: tuple[str, ...]
    concurrently_ready_pairs: int
    mapping_plan: MappingPlan

    def to_dict(self) -> dict[str, object]:
        return {
            "feasible": self.feasible,
            "violations": list(self.violations),
            "makespan_cycles": self.makespan_cycles,
            "makespan_ns": self.makespan_ns,
            "tasks": [task.to_dict() for task in self.tasks],
            "critical_path": list(self.critical_path),
            "concurrently_ready_pairs": self.concurrently_ready_pairs,
            "mapping_plan": self.mapping_plan.to_dict(),
        }


def schedule_block_dag(
    model: ModelSpec,
    groups: tuple[tuple[int, int, int], ...],
    cfg: EvalConfig,
    topology: Topology,
    mapping_strategies: tuple[str, ...] | None = None,
    network_context: NetworkContext | None = None,
) -> ScheduleResult:
    """Schedule blocks at their earliest legal start time.

    A group owns a contiguous set of chiplets. Every block in that group is
    replicated across those chiplets according to its mapping strategy. A
    block can start only after all graph predecessors and its assigned
    chiplets are ready.
    """

    mapping_plan = build_mapping_plan(
        model,
        groups,
        mapping_strategies,
        sram_mb=cfg.chiplet.sram_mb,
    )
    if not mapping_plan.feasible:
        return ScheduleResult(
            feasible=False,
            violations=mapping_plan.violations,
            makespan_cycles=float("inf"),
            makespan_ns=float("inf"),
            tasks=(),
            critical_path=(),
            concurrently_ready_pairs=0,
            mapping_plan=mapping_plan,
        )

    group_for_block: dict[int, int] = {}
    chiplets_for_group: dict[int, tuple[int, ...]] = {}
    for group_index, (start, end, parts) in enumerate(groups):
        chiplet_start = sum(previous[2] for previous in groups[:group_index])
        chiplet_ids = tuple(range(chiplet_start, chiplet_start + parts))
        chiplets_for_group[group_index] = chiplet_ids
        for block_index in range(start, end):
            group_for_block[block_index] = group_index

    order = _topological_order(model)
    finish_by_block: dict[str, float] = {}
    task_by_block: dict[str, ScheduledBlock] = {}
    chiplet_ready = [0.0] * topology.node_count
    strategies = (
        tuple(group.strategy for group in mapping_plan.groups)
        if mapping_strategies is None
        else tuple(mapping_strategies)
    )
    base_ops_per_cycle = (
        cfg.chiplet.pe_rows
        * cfg.chiplet.pe_cols
        * cfg.chiplet.ops_per_pe_per_cycle
        * cfg.chiplet.utilization
    )
    ready_counts: list[int] = []

    for block_index in order:
        block = model.blocks[block_index]
        group_index = group_for_block[block_index]
        mapping_group = mapping_plan.groups[group_index]
        chiplet_ids = chiplets_for_group[group_index]
        predecessors = tuple(block.depends_on)
        ready_predecessors = sum(name in finish_by_block for name in predecessors)
        ready_counts.append(ready_predecessors)
        predecessor_ready = 0.0
        incoming_comm = 0.0
        latest_predecessor: str | None = None
        for predecessor in predecessors:
            predecessor_finish = finish_by_block[predecessor]
            predecessor_index = next(
                index for index, candidate in enumerate(model.blocks) if candidate.name == predecessor
            )
            same_group = group_for_block[predecessor_index] == group_index
            comm_cycles = 0.0 if same_group else _communication_cycles(
                model.blocks[predecessor_index].output_mb,
                cfg,
                topology,
                chiplets_for_group[group_for_block[predecessor_index]],
                chiplet_ids,
                network_context,
            )
            candidate_ready = predecessor_finish + comm_cycles
            if candidate_ready >= predecessor_ready:
                predecessor_ready = candidate_ready
                latest_predecessor = predecessor
            incoming_comm = max(incoming_comm, comm_cycles)

        duration = (
            block.macs_g
            * 1e9
            * cfg.chiplet.op_per_mac
            / max(1, mapping_group.chiplets)
            / max(base_ops_per_cycle, 1e-12)
        )
        resource_ready = max((chiplet_ready[chiplet] for chiplet in chiplet_ids), default=0.0)
        start = max(predecessor_ready, resource_ready)
        finish = start + duration
        for chiplet in chiplet_ids:
            chiplet_ready[chiplet] = finish

        task = ScheduledBlock(
            name=block.name,
            block_index=block_index,
            group_index=group_index,
            chiplet_ids=chiplet_ids,
            strategy=strategies[group_index],
            predecessors=predecessors,
            start_cycles=start,
            finish_cycles=finish,
            duration_cycles=duration,
            incoming_comm_cycles=incoming_comm,
        )
        finish_by_block[block.name] = finish
        task_by_block[block.name] = task

    makespan = max(finish_by_block.values(), default=0.0)
    critical_path = _critical_path(task_by_block)
    concurrently_ready_pairs = sum(max(0, count - 1) for count in ready_counts)
    return ScheduleResult(
        feasible=True,
        violations=(),
        makespan_cycles=makespan,
        makespan_ns=makespan / cfg.chiplet.frequency_hz * 1e9,
        tasks=tuple(task_by_block[model.blocks[index].name] for index in order),
        critical_path=critical_path,
        concurrently_ready_pairs=concurrently_ready_pairs,
        mapping_plan=mapping_plan,
    )


def _communication_cycles(output_mb, cfg, topology, sources, destinations, context=None) -> float:
    """Diagnostic block transfer with actual routes, never an artificial hop factor.

    Producer volume still uses legacy equal-pair semantics. The physical
    operator evaluator supersedes this semantic-block schedule.
    """
    pairs = {(src, dst): output_mb * BYTES_PER_MB * 8 / (len(sources) * len(destinations))
             for src in sources for dst in destinations if src != dst and output_mb > 0}
    routes = shortest_paths_for_pairs(topology, pairs)
    loads = {}
    max_path = 0.0
    for pair, bits in pairs.items():
        route = context.routes.get(pair, routes[pair]) if context else routes[pair]
        latency = (context.path_latencies_cycles[pair] if context and pair in context.path_latencies_cycles
                   else compute_latency(topology, routes, {pair: bits}, cfg.chiplet, cfg.network).avg_cycles)
        max_path = max(max_path, latency)
        for link in zip(route, route[1:]):
            loads[link] = loads.get(link, 0.0) + bits
    return max_path + max((bits / (context.link_bandwidths_bits_per_cycle[link] if context
                         else cfg.network.link_bandwidth_bits_per_cycle)
                         for link, bits in loads.items()), default=0.0)


def _topological_order(model: ModelSpec) -> tuple[int, ...]:
    name_to_index = {block.name: index for index, block in enumerate(model.blocks)}
    indegree = {block.name: len(block.depends_on) for block in model.blocks}
    dependents: dict[str, list[str]] = {block.name: [] for block in model.blocks}
    for block in model.blocks:
        for dependency in block.depends_on:
            dependents[dependency].append(block.name)
    ready = sorted(name for name, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while ready:
        name = ready.pop(0)
        order.append(name_to_index[name])
        for dependent in sorted(dependents[name]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort()
    if len(order) != len(model.blocks):
        raise ValueError(f"Model {model.name} block graph is cyclic or incomplete")
    return tuple(order)


def _critical_path(tasks: dict[str, ScheduledBlock]) -> tuple[str, ...]:
    if not tasks:
        return ()
    current = max(tasks.values(), key=lambda task: task.finish_cycles)
    path = [current.name]
    while current.predecessors:
        predecessor = max(
            (tasks[name] for name in current.predecessors),
            key=lambda task: task.finish_cycles,
        )
        path.append(predecessor.name)
        current = predecessor
    path.reverse()
    return tuple(path)
