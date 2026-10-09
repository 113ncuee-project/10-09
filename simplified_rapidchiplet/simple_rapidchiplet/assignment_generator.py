"""Bounded ordered assignments shared by RL and exhaustive search."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .mapping import integer_channel_ranges
from .tensor_ownership import missing_ranges
from .traffic_model import input_requirements


@dataclass(frozen=True)
class AssignmentAction:
    chiplets: tuple[int, ...]
    family: str
    boundary_hop_byte_lower_bound: float


def legal_assignments(graph, block_index, total_chiplets, previous, residency, primary, top_k,
                      *, fixed_assignment=False, fixed_active_count=None):
    block = graph.blocks[block_index]
    ops, tensors = graph.operation_map, graph.tensor_map
    first = ops[block.operators[0]]
    max_active = min(total_chiplets, *(tensors[ops[name].output].channels for name in block.operators))
    if fixed_active_count is not None and not 1 <= fixed_active_count <= max_active:
        return ()
    counts = (fixed_active_count,) if fixed_active_count is not None else range(1, max_active+1)
    columns = math.ceil(math.sqrt(total_chiplets))
    def distance(a, b):
        return abs(a//columns-b//columns)+abs(a%columns-b%columns)
    available = {}
    for name, node, start, end in residency:
        available.setdefault((name, node), []).append((start, end))
    owners = {}
    for name, node, start, end in primary:
        owners.setdefault(name, []).append((node, start, end))
    def bound(ids):
        cost = 0
        for dst, (start, end) in zip(ids, integer_channel_ranges(tensors[first.output].channels, len(ids))):
            for name in first.inputs:
                tensor = tensors[name]
                if tensor.source == "EXTERNAL":
                    continue
                for low, high in input_requirements(first, start, end, tensor):
                    for left, right in missing_ranges(low, high, available.get((name, dst), ())):
                        for src, own_left, own_right in owners.get(name, ()):
                            amount = max(0, min(right, own_right)-max(left, own_left))
                            cost += amount * tensor.elements_per_channel * tensor.bytes_per_element * distance(src, dst)
        return cost
    actions = []
    for count in counts:
        if fixed_assignment:
            ids = tuple(range(count))
            actions.append(AssignmentAction(ids, "FIXED_ROW_MAJOR", bound(ids)))
            continue
        proposals = {}
        def add(ids, family):
            ids = tuple(ids)
            if len(ids) == count and len(set(ids)) == count:
                proposals.setdefault(ids, family)
        if previous:
            reuse = list(previous[:count])
            remaining = sorted((node for node in range(total_chiplets) if node not in reuse),
                               key=lambda node:(min(distance(node, old) for old in previous), node))
            add(reuse+remaining[:count-len(reuse)], "REUSE_PREVIOUS")
        for anchor in range(total_chiplets):
            compact = sorted(range(total_chiplets), key=lambda node:(distance(anchor, node), node))[:count]
            add(compact, "COMPACT")
            add(reversed(compact), "COMPACT")
        # Physical ownership determines placement ordering; no family bonus.
        relevant = [item for name in first.inputs for item in owners.get(name, ())]
        ranked = sorted(range(total_chiplets), key=lambda node:(
            sum((end-start)*distance(src, node) for src,start,end in relevant), node))[:count]
        add(ranked, "COMM_AWARE")
        add(reversed(ranked), "COMM_AWARE")
        scored = sorted((AssignmentAction(ids, family, bound(ids)) for ids,family in proposals.items()),
                        key=lambda action:(action.boundary_hop_byte_lower_bound, action.chiplets))
        actions.extend(scored[:top_k])
    return tuple(actions)
