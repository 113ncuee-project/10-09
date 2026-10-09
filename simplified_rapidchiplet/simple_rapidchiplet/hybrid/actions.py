"""Gemini's five LP spatial-mapping operators, with explicit legal masks.

The public partition order is B,K,H,W. Gemini's paper prints H,W,B,K;
its output-shard-to-ordered-core correspondence is implemented by mapping.py.
Layer grouping and microbatch changes are separate decisions, not OP1-OP5.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
import math
import random

from .mapping import MappingPlan, Partition


OPERATORS = ("OP1", "OP2", "OP3", "OP4", "OP5")


@dataclass(frozen=True)
class MappingAction:
    operator: str
    layer_id: str
    other_layer_id: str = ""
    index: int = 0
    other_index: int = 0
    partition: tuple[int, int, int, int] = ()
    other_partition: tuple[int, int, int, int] = ()
    field: str = ""
    memory: str = ""
    variant: str = "gemini"
    idle_core: str = ""
    core_delta: int = 0
    resize_cores: tuple[str, ...] = ()

    def to_dict(self):
        data = asdict(self)
        for key in ("partition", "other_partition", "resize_cores"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data):
        copied = dict(data)
        for key in ("partition", "other_partition", "resize_cores"):
            copied[key] = tuple(copied.get(key, ()))
        return cls(**copied)


def partition_tuple(partition):
    return (partition.b, partition.k, partition.h, partition.w)


def _factors(number):
    return tuple(value for value in range(1, math.isqrt(number) + 1) if number % value == 0)


def _divisors(number):
    lower = _factors(number)
    return tuple(sorted(set(lower) | {number // value for value in lower}))


def partition_options(workload, layer_id, core_count, microbatch_size):
    """All exact legal B,K,H,W factors, without utilization heuristics.

    Uneven integer shards are legal; partitions exceeding tensor extents are not.
    MAC utilization and padding/tail cost belong in the physical evaluator.
    """
    if core_count < 1 or microbatch_size < 1:
        return ()
    layer = workload.layer_map[layer_id]
    shape = workload.tensor_map[layer.output].shape
    bounds = (min(microbatch_size, shape[0]), shape[1], shape[2], shape[3])
    options = []
    for b in _divisors(core_count):
        if b > bounds[0]:
            continue
        rest = core_count // b
        for k in _divisors(rest):
            if k > bounds[1]:
                continue
            rest_h = rest // k
            for h in _divisors(rest_h):
                w = rest_h // h
                if h <= bounds[2] and w <= bounds[3]:
                    options.append((b, k, h, w))
    return tuple(sorted(set(options)))


def _mapping_map(plan):
    return {item.layer_id: item for item in plan.layer_mappings}


def _group_ids(plan):
    return {layer: group.id for group in plan.groups for layer in group.layer_ids}


def _spatial_group_ids(plan):
    return {group.id for group in plan.groups if group.mode == "spatial"}


def validate_action_structure(workload, plan, memory_ids=(), core_ids=()):
    """Cheap mask contract; capacity and scheduled lifetime use the oracle.

    This intentionally does not call the expensive traffic/physical evaluator.
    """
    mapping = _mapping_map(plan)
    group_ids = _group_ids(plan)
    if len(mapping) != len(plan.layer_mappings) or set(mapping) != set(workload.layer_map):
        raise ValueError("mapping must assign every workload layer exactly once")
    if set(group_ids) != set(mapping):
        raise ValueError("layer groups must cover every mapped layer")
    if sum(len(group.layer_ids) for group in plan.groups) != len(group_ids):
        raise ValueError("duplicate layer group membership")
    if plan.microbatch_size < 1 or workload.batch_size % plan.microbatch_size:
        raise ValueError("microbatch must divide the fixed workload batch")
    allowed_cores = set(core_ids)
    allowed_memory = set(memory_ids) | {"interleave", "direct"}
    for layer_id, item in mapping.items():
        if not item.chiplets or len(set(item.chiplets)) != len(item.chiplets):
            raise ValueError("core group must be nonempty and unique")
        if allowed_cores and not set(item.chiplets) <= allowed_cores:
            raise ValueError("unknown compute core")
        if partition_tuple(item.partition) not in partition_options(
                workload, layer_id, len(item.chiplets), plan.microbatch_size):
            raise ValueError("partition does not match core group/tensor dimensions")
        if memory_ids and any(getattr(item, field) not in allowed_memory
                              for field in ("input_dram", "weight_dram", "output_dram")):
            raise ValueError("unknown memory placement")
    for group in plan.groups:
        if group.mode == "spatial":
            occupied = [core for layer in group.layer_ids for core in mapping[layer].chiplets]
            if len(occupied) != len(set(occupied)):
                raise ValueError("spatial layers cannot share a mapping core")
    return True


def _occupied_for_layer(plan, layer_id):
    group = plan.group_map[layer_id]
    layers = group.layer_ids if group.mode == "spatial" else (layer_id,)
    return {core for layer in layers for core in plan.mapping_map[layer].chiplets}


def apply_action(plan, action, *, core_ids=()):
    """Apply a parameterized operator; rejected actions never mutate the plan."""
    if action.operator not in OPERATORS:
        raise ValueError(f"unknown Gemini operator {action.operator}")
    mapping = _mapping_map(plan)
    if action.layer_id not in mapping:
        raise ValueError("unknown action layer")
    item = mapping[action.layer_id]
    cores = list(item.chiplets)
    if action.variant not in ("gemini", "hybrid_idle_acquire", "hybrid_idle_release", "hybrid_legal_resize"):
        raise ValueError("unknown mapping action variant")
    if action.variant != "gemini":
        if action.operator != "OP4" or action.other_layer_id:
            raise ValueError("idle-pool extensions belong to OP4 and target one layer")
        if action.variant == "hybrid_legal_resize":
            delta, changed = action.core_delta, tuple(action.resize_cores)
            if abs(delta) < 2 or len(changed) != abs(delta) or len(set(changed)) != len(changed):
                raise ValueError("legal resize must identify the exact multi-core delta")
            if delta > 0:
                if (core_ids and not set(changed) <= set(core_ids)) or set(changed) & _occupied_for_layer(plan, item.layer_id):
                    raise ValueError("legal resize acquisition requires known idle cores")
                if not 0 <= action.index <= len(cores):
                    raise ValueError("invalid legal resize insertion index")
                cores[action.index:action.index] = changed
            else:
                if not set(changed) <= set(cores):
                    raise ValueError("legal resize release requires allocated cores")
                cores = [core for core in cores if core not in changed]
                if not cores:
                    raise ValueError("legal resize release cannot empty a layer")
        elif action.variant == "hybrid_idle_acquire":
            if not action.idle_core or (core_ids and action.idle_core not in core_ids):
                raise ValueError("unknown idle compute core")
            if action.idle_core in _occupied_for_layer(plan, item.layer_id):
                raise ValueError("idle acquire requires an unoccupied core")
            if not 0 <= action.index <= len(cores):
                raise ValueError("invalid idle acquire insertion index")
            cores.insert(action.index, action.idle_core)
        else:
            if len(cores) < 2:
                raise ValueError("idle release cannot empty a layer")
            if not 0 <= action.index < len(cores):
                raise ValueError("idle release core index is out of range")
            if action.idle_core and action.idle_core != cores[action.index]:
                raise ValueError("idle release core identity differs")
            cores.pop(action.index)
        if len(action.partition) != 4 or any(value < 1 for value in action.partition) or math.prod(action.partition) != len(cores):
            raise ValueError("idle-pool partition must match new cardinality")
        mapping[item.layer_id] = replace(item, chiplets=tuple(cores), partition=Partition(*action.partition))
        return replace(plan, layer_mappings=tuple(mapping[item.layer_id] for item in plan.layer_mappings))
    if action.operator in ("OP2", "OP3", "OP4") and not 0 <= action.index < len(cores):
        raise ValueError("action core index is out of range")
    if action.operator == "OP1":
        if len(action.partition) != 4 or math.prod(action.partition) != len(cores):
            raise ValueError("OP1 must preserve exact core cardinality")
        mapping[item.layer_id] = replace(item, partition=Partition(*action.partition))
    elif action.operator == "OP2":
        if not 0 <= action.other_index < len(cores):
            raise ValueError("OP2 core index is out of range")
        if action.index == action.other_index:
            raise ValueError("OP2 requires two distinct core positions")
        cores[action.index], cores[action.other_index] = cores[action.other_index], cores[action.index]
        mapping[item.layer_id] = replace(item, chiplets=tuple(cores))
    elif action.operator in ("OP3", "OP4"):
        if action.other_layer_id == action.layer_id or action.other_layer_id not in mapping:
            raise ValueError("operator requires two distinct layers")
        other = mapping[action.other_layer_id]
        other_cores = list(other.chiplets)
        group_ids = _group_ids(plan)
        if group_ids[item.layer_id] != group_ids[other.layer_id] or group_ids[item.layer_id] not in _spatial_group_ids(plan):
            raise ValueError("inter-layer core changes require one spatial group")
        if action.operator == "OP3":
            if not 0 <= action.other_index < len(other_cores):
                raise ValueError("OP3 core index is out of range")
            cores[action.index], other_cores[action.other_index] = other_cores[action.other_index], cores[action.index]
            mapping[item.layer_id] = replace(item, chiplets=tuple(cores))
            mapping[other.layer_id] = replace(other, chiplets=tuple(other_cores))
        else:
            if len(cores) < 2:
                raise ValueError("OP4 cannot empty a donor layer")
            moved = cores.pop(action.index)
            if action.other_index < 0 or action.other_index > len(other_cores):
                raise ValueError("invalid OP4 destination insertion index")
            other_cores.insert(action.other_index, moved)
            if len(action.partition) != 4 or math.prod(action.partition) != len(cores):
                raise ValueError("OP4 donor partition must match new cardinality")
            if len(action.other_partition) != 4 or math.prod(action.other_partition) != len(other_cores):
                raise ValueError("OP4 recipient partition must match new cardinality")
            mapping[item.layer_id] = replace(item, chiplets=tuple(cores), partition=Partition(*action.partition))
            mapping[other.layer_id] = replace(other, chiplets=tuple(other_cores), partition=Partition(*action.other_partition))
    else:
        if action.field not in ("input_dram", "weight_dram", "output_dram") or not action.memory:
            raise ValueError("invalid OP5 external flow selector")
        mapping[item.layer_id] = replace(item, **{action.field: action.memory})
    return replace(plan, layer_mappings=tuple(mapping[item.layer_id] for item in plan.layer_mappings))


def _sample(items, limit, rng):
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    return rng.sample(items, limit)


def _bounded_core_subsets(cores, count, limit, rng):
    if not limit or math.comb(len(cores), count) <= limit:
        return tuple(combinations(cores, count))
    order = {core: index for index, core in enumerate(cores)}
    selected = set()
    while len(selected) < limit:
        selected.add(tuple(sorted(rng.sample(cores, count), key=order.__getitem__)))
    return tuple(sorted(selected, key=lambda subset: tuple(order[core] for core in subset)))


def legal_actions(workload, plan, memory_ids, *, core_ids=(), max_per_operator=24, rng=None):
    """Generate parameterized legal candidates and mask absent operators.

    A bounded uniform proposal pool is used for large search spaces. Each OP4
    core transfer proposes legal partition pairs, not an arbitrary clamped shape.
    Set max_per_operator=0 for exhaustive proposal enumeration on small tests.
    The bounds are reported by search; bounded pools do not claim completeness.
    """
    validate_action_structure(workload, plan, memory_ids, core_ids)
    rng = rng or random.Random(0)
    by_op = {operator: [] for operator in OPERATORS}
    mapping = _mapping_map(plan)
    group_ids = _group_ids(plan)
    spatial = _spatial_group_ids(plan)
    layer_ids = tuple(item.layer_id for item in plan.layer_mappings)
    for layer_id in layer_ids:
        item = mapping[layer_id]
        for part in partition_options(workload, layer_id, len(item.chiplets), plan.microbatch_size):
            if part != partition_tuple(item.partition):
                by_op["OP1"].append(MappingAction("OP1", layer_id, partition=part))
        for a, b in combinations(range(len(item.chiplets)), 2):
            by_op["OP2"].append(MappingAction("OP2", layer_id, index=a, other_index=b))
        # Our balanced planner can leave cores idle. Gemini's original OP4
        # preserves the allocated union, so explicitly labelled extensions
        # allow activating and releasing cores without changing its five
        # operator families. They are shared by every search baseline.
        if core_ids:
            occupied = _occupied_for_layer(plan, layer_id)
            unused = tuple(core for core in core_ids if core not in occupied)
            acquire_parts = partition_options(workload, layer_id, len(item.chiplets) + 1, plan.microbatch_size)
            acquire_parts = _sample(list(acquire_parts), max_per_operator, rng)
            for core, part in product(unused, acquire_parts):
                insertion_points = range(len(item.chiplets) + 1)
                if max_per_operator:
                    insertion_points = (rng.randrange(len(item.chiplets) + 1),)
                for insertion in insertion_points:
                    by_op["OP4"].append(MappingAction("OP4", layer_id, index=insertion, partition=part,
                                                    variant="hybrid_idle_acquire", idle_core=core,
                                                    core_delta=1, resize_cores=(core,)))
            release_parts = partition_options(workload, layer_id, len(item.chiplets) - 1, plan.microbatch_size)
            release_parts = _sample(list(release_parts), max_per_operator, rng)
            for index, part in product(range(len(item.chiplets)), release_parts):
                by_op["OP4"].append(MappingAction("OP4", layer_id, index=index, partition=part,
                                                variant="hybrid_idle_release", idle_core=item.chiplets[index],
                                                core_delta=-1, resize_cores=(item.chiplets[index],)))
            # Strict integer factors can make one-core transitions impossible.
            # Jump only to the nearest feasible cardinality on each side;
            # record this hybrid extension separately from paper OP4.
            current_count = len(item.chiplets)
            for direction, available in ((-1, item.chiplets), (1, unused)):
                for amount in range(1, len(available) + 1):
                    new_count = current_count + direction * amount
                    if new_count < 1:
                        break
                    parts = partition_options(workload, layer_id, new_count, plan.microbatch_size)
                    if not parts:
                        continue
                    if amount > 1:
                        subsets = _bounded_core_subsets(available, amount, max_per_operator, rng)
                        parts = _sample(list(parts), max_per_operator, rng)
                        for subset, part in product(subsets, parts):
                            insertion_points = range(current_count + 1) if direction > 0 and not max_per_operator else (
                                rng.randrange(current_count + 1) if direction > 0 else item.chiplets.index(subset[0]),)
                            for index in insertion_points:
                                by_op["OP4"].append(MappingAction("OP4", layer_id, index=index, partition=part,
                                                                variant="hybrid_legal_resize", core_delta=direction * amount,
                                                                resize_cores=subset))
                    break
        layer = workload.layer_map[layer_id]
        fields = []
        if any(name in workload.input_ids for name in layer.inputs):
            fields.append("input_dram")
        if layer.weight_bytes:
            fields.append("weight_dram")
        if layer.output in workload.output_ids or any(group_ids[consumer] != group_ids[layer_id]
                                                    for consumer in workload.consumers[layer.output]):
            fields.append("output_dram")
        for field in fields:
            for memory in (*memory_ids, "interleave"):
                if memory != getattr(item, field):
                    by_op["OP5"].append(MappingAction("OP5", layer_id, field=field, memory=memory))
    for left, right in combinations(layer_ids, 2):
        if group_ids[left] != group_ids[right] or group_ids[left] not in spatial:
            continue
        for a, b in product(range(len(mapping[left].chiplets)), range(len(mapping[right].chiplets))):
            by_op["OP3"].append(MappingAction("OP3", left, right, a, b))
        for donor, recipient in ((left, right), (right, left)):
            donor_count, recipient_count = len(mapping[donor].chiplets), len(mapping[recipient].chiplets)
            if donor_count < 2:
                continue
            donor_parts = partition_options(workload, donor, donor_count - 1, plan.microbatch_size)
            recipient_parts = partition_options(workload, recipient, recipient_count + 1, plan.microbatch_size)
            pairs = list(product(donor_parts, recipient_parts))
            # Bound expensive combinatorial proposal construction before the
            # final family sample; use seeded randomness so all pairs remain
            # reachable over repeated proposals.
            pairs = _sample(pairs, max_per_operator, rng)
            for a in range(donor_count):
                insertion_points = range(recipient_count + 1)
                if max_per_operator:
                    insertion_points = (rng.randrange(recipient_count + 1),)
                for insertion, (donor_part, recipient_part) in product(insertion_points, pairs):
                    by_op["OP4"].append(MappingAction("OP4", donor, recipient, a, insertion,
                                                    donor_part, recipient_part))
    selected = []
    for operator in OPERATORS:
        candidates = by_op[operator]
        if operator == "OP4" and max_per_operator:
            # Keep each available OP4 variant reachable in a bounded family
            # pool rather than letting the larger transfer pool starve idle
            # proposals. When the bound is smaller than the variant count,
            # seeded selection rotates which variants appear.
            variants = sorted({action.variant for action in candidates})
            variants = _sample(variants, max_per_operator, rng)
            guaranteed = [rng.choice([action for action in candidates if action.variant == variant])
                          for variant in variants]
            remaining = [action for action in candidates if action not in guaranteed]
            selected.extend(guaranteed)
            selected.extend(_sample(remaining, max_per_operator - len(guaranteed), rng)
                            if max_per_operator > len(guaranteed) else [])
        else:
            selected.extend(_sample(candidates, max_per_operator, rng))
    return tuple(selected)


def operator_mask(actions):
    present = {action.operator for action in actions}
    return {operator: operator in present for operator in OPERATORS}
