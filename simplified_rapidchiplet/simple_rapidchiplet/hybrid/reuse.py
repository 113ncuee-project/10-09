"""Capacity-visible weight retention and DRAM tensor residency.

Persistent weights are retained in the receiving core's unified-buffer namespace
for one layer group. Exact region copies may serve later tiles/microbatches;
their full lifetimes are checked by the evaluator. No reuse discount is applied
to MACs, PE/L1 accesses or unrelated overlapping regions.
"""
from collections import defaultdict
from dataclasses import replace

from .communication import _memory_regions, TensorLifetime
from .mapping import TensorRegion
from .scheduler import peak_interval_bytes


def apply_weight_reuse(compiled, communication):
    unit_map = compiled.unit_map
    canonical, replacements, selected = {}, {}, {}
    before = sum(transfer.bytes for transfer in communication.transfers if transfer.kind == "dram_weight")
    for transfer in communication.transfers:
        if transfer.kind != "dram_weight":
            selected[transfer.id] = transfer
            continue
        layer_id = unit_map[transfer.consumer_units[0]].layer_id
        mapping = compiled.plan.mapping_map[layer_id]
        if mapping.weight_policy == "cold":
            selected[transfer.id] = transfer
            continue
        if mapping.weight_policy != "persistent":
            raise ValueError("unknown weight residency policy")
        key = (unit_map[transfer.consumer_units[0]].group_id, transfer.src,
               transfer.dsts, transfer.tensor, transfer.region)
        if key not in canonical:
            canonical[key] = transfer.id
            selected[transfer.id] = transfer
        else:
            existing_id = canonical[key]
            replacements[transfer.id] = existing_id
            old = selected[existing_id]
            selected[existing_id] = replace(old, consumer_units=tuple(dict.fromkeys(
                old.consumer_units + transfer.consumer_units)))
    transfers = tuple(replace(transfer, dependencies=tuple(dict.fromkeys(replacements.get(dep,dep)
                      for dep in transfer.dependencies))) for transfer in selected.values())
    dependencies = tuple((unit,tuple(dict.fromkeys(replacements.get(dep,dep) for dep in deps)))
                         for unit,deps in communication.unit_dependencies)
    after = sum(transfer.bytes for transfer in transfers if transfer.kind == "dram_weight")
    lifetimes = [lifetime for lifetime in communication.lifetimes if lifetime.kind != 'weight']
    for transfer in transfers:
        if transfer.kind != 'dram_weight':
            continue
        for destination in transfer.dsts:
            users = tuple(name for name in transfer.consumer_units if unit_map[name].chiplet == destination)
            lifetimes.append(TensorLifetime(transfer.tensor,transfer.microbatch,destination,transfer.region,
                                           transfer.bytes,transfer.producer_units,users,'weight'))
    return replace(communication,transfers=transfers,unit_dependencies=dependencies,lifetimes=tuple(lifetimes)), {
        "policy_by_layer":{layer.layer_id:layer.weight_policy for layer in compiled.plan.layer_mappings},
        "cold_weight_read_bytes":before,"actual_weight_read_bytes":after,
        "saved_weight_read_bytes":before-after,"removed_transfers":len(replacements),
        "storage":"full exact region per receiving core until last reader; capacity checked; group scope",
    }


def _union_elements(regions, axis=0):
    """Exact axis sweep for a small set of rectangular initial placements."""
    boxes = tuple(set(region.bounds for region in regions))
    if not boxes:
        return 0
    def sweep(boxes, axis):
        if axis == 4:
            return 1
        cuts = sorted({value for box in boxes for value in box[axis]})
        return sum((right-left)*sweep(tuple(box for box in boxes
                       if box[axis][0] <= left and right <= box[axis][1]),axis+1)
                   for left,right in zip(cuts,cuts[1:])
                   if any(box[axis][0] <= left and right <= box[axis][1] for box in boxes))
    return sweep(boxes,axis)


def dram_residency(compiled, communication, schedule, hardware):
    """Initial model/input placements plus boundary/output write-to-last-read."""
    workload, plan = compiled.workload, compiled.plan
    placements = defaultdict(list)
    bits = {}
    for layer in workload.layers:
        mapping = plan.mapping_map[layer.id]
        if layer.weight_shape:
            region = TensorRegion(((0,layer.weight_shape[0]),(0,layer.weight_shape[1]),
                                   (0,layer.kernel[0]),(0,layer.kernel[1])))
            tensor_id = f"weights:{layer.id}"
            bits[tensor_id] = layer.weight_bits
            for memory, part in _memory_regions(region,mapping.weight_dram,plan.memory_ids,region):
                placements[(memory,tensor_id)].append(part)
        for tensor_id in layer.inputs:
            tensor = workload.tensor_map[tensor_id]
            if tensor.producer:
                continue
            region = TensorRegion.full(tensor.shape)
            bits[tensor_id] = tensor.bits
            for memory, part in _memory_regions(region,mapping.input_dram,plan.memory_ids,region):
                placements[(memory,tensor_id)].append(part)
    persistent = defaultdict(int)
    for (memory,tensor), regions in placements.items():
        persistent[memory] += _union_elements(regions)*bits[tensor]//8
    read_users = defaultdict(list)
    for transfer in communication.transfers:
        for dependency in transfer.dependencies:
            read_users[dependency].append(transfer.id)
    events = schedule.event_map
    intervals = defaultdict(list)
    for transfer in communication.transfers:
        for memory in transfer.dsts:
            if not memory.startswith("memory:"):
                continue
            finish = max((events[name].finish_s for name in read_users[transfer.id]),
                         default=schedule.makespan_s)
            intervals[memory].append((events[transfer.id].finish_s,finish,transfer.bytes))
    peaks = {memory:persistent[memory]+peak_interval_bytes(intervals[memory]) for memory in plan.memory_ids}
    # At completion the entire output batch exists in DRAM, including the last
    # write whose zero-duration endpoint would disappear in a half-open sweep.
    output_at_completion = defaultdict(int)
    for transfer in communication.transfers:
        if transfer.kind == 'output_write':
            for memory in transfer.dsts:
                output_at_completion[memory] += transfer.bytes
    peaks = {memory:max(amount,persistent[memory]+output_at_completion[memory])
             for memory,amount in peaks.items()}
    violations = [f"{memory} DRAM needs {amount} B > {hardware.dram_capacity_bytes} B"
                  for memory,amount in peaks.items() if amount > hardware.dram_capacity_bytes]
    return {"initial_resident_bytes":dict(persistent),"peak_by_memory":peaks,
            "capacity_bytes_per_memory":hardware.dram_capacity_bytes,"violations":violations,
            "policy":"weights and external inputs resident initially; boundary tensors until last read; outputs until batch completes"}
