"""Bounded graph-group proposals followed by optional full-plan evaluation.

The beam carries delay and energy separately. Its local estimates are proposal
heuristics, not an exact global EDP optimizer; the event evaluator ranks final
plans including routing, DRAM contention, residual liveness and fill/drain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from .hardware import HardwareSpec
from .costs import sram_bank
from .mapping import LayerGroup, LayerMapping, MappingPlan, Partition, enumerate_partitions, validate_plan


@dataclass(frozen=True)
class PlanningCandidate:
    plan: MappingPlan
    estimated_delay_s: float
    estimated_energy_j: float
    boundary_bytes: int
    method: str = "bounded graph-partition beam + balanced core allocation"

    def to_dict(self):
        return {"plan": self.plan.to_dict(), "estimated_delay_s": self.estimated_delay_s,
                "estimated_energy_j": self.estimated_energy_j, "boundary_bytes": self.boundary_bytes,
                "method": self.method, "estimate_is_final_evaluation": False}


def _partition_service(workload, layer, partition, microbatch_size, hardware):
    output = workload.tensor_map[layer.output]
    shape = (microbatch_size, *output.shape[1:])
    b, k, h, w = (math.ceil(extent / factor) for extent, factor in zip(shape, partition.as_tuple()))
    pes = hardware.pes_per_core
    if layer.macs_per_sample:
        cin = layer.weight_shape[1]
        kernel = math.prod(layer.kernel)
        cycles = math.ceil(b * h * w / pes) * math.ceil(k / hardware.lanes) * math.ceil(cin / hardware.vector_size) * kernel
        if layer.kind == "Linear":
            cycles = math.ceil(b / pes) * math.ceil(k / hardware.lanes) * math.ceil(cin / hardware.vector_size)
        compute = cycles / hardware.compute_clock_hz
    else:
        compute = 0.0
    compute += math.ceil(b * k * h * w * layer.vector_ops_per_element / (pes * hardware.lanes)) / hardware.compute_clock_hz
    input_bytes = 0
    for tensor_id in layer.inputs:
        tensor = workload.tensor_map[tensor_id]
        if layer.kind in {"Conv2d", "MaxPool2d", "AvgPool2d"}:
            ih = min(tensor.shape[2], (h - 1) * layer.stride[0] + layer.dilation[0] * (layer.kernel[0] - 1) + 1)
            iw = min(tensor.shape[3], (w - 1) * layer.stride[1] + layer.dilation[1] * (layer.kernel[1] - 1) + 1)
            channels = tensor.shape[1] if layer.kind == "Conv2d" else k
            if layer.kind == "Conv2d" and layer.groups > 1:
                per_out = output.shape[1] // layer.groups
                channels = min(tensor.shape[1], (math.ceil(k / per_out) + 1) * tensor.shape[1] // layer.groups)
            elements = b * channels * ih * iw
        elif layer.kind == "Linear":
            elements = b * math.prod(tensor.shape[1:])
        elif layer.kind == "AdaptiveAvgPool2d":
            ih = min(tensor.shape[2], math.ceil(h * tensor.shape[2] / output.shape[2]) + (output.shape[2] > 1))
            iw = min(tensor.shape[3], math.ceil(w * tensor.shape[3] / output.shape[3]) + (output.shape[3] > 1))
            elements = b * k * ih * iw
        elif layer.kind == "Flatten":
            elements = b * k
        else:
            elements = b * k * h * w
        input_bytes += elements * tensor.bytes_per_element
    weight_bytes = k * math.prod(layer.weight_shape[1:]) * layer.weight_bits // 8 if layer.weight_shape else 0
    output_bytes = b * k * h * w * output.bytes_per_element
    global_bytes = input_bytes + weight_bytes + output_bytes
    # Proposal stage uses the same logical data and injection lower bound for
    # both fabrics. Actual directed ring/mesh multicast routes determine link
    # loads in the full evaluator; mesh is never charged artificial PE copies.
    memory = max(global_bytes / min(hardware.l2_bandwidth_Bps, hardware.unified_bandwidth_Bps),
                 global_bytes / hardware.noc_bandwidth_Bps)
    # C streaming reduces local scratch, but full received data live in the
    # shared unified buffer. Penalize replication against a conservative core
    # share with two microbatches in flight; final liveness is checked globally.
    accum_bytes = b * k * h * w * hardware.psum_bytes if layer.weight_shape else 0
    quota = hardware.unified_buffer_bytes / hardware.cores_per_die
    simultaneous_bytes = weight_bytes + 2 * (input_bytes + output_bytes + accum_bytes)
    pressure = max(1.0, simultaneous_bytes / quota)
    return (compute + memory) * pressure * pressure


def _best_partition(workload, layer, count, microbatch, hardware, cache):
    key = layer.id, count, microbatch
    if key not in cache:
        choices = enumerate_partitions(workload, layer.id, count, microbatch)
        if not choices:
            cache[key] = None
        else:
            part = min(choices, key=lambda value: (_partition_service(workload, layer, value, microbatch, hardware),
                                                    value.k, value.h * value.w, value.as_tuple()))
            cache[key] = part, _partition_service(workload, layer, part, microbatch, hardware)
    return cache[key]


def balanced_core_allocation(workload, layer_ids, compute_ids, microbatch_size=1, hardware=None, _cache=None):
    """Integer allocation minimizes the estimated pipeline bottleneck.

    Each layer starts with one core. Lookahead can jump to a useful factorized
    count, avoiding the plateaus caused by SIMD rounds and prime allocations.
    Unused cores are legal when adding them cannot improve estimated service.
    """
    hardware = hardware or HardwareSpec()
    compute_ids = tuple(map(str, compute_ids))
    layers = tuple(workload.layer_map[layer] for layer in layer_ids)
    if not layers or len(layers) > len(compute_ids):
        raise ValueError("a spatial group needs at least one core per layer")
    cache = {} if _cache is None else _cache
    counts = [1] * len(layers)
    values = [_best_partition(workload, layer, 1, microbatch_size, hardware, cache) for layer in layers]
    microbatches = workload.batch_size // microbatch_size
    def pipeline_cost(service):
        return sum(service) + max(service) * max(0, microbatches - 1)
    while sum(counts) < len(compute_ids):
        service = [value[1] for value in values]
        current = pipeline_cost(service)
        remaining = len(compute_ids) - sum(counts)
        options = []
        for index, layer in enumerate(layers):
            for additional in range(1, remaining + 1):
                candidate = _best_partition(workload, layer, counts[index] + additional, microbatch_size, hardware, cache)
                if candidate is None:
                    continue
                changed = list(service)
                changed[index] = candidate[1]
                improvement = current - pipeline_cost(changed)
                if improvement > 1e-18:
                    options.append((-improvement / additional, -improvement, index, additional, candidate))
        if not options:
            break
        _, _, index, additional, candidate = min(options)
        counts[index] += additional
        values[index] = candidate
    result, offset = [], 0
    for layer, count, (partition, _) in zip(layers, counts, values):
        result.append((layer.id, compute_ids[offset:offset + count], partition))
        offset += count
    return tuple(result)


def _nearest_memory(cores, memory_ids, hardware):
    clusters = []
    for core in cores:
        try:
            die = int(core.split("/")[0].split(":")[-1])
            clusters.append(die // hardware.compute_per_cluster)
        except ValueError:
            return memory_ids[0]
    best = max(set(clusters), key=lambda cluster: (clusters.count(cluster), -cluster))
    return f"memory:{best}" if f"memory:{best}" in memory_ids else memory_ids[0]


def balanced_mapping(workload, compute_ids, memory_ids=("memory:0",), microbatch_size=1,
                     max_group_layers=3, pipeline=True, hardware=None, fd_policy="nearest", groups=None,
                     weight_policy="cold"):
    hardware = hardware or HardwareSpec()
    compute_ids, memory_ids = tuple(map(str, compute_ids)), tuple(map(str, memory_ids))
    if not compute_ids or not memory_ids or max_group_layers < 1:
        raise ValueError("planning needs compute and memory resources")
    if workload.batch_size % microbatch_size:
        raise ValueError("microbatch size must divide the workload batch")
    if fd_policy not in {"nearest", "interleave"}:
        raise ValueError("FD policy must be nearest or interleave")
    if groups is None:
        size = min(max_group_layers, len(compute_ids)) if pipeline else max_group_layers
        groups = tuple(LayerGroup(f"group{index}", tuple(layer.id for layer in workload.layers[start:start + size]),
                                 "spatial" if pipeline else "temporal")
                       for index, start in enumerate(range(0, len(workload.layers), size)))
    cache, mappings = {}, []
    for group in groups:
        if group.mode == "spatial":
            allocations = balanced_core_allocation(workload, group.layer_ids, compute_ids, microbatch_size, hardware, cache)
        else:
            allocations = []
            for layer_id in group.layer_ids:
                layer = workload.layer_map[layer_id]
                choices = [(count, _best_partition(workload, layer, count, microbatch_size, hardware, cache))
                           for count in range(1, len(compute_ids) + 1)]
                count, choice = min((item for item in choices if item[1] is not None), key=lambda item: (item[1][1], item[0]))
                allocations.append((layer_id, compute_ids[:count], choice[0]))
        for layer_id, cores, partition in allocations:
            memory = "interleave" if fd_policy == "interleave" else _nearest_memory(cores, memory_ids, hardware)
            mappings.append(LayerMapping(layer_id, tuple(cores), partition, memory, memory, memory,
                                         weight_policy=weight_policy))
    plan = MappingPlan(tuple(groups), tuple(mappings), microbatch_size, "multicast", memory_ids)
    validate_plan(workload, plan, compute_ids)
    return plan


def boundary_tensors(workload, layer_ids):
    selected = set(layer_ids)
    layers, tensors = workload.layer_map, workload.tensor_map
    inputs = tuple(dict.fromkeys(tensor for layer in layer_ids for tensor in layers[layer].inputs
                                if tensors[tensor].producer not in selected))
    outputs = tuple(layer.output for layer in workload.layers if layer.id in selected and
                    (layer.output in workload.output_ids or any(user not in selected for user in workload.consumers[layer.output])))
    return inputs, outputs


def _estimate_group(workload, group, allocations, microbatch, hardware, weight_policy="cold"):
    by_layer = {layer: (cores, part) for layer, cores, part in allocations}
    service = {layer: _partition_service(workload, workload.layer_map[layer], by_layer[layer][1], microbatch, hardware)
               for layer in group.layer_ids}
    finish = {}
    for layer_id in group.layer_ids:
        predecessors = [workload.tensor_map[tensor].producer for tensor in workload.layer_map[layer_id].inputs]
        finish[layer_id] = max((finish.get(previous, 0) for previous in predecessors), default=0) + service[layer_id]
    count = workload.batch_size // microbatch
    if group.mode == "spatial":
        delay = max(finish.values()) + (count - 1) * max(service.values())
    else:
        delay = count * sum(service.values())
    inputs, outputs = boundary_tensors(workload, group.layer_ids)
    per_batch_bytes = sum(workload.tensor_map[tensor].bytes for tensor in (*inputs, *outputs))
    weight_bytes = sum(workload.layer_map[layer].weight_bytes for layer in group.layer_ids) * (1 if weight_policy == "persistent" else count)
    dram_bytes = per_batch_bytes + weight_bytes
    # Aggregate DRAM is a lower-bound resource, final evaluator includes exact
    # FD placement, per-controller contention, and routed shared network links.
    delay += dram_bytes / (hardware.cluster_count * hardware.dram_bandwidth_Bps)
    macs = sum(workload.layer_map[layer].macs_per_sample for layer in group.layer_ids) * workload.batch_size
    local_io = sum((workload.tensor_map[workload.layer_map[layer].output].bytes +
                    sum(workload.tensor_map[tensor].bytes for tensor in workload.layer_map[layer].inputs))
                   for layer in group.layer_ids) + weight_bytes
    # UB traffic proxy for proposing groups only. Final named L1/L2/UB access
    # costs and exact mapped MAC counts are recomputed by the physical oracle.
    energy = (macs * hardware.effective_mac_energy_pj + dram_bytes * 8 * hardware.dram_energy_pj_per_bit +
              local_io * 8 * (sram_bank(hardware,'UB')['read_pj_per_bit'] + hardware.noc_energy_pj_per_bit)) * 1e-12
    return delay, energy, per_batch_bytes


def _score(delay, energy, objective):
    if objective == "latency":
        return delay
    if objective == "energy":
        return energy
    if objective == "edp":
        return delay * energy
    raise ValueError("planner objective must be latency, energy or edp")


def propose_plans(workload, compute_ids, memory_ids=("memory:0",), hardware=None,
                  microbatch_sizes=(1, 2, 4), max_group_layers=4, beam_width=4,
                  max_candidates=12, objective="edp", weight_policy="cold"):
    """Beam over contiguous DAG cuts; rank full plans with a separate evaluator."""
    hardware = hardware or HardwareSpec()
    compute_ids, memory_ids = tuple(compute_ids), tuple(memory_ids)
    if beam_width < 1 or max_candidates < 1 or max_group_layers < 1:
        raise ValueError("planner bounds must be positive")
    candidates = []
    n_layers = len(workload.layers)
    for microbatch in sorted(set(microbatch_sizes)):
        if microbatch < 1 or workload.batch_size % microbatch:
            continue
        # Entries retain the whole cut history; frontier pruning is bounded
        # rather than a claim of DP optimality under nonlinear global EDP.
        frontiers = {0: [((), (), 0.0, 0.0, 0)]}
        allocation_cache, partition_cache = {}, {}
        for start in range(n_layers):
            states = frontiers.get(start, ())
            if not states:
                continue
            for length in range(1, min(max_group_layers, n_layers - start) + 1):
                layer_ids = tuple(layer.id for layer in workload.layers[start:start + length])
                for mode in (("spatial",) if length == 1 else ("spatial", "temporal")):
                    if mode == "spatial" and length > len(compute_ids):
                        continue
                    key = layer_ids, mode
                    if key not in allocation_cache:
                        if mode == "spatial":
                            allocations = balanced_core_allocation(workload, layer_ids, compute_ids,
                                                                   microbatch, hardware, partition_cache)
                        else:
                            allocations = []
                            for layer_id in layer_ids:
                                layer = workload.layer_map[layer_id]
                                choices = [(count, _best_partition(workload, layer, count, microbatch, hardware, partition_cache))
                                           for count in range(1, len(compute_ids) + 1)]
                                count, value = min((choice for choice in choices if choice[1] is not None),
                                                   key=lambda item: (item[1][1], item[0]))
                                allocations.append((layer_id, compute_ids[:count], value[0]))
                            allocations = tuple(allocations)
                        estimate = _estimate_group(workload, LayerGroup("estimate", layer_ids, mode), allocations, microbatch, hardware, weight_policy)
                        allocation_cache[key] = allocations, estimate
                    allocations, (delay, energy, spill) = allocation_cache[key]
                    destination = frontiers.setdefault(start + length, [])
                    for groups, mappings, old_delay, old_energy, old_spill in states:
                        group = LayerGroup(f"group{len(groups)}", layer_ids, mode)
                        added = tuple(LayerMapping(layer, tuple(cores), part,
                                                   *( (_nearest_memory(cores, memory_ids, hardware),) * 3),
                                                   weight_policy=weight_policy)
                                      for layer, cores, part in allocations)
                        destination.append(((*groups, group), (*mappings, *added), old_delay + delay,
                                            old_energy + energy, old_spill + spill))
                    destination.sort(key=lambda state: (_score(state[2], state[3], objective),
                                                         state[4], tuple((group.layer_ids, group.mode) for group in state[0])))
                    del destination[beam_width:]
        for groups, mappings, delay, energy, spill in frontiers.get(n_layers, ()):
            for fd in ("nearest", "interleave"):
                selected = mappings if fd == "nearest" else tuple(replace(mapping, input_dram="interleave",
                                                                         weight_dram="interleave", output_dram="interleave")
                                                                  for mapping in mappings)
                plan = MappingPlan(groups, selected, microbatch, "multicast", memory_ids)
                validate_plan(workload, plan, compute_ids)
                candidates.append(PlanningCandidate(plan, delay, energy, spill))
    if not candidates:
        raise ValueError("no supplied microbatch size divides this workload")
    candidates.sort(key=lambda candidate: (_score(candidate.estimated_delay_s, candidate.estimated_energy_j, objective),
                                            candidate.boundary_bytes, repr(candidate.plan)))
    unique = []
    seen = set()
    for candidate in candidates:
        key = repr(candidate.plan)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    # Keep microbatch and graph-cut alternatives visible to the full oracle.
    # A lower-bound estimate must not consume the entire budget with one cut's
    # near-identical FD variants before another microbatch gets evaluated.
    selected = []
    represented_microbatches = set()
    for candidate in unique:
        if candidate.plan.microbatch_size not in represented_microbatches:
            selected.append(candidate)
            represented_microbatches.add(candidate.plan.microbatch_size)
            if len(selected) == max_candidates:
                return tuple(selected)
    represented_cuts = {tuple((group.layer_ids, group.mode) for group in candidate.plan.groups) for candidate in selected}
    for candidate in unique:
        cuts = tuple((group.layer_ids, group.mode) for group in candidate.plan.groups)
        if candidate not in selected and cuts not in represented_cuts:
            selected.append(candidate)
            represented_cuts.add(cuts)
            if len(selected) == max_candidates:
                return tuple(selected)
    for candidate in unique:
        if candidate not in selected:
            selected.append(candidate)
            if len(selected) == max_candidates:
                break
    return tuple(selected)


def rank_plans(candidates, evaluator, objective="edp", max_evaluations=None):
    """Evaluate proposals as complete plans; return best feasible plan and rows."""
    rows = []
    candidates = tuple(candidates)
    limit = len(candidates) if max_evaluations is None else max_evaluations
    for candidate in candidates[:limit]:
        result = evaluator(candidate.plan) if callable(evaluator) else evaluator.evaluate(candidate.plan)
        metrics = result.metrics if hasattr(result, "metrics") else result
        feasible = metrics.get("feasible", True)
        cost = _score(metrics["latency_s"], metrics["energy_j"], objective) if feasible else math.inf
        rows.append({"candidate": candidate, "metrics": metrics, "cost": cost})
    feasible = [row for row in rows if math.isfinite(row["cost"])]
    best = min(feasible, key=lambda row: row["cost"]) if feasible else None
    return best, tuple(rows)


def choose_plan(candidates, evaluator, objective="edp", max_evaluations=None):
    """JSON-ready full-evaluation summary, keeping estimates visibly separate."""
    candidates = tuple(candidates)
    best, rows = rank_plans(candidates, evaluator, objective, max_evaluations)
    summaries = []
    for index, row in enumerate(rows):
        candidate, metrics = row["candidate"], row["metrics"]
        summaries.append({"index": index, "microbatch_size": candidate.plan.microbatch_size,
                          "group_count": len(candidate.plan.groups),
                          "estimated_delay_s": candidate.estimated_delay_s,
                          "estimated_energy_j": candidate.estimated_energy_j,
                          "boundary_bytes": candidate.boundary_bytes,
                          "feasible": metrics.get("feasible", True),
                          "cost": row["cost"] if math.isfinite(row["cost"]) else None,
                          "latency_s": metrics["latency_s"], "energy_j": metrics["energy_j"],
                          "violations": metrics.get("violations", [])})
    summary = {"method": "bounded graph partition/microbatch proposals, full event evaluation ranking",
               "globally_exact": False, "objective": objective,
               "proposal_count": len(candidates), "evaluated_count": len(rows),
               "feasible_count": sum(row["cost"] is not None for row in summaries),
               "candidates": summaries, "best_metrics": None if best is None else best["metrics"]}
    return (None if best is None else best["candidate"].plan), summary
