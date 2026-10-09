"""Data requirements, DRAM materialization and executable collectives.

This module preserves logical tensor regions. Multicast shares each common
region once; ring mode explicitly forwards chunks with transfer dependencies.
Routing and timing belong to the hardware/evaluator layer.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

from .mapping import CompiledMapping, TensorRegion


@dataclass(frozen=True)
class Transfer:
    id: str
    src: str
    dsts: tuple[str, ...]
    bytes: int
    producer_units: tuple[str, ...]
    consumer_units: tuple[str, ...]
    kind: str
    tensor: str
    region: TensorRegion
    microbatch: int
    dependencies: tuple[str, ...] = ()

    @property
    def source_die(self):
        return self.src.split("/")[0]

    @property
    def destination_dies(self):
        return tuple(dict.fromkeys(destination.split("/")[0] for destination in self.dsts))


@dataclass(frozen=True)
class TensorLifetime:
    tensor: str
    microbatch: int
    endpoint: str
    region: TensorRegion
    bytes: int
    producer_units: tuple[str, ...]
    consumer_units: tuple[str, ...]
    kind: str = "activation"


@dataclass(frozen=True)
class CommunicationPlan:
    transfers: tuple[Transfer, ...]
    lifetimes: tuple[TensorLifetime, ...]
    unit_dependencies: tuple[tuple[str, tuple[str, ...]], ...]
    collective: str

    @property
    def transfer_map(self):
        return {transfer.id: transfer for transfer in self.transfers}

    @property
    def dependency_map(self):
        return dict(self.unit_dependencies)

    @property
    def total_bytes(self):
        return sum(transfer.bytes for transfer in self.transfers)

    @property
    def dram_read_bytes(self):
        return sum(transfer.bytes for transfer in self.transfers if transfer.src.startswith("memory:"))

    @property
    def dram_write_bytes(self):
        return sum(transfer.bytes for transfer in self.transfers if any(dst.startswith("memory:") for dst in transfer.dsts))


@dataclass(frozen=True)
class _Demand:
    src: str
    dst: str
    tensor: str
    region: TensorRegion
    bits: int
    microbatch: int
    producer_units: tuple[str, ...]
    consumer_unit: str
    kind: str
    dependencies: tuple[str, ...] = ()
    reload_cohort: int = 0


def _memory_regions(region, selector, memory_ids, placement_region=None):
    if selector == "direct":
        raise ValueError("a materialized/external tensor requires a DRAM endpoint")
    if selector != "interleave":
        return ((selector, region),)
    # Stripes are whole tensor elements, not fractional byte approximations.
    placement = placement_region or region
    axis = max(range(4), key=lambda index: placement.shape[index])
    count = min(len(memory_ids), placement.shape[axis])
    left, right = placement.bounds[axis]
    result = []
    for index in range(count):
        bounds = list(placement.bounds)
        bounds[axis] = (left + (right - left) * index // count,
                        left + (right - left) * (index + 1) // count)
        intersection = TensorRegion(tuple(bounds)).intersection(region)
        if intersection is not None:
            result.append((memory_ids[index], intersection))
    return tuple(result)


def _atomic_demands(demands):
    """Partition overlapping consumer rectangles into unique exact regions."""
    groups = {}
    for demand in demands:
        key = (demand.src, demand.tensor, demand.bits, demand.microbatch, demand.producer_units,
               demand.kind, demand.dependencies, demand.reload_cohort)
        groups.setdefault(key, []).append(demand)
    result = []
    for key, selected in groups.items():
        cuts = [sorted({value for demand in selected for value in demand.region.bounds[axis]})
                for axis in range(4)]
        ranges = [tuple(zip(axis_cuts[:-1], axis_cuts[1:])) for axis_cuts in cuts]
        for bounds in product(*ranges):
            users = [demand for demand in selected if all(lo <= left and right <= hi
                     for (lo,hi),(left,right) in zip(demand.region.bounds,bounds))]
            if not users:
                continue
            region = TensorRegion(tuple(bounds))
            src, tensor, bits, microbatch, producers, kind, dependencies, _ = key
            # Local consumers still depend on the producer but do not transmit.
            remote = [demand for demand in users if demand.dst != src]
            if remote:
                result.append((src, tuple(dict.fromkeys(demand.dst for demand in remote)),
                               region.elements * bits // 8, producers,
                               tuple(dict.fromkeys(demand.consumer_unit for demand in remote)),
                               kind, tensor, region, microbatch, dependencies))
    return result


def build_communication(workload, compiled: CompiledMapping, collective="multicast"):
    if collective not in {"unicast", "multicast", "ring"}:
        raise ValueError("collective must be unicast, multicast, or ring")
    if workload.fingerprint != compiled.workload.fingerprint:
        raise ValueError("communication workload differs from compiled mapping")
    plan, tensors = compiled.plan, workload.tensor_map
    unit_map = compiled.unit_map
    owners = {}
    requirements = {}
    for unit in compiled.units:
        owners.setdefault((unit.output_tensor, unit.microbatch), []).append(unit)
        for req in unit.inputs:
            requirements.setdefault((req.tensor_id, unit.microbatch), []).append((unit, req.region))
    transfers, demands, lifetimes = [], [], []
    writes = {}
    group_map, consumers = plan.group_map, workload.consumers
    # Write every boundary/model-output region exactly once. A residual value
    # with both local and external-group users has one shared materialization.
    for owner in compiled.units:
        tensor = tensors[owner.output_tensor]
        boundary = (owner.output_tensor in workload.output_ids or
                    any(group_map[layer].id != owner.group_id for layer in consumers[owner.output_tensor]))
        if boundary:
            selector = plan.mapping_map[owner.layer_id].output_dram
            for memory, region in _memory_regions(owner.output_region, selector, plan.memory_ids,
                                                  TensorRegion.full(tensor.shape)):
                transfer = Transfer(f"transfer{len(transfers)}", owner.chiplet, (memory,),
                                    region.elements * tensor.bytes_per_element, (owner.id,), (),
                                    "output_write" if owner.output_tensor in workload.output_ids else "inter_group_write",
                                    owner.output_tensor, region, owner.microbatch)
                transfers.append(transfer)
                writes.setdefault(owner.id, []).append((memory, region, transfer.id))
        users = tuple(dict.fromkeys(unit.id for unit, region in requirements.get((owner.output_tensor, owner.microbatch), ())
                                   if unit.chiplet == owner.chiplet and region.intersection(owner.output_region) is not None))
        lifetimes.append(TensorLifetime(owner.output_tensor, owner.microbatch, owner.chiplet, owner.output_region,
                                       owner.output_bytes, (owner.id,), users))
    for unit in compiled.units:
        mapping = plan.mapping_map[unit.layer_id]
        for req in unit.inputs:
            tensor = tensors[req.tensor_id]
            if not tensor.producer:
                for memory, region in _memory_regions(req.region, mapping.input_dram, plan.memory_ids,
                                                      TensorRegion.full(tensor.shape)):
                    demands.append(_Demand(memory, unit.chiplet, req.tensor_id, region, tensor.bits,
                                           unit.microbatch, (), unit.id, "dram_input"))
                continue
            covered = 0
            for owner in owners[(req.tensor_id, unit.microbatch)]:
                intersection = owner.output_region.intersection(req.region)
                if intersection is None:
                    continue
                covered += intersection.elements
                if owner.group_id == unit.group_id:
                    demands.append(_Demand(owner.chiplet, unit.chiplet, req.tensor_id, intersection,
                                           tensor.bits, unit.microbatch, (owner.id,), unit.id, "activation"))
                else:
                    # Cross-group reads follow the producer's materialization
                    # endpoint, rather than inventing a new input placement.
                    for memory, write_region, write_id in writes[owner.id]:
                        read_region = write_region.intersection(intersection)
                        if read_region is not None:
                            demands.append(_Demand(memory, unit.chiplet, req.tensor_id, read_region,
                                                   tensor.bits, unit.microbatch, (owner.id,), unit.id,
                                                   "inter_group_read", (write_id,)))
            if covered != req.region.elements:
                raise ValueError(f"producer coverage missing for {unit.id}/{req.tensor_id}")
        if unit.weight_region is not None:
            layer = workload.layer_map[unit.layer_id]
            full_weight = TensorRegion(((0, layer.weight_shape[0]), (0, layer.weight_shape[1]),
                                        (0, layer.kernel[0]), (0, layer.kernel[1])))
            for memory, region in _memory_regions(unit.weight_region, mapping.weight_dram, plan.memory_ids, full_weight):
                demands.append(_Demand(memory, unit.chiplet, f"weights:{unit.layer_id}", region, layer.weight_bits,
                                       unit.microbatch, (), unit.id, "dram_weight", reload_cohort=unit.tile_index))
    # Share repeated reads within a microbatch. Different microbatches remain
    # cold unless the evaluator explicitly enables a capacity-backed cache.
    for src, dsts, size, producers, users, kind, tensor, region, microbatch, dependencies in _atomic_demands(demands):
        if collective == "unicast":
            for destination in dsts:
                selected = tuple(unit for unit in users if unit_map[unit].chiplet == destination)
                transfers.append(Transfer(f"transfer{len(transfers)}", src, (destination,), size, producers,
                                          selected, kind, tensor, region, microbatch, dependencies))
        elif collective == "ring" and kind == "activation" and len(dsts) > 1:
            current, previous = src, None
            # Each independent chunk travels through an explicit forwarding
            # chain. Its next hop cannot start before the previous hop finishes.
            for destination in dsts:
                selected = tuple(unit for unit in users if unit_map[unit].chiplet == destination)
                transfer = Transfer(f"transfer{len(transfers)}", current, (destination,), size,
                                    producers if previous is None else (), selected, "ring_forward", tensor,
                                    region, microbatch, dependencies if previous is None else (previous,))
                transfers.append(transfer)
                previous, current = transfer.id, destination
        else:
            transfers.append(Transfer(f"transfer{len(transfers)}", src, dsts, size, producers, users,
                                      kind, tensor, region, microbatch, dependencies))
    dependency_map = {unit.id: list(unit.dependencies) for unit in compiled.units}
    for transfer in transfers:
        for consumer in transfer.consumer_units:
            dependency_map[consumer].append(transfer.id)
        if transfer.dsts and not transfer.dsts[0].startswith("memory:"):
            for destination in transfer.dsts:
                users = tuple(unit for unit in transfer.consumer_units if unit_map[unit].chiplet == destination)
                lifetimes.append(TensorLifetime(transfer.tensor, transfer.microbatch, destination, transfer.region,
                                               transfer.bytes, transfer.producer_units, users,
                                               "weight" if transfer.kind == "dram_weight" else "received"))
    return CommunicationPlan(tuple(transfers), tuple(lifetimes),
                             tuple((unit, tuple(dict.fromkeys(dependencies))) for unit, dependencies in dependency_map.items()),
                             collective)
