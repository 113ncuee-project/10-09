"""Directed on-die fabrics and exact region multicast for the hybrid model.

Ring packets stay on one directed ring. Different rings communicate only by
returning to shared L2 and explicitly reinjecting. Timing is a conservative
phase link-service bound, rather than packet arbitration or mean-hop scaling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections import OrderedDict
from functools import lru_cache
from itertools import product, zip_longest
import math

from .mapping import TensorRegion, required_input_regions
from .costs import sram_access_energy


@dataclass(frozen=True)
class CoreService:
    compute_s: float
    buffer_s: float
    noc_s: float
    service_s: float
    mac_energy_j: float
    sram_energy_j: float
    noc_energy_j: float
    sram_bits: int
    noc_injected_bytes: int
    noc_hop_bytes: int
    mac_cycles: int
    vector_cycles: int
    utilization: float
    input_noc_s: float
    output_noc_s: float
    l1_s: float
    input_buffer_s: float
    output_buffer_s: float
    ring_resources: tuple[str, ...]
    traffic_evidence: dict
    ub_read_bytes: int

    @property
    def energy_j(self):
        return self.mac_energy_j + self.sram_energy_j + self.noc_energy_j

    @property
    def input_noc_energy_j(self):
        return (self.noc_energy_j * self.traffic_evidence["input_hop_bytes"] / self.noc_hop_bytes
                if self.noc_hop_bytes else 0.0)

    @property
    def output_noc_energy_j(self):
        return self.noc_energy_j - self.input_noc_energy_j

    def to_dict(self):
        return {**asdict(self), "energy_j": self.energy_j}


@dataclass(frozen=True)
class OnDieFabric:
    topology: str
    pe_count: int
    ring_count: int
    edges: tuple[tuple[str, str], ...]
    rings: tuple[tuple[str, ...], ...]
    coordinates: dict[str, tuple[int, int]]

    def ring_for_pe(self, pe: str) -> int:
        for index, ring in enumerate(self.rings):
            if pe in ring:
                return index
        raise ValueError("PE is not part of the directed ring fabric")

    def route(self, src: str, dst: str) -> tuple[str, ...]:
        if src == dst:
            return (src,)
        if self.topology == "multi_ring":
            selected = dst if src == "l2" else src
            ring = self.rings[self.ring_for_pe(selected)]
            if src not in ring or dst not in ring:
                raise ValueError("packets cannot cross rings; stage through L2 and reinject explicitly")
            position = ring.index(src)
            path = [src]
            while path[-1] != dst:
                position = (position + 1) % len(ring)
                path.append(ring[position])
                if len(path) > len(ring):
                    raise ValueError("invalid directed ring route")
            return tuple(path)
        path = [src]
        current = "pe:0" if src == "l2" else src
        if src == "l2":
            path.append(current)
        target = "pe:0" if dst == "l2" else dst
        row, col = self.coordinates[current]
        dest_row, dest_col = self.coordinates[target]
        inverse = {coordinate: pe for pe, coordinate in self.coordinates.items()}
        while col != dest_col:
            col += 1 if dest_col > col else -1
            if (row, col) not in inverse:
                raise ValueError("XY route reaches an absent PE in an incomplete mesh row")
            path.append(inverse[(row, col)])
        while row != dest_row:
            row += 1 if dest_row > row else -1
            path.append(inverse[(row, col)])
        if dst == "l2":
            path.append("l2")
        if any(edge not in self.edges for edge in zip(path, path[1:])):
            raise ValueError("mesh route uses an absent physical edge")
        return tuple(path)


@lru_cache(maxsize=64)
def build_on_die_fabric(hardware) -> OnDieFabric:
    count = hardware.pe_count
    side = math.ceil(math.sqrt(count))
    coords = {f"pe:{pe}": divmod(pe, side) for pe in range(count)}
    if hardware.noc_topology == "multi_ring":
        rings = tuple(("l2", *(f"pe:{pe}" for pe in range(count * ring // hardware.ring_count,
                                                               count * (ring + 1) // hardware.ring_count)))
                      for ring in range(hardware.ring_count))
        edges = tuple(edge for ring in rings for edge in zip(ring, (*ring[1:], ring[0])))
    else:
        rings = ()
        inverse = {coordinate: pe for pe, coordinate in coords.items()}
        directed = [("l2", "pe:0"), ("pe:0", "l2")]
        for pe, (row, col) in coords.items():
            for other in ((row + 1, col), (row, col + 1)):
                if other in inverse:
                    directed.extend(((pe, inverse[other]), (inverse[other], pe)))
        edges = tuple(directed)
    return OnDieFabric(hardware.noc_topology, count, hardware.ring_count, edges, rings, coords)


def _split_pe_outputs(region: TensorRegion, pes: int) -> tuple[TensorRegion, ...]:
    """Partition nonempty B/H/W positions first, K next; never split C sums."""
    boxes = [region]
    while len(boxes) < pes:
        choices = []
        for index, box in enumerate(boxes):
            for rank, axis in enumerate((2, 3, 0, 1)):
                if box.shape[axis] > 1:
                    choices.append((axis == 1, -box.elements, rank, index, axis))
        if not choices:
            break
        _, _, _, index, axis = min(choices)
        box = boxes.pop(index)
        lo, hi = box.bounds[axis]
        midpoint = lo + (hi - lo) // 2
        left, right = list(box.bounds), list(box.bounds)
        left[axis], right[axis] = (lo, midpoint), (midpoint, hi)
        boxes[index:index] = [TensorRegion(tuple(left)), TensorRegion(tuple(right))]
    return tuple(boxes)


def _atomic_multicast(demands):
    """Atomic region -> destinations; shared by mesh and ring without bias."""
    groups = {}
    for tensor, region, bits, destination in demands:
        groups.setdefault((tensor, bits), []).append((region, destination))
    result = []
    for (tensor, bits), members in groups.items():
        cuts = [sorted({v for region, _ in members for v in region.bounds[axis]}) for axis in range(4)]
        ranges = [tuple(zip(axis[:-1], axis[1:])) for axis in cuts]
        for bounds in product(*ranges):
            users = tuple(dict.fromkeys(destination for region, destination in members
                                         if all(lo <= left and right <= hi for (lo,hi),(left,right) in zip(region.bounds,bounds))))
            if users:
                atomic = TensorRegion(tuple(bounds))
                result.append((tensor, atomic, bits, users))
    return tuple(result)


def _phase_routes(fabric, packets, source="l2", output=False):
    loads = {}
    injected = delivered = hop_bytes = longest_hops = 0
    packet_evidence = []
    for tensor, region, bits, destinations in packets:
        amount = region.elements * bits // 8
        if not amount:
            continue
        groups = {}
        for destination in destinations:
            ring = fabric.ring_for_pe(destination) if fabric.topology == "multi_ring" else 0
            groups.setdefault(ring, []).append(destination)
        if output:
            groups = {0: list(destinations)}
        copies = len(groups)
        injected += amount * copies
        delivered += amount * len(destinations)
        union, routes = set(), []
        for group in groups.values():
            group_union = set()
            for destination in group:
                route = fabric.route(destination, "l2") if output else fabric.route(source, destination)
                routes.append(route)
                longest_hops = max(longest_hops, len(route) - 1)
                group_union.update(zip(route, route[1:]))
            for edge in group_union:
                loads[edge] = loads.get(edge, 0) + amount
            union.update(group_union)
            hop_bytes += amount * len(group_union)
        packet_evidence.append({"tensor": tensor, "bounds": region.bounds, "bytes": amount,
                                "destinations": tuple(destinations), "injected_copies": copies,
                                "routes": tuple(routes), "union_edges": tuple(sorted(union))})
    return loads, injected, delivered, hop_bytes, longest_hops, packet_evidence


def _phase_time(loads, hops, hardware):
    if not loads:
        return 0.0
    return max(loads.values()) / hardware.noc_bandwidth_Bps + hops * hardware.pe_router_latency_cycles / hardware.noc_clock_hz


def _pe_cycles(region, layer, workload, hardware, c_extent=None, include_vector=True):
    b, k, h, w = region.shape
    if not layer.weight_shape:
        mac_cycles = 0
    elif layer.kind == "Conv2d":
        cin = layer.weight_shape[1] if c_extent is None else c_extent
        cout = workload.tensor_map[layer.output].shape[1]
        out_group = cout // layer.groups
        lo, hi = region.bounds[1]
        k_rounds = sum(math.ceil(max(0, min(hi, (group + 1) * out_group) - max(lo, group * out_group)) / hardware.lanes)
                       for group in range(lo // out_group, (hi - 1) // out_group + 1))
        mac_cycles = b * h * w * k_rounds * math.ceil(cin / hardware.vector_size) * math.prod(layer.kernel)
    else:
        cin = math.prod(layer.weight_shape[1:]) if c_extent is None else c_extent
        mac_cycles = b * math.ceil(k / hardware.lanes) * math.ceil(cin / hardware.vector_size)
    vector_cycles = math.ceil(region.elements * layer.vector_ops_per_element / hardware.lanes) if include_vector else 0
    return mac_cycles, vector_cycles


class _BankCache:
    """Bounded whole-region LRU; containment hits never consume pooled SRAM."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.shapes = {}
        self.order = {}
        self.starts = {}
        self.stamp = 0
        self.bytes = 0
        self.peak = 0

    def ensure(self, tensor, region, bits):
        amount = region.elements * bits // 8
        if amount > self.capacity:
            raise ValueError("internal microkernel exceeds a named SRAM bank")
        key = tensor,bits,region.bounds
        self.stamp += 1
        if key in self.entries:
            self.entries.move_to_end(key)
            self.order[key] = self.stamp
            return False
        groups = self.shapes.get((tensor,bits),{})
        shape = region.shape
        covering = []
        for old_shape,keys in groups.items():
            # Equal extents can contain only at identical coordinates, already
            # checked by the hash lookup. Avoid scanning hundreds of tiny tiles.
            if old_shape!=shape and all(a>=b for a,b in zip(old_shape,shape)):
                for old_key in self._candidates(keys,tensor,bits,old_shape,shape,region.bounds):
                    if all(lo <= left and right <= hi for (lo,hi),(left,right) in zip(old_key[2],region.bounds)):
                        covering.append(old_key)
        if covering:
            oldest = min(covering,key=self.order.__getitem__)
            self.entries.move_to_end(oldest)
            self.order[oldest] = self.stamp
            return False
        # Remove subsets before retaining a covering region. Partially overlapping
        # boxes stay distinct and pay capacity; this conservative cache does not
        # pretend that a halo overlap guarantees all data is still resident.
        subsets = []
        for old_shape,keys in groups.items():
            if old_shape!=shape and all(a>=b for a,b in zip(shape,old_shape)):
                for old_key in self._candidates(keys,tensor,bits,old_shape,shape,region.bounds):
                    if all(lo <= left and right <= hi for (lo,hi),(left,right) in zip(region.bounds,old_key[2])):
                        subsets.append(old_key)
        for old_key in subsets:
            self._remove(old_key)
        while self.bytes + amount > self.capacity and self.entries:
            self._remove(next(iter(self.entries)))
        self.entries[key] = (tensor, region, bits, amount)
        self.order[key] = self.stamp
        self.shapes.setdefault((tensor,bits),{}).setdefault(shape,set()).add(key)
        for axis,(lo,hi) in enumerate(region.bounds):
            self.starts.setdefault((tensor,bits,axis,lo),set()).add(key)
        self.bytes += amount
        self.peak = max(self.peak, self.bytes)
        return True

    def _remove(self,key):
        tensor,region,bits,amount = self.entries.pop(key)
        self.bytes -= amount
        self.order.pop(key)
        shapes = self.shapes[(tensor,bits)]
        group = shapes[region.shape]
        group.remove(key)
        if not group:
            del shapes[region.shape]
        if not shapes:
            del self.shapes[(tensor,bits)]
        for axis,(lo,hi) in enumerate(region.bounds):
            index = tensor,bits,axis,lo
            group = self.starts[index]
            group.remove(key)
            if not group:
                del self.starts[index]

    def _candidates(self,keys,tensor,bits,old_shape,shape,bounds):
        # Equal-sized axes require equal origins for containment. In flattened
        # spatial tensors this removes unrelated channels from the candidate set.
        buckets = [self.starts.get((tensor,bits,axis,bounds[axis][0]),set())
                   for axis in range(4) if old_shape[axis]==shape[axis]]
        return keys.intersection(min(buckets,key=len)) if buckets else keys


def _internal_footprints(unit, layer, workload, hardware, dims, c_tile, active_pes):
    b, k, h, w = dims
    output = b * k * h * w
    if layer.kind == "Conv2d":
        shape = workload.tensor_map[layer.inputs[0]].shape
        out_group = workload.tensor_map[layer.output].shape[1] // layer.groups
        group_count = min(layer.groups, math.ceil((k + out_group - 1) / out_group))
        ih = min(shape[2], (h - 1) * layer.stride[0] + layer.dilation[0] * (layer.kernel[0] - 1) + 1)
        iw = min(shape[3], (w - 1) * layer.stride[1] + layer.dilation[1] * (layer.kernel[1] - 1) + 1)
        activation = b * c_tile * group_count * ih * iw * hardware.activation_bytes
    elif layer.kind == "Linear":
        activation = b * c_tile * hardware.activation_bytes
    elif layer.kind in ("MaxPool2d", "AvgPool2d"):
        shape = workload.tensor_map[layer.inputs[0]].shape
        ih = min(shape[2], (h - 1) * layer.stride[0] + layer.dilation[0] * (layer.kernel[0] - 1) + 1)
        iw = min(shape[3], (w - 1) * layer.stride[1] + layer.dilation[1] * (layer.kernel[1] - 1) + 1)
        activation = b * k * ih * iw * hardware.activation_bytes
    elif layer.kind == "AdaptiveAvgPool2d":
        shape, out = workload.tensor_map[layer.inputs[0]].shape, workload.tensor_map[layer.output].shape
        ih, iw = min(shape[2], math.ceil((h + 1) * shape[2] / out[2])), min(shape[3], math.ceil((w + 1) * shape[3] / out[3]))
        activation = b * k * ih * iw * hardware.activation_bytes
    else:
        activation = output * hardware.activation_bytes * len(layer.inputs)
    weight = k * c_tile * math.prod(layer.kernel) * hardware.weight_bytes if layer.weight_shape else 0
    accumulator = output * (hardware.psum_bytes if unit.macs else hardware.activation_bytes)
    return {
        "AL1": activation, "WL1": weight, "OL1": accumulator,
        "AL2": min(unit.input_bytes, activation * active_pes),
        "WL2": min(unit.weight_bytes, weight * active_pes),
        "OL2": output * active_pes * hardware.activation_bytes,
    }


def _internal_tiling(unit, layer, workload, hardware, regions):
    dims = tuple(max(region.shape[axis] for region in regions) for axis in range(4))
    dims = tuple(min(extent, hint or extent) for extent, hint in zip(dims, unit.microtile_shape))
    cin = (layer.weight_shape[1] if layer.kind == "Conv2d" else math.prod(layer.weight_shape[1:])) if layer.weight_shape else 1
    c_tile = min(cin, unit.channel_tile or cin)
    capacities = {"AL1": hardware.l1_activation_bytes, "WL1": hardware.l1_weight_bytes,
                  "OL1": hardware.l1_output_bytes, "AL2": hardware.l2_activation_bytes,
                  "WL2": hardware.l2_weight_bytes, "OL2": hardware.l2_output_bytes}
    def score(shape, channels):
        footprint = _internal_footprints(unit, layer, workload, hardware, shape, channels, len(regions))
        ratios = [footprint[bank] / capacities[bank] for bank in capacities]
        return max(ratios), sum(ratios), footprint
    while score(dims, c_tile)[0] > 1:
        candidates = []
        for axis, extent in enumerate(dims):
            if extent > 1:
                trial = list(dims)
                trial[axis] = math.ceil(extent / 2)
                trial = tuple(trial)
                candidates.append((*score(trial, c_tile)[:2], trial, c_tile))
        if c_tile > 1 and layer.weight_shape:
            channels = math.ceil(c_tile / 2)
            candidates.append((*score(dims, channels)[:2], dims, channels))
        if not candidates:
            raise ValueError("even a minimal internal microkernel cannot fit the named SRAM banks")
        _, _, dims, c_tile = min(candidates)
    return dims, c_tile, capacities, score(dims, c_tile)[2], cin


def _internal_tiles(region, dims, dataflow='output_stationary'):
    axes = [tuple((start, min(end, start + extent)) for start in range(lo, end, extent))
            for (lo, end), extent in zip(region.bounds, dims)]
    # C reduction remains inside one output microtile. Changing K/spatial order
    # changes real containment-cache refills, never an efficiency multiplier.
    order = (0, 2, 3, 1) if dataflow == 'activation_reuse' else (0, 1, 3, 2) if dataflow == 'weight_reuse' else (0, 1, 2, 3)
    def ordered():
        for selected in product(*(axes[axis] for axis in order)):
            bounds = [None] * 4
            for axis, value in zip(order, selected):
                bounds[axis] = value
            yield TensorRegion(tuple(bounds))
    return ordered()


def _chunk_inputs(workload, layer, output, c_start, c_end):
    full = required_input_regions(workload, layer, output)
    if layer.kind == "Conv2d":
        shape = workload.tensor_map[layer.inputs[0]].shape
        per_input = shape[1] // layer.groups
        per_output = workload.tensor_map[layer.output].shape[1] // layer.groups
        lo, hi = output.bounds[1]
        return tuple((full[0].tensor_id, TensorRegion((full[0].region.bounds[0],
                      (group * per_input + c_start, group * per_input + c_end),
                      *full[0].region.bounds[2:])), workload.tensor_map[full[0].tensor_id].bits)
                     for group in range(lo // per_output, (hi - 1) // per_output + 1))
    if layer.kind == "Linear":
        req = full[0]
        shape = workload.tensor_map[req.tensor_id].shape
        if shape[2:] == (1, 1):
            return ((req.tensor_id, TensorRegion((output.bounds[0], (c_start, c_end), (0, 1), (0, 1))),
                     workload.tensor_map[req.tensor_id].bits),)
        # Feature-C uses flattened C/H/W indices if the input is not 1x1.
        result, cursor = [], c_start
        while cursor < c_end:
            channel, rem = divmod(cursor, shape[2] * shape[3])
            row, column = divmod(rem, shape[3])
            length = min(c_end - cursor, shape[3] - column)
            region = TensorRegion((output.bounds[0], (channel, channel + 1), (row, row + 1), (column, column + length)))
            result.append((req.tensor_id, region, workload.tensor_map[req.tensor_id].bits))
            cursor += length
        return tuple(result)
    return tuple((req.tensor_id, req.region, workload.tensor_map[req.tensor_id].bits) for req in full)


def _logical_service(unit, layer, workload, hardware):
    """Exact per-PE demand/ownership and separate fill/compute/drain stages."""
    fabric = build_on_die_fabric(hardware)
    die = unit.chiplet.split("/")[0]
    core = int(unit.chiplet.split("/core:")[1]) if "/core:" in unit.chiplet else 0
    if not 0 <= core < hardware.cores_per_die:
        raise ValueError("work unit refers to an unavailable core slot")
    regions = _split_pe_outputs(unit.output_region, hardware.pes_per_core)
    pes = tuple(f"pe:{core * hardware.pes_per_core + index}" for index in range(len(regions)))
    activation_demands, weight_demands, output_packets = [], [], []
    pe_evidence, cycles_by_pe = [], []
    l1_transfer_bytes = 0
    for pe, region in zip(pes, regions):
        inputs = required_input_regions(workload, layer, region)
        for req in inputs:
            bits = workload.tensor_map[req.tensor_id].bits
            activation_demands.append((req.tensor_id, req.region, bits, pe))
            l1_transfer_bytes += req.region.elements * bits // 8
        weight = None
        if unit.weight_region is not None:
            weight = TensorRegion((region.bounds[1], *unit.weight_region.bounds[1:]))
            weight_demands.append((f"weight:{layer.id}", weight, layer.weight_bits, pe))
            l1_transfer_bytes += weight.elements * layer.weight_bits // 8
        output_bits = workload.tensor_map[layer.output].bits
        output_packets.append((layer.output, region, output_bits, (pe,)))
        l1_transfer_bytes += region.elements * output_bits // 8
        mac_cycles, vector_cycles = _pe_cycles(region, layer, workload, hardware)
        cycles_by_pe.append((mac_cycles, vector_cycles))
        pe_evidence.append({"pe": pe, "output_bounds": region.bounds, "output_elements": region.elements,
                            "mac_cycles": mac_cycles, "vector_cycles": vector_cycles,
                            "inputs": [{"tensor": req.tensor_id, "bounds": req.region.bounds} for req in inputs],
                            "weight_bounds": weight.bounds if weight is not None else None})
    input_packets = (*_atomic_multicast(activation_demands), *_atomic_multicast(weight_demands))
    in_loads, in_injected, in_delivered, in_hops, in_distance, in_evidence = _phase_routes(fabric, input_packets)
    out_loads, out_injected, out_delivered, out_hops, out_distance, out_evidence = _phase_routes(fabric, output_packets, output=True)
    unique_input_bytes = sum(region.elements * bits // 8 for _, region, bits, _ in input_packets)
    # Compiler envelopes can include unused gaps for stride > kernel; their
    # payload is conservative. PE union may be smaller, but never larger.
    if unique_input_bytes > unit.input_bytes + unit.weight_bytes:
        raise ValueError("PE input union exceeds compiled work-unit payload")
    if out_delivered != unit.output_bytes:
        raise ValueError("PE output ownership does not cover work-unit output bytes")
    mac_cycles = max((value[0] for value in cycles_by_pe), default=0)
    vector_cycles = max((value[1] for value in cycles_by_pe), default=0)
    compute_s = max((sum(value) for value in cycles_by_pe), default=0) / hardware.compute_clock_hz
    activation_reads = math.ceil(unit.macs / hardware.lanes) if unit.macs else unit.input_bytes
    weight_reads = unit.macs
    accumulator_accesses = 2 * math.ceil(unit.macs / hardware.vector_size) if unit.macs else 0
    l1_compute_bits = activation_reads * hardware.activation_bits + weight_reads * hardware.weight_bits + accumulator_accesses * hardware.psum_bits
    global_input_bits = 16 * (unit.input_bytes + unit.weight_bytes)
    global_output_bits = 16 * unit.output_bytes
    sram_bits = l1_compute_bits + 8 * l1_transfer_bytes + global_input_bits + global_output_bits
    # Logical ownership helper only. Final physical service below reconstructs
    # every bank miss/access; this coarse projection is never the PPA oracle.
    coarse_sram_energy, _ = sram_access_energy(hardware,{
        'AL1':{'read_bits':activation_reads*hardware.activation_bits},
        'WL1':{'read_bits':weight_reads*hardware.weight_bits},
        'OL1':{'read_bits':accumulator_accesses//2*hardware.psum_bits,
               'write_bits':accumulator_accesses//2*hardware.psum_bits},
        'UB':{'read_bits':(global_input_bits+global_output_bits)//2,
              'write_bits':(global_input_bits+global_output_bits)//2+8*l1_transfer_bytes}})
    l1_s = l1_compute_bits / 8 / (hardware.l1_bandwidth_Bps * max(1, len(pes)))
    buffer_rate = min(hardware.l2_bandwidth_Bps, hardware.unified_bandwidth_Bps)
    input_buffer_s = global_input_bits / 8 / buffer_rate
    output_buffer_s = global_output_bits / 8 / buffer_rate
    input_noc_s, output_noc_s = _phase_time(in_loads, in_distance, hardware), _phase_time(out_loads, out_distance, hardware)
    buffer_s = input_buffer_s + l1_s + output_buffer_s
    noc_s = input_noc_s + output_noc_s
    service_s = max(input_buffer_s, input_noc_s) + max(compute_s, l1_s) + max(output_buffer_s, output_noc_s)
    if hardware.noc_topology == "multi_ring":
        ring_resources = tuple(f"noc:{die}/ring:{index}" for index in sorted({fabric.ring_for_pe(pe) for pe in pes}))
    else:
        ring_resources = (f"noc:{die}/mesh",)
    hop_bytes = in_hops + out_hops
    evidence = {
        "topology": hardware.noc_topology, "model": "directed routes with atomic-region multicast union edges",
        "pe_output_regions": pe_evidence, "input_packets": in_evidence, "output_packets": out_evidence,
        "input_link_loads": [{"source": a, "destination": b, "bytes": amount} for (a, b), amount in sorted(in_loads.items())],
        "output_link_loads": [{"source": a, "destination": b, "bytes": amount} for (a, b), amount in sorted(out_loads.items())],
        "unique_input_weight_bytes": unique_input_bytes, "input_delivered_bytes": in_delivered,
        "output_delivered_bytes": out_delivered, "input_injected_bytes": in_injected,
        "output_injected_bytes": out_injected, "input_hop_bytes": in_hops,
        "output_hop_bytes": out_hops, "ring_resources": ring_resources,
        "limitations": "dense receptive-field envelopes; atomic packets/phase service; no VC/arbitration model; shared L2 source",
    }
    ideal = unit.macs / hardware.per_core_mac_rate if unit.macs else 0.0
    return CoreService(compute_s, buffer_s, noc_s, service_s,
                       unit.macs * hardware.effective_mac_energy_pj * 1e-12,
                       coarse_sram_energy,
                       hop_bytes * 8 * hardware.noc_energy_pj_per_bit * 1e-12,
                       sram_bits, in_injected + out_injected, hop_bytes, mac_cycles, vector_cycles,
                       ideal / compute_s if compute_s else 0.0,
                       input_noc_s, output_noc_s, l1_s, input_buffer_s, output_buffer_s, ring_resources, evidence,
                       unit.input_bytes + unit.weight_bytes)


@lru_cache(maxsize=4096)
def _cached_core_service(unit, layer, workload, hardware):
    """Bank-backed internal microkernels, projected into macro three-stage cost.

    Full received operands stay backed in unified SRAM until the unit finishes.
    Internal bank misses reload from that backing; they never assume free DRAM
    or cross-bank capacity. Microkernel streaming has no fine-grained overlap
    credit in the read/compute/write projection used by the outer scheduler.
    """
    logical = _logical_service(unit, layer, workload, hardware)
    records = logical.traffic_evidence["pe_output_regions"]
    pes = tuple(record["pe"] for record in records)
    regions = tuple(TensorRegion(tuple(record["output_bounds"])) for record in records)
    fabric = build_on_die_fabric(hardware)
    dims, channel_tile, capacities, envelope, cin = _internal_tiling(unit, layer, workload, hardware, regions)
    l1a = {pe: _BankCache(capacities["AL1"]) for pe in pes}
    l1w = {pe: _BankCache(capacities["WL1"]) for pe in pes}
    l2a, l2w = _BankCache(capacities["AL2"]), _BankCache(capacities["WL2"])
    input_loads, output_loads = {}, {}
    input_records, output_records = [], []
    input_packets_count = output_packets_count = 0
    input_injected = input_delivered = input_hops = 0
    output_injected = output_delivered = output_hops = 0
    input_noc_s = output_noc_s = 0.0
    ub_reads = l2_writes = l2_reads = l1_writes = l1_output_reads = 0
    mac_cycles = vector_cycles = internal_macs = 0
    local_compute_bits = 0
    accesses = {name:{'read_bits':0,'write_bits':0} for name in ('WL1','AL1','OL1','WL2','AL2','OL2','UB')}
    l1_compute_bits_by_pe = {pe: 0 for pe in pes}
    l1_refill_bytes_by_pe = {pe: 0 for pe in pes}
    l1_output_bytes_by_pe = {pe: 0 for pe in pes}
    rounds = channel_rounds = 0
    observed_ol1 = observed_ol2 = 0
    per_pe_macro_cycles = {pe: [0, 0] for pe in pes}
    pe_microtiles = {pe: 0 for pe in pes}
    iterators = [_internal_tiles(region, dims, unit.dataflow) for region in regions]
    for outputs in zip_longest(*iterators):
        rounds += 1
        for pe, output in zip(pes, outputs):
            if output is not None:
                pe_microtiles[pe] += 1
                observed_ol1 = max(observed_ol1, output.elements * (hardware.psum_bytes if unit.macs else hardware.activation_bytes))
        observed_ol2 = max(observed_ol2, sum(output.elements * hardware.activation_bytes for output in outputs if output is not None))
        for c_start in range(0, cin, channel_tile):
            c_end = min(cin, c_start + channel_tile)
            channel_rounds += 1
            demands = []
            phase_cycles = []
            for pe, output in zip(pes, outputs):
                if output is None:
                    continue
                for tensor, region, bits in _chunk_inputs(workload, layer, output, c_start, c_end):
                    if l1a[pe].ensure(tensor, region, bits):
                        demands.append((tensor, region, bits, pe))
                        l1_writes += region.elements * bits // 8
                        accesses['AL1']['write_bits'] += region.elements*bits
                        l1_refill_bytes_by_pe[pe] += region.elements * bits // 8
                if unit.weight_region is not None:
                    weight = TensorRegion((output.bounds[1], (c_start, c_end), *unit.weight_region.bounds[2:]))
                    if l1w[pe].ensure(f"weight:{layer.id}", weight, layer.weight_bits):
                        demands.append((f"weight:{layer.id}", weight, layer.weight_bits, pe))
                        l1_writes += weight.elements * layer.weight_bits // 8
                        accesses['WL1']['write_bits'] += weight.elements*layer.weight_bits
                        l1_refill_bytes_by_pe[pe] += weight.elements * layer.weight_bits // 8
                    actual_macs = output.elements * (c_end - c_start) * math.prod(layer.kernel)
                    internal_macs += actual_macs
                    # Integer K-lane rounds and vector-C rounds include tails.
                    pe_mac, pe_vector = _pe_cycles(output, layer, workload, hardware,
                                                   c_end - c_start, c_end == cin)
                    if layer.kind == "Conv2d":
                        out_group = workload.tensor_map[layer.output].shape[1] // layer.groups
                        lo, hi = output.bounds[1]
                        k_rounds = sum(math.ceil((min(hi, (g + 1) * out_group) - max(lo, g * out_group)) / hardware.lanes)
                                       for g in range(lo // out_group, (hi - 1) // out_group + 1))
                    else:
                        k_rounds = math.ceil(output.shape[1] / hardware.lanes)
                    positions = output.shape[0] * output.shape[2] * output.shape[3]
                    activation_reads = positions * k_rounds * (c_end - c_start) * math.prod(layer.kernel)
                    accumulator_rw = 2 * output.elements * math.ceil((c_end - c_start) / hardware.vector_size) * math.prod(layer.kernel)
                    pe_compute_bits = (activation_reads * hardware.activation_bits +
                                       actual_macs * hardware.weight_bits + accumulator_rw * hardware.psum_bits)
                    accesses['AL1']['read_bits'] += activation_reads*hardware.activation_bits
                    accesses['WL1']['read_bits'] += actual_macs*hardware.weight_bits
                    accesses['OL1']['read_bits'] += accumulator_rw//2*hardware.psum_bits
                    accesses['OL1']['write_bits'] += accumulator_rw//2*hardware.psum_bits
                else:
                    pe_mac, pe_vector = _pe_cycles(output, layer, workload, hardware)
                    input_elements = sum(region.elements for _, region, _ in _chunk_inputs(workload, layer, output, 0, 1))
                    pe_compute_bits = max(input_elements, output.elements * layer.vector_ops_per_element) * hardware.activation_bits
                    accesses['AL1']['read_bits'] += pe_compute_bits
                local_compute_bits += pe_compute_bits
                l1_compute_bits_by_pe[pe] += pe_compute_bits
                phase_cycles.append((pe_mac, pe_vector))
                per_pe_macro_cycles[pe][0] += pe_mac
                per_pe_macro_cycles[pe][1] += pe_vector
            mac_cycles += max((value[0] for value in phase_cycles), default=0)
            vector_cycles += max((value[1] for value in phase_cycles), default=0)
            packets = _atomic_multicast(demands)
            for tensor, region, bits, _ in packets:
                bank = l2w if tensor.startswith("weight:") else l2a
                if bank.ensure(tensor, region, bits):
                    amount = region.elements * bits // 8
                    ub_reads += amount
                    l2_writes += amount
                    accesses['UB']['read_bits'] += amount*8
                    accesses['WL2' if tensor.startswith('weight:') else 'AL2']['write_bits'] += amount*8
            loads, injected, delivered, hops, distance, packet_records = _phase_routes(fabric, packets)
            input_injected += injected
            input_delivered += delivered
            input_hops += hops
            l2_reads += injected
            for packet in packet_records:
                accesses['WL2' if packet['tensor'].startswith('weight:') else 'AL2']['read_bits'] += packet['bytes']*packet['injected_copies']*8
            input_noc_s += _phase_time(loads, distance, hardware)
            for edge, amount in loads.items():
                input_loads[edge] = input_loads.get(edge, 0) + amount
            input_packets_count += len(packet_records)
            input_records.extend(packet_records[:max(0, 128 - len(input_records))])
        packets = tuple((layer.output, output, workload.tensor_map[layer.output].bits, (pe,))
                        for pe, output in zip(pes, outputs) if output is not None)
        loads, injected, delivered, hops, distance, packet_records = _phase_routes(fabric, packets, output=True)
        output_injected += injected
        output_delivered += delivered
        output_hops += hops
        output_noc_s += _phase_time(loads, distance, hardware)
        l1_output_reads += delivered
        for _, region, bits, destinations in packets:
            l1_output_bytes_by_pe[destinations[0]] += region.elements * bits // 8
        for edge, amount in loads.items():
            output_loads[edge] = output_loads.get(edge, 0) + amount
        output_packets_count += len(packet_records)
        output_records.extend(packet_records[:max(0, 128 - len(output_records))])
    if internal_macs != unit.macs or output_delivered != unit.output_bytes:
        raise ValueError("internal microkernels do not preserve macro MAC/output coverage")
    observed = {"AL1": max(bank.peak for bank in l1a.values()), "WL1": max(bank.peak for bank in l1w.values()),
                "OL1": observed_ol1, "AL2": l2a.peak, "WL2": l2w.peak, "OL2": observed_ol2}
    if any(observed[bank] > capacities[bank] for bank in capacities):
        raise ValueError("internal microkernel/cache exceeds an independent SRAM bank")
    output_l2_writes = output_delivered
    output_ub_writes = output_delivered
    sram_bits = local_compute_bits + 8 * (ub_reads + l2_writes + l2_reads + l1_writes +
                                         l1_output_reads + output_l2_writes + output_ub_writes)
    accesses['OL1']['read_bits'] += l1_output_reads*8
    accesses['OL2']['write_bits'] += output_l2_writes*8
    accesses['UB']['write_bits'] += output_ub_writes*8
    if sum(sum(row.values()) for row in accesses.values()) != sram_bits:
        raise ValueError('named SRAM access ledger differs from total physical access bits')
    sram_energy, access_costs = sram_access_energy(hardware,accesses)
    # Each PE owns an independent port. Aggregate bandwidth cannot hide the
    # busiest PE's tail, halo or grouped/vector accumulation accesses.
    l1_s = max(l1_compute_bits_by_pe.values(), default=0) / 8 / hardware.l1_bandwidth_Bps
    input_buffer_s = max(ub_reads / hardware.unified_bandwidth_Bps,
                         (l2_reads + l2_writes) / hardware.l2_bandwidth_Bps,
                         max(l1_refill_bytes_by_pe.values(), default=0) / hardware.l1_bandwidth_Bps)
    output_buffer_s = max(output_ub_writes / hardware.unified_bandwidth_Bps,
                          output_l2_writes / hardware.l2_bandwidth_Bps,
                          max(l1_output_bytes_by_pe.values(), default=0) / hardware.l1_bandwidth_Bps)
    compute_s = (mac_cycles + vector_cycles) / hardware.compute_clock_hz
    noc_s = input_noc_s + output_noc_s
    buffer_s = input_buffer_s + l1_s + output_buffer_s
    service_s = max(input_buffer_s, input_noc_s) + max(compute_s, l1_s) + max(output_buffer_s, output_noc_s)
    evidence = {**logical.traffic_evidence,
        "model": "capacity-backed independent SRAM banks, directed multicast and integer internal microkernels",
        "input_packets": input_records, "output_packets": output_records,
        "input_packet_count": input_packets_count, "output_packet_count": output_packets_count,
        "packet_records_truncated": input_packets_count > len(input_records) or output_packets_count > len(output_records),
        "input_link_loads": [{"source": a, "destination": b, "bytes": amount} for (a, b), amount in sorted(input_loads.items())],
        "output_link_loads": [{"source": a, "destination": b, "bytes": amount} for (a, b), amount in sorted(output_loads.items())],
        "input_injected_bytes": input_injected, "output_injected_bytes": output_injected,
        "input_delivered_bytes": input_delivered, "output_delivered_bytes": output_delivered,
        "input_hop_bytes": input_hops, "output_hop_bytes": output_hops,
        "microtile_shape": dims, "micro_channel_tile": channel_tile,
        "dataflow": unit.dataflow,
        "loop_order": 'B,H,W,K,C' if unit.dataflow == 'activation_reuse' else 'B,K,W,H,C' if unit.dataflow == 'weight_reuse' else 'B,K,H,W,C',
        "output_microtile_rounds": rounds, "channel_microkernel_rounds": channel_rounds,
        "pe_microtile_counts": pe_microtiles, "pe_macro_cycles": per_pe_macro_cycles,
        "bank_capacities_bytes": capacities, "bank_peak_bytes": observed,
        "microkernel_footprint_envelope_bytes": envelope,
        "ub_read_bytes": ub_reads, "l2_refill_write_bytes": l2_writes,
        "sram_access_ledger": access_costs,
        "l2_read_bytes": l2_reads, "l1_refill_write_bytes": l1_writes,
        "l1_compute_bits_by_pe": l1_compute_bits_by_pe,
        "l1_refill_bytes_by_pe": l1_refill_bytes_by_pe,
        "l1_output_read_bytes_by_pe": l1_output_bytes_by_pe,
        "partial_sum_spill_bytes": 0,
        "partial_sum_policy": "each output microtile's INT24 sums fit OL1 and stay local over all C chunks",
        "backing_policy": "full received activation/weight macroregions remain in UB until unit finish; bank misses reread UB",
        "limitations": "whole-region containment LRU, conservative envelope fitting; dense receptive-field gaps; internal streaming costs projected to macro read/compute/write without microkernel overlap credit; no RTL/VC/packet arbitration",
    }
    hop_bytes = input_hops + output_hops
    ideal = unit.macs / hardware.per_core_mac_rate if unit.macs else 0.0
    return CoreService(compute_s, buffer_s, noc_s, service_s,
                       unit.macs * hardware.effective_mac_energy_pj * 1e-12,
                       sram_energy,
                       hop_bytes * 8 * hardware.noc_energy_pj_per_bit * 1e-12,
                       sram_bits, input_injected + output_injected, hop_bytes,
                       mac_cycles, vector_cycles, ideal / compute_s if compute_s else 0.0,
                       input_noc_s, output_noc_s, l1_s, input_buffer_s, output_buffer_s,
                       logical.ring_resources, evidence, ub_reads)


def core_service(unit, layer, workload, hardware):
    """Reuse identical physical microkernels across batch instances and dies.

    The cache preserves layer/precision, K/H/W coordinates, payload sizes,
    hardware and core slot. Only batch origin, physical die and scheduling
    identities are normalized. Evidence explicitly records the coordinate
    frame and restores the caller's real ring-resource identity.
    """
    batch_origin = unit.output_region.bounds[0][0]
    physical_die = unit.chiplet.split("/")[0]
    slot = unit.chiplet.split("/", 1)[1] if "/" in unit.chiplet else "core:0"
    def relative_batch(region):
        lo, hi = region.bounds[0]
        return TensorRegion(((lo - batch_origin, hi - batch_origin), *region.bounds[1:]))
    normalized = replace(unit, id="", group_id="", microbatch=0, shard_index=0, tile_index=0,
                         chiplet=f"compute:0/{slot}", dependencies=(),
                         output_region=relative_batch(unit.output_region),
                         inputs=tuple(replace(req, region=relative_batch(req.region)) for req in unit.inputs))
    # Weight regions are K/C/R/S; their leading K axis must never be treated as B.
    cached = _cached_core_service(normalized, layer, workload, hardware)
    resources = tuple(resource.replace("noc:compute:0/", f"noc:{physical_die}/", 1)
                      for resource in cached.ring_resources)
    evidence = {**cached.traffic_evidence,
                "coordinate_frame": "activation/output B is unit-relative; weight K/C/R/S and activation K/H/W are absolute tensor coordinates",
                "batch_origin": batch_origin, "actual_output_bounds": unit.output_region.bounds,
                "physical_die": physical_die, "ring_resources": resources,
                "cache_normalization": "batch origin, physical die and schedule IDs only; physical core slot, K/H/W, precision, bank sizes and payload counts preserved"}
    return replace(cached, ring_resources=resources, traffic_evidence=evidence)


# Preserve the previous decorated public API's useful diagnostics without making
# the physical service object depend on a cache hit/miss or evaluation order.
core_service.cache_info = _cached_core_service.cache_info
core_service.cache_clear = _cached_core_service.cache_clear
