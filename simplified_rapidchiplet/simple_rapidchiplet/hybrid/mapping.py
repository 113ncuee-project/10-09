"""Integer output ownership and layer-pipeline mapping contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import cached_property, lru_cache
from itertools import product
import math

from .workload import Workload


@dataclass(frozen=True)
class Partition:
    b: int = 1
    k: int = 1
    h: int = 1
    w: int = 1

    def __post_init__(self):
        if any(not isinstance(value, int) or value < 1 for value in self.as_tuple()):
            raise ValueError("partition factors must be positive integers")

    def as_tuple(self):
        return self.b, self.k, self.h, self.w

    @property
    def size(self):
        return math.prod(self.as_tuple())

    @property
    def product(self):
        return self.size


@dataclass(frozen=True)
class TensorRegion:
    """Half-open B,K,H,W box; weight boxes use K,C,R,S axes."""
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]

    def __post_init__(self):
        if len(self.bounds) != 4 or any(left < 0 or right <= left for left, right in self.bounds):
            raise ValueError(f"invalid nonempty tensor region: {self.bounds}")

    @property
    def shape(self):
        return tuple(right - left for left, right in self.bounds)

    @property
    def elements(self):
        return math.prod(self.shape)

    def intersection(self, other):
        bounds = tuple((max(a, c), min(b, d)) for (a, b), (c, d) in zip(self.bounds, other.bounds))
        return None if any(left >= right for left, right in bounds) else TensorRegion(bounds)

    @classmethod
    def full(cls, shape):
        return cls(tuple((0, extent) for extent in shape))


@dataclass(frozen=True)
class LayerMapping:
    layer_id: str
    chiplets: tuple[str, ...]
    partition: Partition = Partition()
    input_dram: str = "memory:0"
    weight_dram: str = "memory:0"
    output_dram: str = "memory:0"
    tile_shape: tuple[int, int, int, int] = (0, 0, 0, 0)
    channel_tile: int = 128
    weight_policy: str = "cold"
    dataflow: str = "output_stationary"
    microtile_shape: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def core_ids(self):
        return self.chiplets

    @property
    def fd(self):
        return self.input_dram, self.weight_dram, self.output_dram


@dataclass(frozen=True)
class LayerGroup:
    id: str
    layer_ids: tuple[str, ...]
    mode: str = "spatial"


@dataclass(frozen=True)
class MappingPlan:
    groups: tuple[LayerGroup, ...]
    layer_mappings: tuple[LayerMapping, ...]
    microbatch_size: int = 1
    collective: str = "multicast"
    memory_ids: tuple[str, ...] = ("memory:0",)

    @cached_property
    def mapping_map(self):
        return {mapping.layer_id: mapping for mapping in self.layer_mappings}

    @cached_property
    def group_map(self):
        return {layer: group for group in self.groups for layer in group.layer_ids}

    def replace_layer(self, mapping):
        if mapping.layer_id not in self.mapping_map:
            raise KeyError(mapping.layer_id)
        return replace(self, layer_mappings=tuple(mapping if old.layer_id == mapping.layer_id else old
                                                 for old in self.layer_mappings))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TensorRequirement:
    tensor_id: str
    region: TensorRegion


@dataclass(frozen=True)
class WorkUnit:
    id: str
    layer_id: str
    group_id: str
    microbatch: int
    chiplet: str
    shard_index: int
    tile_index: int
    output_tensor: str
    output_region: TensorRegion
    inputs: tuple[TensorRequirement, ...]
    weight_region: TensorRegion | None
    macs: int
    vector_ops: int
    weight_bytes: int
    input_bytes: int
    output_bytes: int
    buffer_bytes: int
    channel_tile: int
    dependencies: tuple[str, ...] = ()
    dataflow: str = "output_stationary"
    microtile_shape: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def physical_die_id(self):
        return self.chiplet.split("/")[0]

    @property
    def output(self):
        return self.output_region


@dataclass(frozen=True)
class CompiledMapping:
    workload: Workload
    plan: MappingPlan
    units: tuple[WorkUnit, ...]

    @cached_property
    def unit_map(self):
        return {unit.id: unit for unit in self.units}

    @cached_property
    def tensor_owners(self):
        return {tensor.id: tuple(unit for unit in self.units if unit.output_tensor == tensor.id)
                for tensor in self.workload.tensors}

    @property
    def microbatch_count(self):
        return math.ceil(self.workload.batch_size / self.plan.microbatch_size)


def enumerate_partitions(workload, layer_id, cardinality, microbatch_size=1):
    """All exact-cardinality B/K/H/W factorizations with nonempty shards."""
    if cardinality < 1:
        return ()
    output = workload.tensor_map[workload.layer_map[layer_id].output]
    bounds = (min(microbatch_size, workload.batch_size), *output.shape[1:])
    return _partition_factors(bounds, cardinality)


@lru_cache(maxsize=4096)
def _partition_factors(bounds, cardinality):
    divisors = [value for value in range(1, cardinality + 1) if cardinality % value == 0]
    return tuple(Partition(*factors) for factors in product(divisors, repeat=4)
                 if math.prod(factors) == cardinality and all(factor <= bound for factor, bound in zip(factors, bounds)))


def validate_plan(workload, plan, compute_ids=None):
    workload.validate()
    if plan.microbatch_size < 1 or workload.batch_size % plan.microbatch_size:
        raise ValueError("microbatch size must divide the workload batch size")
    layer_order = tuple(layer.id for layer in workload.layers)
    grouped = tuple(layer for group in plan.groups for layer in group.layer_ids)
    if grouped != layer_order or len(set(group.id for group in plan.groups)) != len(plan.groups):
        raise ValueError("groups must partition the topological layer order exactly once")
    mappings = plan.mapping_map
    if set(mappings) != set(layer_order) or len(mappings) != len(plan.layer_mappings):
        raise ValueError("every layer requires exactly one mapping")
    if not plan.memory_ids or len(set(plan.memory_ids)) != len(plan.memory_ids):
        raise ValueError("memory endpoints must be unique and nonempty")
    legal_fd = {*plan.memory_ids, "interleave", "direct"}
    allowed = set(compute_ids) if compute_ids is not None else None
    for layer in workload.layers:
        mapping = mappings[layer.id]
        if not mapping.chiplets or len(set(mapping.chiplets)) != len(mapping.chiplets):
            raise ValueError(f"compute group is empty or has duplicate cores: {layer.id}")
        if allowed is not None and not set(mapping.chiplets) <= allowed:
            raise ValueError("mapping refers to an unavailable compute core")
        if not isinstance(mapping.partition, Partition):
            raise ValueError("layer partition must be a Partition")
        if mapping.partition.size != len(mapping.chiplets):
            raise ValueError("partition product must equal ordered compute-group size")
        if mapping.partition not in enumerate_partitions(workload, layer.id, len(mapping.chiplets), plan.microbatch_size):
            raise ValueError(f"partition creates empty output shards: {layer.id}")
        if any(fd not in legal_fd for fd in mapping.fd):
            raise ValueError("FD must reference a memory endpoint, interleave, or direct")
        if layer.weight_shape and mapping.weight_dram == "direct":
            raise ValueError("weighted layers require an external weight source")
        if any(name in workload.input_ids for name in layer.inputs) and mapping.input_dram == "direct":
            raise ValueError("external model input needs a DRAM source")
        if any(value < 0 for value in mapping.tile_shape) or mapping.channel_tile < 0:
            raise ValueError("tile extents cannot be negative")
        if mapping.weight_policy not in {"cold", "persistent"}:
            raise ValueError("weight policy must be cold or persistent")
        if mapping.dataflow not in {'output_stationary', 'activation_reuse', 'weight_reuse'}:
            raise ValueError('unsupported intra-die dataflow')
        if len(mapping.microtile_shape) != 4 or any(not isinstance(v, int) or v < 0 for v in mapping.microtile_shape):
            raise ValueError('invalid microtile shape')
    for group in plan.groups:
        if group.mode not in {"spatial", "temporal"} or not group.layer_ids:
            raise ValueError("layer group mode must be spatial or temporal")
        if group.mode == "spatial":
            cores = [core for layer in group.layer_ids for core in mappings[layer].chiplets]
            if len(cores) != len(set(cores)):
                raise ValueError("spatial pipeline stages must have disjoint compute groups")
    return True


def initial_mapping(workload, compute_ids, memory_ids=("memory:0",), microbatch_size=1,
                    max_group_layers=3, pipeline=True):
    """Deterministic balanced stripes; search can subsequently mutate all fields."""
    compute_ids, memory_ids = tuple(map(str, compute_ids)), tuple(map(str, memory_ids))
    if not compute_ids or max_group_layers < 1:
        raise ValueError("initial mapping requires cores and positive group size")
    groups, mappings = [], []
    group_size = min(max_group_layers, len(compute_ids)) if pipeline else max_group_layers
    for start in range(0, len(workload.layers), group_size):
        layers = workload.layers[start:start + group_size]
        groups.append(LayerGroup(f"group{len(groups)}", tuple(layer.id for layer in layers),
                                 "spatial" if pipeline else "temporal"))
        for index, layer in enumerate(layers):
            if pipeline:
                lo, hi = len(compute_ids) * index // len(layers), len(compute_ids) * (index + 1) // len(layers)
                allocated = compute_ids[lo:hi]
            else:
                allocated = compute_ids
            # Some small-channel/vector layers cannot use every allocated core.
            partitions = enumerate_partitions(workload, layer.id, len(allocated), microbatch_size)
            while not partitions and len(allocated) > 1:
                allocated = allocated[:-1]
                partitions = enumerate_partitions(workload, layer.id, len(allocated), microbatch_size)
            partition = min(partitions, key=lambda part: (part.h * part.w, part.b, -part.k, part.as_tuple()))
            memory = memory_ids[index % len(memory_ids)]
            mappings.append(LayerMapping(layer.id, allocated, partition, memory, memory, memory))
    plan = MappingPlan(tuple(groups), tuple(mappings), microbatch_size, "multicast", memory_ids)
    validate_plan(workload, plan, compute_ids)
    return plan


def _ranges(start, end, parts):
    return tuple((start + (end - start) * index // parts, start + (end - start) * (index + 1) // parts)
                 for index in range(parts))


def _shards(shape, partition, batch_start, batch_end):
    br = _ranges(batch_start, batch_end, partition.b)
    kr, hr, wr = (_ranges(0, extent, parts) for extent, parts in zip(shape[1:], partition.as_tuple()[1:]))
    # Correspondence rule in Gemini: h -> w -> b -> k, with k fastest.
    return tuple(TensorRegion((b, k, h, w)) for h, w, b, k in product(hr, wr, br, kr))


def _tiles(region, tile_shape):
    ranges = []
    for (start, end), tile in zip(region.bounds, tile_shape):
        step = tile or end - start
        ranges.append(tuple((left, min(end, left + step)) for left in range(start, end, step)))
    return tuple(TensorRegion(tuple(bounds)) for bounds in product(*ranges))


def required_input_regions(workload, layer, output):
    """Derive exact rectangles (a dense receptive-field envelope for dilation)."""
    result = []
    for input_id in layer.inputs:
        shape = workload.tensor_map[input_id].shape
        b, k, h, w = output.bounds
        if layer.kind == "Conv2d":
            out_channels = workload.tensor_map[layer.output].shape[1]
            per_out, per_in = out_channels // layer.groups, shape[1] // layer.groups
            first_group, last_group = k[0] // per_out, (k[1] - 1) // per_out
            channels = (first_group * per_in, (last_group + 1) * per_in)
            spatial = []
            for interval, stride, pad, dilation, kernel, extent in zip((h, w), layer.stride, layer.padding,
                                                                     layer.dilation, layer.kernel, shape[2:]):
                spatial.append((max(0, interval[0] * stride - pad),
                                min(extent, (interval[1] - 1) * stride - pad + dilation * (kernel - 1) + 1)))
            result.append(TensorRequirement(input_id, TensorRegion((b, channels, *spatial))))
        elif layer.kind == "Linear":
            result.append(TensorRequirement(input_id, TensorRegion((b, (0, shape[1]), (0, shape[2]), (0, shape[3])))))
        elif layer.kind in {"MaxPool2d", "AvgPool2d"}:
            spatial = tuple((max(0, interval[0] * stride - pad),
                             min(extent, (interval[1] - 1) * stride - pad + dilation * (kernel - 1) + 1))
                            for interval, stride, pad, dilation, kernel, extent in zip((h, w), layer.stride,
                                layer.padding, layer.dilation, layer.kernel, shape[2:]))
            result.append(TensorRequirement(input_id, TensorRegion((b, k, *spatial))))
        elif layer.kind == "AdaptiveAvgPool2d":
            out_shape = workload.tensor_map[layer.output].shape
            spatial = tuple((interval[0] * extent // out_extent,
                             math.ceil(interval[1] * extent / out_extent))
                            for interval, extent, out_extent in zip((h, w), shape[2:], out_shape[2:]))
            result.append(TensorRequirement(input_id, TensorRegion((b, k, *spatial))))
        elif layer.kind == "Flatten":
            # Flattened K indices encode C,H,W. Split at row boundaries so the
            # region transformation remains exact for general spatial flatten.
            cursor = k[0]
            while cursor < k[1]:
                channel, rem = divmod(cursor, shape[2] * shape[3])
                row, column = divmod(rem, shape[3])
                plane = shape[2] * shape[3]
                if rem == 0 and k[1] - cursor >= plane:
                    count = (k[1] - cursor) // plane
                    result.append(TensorRequirement(input_id, TensorRegion((b, (channel, channel + count), (0, shape[2]), (0, shape[3])))))
                    cursor += count * plane
                    continue
                if column == 0 and k[1] - cursor >= shape[3]:
                    count = min(shape[2] - row, (k[1] - cursor) // shape[3])
                    result.append(TensorRequirement(input_id, TensorRegion((b, (channel, channel + 1), (row, row + count), (0, shape[3])))))
                    cursor += count * shape[3]
                    continue
                length = min(shape[3] - column, k[1] - cursor)
                result.append(TensorRequirement(input_id, TensorRegion((b, (channel, channel + 1),
                                             (row, row + 1), (column, column + length)))))
                cursor += length
        else:
            result.append(TensorRequirement(input_id, output))
    return tuple(result)


def compile_mapping(workload, plan, hardware=None):
    """Compile ownership/tiles and exact data dependencies; no timing estimates."""
    validate_plan(workload, plan)
    units = []
    spec = getattr(hardware, "spec", hardware)
    quota = (spec.pes_per_core * spec.l1_bytes_per_pe + spec.l2_bytes_per_die // spec.cores_per_die) if spec is not None else None
    psum_bytes = spec.psum_bytes if spec is not None else 3
    for microbatch, batch_start in enumerate(range(0, workload.batch_size, plan.microbatch_size)):
        batch_end = min(workload.batch_size, batch_start + plan.microbatch_size)
        for layer in workload.layers:
            mapping = plan.mapping_map[layer.id]
            out_tensor = workload.tensor_map[layer.output]
            for shard_index, shard in enumerate(_shards(out_tensor.shape, mapping.partition, batch_start, batch_end)):
                tile_shape = tuple(min(extent, tile or extent) for extent, tile in zip(shard.shape, mapping.tile_shape))
                channel_tile = mapping.channel_tile
                if quota is not None:
                    tile_shape, channel_tile = _fit_tile(workload, layer, shard, tile_shape, channel_tile, quota, psum_bytes)
                for tile_index, output in enumerate(_tiles(shard, tile_shape)):
                    inputs = required_input_regions(workload, layer, output)
                    input_bytes = sum(req.region.elements * workload.tensor_map[req.tensor_id].bytes_per_element for req in inputs)
                    weight_region = None
                    weight_bytes = 0
                    if layer.weight_shape:
                        channels = layer.weight_shape[1]
                        weight_region = TensorRegion((output.bounds[1], (0, channels),
                                                      (0, layer.kernel[0]), (0, layer.kernel[1])))
                        weight_bytes = weight_region.elements * layer.weight_bits // 8
                    macs = (output.elements * layer.weight_shape[1] * math.prod(layer.kernel)
                            if layer.kind == "Conv2d" else output.elements * math.prod(layer.weight_shape[1:])
                            if layer.kind == "Linear" else 0)
                    c_extent = max((req.region.shape[1] for req in inputs), default=1)
                    c_tile = min(c_extent, channel_tile or c_extent)
                    streamed_input = sum(math.ceil(req.region.elements * min(c_tile, req.region.shape[1]) / req.region.shape[1])
                                         * workload.tensor_map[req.tensor_id].bytes_per_element for req in inputs)
                    streamed_weight = (weight_region.shape[0] * min(c_tile, weight_region.shape[1])
                                       * math.prod(weight_region.shape[2:]) * layer.weight_bits // 8) if weight_region else 0
                    output_bytes = output.elements * out_tensor.bytes_per_element
                    # Explicit C tiling only shrinks concurrent footprint. Full
                    # input/weight traffic and MAC counts remain unchanged.
                    accumulator = output.elements * psum_bytes if macs else output_bytes
                    buffer_bytes = streamed_input + streamed_weight + accumulator + output_bytes
                    units.append(WorkUnit(f"{layer.id}:mb{microbatch}:s{shard_index}:t{tile_index}", layer.id,
                                 plan.group_map[layer.id].id, microbatch, mapping.chiplets[shard_index],
                                 shard_index, tile_index, layer.output, output, inputs, weight_region, macs,
                                 output.elements * layer.vector_ops_per_element, weight_bytes, input_bytes,
                                 output_bytes, buffer_bytes, c_tile, dataflow=mapping.dataflow,
                                 microtile_shape=mapping.microtile_shape))
    owners = {}
    for unit in units:
        owners.setdefault((unit.output_tensor, unit.microbatch), []).append(unit)
    result = []
    for unit in units:
        dependencies = tuple(dict.fromkeys(owner.id for req in unit.inputs
                              for owner in owners.get((req.tensor_id, unit.microbatch), ())
                              if owner.output_region.intersection(req.region) is not None))
        result.append(replace(unit, dependencies=dependencies))
    compiled = CompiledMapping(workload, plan, tuple(result))
    if sum(unit.macs for unit in compiled.units) != workload.total_macs:
        raise ValueError("compiled shard MACs do not cover the workload exactly")
    return compiled


def _footprint(workload, layer, output, channel_tile, psum_bytes):
    requirements = required_input_regions(workload, layer, output)
    c_extent = max(req.region.shape[1] for req in requirements)
    c_tile = min(c_extent, channel_tile or c_extent)
    inputs = sum(math.ceil(req.region.elements * min(c_tile, req.region.shape[1]) / req.region.shape[1])
                 * workload.tensor_map[req.tensor_id].bytes_per_element for req in requirements)
    weights = (output.shape[1] * min(c_tile, layer.weight_shape[1]) * math.prod(layer.kernel)
               * layer.weight_bits // 8) if layer.weight_shape else 0
    output_bytes = output.elements * workload.tensor_map[layer.output].bytes_per_element
    return inputs + weights + output_bytes + output.elements * (psum_bytes if layer.weight_shape else
                                                               workload.tensor_map[layer.output].bytes_per_element)


def _fit_tile(workload, layer, shard, tile_shape, channel_tile, quota, psum_bytes):
    """Choose a bounded tile; changing C footprint never clips transfer traffic."""
    dimensions = tuple(tile_shape)
    c_tile = channel_tile or max(workload.tensor_map[layer.inputs[0]].shape[1], 1)
    def region_for(shape):
        return TensorRegion(tuple((left, min(right, left + extent)) for (left, right), extent in zip(shard.bounds, shape)))
    while _footprint(workload, layer, region_for(dimensions), c_tile, psum_bytes) > quota:
        candidates = []
        for axis, extent in enumerate(dimensions):
            if extent > 1:
                trial = list(dimensions)
                trial[axis] = math.ceil(extent / 2)
                trial = tuple(trial)
                candidates.append((_footprint(workload, layer, region_for(trial), c_tile, psum_bytes),
                                   trial, c_tile))
        if c_tile > 1:
            trial_c = math.ceil(c_tile / 2)
            candidates.append((_footprint(workload, layer, region_for(dimensions), trial_c, psum_bytes),
                               dimensions, trial_c))
        if not candidates:
            raise ValueError(f"even one output element cannot fit core buffer quota: {layer.id}")
        _, dimensions, c_tile = min(candidates, key=lambda candidate: (candidate[0], candidate[1], candidate[2]))
    return dimensions, c_tile
