"""Precision-aware operator DAG for the INDM/Gemini hybrid accelerator.

Shapes are canonical B,K,H,W. FX is used for shape extraction only; selecting
INT8 here is a hardware workload assumption, not a quantized accuracy claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import cached_property
import hashlib
import json
import math


@dataclass(frozen=True)
class TensorSpec:
    id: str
    shape: tuple[int, int, int, int]
    bits: int = 8
    producer: str = ""

    def __post_init__(self):
        if len(self.shape) != 4 or any(value < 1 for value in self.shape):
            raise ValueError(f"invalid B,K,H,W shape for {self.id}: {self.shape}")
        if self.bits < 1 or self.bits % 8:
            raise ValueError("activation precision must use an integer number of bytes")

    @property
    def elements(self):
        return math.prod(self.shape)

    @property
    def bytes_per_element(self):
        return self.bits // 8

    @property
    def bytes(self):
        return self.elements * self.bytes_per_element


@dataclass(frozen=True)
class LayerSpec:
    id: str
    kind: str
    inputs: tuple[str, ...]
    output: str
    kernel: tuple[int, int] = (1, 1)
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    dilation: tuple[int, int] = (1, 1)
    groups: int = 1
    weight_shape: tuple[int, ...] = ()
    weight_bits: int = 8
    macs_per_sample: int = 0
    vector_ops_per_element: int = 0
    semantic_block: str = ""
    post_ops: tuple[str, ...] = ()

    @property
    def weight_bytes(self):
        return math.prod(self.weight_shape) * (self.weight_bits // 8) if self.weight_shape else 0


@dataclass(frozen=True)
class Workload:
    name: str
    tensors: tuple[TensorSpec, ...]
    layers: tuple[LayerSpec, ...]
    input_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    batch_size: int = 1
    source: str = "explicit"

    @cached_property
    def tensor_map(self):
        return {tensor.id: tensor for tensor in self.tensors}

    @cached_property
    def layer_map(self):
        return {layer.id: layer for layer in self.layers}

    @cached_property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @cached_property
    def consumers(self):
        return {tensor.id: tuple(layer.id for layer in self.layers if tensor.id in layer.inputs)
                for tensor in self.tensors}

    @property
    def total_macs(self):
        return sum(layer.macs_per_sample for layer in self.layers) * self.batch_size

    def validate(self):
        tensors = self.tensor_map
        if self.batch_size < 1 or len(tensors) != len(self.tensors) or len(self.layer_map) != len(self.layers):
            raise ValueError("invalid batch size or duplicate workload IDs")
        produced = set(self.input_ids)
        if any(name not in tensors or tensors[name].producer for name in produced):
            raise ValueError("workload inputs require external tensors")
        supported = {"Conv2d", "Linear", "Add", "ReLU", "BatchNorm2d", "Identity",
                     "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "Flatten"}
        for layer in self.layers:
            if layer.kind not in supported:
                raise ValueError(f"unsupported hybrid operator: {layer.kind}")
            if not layer.inputs or any(name not in produced for name in layer.inputs):
                raise ValueError(f"DAG is not topologically ordered at {layer.id}")
            if layer.output in produced or layer.output not in tensors or tensors[layer.output].producer != layer.id:
                raise ValueError(f"invalid output owner for {layer.id}")
            output = tensors[layer.output]
            if output.shape[0] != self.batch_size:
                raise ValueError("all tensor batch extents must match the workload")
            if layer.kind == "Add" and any(tensors[name].shape != output.shape for name in layer.inputs):
                raise ValueError("residual Add requires equal input/output shapes")
            if layer.kind == "Conv2d":
                cin, cout = tensors[layer.inputs[0]].shape[1], output.shape[1]
                if layer.groups < 1 or cin % layer.groups or cout % layer.groups:
                    raise ValueError("invalid grouped convolution")
                expected = cout * output.shape[2] * output.shape[3] * (cin // layer.groups) * math.prod(layer.kernel)
                if layer.macs_per_sample != expected:
                    raise ValueError("Conv MAC metadata disagrees with shapes")
            produced.add(layer.output)
        if any(name not in produced for name in self.output_ids):
            raise ValueError("unknown workload output")

    def to_dict(self):
        return {**asdict(self), "fingerprint": self.fingerprint, "total_macs": self.total_macs}


def from_operator_graph(graph, activation_bits=8, weight_bits=8, batch_size=1, fuse_postops=True):
    """Reuse verified FX shape metadata without its old FP32/K-only restrictions."""
    tensors = []
    for tensor in graph.tensors:
        shape = tuple(tensor.shape)
        if len(shape) == 2:
            shape = (*shape, 1, 1)
        tensors.append(TensorSpec(tensor.id, (batch_size, *shape[1:]), activation_bits, tensor.producer))
    layers = []
    tm = {tensor.id: tensor for tensor in tensors}
    for op in graph.operators:
        kernel, stride, padding = op.kernel or (1, 1), op.stride or (1, 1), op.padding or (0, 0)
        # The earlier metadata omitted pool attributes; ResNet's exact stem pool
        # semantics are 3x3, stride 2, padding 1 and must not become a 1x1 copy.
        if op.kind == "MaxPool2d" and op.target == "maxpool":
            kernel, stride, padding = (3, 3), (2, 2), (1, 1)
        vector = {"Add": 1, "ReLU": 1, "BatchNorm2d": 2, "MaxPool2d": math.prod(kernel),
                  "AvgPool2d": math.prod(kernel)}.get(op.kind, 0)
        if op.kind == "AdaptiveAvgPool2d":
            inp, out = tm[op.inputs[0]].shape, tm[op.output].shape
            vector = math.ceil(inp[2] / out[2]) * math.ceil(inp[3] / out[3])
        layers.append(LayerSpec(op.id, op.kind, op.inputs, op.output, tuple(kernel), tuple(stride),
                               tuple(padding), (1, 1), op.groups, op.weight_shape, weight_bits,
                               op.macs // graph.tensor_map[op.output].shape[0], vector, op.block_id))
    if fuse_postops:
        tensors, layers = _fuse_postops(tensors, layers)
    result = Workload(graph.model_name, tuple(tensors), tuple(layers), (graph.input_id,),
                      (graph.output_id,), batch_size, "FX shapes + explicit hardware precision")
    result.validate()
    return result


def extract_resnet50(batch_size=8, activation_bits=8, weight_bits=8, fuse_postops=True):
    """Extract the real ResNet-50 v1.5 residual DAG; precision is configurable."""
    from ..config import ReferenceModelConfig
    from ..model_parser import extract_reference_graph
    reference = ReferenceModelConfig("torchvision.models.resnet50", "ResNet-50 v1.5",
                                     (1, 3, 224, 224), 1000, "FP32", "FP32", 4, 4,
                                     "EXTERNAL", "EXTERNAL")
    return from_operator_graph(extract_reference_graph(reference), activation_bits, weight_bits, batch_size, fuse_postops)


def _fuse_postops(tensors, layers):
    """Fuse only single-consumer shape-preserving inference post-processing."""
    consumers = {tensor.id: [layer.id for layer in layers if tensor.id in layer.inputs] for tensor in tensors}
    tensor_map = {tensor.id: tensor for tensor in tensors}
    compact = []
    retired = set()
    for layer in layers:
        predecessor = tensor_map[layer.inputs[0]].producer if len(layer.inputs) == 1 else ""
        index = next((index for index, old in enumerate(compact) if old.id == predecessor), None)
        if (layer.kind in {"BatchNorm2d", "ReLU"} and index is not None and
                compact[index].kind in {"Conv2d", "Linear", "Add"} and
                len(consumers[layer.inputs[0]]) == 1 and
                tensor_map[layer.inputs[0]].shape == tensor_map[layer.output].shape):
            old = compact[index]
            compact[index] = replace(old, output=layer.output,
                                     vector_ops_per_element=old.vector_ops_per_element + layer.vector_ops_per_element,
                                     post_ops=(*old.post_ops, layer.kind))
            tensor_map[layer.output] = replace(tensor_map[layer.output], producer=old.id)
            retired.add(layer.inputs[0])
        else:
            compact.append(layer)
    return [tensor_map[tensor.id] for tensor in tensors if tensor.id not in retired], compact


def make_toy_workload(residual=True, batch=4, channels=8, spatial=8, batch_size=None):
    """Hand-checkable two-convolution DAG with optional residual join."""
    if batch_size is not None:
        batch = batch_size
    shape = (batch, channels, spatial, spatial)
    tensors = [TensorSpec("image", shape)]
    layers = []
    for index, kernel in enumerate(((3, 3), (1, 1))):
        name = f"conv{index}"
        inputs = ("image" if index == 0 else "conv0",)
        layers.append(LayerSpec(name, "Conv2d", inputs, name, kernel, (1, 1),
                               (1, 1) if index == 0 else (0, 0), weight_shape=(channels, channels, *kernel),
                               macs_per_sample=channels * channels * spatial * spatial * math.prod(kernel),
                               semantic_block=f"block{index}"))
        tensors.append(TensorSpec(name, shape, producer=name))
    output = "conv1"
    if residual:
        layers.append(LayerSpec("add", "Add", ("image", "conv1"), "add", vector_ops_per_element=1,
                               semantic_block="residual"))
        tensors.append(TensorSpec("add", shape, producer="add"))
        output = "add"
    result = Workload("toy_residual" if residual else "toy_chain", tuple(tensors), tuple(layers),
                      ("image",), (output,), batch, "hand-checkable fixture")
    result.validate()
    return result
