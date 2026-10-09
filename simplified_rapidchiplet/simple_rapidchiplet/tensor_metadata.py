"""Exact operator IR. Tensor dimensions come from FX ShapeProp, never MiB."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import cached_property


@dataclass(frozen=True)
class TensorSpec:
    id: str
    shape: tuple[int, ...]
    bytes_per_element: int
    producer: str
    source: str = "COMPUTE"

    def __post_init__(self):
        if len(self.shape) not in (2, 4) or any(n <= 0 for n in self.shape):
            raise ValueError(f"unsupported activation shape: {self.shape}")
        if self.bytes_per_element <= 0:
            raise ValueError("precision bytes must be positive")

    @property
    def channels(self):
        return self.shape[1]

    @property
    def elements_per_channel(self):
        return math.prod(self.shape) // self.channels

    @property
    def bytes(self):
        return math.prod(self.shape) * self.bytes_per_element


@dataclass(frozen=True)
class OperationSpec:
    id: str
    kind: str
    block_id: str
    inputs: tuple[str, ...]
    output: str
    dependencies: tuple[str, ...]
    cin: int = 0
    cout: int = 0
    kernel: tuple[int, ...] = ()
    stride: tuple[int, ...] = ()
    padding: tuple[int, ...] = ()
    groups: int = 1
    weight_shape: tuple[int, ...] = ()
    parameter_bytes: int = 0
    macs: int = 0
    fused: bool = False
    target: str = ""


@dataclass(frozen=True)
class SemanticBlockSpec:
    id: str
    kind: str
    operators: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class OperatorGraph:
    model_name: str
    tensors: tuple[TensorSpec, ...]
    operators: tuple[OperationSpec, ...]
    blocks: tuple[SemanticBlockSpec, ...]
    input_id: str
    output_id: str
    weight_bytes: int
    environment: tuple[tuple[str, str], ...] = ()
    source: str = "torch.fx.symbolic_trace+ShapeProp"

    @property
    def tensor_map(self):
        return {tensor.id: tensor for tensor in self.tensors}

    @property
    def operation_map(self):
        return {op.id: op for op in self.operators}

    @cached_property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def to_dict(self):
        result = asdict(self)
        tensors = self.tensor_map
        for op in result["operators"]:
            op["input_shapes"] = [tensors[name].shape for name in op["inputs"]]
            op["output_shape"] = tensors[op["output"]].shape
        result["fingerprint"] = self.fingerprint
        result["total_macs"] = sum(op.macs for op in self.operators)
        return result

    def validate(self):
        tensors = self.tensor_map
        ops = self.operation_map
        if len(tensors) != len(self.tensors) or len(ops) != len(self.operators):
            raise ValueError("duplicate operator/tensor ID")
        if tensors[self.input_id].source != "EXTERNAL":
            raise ValueError("input must be EXTERNAL")
        produced = {self.input_id}
        seen = set()
        for op in self.operators:
            if any(name not in produced for name in op.inputs) or any(dep not in seen for dep in op.dependencies):
                raise ValueError(f"graph not in dependency order: {op.id}")
            if op.output not in tensors or tensors[op.output].producer != op.id:
                raise ValueError(f"invalid output producer: {op.id}")
            if op.kind == "Conv2d":
                if op.groups < 1 or op.cin % op.groups or op.cout % op.groups:
                    raise ValueError("invalid grouped convolution")
                output = tensors[op.output]
                if tensors[op.inputs[0]].channels != op.cin or output.channels != op.cout:
                    raise ValueError("convolution channel metadata disagrees with tensor shape")
                expected = math.prod(output.shape) * (op.cin // op.groups) * math.prod(op.kernel)
                if op.macs != expected:
                    raise ValueError("convolution MAC count disagrees with tensor shape")
            if op.kind == "Add" and any(tensors[name].shape != tensors[op.output].shape for name in op.inputs):
                raise ValueError("Add requires equal activation shapes")
            if op.kind == "Flatten" and tensors[op.inputs[0]].channels != tensors[op.output].channels:
                raise ValueError("K ownership supports flatten after 1x1 pooling only")
            if op.macs < 0:
                raise ValueError("negative MAC count")
            produced.add(op.output)
            seen.add(op.id)
        block_ops = [name for block in self.blocks for name in block.operators]
        if len(block_ops) != len(set(block_ops)) or set(block_ops) != set(ops):
            raise ValueError("each operator must belong to exactly one semantic block")
        if self.output_id not in produced:
            raise ValueError("missing model output")


def build_blocks(operators, tensors, block_kinds):
    """Derive boundary tensors from actual graph users, including skip edges."""
    by_tensor = {tensor.id: tensor for tensor in tensors}
    by_op = {op.id: op for op in operators}
    result = []
    for block_id, kind in block_kinds:
        selected = tuple(op for op in operators if op.block_id == block_id)
        inputs = tuple(dict.fromkeys(name for op in selected for name in op.inputs
                       if by_tensor[name].source == "EXTERNAL" or by_op[by_tensor[name].producer].block_id != block_id))
        outputs = tuple(op.output for op in selected if not any(op.output in user.inputs for user in operators)
                        or any(op.output in user.inputs and user.block_id != block_id for user in operators))
        result.append(SemanticBlockSpec(block_id, kind, tuple(op.id for op in selected), inputs, outputs))
    return tuple(result)
