"""TorchVision ResNet-50 v1.5 extraction; preserves 18 semantic decisions."""
from __future__ import annotations

from functools import lru_cache

from .config import ReferenceModelConfig
from .tensor_metadata import OperationSpec, OperatorGraph, TensorSpec, build_blocks


@lru_cache(maxsize=4)
def extract_reference_graph(reference: ReferenceModelConfig) -> OperatorGraph:
    if reference.name != "torchvision.models.resnet50" or reference.variant != "ResNet-50 v1.5":
        raise ValueError("physical reference parser currently supports TorchVision ResNet-50 v1.5")
    if reference.activation_precision != "FP32" or reference.weight_precision != "FP32":
        raise ValueError("reference parser requires the configured FP32 baseline")
    if (reference.activation_bytes_per_element, reference.weight_bytes_per_element) != (4, 4):
        raise ValueError("FP32 must use four bytes per element")
    if reference.initial_input_source != "EXTERNAL" or reference.weights_source != "EXTERNAL":
        raise ValueError("compute-only mesh requires EXTERNAL input/weights")
    if reference.input_shape != (1, 3, 224, 224) or reference.num_classes != 1000:
        raise ValueError("this reference requires batch=1, 3x224x224, ImageNet-1K/1000 classes")
    try:
        import torch
        import torchvision
        from torch.fx import symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp
        from torchvision.models.resnet import Bottleneck
    except ImportError as exc:
        raise RuntimeError("Exact graph extraction requires torch/torchvision; use the project .venv. No legacy shape fallback is permitted.") from exc
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = torchvision.models.resnet50(weights=None, num_classes=reference.num_classes).eval()
    module_blocks = tuple(name for name, module in model.named_modules() if isinstance(module, Bottleneck))
    if len(module_blocks) != 16:
        raise ValueError("reference hierarchy must contain 16 Bottlenecks")
    block_kinds = (("stem", "Stem"), *((name, "Bottleneck") for name in module_blocks), ("head", "Head"))
    traced = symbolic_trace(model)
    with torch.no_grad():
        ShapeProp(traced).propagate(torch.zeros(reference.input_shape, dtype=torch.float32))
    modules = dict(traced.named_modules())
    tensors, operators = [], []
    input_id = output_id = ""
    for node in traced.graph.nodes:
        if node.op == "output":
            output_id = node.all_input_nodes[0].name
            continue
        shape = tuple(int(n) for n in node.meta["tensor_meta"].shape)
        if node.op == "placeholder":
            input_id = node.name
            tensors.append(TensorSpec(node.name, shape, reference.activation_bytes_per_element, "", "EXTERNAL"))
            continue
        stack = tuple(node.meta.get("nn_module_stack", {}))
        # torch built-in repr includes a process-dependent memory address.
        # Store the qualified callable name so metadata/cache hashes reproduce.
        target = (node.target if isinstance(node.target, str) else
                  f"{node.target.__module__}.{node.target.__qualname__}")
        candidates = [prefix for prefix in module_blocks if target == prefix or target.startswith(prefix + ".")
                      or any(name == prefix or name.startswith(prefix + ".") for name in stack)]
        block_id = max(candidates, key=len) if candidates else (
            "head" if target in ("avgpool", "fc") or node.op == "call_function" else "stem")
        inputs = tuple(dep.name for dep in node.all_input_nodes)
        dependencies = tuple(name for name in inputs if name != input_id)
        module = modules.get(target) if node.op == "call_module" else None
        kind = type(module).__name__ if module is not None else ""
        kwargs = {}
        if isinstance(module, torch.nn.Conv2d):
            kwargs = dict(cin=module.in_channels, cout=module.out_channels, kernel=tuple(module.kernel_size),
                          stride=tuple(module.stride), padding=tuple(module.padding), groups=module.groups,
                          weight_shape=tuple(module.weight.shape),
                          macs=shape[0]*shape[2]*shape[3]*module.out_channels*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1])
        elif isinstance(module, torch.nn.Linear):
            kwargs = dict(cin=module.in_features, cout=module.out_features, weight_shape=tuple(module.weight.shape),
                          macs=shape[0]*module.in_features*module.out_features)
        elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.ReLU)):
            kwargs = dict(fused=True)
        elif isinstance(module, (torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d)):
            pass
        elif node.op == "call_function" and node.target in (__import__("operator").add, torch.add):
            kind = "Add"
        elif node.op == "call_function" and node.target == torch.flatten:
            kind = "Flatten"
        else:
            raise ValueError(f"unsupported FX operator {node.op}: {node.target}")
        if module is not None:
            kwargs["parameter_bytes"] = sum(parameter.numel() for parameter in module.parameters(recurse=False)) * reference.weight_bytes_per_element
        operators.append(OperationSpec(node.name, kind, block_id, inputs, node.name, dependencies, target=target, **kwargs))
        tensors.append(TensorSpec(node.name, shape, reference.activation_bytes_per_element, node.name))
    graph = OperatorGraph("resnet50", tuple(tensors), tuple(operators), build_blocks(operators, tensors, block_kinds),
                          input_id, output_id, sum(parameter.numel() for parameter in model.parameters()) * reference.weight_bytes_per_element,
                          (("torch", torch.__version__), ("torchvision", torchvision.__version__)))
    graph.validate()
    return graph
