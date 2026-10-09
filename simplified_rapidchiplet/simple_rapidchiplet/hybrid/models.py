"""Real torchvision inference DAGs; meta tensors avoid allocating model weights.

No downloads, pretrained accuracy claim, or model-specific latency constants.
INT8 is the accelerator's storage/arithmetic assumption, not PTQ validation.
"""
from __future__ import annotations

import math
import operator
from functools import lru_cache

from .workload import LayerSpec, TensorSpec, Workload, _fuse_postops

MODEL_NAMES = ('resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19', 'alexnet', 'mobilenet_v2')


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


@lru_cache(maxsize=24)
def extract_model(name, batch_size=1, input_size=224, fuse_postops=True):
    if name not in MODEL_NAMES or batch_size < 1 or input_size < 32:
        raise ValueError('unsupported model or invalid input dimensions')
    import torch
    import torchvision
    from torch.fx import symbolic_trace, Node
    from torch.fx.passes.shape_prop import ShapeProp

    with torch.device('meta'):
        model = getattr(torchvision.models, name)(weights=None).eval()
    graph = symbolic_trace(model)
    ShapeProp(graph).propagate(torch.empty(batch_size, 3, input_size, input_size, device='meta'))
    modules = dict(graph.named_modules())
    tensors, layers, input_ids, output_ids = [], [], [], []
    ids = {}
    def input_nodes(value):
        if isinstance(value, Node):
            return [value]
        if isinstance(value, (tuple, list)):
            return [node for item in value for node in input_nodes(item)]
        return []
    for node in graph.graph.nodes:
        if node.op == 'output':
            output_ids = [ids[item] for item in input_nodes(node.args)]
            continue
        shape = tuple(node.meta['tensor_meta'].shape)
        if len(shape) == 2:
            shape = (*shape, 1, 1)
        if len(shape) != 4:
            raise ValueError(f'{name}: unsupported tensor rank at {node.name}')
        ident = node.name
        ids[node] = ident
        if node.op == 'placeholder':
            tensors.append(TensorSpec(ident, shape))
            input_ids.append(ident)
            continue
        inputs = tuple(ids[item] for item in input_nodes(node.args))
        kind, kernel, stride, padding, dilation, groups = 'Identity', (1, 1), (1, 1), (0, 0), (1, 1), 1
        weights, macs, vector = (), 0, 0
        if node.op == 'call_module':
            module = modules[node.target]
            if isinstance(module, torch.nn.Conv2d):
                kind = 'Conv2d'
                kernel, stride, padding, dilation = map(_pair, (module.kernel_size, module.stride, module.padding, module.dilation))
                groups = module.groups
                weights = tuple(module.weight.shape)
                macs = math.prod(shape[1:]) * weights[1] * math.prod(kernel)
                vector = int(module.bias is not None)
            elif isinstance(module, torch.nn.Linear):
                kind, weights = 'Linear', tuple(module.weight.shape)
                macs = weights[0] * weights[1]
                vector = int(module.bias is not None)
            elif isinstance(module, (torch.nn.ReLU, torch.nn.ReLU6)):
                kind, vector = 'ReLU', 2 if isinstance(module, torch.nn.ReLU6) else 1
            elif isinstance(module, torch.nn.BatchNorm2d):
                kind, vector = 'BatchNorm2d', 2
            elif isinstance(module, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
                kind = type(module).__name__
                kernel, stride, padding = map(_pair, (module.kernel_size, module.stride or module.kernel_size, module.padding))
                dilation = _pair(getattr(module, 'dilation', 1))
                vector = math.prod(kernel)
            elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
                kind = 'AdaptiveAvgPool2d'
            elif not isinstance(module, (torch.nn.Dropout, torch.nn.Identity)):
                raise ValueError(f'{name}: unsupported module {type(module).__name__}')
        elif node.op == 'call_function' and node.target in (operator.add, torch.add):
            kind, vector = 'Add', 1
        elif node.op == 'call_function' and node.target is torch.flatten:
            if node.args[1:] not in ((), (1,)):
                raise ValueError('only feature flatten is supported')
            kind = 'Flatten'
        elif node.op == 'call_function' and node.target is torch.nn.functional.adaptive_avg_pool2d:
            kind = 'AdaptiveAvgPool2d'
        elif node.op == 'call_method' and node.target == 'mean':
            if tuple(node.args[1]) != (2, 3):
                raise ValueError('only global spatial mean is supported')
            kind = 'AdaptiveAvgPool2d'
        else:
            raise ValueError(f'{name}: unsupported FX operation {node.op}: {node.target}')
        if kind == 'AdaptiveAvgPool2d':
            inp = next(tensor for tensor in tensors if tensor.id == inputs[0])
            vector = math.ceil(inp.shape[2] / shape[2]) * math.ceil(inp.shape[3] / shape[3])
        tensors.append(TensorSpec(ident, shape, producer=ident))
        target_name = (f'{node.target.__module__}.{node.target.__qualname__}'
                       if callable(node.target) else str(node.target))
        layers.append(LayerSpec(ident, kind, inputs, ident, kernel, stride, padding, dilation,
                                groups, weights, 8, macs, vector, target_name.rsplit('.', 1)[0]))
    if fuse_postops:
        tensors, layers = _fuse_postops(tensors, layers)
    workload = Workload(name, tuple(tensors), tuple(layers), tuple(input_ids), tuple(output_ids), batch_size,
                        f'torchvision {torchvision.__version__} FX/meta shapes; weights=None; INT8/INT24 assumption; input={input_size}')
    workload.validate()
    return workload
