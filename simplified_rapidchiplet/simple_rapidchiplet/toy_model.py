"""Explicit tiny fixture for hand calculations and exhaustive validation."""
from .tensor_metadata import OperationSpec, OperatorGraph, TensorSpec, build_blocks


def make_toy_graph(block_count=2, channels=4):
    tensors = [TensorSpec("image",(1,channels,2,2),4,"","EXTERNAL")]
    operators = []
    for index in range(block_count):
        name, block = f"conv{index}", f"block{index}"
        previous = tensors[-1]
        operators.append(OperationSpec(name,"Conv2d",block,(previous.id,),name,
                         () if index == 0 else (previous.producer,),cin=channels,cout=channels,
                         kernel=(1,1),stride=(1,1),padding=(0,0),weight_shape=(channels,channels,1,1),
                         parameter_bytes=channels*channels*4,macs=4*channels*channels))
        tensors.append(TensorSpec(name,(1,channels,2,2),4,name))
    blocks = build_blocks(operators,tensors,tuple((f"block{i}","Toy") for i in range(block_count)))
    graph = OperatorGraph("toy",tuple(tensors),tuple(operators),blocks,"image",tensors[-1].id,
                          sum(op.parameter_bytes for op in operators),source="explicit hand-checkable toy fixture")
    graph.validate()
    return graph
