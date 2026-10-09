"""Operator requirements -> missing resident tensor slices -> canonical flows."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from .flows import Flow
from .tensor_ownership import OwnedSlice, TensorOwnership
from .workload import WorkloadPartition


def input_requirements(op, output_start, output_end, input_tensor):
    if op.kind == "Conv2d":
        # Output groups determine exactly which input channel groups are read.
        # This covers general grouped Conv and depthwise with a multiplier.
        outputs_per_group = op.cout // op.groups
        inputs_per_group = op.cin // op.groups
        first = output_start // outputs_per_group
        last = (output_end-1) // outputs_per_group
        return ((first*inputs_per_group, (last+1)*inputs_per_group),)
    if op.kind == "Linear":
        return ((0, input_tensor.channels),)
    if op.kind in ("Add", "MaxPool2d", "AdaptiveAvgPool2d", "Flatten"):
        return ((output_start, output_end),)
    raise ValueError(f"unsupported physical consumer: {op.kind}")


@dataclass(frozen=True)
class DerivedWorkload:
    workload: WorkloadPartition
    operator_workloads: tuple[dict, ...]
    ownership_trace: tuple[dict, ...]
    peak_sram_bytes: tuple[int, ...]
    violations: tuple[str, ...]
    final_residency: tuple[tuple[str, int, int, int], ...]
    final_primary: tuple[tuple[str, int, int, int], ...]


def derive_operator_workload(graph, plan, cfg):
    graph.validate()
    tensors, operators = graph.tensor_map, graph.operation_map
    block_index = {block.id:index for index,block in enumerate(graph.blocks)}
    shards_by_op = {op_id:shards for block in plan.block_mappings for op_id,shards in block.operator_shards}
    ownership = TensorOwnership(tensors, plan.total_chiplets)
    remaining_users = Counter(name for op in graph.operators for name in op.inputs)
    flows, work_records, ownership_trace = [], [], []
    total_ops = [0.0]*plan.total_chiplets
    peak = [0]*plan.total_chiplets
    violations = []
    for op in graph.operators:
        if op.id not in shards_by_op:
            break  # Prefix evaluation retains future users for correct liveness.
        shards = shards_by_op[op.id]
        input_records = []
        if op.fused:
            # New logical values, identical PRIMARY ownership. Never inherit
            # cached replicas of a previous value that has since been changed.
            owners = ownership.primary[op.inputs[0]]
            expected = tuple((shard.chiplet,shard.start,shard.end) for shard in shards)
            if set(expected) != {(owner.chiplet,owner.start,owner.end) for owner in owners}:
                raise ValueError("a fused BN/ReLU cannot silently change physical assignment")
        else:
            for shard in shards:
                for tensor_id in op.inputs:
                    tensor = tensors[tensor_id]
                    required = input_requirements(op, shard.start, shard.end, tensor)
                    input_records.append({"tensor_id":tensor_id,"destination":shard.chiplet,
                                          "required_ranges":required,"source":tensor.source})
                    for src,dst,start,end,amount in ownership.required_transfers(tensor_id,shard.chiplet,required):
                        producer = operators[tensor.producer]
                        kind = "intra_block" if producer.block_id == op.block_id else "transition"
                        flows.append(Flow(tensor_id,src,dst,amount,kind,block_index[producer.block_id],block_index[op.block_id],
                                          producer.block_id,op.block_id,producer_operator=producer.id,consumer_operator=op.id,
                                          channel_start=start,channel_end=end,event_id=op.id))
            owners = tuple(OwnedSlice(shard.chiplet,shard.start,shard.end) for shard in shards)
        ownership.produce(op.output,owners)
        weights = {shard.chiplet:shard.weight_bytes for shard in shards}
        resident = ownership.resident_bytes()
        for node in range(plan.total_chiplets):
            footprint = resident[node]+weights.get(node,0)
            peak[node] = max(peak[node],footprint)
            if footprint > cfg.chiplet.sram_mb*1024**2:
                violations.append(f"{op.id}: C{node} requires {footprint} bytes > SRAM {cfg.chiplet.sram_mb} MiB")
        node_ops = [0.0]*plan.total_chiplets
        for shard in shards:
            node_ops[shard.chiplet] = shard.macs*cfg.chiplet.op_per_mac
            total_ops[shard.chiplet] += node_ops[shard.chiplet]
        work_records.append({"operator_id":op.id,"kind":op.kind,"block_id":op.block_id,
                             "macs":op.macs,"ops_per_chiplet":tuple(node_ops),
                             "shards":[asdict(shard) for shard in shards],"input_requirements":input_records,
                             "resident_bytes":resident,"fused":op.fused})
        ownership_trace.append({"operator_id":op.id,"tensor_id":op.output,"block_id":op.block_id,
                                "owners":[asdict(owner) for owner in owners]})
        for tensor_id in op.inputs:
            remaining_users[tensor_id] -= 1
            if remaining_users[tensor_id] == 0:
                ownership.retire(tensor_id)
    workload = WorkloadPartition(tuple(flows),tuple(total_ops),
               tuple(" / ".join(block.block_id for block in plan.block_mappings if node in block.active_chiplets)
                     for node in range(plan.total_chiplets)),"physical-K", " | ".join(
                     f"{block.block_id}:K@{','.join(map(str,block.active_chiplets))}" for block in plan.block_mappings),
               mapping_plan=plan.to_dict())
    return DerivedWorkload(workload,tuple(work_records),tuple(ownership_trace),tuple(peak),
                           tuple(dict.fromkeys(violations)),ownership.snapshot(),
                           tuple((name, owner.chiplet, owner.start, owner.end)
                                 for name, owners in sorted(ownership.primary.items()) for owner in owners))
