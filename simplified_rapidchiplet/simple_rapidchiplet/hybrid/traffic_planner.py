"""Whole-DAG frontier DP proposals for layer-switch communication.

Exact rectangular demands and native package routes inform a bounded proposal
search. Residual producers remain in the frontier until their last consumer.
A chain reduces to the usual O(G*M^2) state recurrence; a general DAG uses a
bounded beam, without claiming that pairwise DP solves arbitrary DAGs exactly.
Only the full event oracle supplies final PPA and feasibility.
"""
from dataclasses import replace
import math

from .communication import _Demand, _atomic_demands
from .mapping import MappingPlan, _shards, required_input_regions, enumerate_partitions
from .planner import _nearest_memory, _partition_service


def _group_states(workload, base, group, evaluator, count):
    hw = evaluator.hardware
    cores = evaluator.package.mapping_core_ids
    original = tuple(base.mapping_map[layer] for layer in group.layer_ids)
    states = [original]
    # Preserve disjoint allocations, vary correspondence/placement and Part.
    for variant in range(1, count):
        offset = (variant // 3 * hw.cores_per_die) % len(cores)
        pool = tuple(sorted(cores,key=lambda core:(int(core.rsplit(':',1)[1]),int(core.split('/')[0].split(':')[1])))) if variant >= count//2 else cores
        mapped = []
        for item in original:
            layer = workload.layer_map[item.layer_id]
            parts = enumerate_partitions(workload, item.layer_id, len(item.chiplets), base.microbatch_size)
            ranked = sorted(parts, key=lambda p:_partition_service(workload,layer,p,base.microbatch_size,hw))
            if variant % 3 == 1:
                part = min(parts, key=lambda p:(p.h*p.w, -p.k, p.as_tuple()))
            elif variant % 3 == 2:
                part = min(parts, key=lambda p:(p.k, -p.h*p.w, p.as_tuple()))
            else:
                part = ranked[0]
            allocated = tuple(pool[(cores.index(core)+offset)%len(cores)] for core in item.chiplets)
            memory = _nearest_memory(allocated, base.memory_ids, hw)
            mapped.append(replace(item,chiplets=allocated,partition=part,
                                  input_dram=memory,weight_dram=memory,output_dram=memory))
        candidate = tuple(mapped)
        if candidate not in states:
            states.append(candidate)
    return tuple(states)


def _edge_cost(workload, base, selected, new, evaluator):
    """Charges every true DAG edge when its consumer becomes known.

    Same-group edges use owner/consumer regions. Cross-group edges are backed
    in DRAM (matching the existing group-barrier contract). Atomic multicast
    shares identical regions; native routes charge every directed hop once.
    """
    hw = evaluator.hardware
    mappings = {item.layer_id:item for item in (*selected,*new)}
    demands = []
    memory_bytes = 0
    for consumer in new:
        layer = workload.layer_map[consumer.layer_id]
        out = workload.tensor_map[layer.output]
        shards = _shards(out.shape,consumer.partition,0,base.microbatch_size)
        for dst, output in zip(consumer.chiplets,shards):
            for requirement in required_input_regions(workload,layer,output):
                tensor = workload.tensor_map[requirement.tensor_id]
                producer = tensor.producer
                prior = mappings.get(producer)
                if prior is None:
                    # External input; interleave is approximated only during
                    # proposals. The final oracle implements exact stripes.
                    src = consumer.input_dram if consumer.input_dram != 'interleave' else base.memory_ids[0]
                    demands.append(_Demand(src,dst,tensor.id,requirement.region,tensor.bits,0,(),consumer.layer_id,'input_read'))
                    memory_bytes += requirement.region.elements * tensor.bytes_per_element
                    continue
                owners = _shards(tensor.shape,prior.partition,0,base.microbatch_size)
                same_group = base.group_map[producer].id == base.group_map[layer.id].id
                for src, owned in zip(prior.chiplets,owners):
                    owner_key = (f'{producer}:{src}',)
                    overlap = owned.intersection(requirement.region)
                    if overlap is None:
                        continue
                    if not same_group:
                        src = prior.output_dram if prior.output_dram != 'interleave' else base.memory_ids[0]
                        memory_bytes += overlap.elements * tensor.bytes_per_element
                    demands.append(_Demand(src,dst,tensor.id,overlap,tensor.bits,0,owner_key,consumer.layer_id,
                                           'direct' if same_group else 'boundary_read'))
    loads, delivered = {}, 0
    for src,dsts,amount,*_ in _atomic_demands(demands):
        delivered += amount * len(dsts)
        edges = set()
        for dst in dsts:
            edges.update(zip((route:=evaluator.context.route(src,dst)),route[1:]))
        for edge in edges:
            loads[edge] = loads.get(edge,0)+amount
    network = max((amount/evaluator.context.link_bandwidth_Bps[edge] for edge,amount in loads.items()),default=0)
    memory = memory_bytes / (hw.cluster_count*hw.dram_bandwidth_Bps)
    return max(network,memory), sum(loads.values()), delivered


def traffic_plans(workload, base, evaluator, *, states_per_group=8, beam_width=8, keep=3):
    hw = evaluator.hardware
    if min(states_per_group,beam_width,keep) < 1:
        raise ValueError('positive DP bounds required')
    layer_group = {layer:idx for idx,group in enumerate(base.groups) for layer in group.layer_ids}
    last_user = {layer.id:max((layer_group[user] for user in workload.consumers[layer.output]),
                              default=layer_group[layer.id]) for layer in workload.layers}
    # (proxy delay, hop bytes, delivered bytes, complete mapping history)
    beam = [(0.0,0,0,())]
    transitions = 0
    edge_cache = {}
    for index, group in enumerate(base.groups):
        by_frontier = {}
        for state in _group_states(workload,base,group,evaluator,states_per_group):
            costs = [_partition_service(workload,workload.layer_map[item.layer_id],item.partition,base.microbatch_size,hw) for item in state]
            count = workload.batch_size//base.microbatch_size
            local = sum(costs) + (count-1)*(max(costs) if group.mode=='spatial' else sum(costs))
            # Whole macro weights must reside in UB; core count alone misses
            # concentrated FC placement. This capacity pressure only guides
            # proposals. The event/lifetime oracle remains the final gate.
            weights_by_die = {}
            for item in state:
                layer = workload.layer_map[item.layer_id]
                if not layer.weight_shape:
                    continue
                output = workload.tensor_map[layer.output]
                for core, shard in zip(item.chiplets,_shards(output.shape,item.partition,0,base.microbatch_size)):
                    amount = shard.shape[1]*math.prod(layer.weight_shape[1:])*layer.weight_bits//8
                    die = core.split('/')[0]
                    weights_by_die[die] = weights_by_die.get(die,0)+amount
            pressure = max([1.0,*(amount/hw.unified_buffer_bytes for amount in weights_by_die.values())])
            local *= pressure*pressure
            for score,hops,delivered,history in beam:
                needed = {workload.tensor_map[tensor].producer for item in state
                          for tensor in workload.layer_map[item.layer_id].inputs}
                edge_key = (tuple(item for item in history if item.layer_id in needed),state)
                if edge_key not in edge_cache:
                    edge_cache[edge_key] = _edge_cost(workload,base,edge_key[0],state,evaluator)
                network,new_hops,new_delivered = edge_cache[edge_key]
                combined = (*history,*state)
                frontier = tuple((item.layer_id,item.chiplets,item.partition,item.output_dram)
                                 for item in combined if last_user[item.layer_id] > index)
                if index == len(base.groups)-1:
                    frontier = (*frontier,('final',state))
                # Retain the minimum additive delay proxy for each live frontier.
                # Never sum per-layer EDP. Full E/D/P/A rank the final proposals.
                row = (score+local+network*count,hops+new_hops*count,delivered+new_delivered*count,combined)
                if frontier not in by_frontier or row[:3] < by_frontier[frontier][:3]:
                    by_frontier[frontier] = row
                transitions += 1
        beam = sorted(by_frontier.values(),key=lambda row:row[:3])[:beam_width]
    plans = [base]
    for row in beam[:keep]:
        candidate = replace(base,layer_mappings=row[3])
        if candidate not in plans:
            plans.append(candidate)
    return tuple(plans), {'method':'whole-DAG live-frontier DP with bounded beam',
        'states_per_group':states_per_group,'beam_width':beam_width,'transitions':transitions,
        'residual_edges_preserved':True,'proxy_is_final_ppa':False,
        'complexity':'chain O(G*M^2); general DAG bounded O(G*W*M*region routing)',
        'proposal_proxy_latency_s':beam[0][0] if beam else None}


def select_traffic_plan(workload, base, evaluator, objective, *, states_per_group=8, beam_width=8):
    candidates, evidence = traffic_plans(workload,base,evaluator,states_per_group=states_per_group,beam_width=beam_width)
    reference = {key:1.0 for key in ('latency_s','energy_j','power_w','area_mm2')}
    rows = []
    for plan in candidates:
        result = evaluator.evaluate(plan)
        rows.append((objective.cost(result.metrics,reference),plan,result.metrics))
    best = min(rows,key=lambda row:row[0])
    evidence['full_oracle_queries'] = len(rows)
    evidence['candidates'] = [{'cost':cost if math.isfinite(cost) else None,'metrics':metrics} for cost,_,metrics in rows]
    # Fall back to base on total infeasibility; caller reports capacity/limits.
    return best[1] if math.isfinite(best[0]) else base, evidence
