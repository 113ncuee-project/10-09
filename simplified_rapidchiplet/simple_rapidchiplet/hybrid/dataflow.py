"""Bounded intra-core loop/tile search, followed by whole-plan oracle acceptance."""
from dataclasses import replace
import math
from .mapping import compile_mapping
from .ondie import core_service

DATAFLOWS = ('output_stationary','activation_reuse','weight_reuse')
OPTIONS = (('output_stationary',(0,0,0,0),128),
           ('activation_reuse',(1,16,8,8),64),
           ('activation_reuse',(1,32,4,4),128),
           ('weight_reuse',(1,16,8,8),64),
           ('weight_reuse',(1,32,4,4),128),
           ('output_stationary',(1,16,4,4),64))

def tune_dataflow(workload,plan,evaluator,objective,samples_per_layer=2):
    compiled = compile_mapping(workload,plan,evaluator.hardware)
    by_layer = {}
    for unit in compiled.units:
        if unit.microbatch == 0:
            by_layer.setdefault(unit.layer_id,[]).append(unit)
    mapped, evidence = [], []
    for mapping in plan.layer_mappings:
        all_units = by_layer[mapping.layer_id]
        # Largest and tail shapes provide bounded proposals. This is explicitly
        # sampled, not an exhaustive optimum over the entire dataflow space.
        chosen = sorted(all_units,key=lambda u:u.output_region.elements)
        samples = [chosen[0]] if len(chosen)==1 else [chosen[0],chosen[-1]][:samples_per_layer]
        layer = workload.layer_map[mapping.layer_id]
        variants = [(mapping.dataflow,mapping.microtile_shape,mapping.channel_tile),*OPTIONS]
        rows = []
        for dataflow,hint,channels in dict.fromkeys(variants):
            services = [core_service(replace(unit,dataflow=dataflow,microtile_shape=hint,channel_tile=channels),
                                     layer,workload,evaluator.hardware) for unit in samples]
            delay = sum(service.service_s for service in services)
            energy = sum(service.energy_j for service in services) + evaluator.context.idle_power_w*delay
            local = {'feasible':True,'latency_s':delay,'energy_j':energy,'power_w':energy/delay,
                     'area_mm2':evaluator.context.area_breakdown['total_silicon_mm2']}
            # Local proposals must not discard candidates on whole-workload
            # latency bounds; only the complete oracle applies those bounds.
            relaxed = replace(objective,max_latency_s=None,max_power_w=None,max_area_mm2=None)
            score = relaxed.cost(local,{key:1.0 for key in ('latency_s','energy_j','power_w','area_mm2')})
            rows.append((score,dataflow,hint,channels))
        best = min(rows,key=lambda row:row[0])
        mapped.append(replace(mapping,dataflow=best[1],microtile_shape=best[2],channel_tile=best[3]))
        evidence.append({'layer':mapping.layer_id,'sampled_units':len(samples),'variants':len(rows),
                         'selected_dataflow':best[1],'microtile_shape':best[2],'channel_tile':best[3]})
    candidate = replace(plan,layer_mappings=tuple(mapped))
    reference = {key:1.0 for key in ('latency_s','energy_j','power_w','area_mm2')}
    old = evaluator.evaluate(plan)
    new = evaluator.evaluate(candidate)
    old_score,new_score = (objective.cost(result.metrics,reference) for result in (old,new))
    accepted = math.isfinite(new_score) and new_score <= old_score
    return candidate if accepted else plan, {'method':'sampled loop/tile proposals + whole-plan PPA acceptance',
        'accepted':accepted,'old_cost':old_score if math.isfinite(old_score) else None,
        'new_cost':new_score if math.isfinite(new_score) else None,'layers':evidence,
        'partial_sum_policy':'INT24 kept in OL1 across C; no cross-die IC reduction or free psum spill'}
