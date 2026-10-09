"""INDM cluster/ring hardware + Gemini pipeline encoding + learned RL search.

Run from repository root with .venv/Scripts/python.exe
simplified_rapidchiplet/hybrid_dse.py --help
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from simple_rapidchiplet.hybrid.architecture import enumerate_architectures, resource_totals
from simple_rapidchiplet.hybrid.evaluator import HybridEvaluator
from simple_rapidchiplet.hybrid.hardware import HardwareSpec
from simple_rapidchiplet.hybrid.mapping import initial_mapping
from simple_rapidchiplet.hybrid.planner import balanced_mapping, choose_plan, propose_plans
from simple_rapidchiplet.hybrid.rapid_backend import DEFAULT_RAPID_ROOT
from simple_rapidchiplet.hybrid.report import save_evaluation, write_json
from simple_rapidchiplet.hybrid.search import SearchObjective, plan_from_dict, run_search
from simple_rapidchiplet.hybrid.workload import Workload, TensorSpec, LayerSpec, extract_resnet50, make_toy_workload
from simple_rapidchiplet.hybrid.models import MODEL_NAMES, extract_model
from simple_rapidchiplet.hybrid.preferences import PREFERENCES, preference_objective, constraint_report
from simple_rapidchiplet.hybrid.dataflow import DATAFLOWS, tune_dataflow
from simple_rapidchiplet.hybrid.traffic_planner import select_traffic_plan


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--mode',choices=('evaluate','search','compare','architecture'),default='search')
    result.add_argument('--model',choices=('toy',*MODEL_NAMES),default='toy')
    result.add_argument('--workload',type=Path,help='explicit supported-operator workload.json; overrides model/batch')
    result.add_argument('--batch',type=int,default=4)
    result.add_argument('--toy-channels',type=int,default=8)
    result.add_argument('--toy-spatial',type=int,default=8)
    result.add_argument('--hardware',type=Path,help='JSON HardwareSpec overrides')
    result.add_argument('--dies',type=int,choices=(4,8,16),default=16)
    result.add_argument('--noc',choices=('multi_ring','mesh'),default='multi_ring')
    result.add_argument('--rapid-root',default=DEFAULT_RAPID_ROOT)
    result.add_argument('--planner',choices=('balanced','beam','uniform','traffic_dp'),default='balanced')
    result.add_argument('--dataflow',choices=(*DATAFLOWS,'auto'),default='output_stationary')
    result.add_argument('--prefetch-slots',type=int,choices=(1,2),default=1)
    result.add_argument('--preference',choices=tuple(PREFERENCES))
    result.add_argument('--groups',type=int,default=3,help='max layers per group/proposal')
    result.add_argument('--microbatch',type=int,default=1)
    result.add_argument('--microbatch-candidates',default='1,2,4')
    result.add_argument('--planning-budget',type=int,default=4)
    result.add_argument('--pipeline-window',type=int,default=2)
    result.add_argument('--temporal',action='store_true',help='disable spatial groups for sequential ablation')
    result.add_argument('--weights',choices=('cold','persistent'),default='cold')
    result.add_argument('--collective',choices=('multicast','unicast','ring'),default='multicast')
    result.add_argument('--mapping',type=Path,help='evaluate/search an existing mapping.json')
    result.add_argument('--method',choices=('rl','random','greedy','sa'),default='rl')
    result.add_argument('--proposal',choices=('uniform','cluster_guided'),default='uniform')
    result.add_argument('--budget',type=int,default=64,help='unique physical mapping queries, includes initial')
    result.add_argument('--seed',type=int,default=7)
    result.add_argument('--seeds',default='7,8,9',help='compare mode seeds')
    result.add_argument('--max-per-operator',type=int,default=12)
    result.add_argument('--horizon',type=int,default=16)
    result.add_argument('--objective',choices=('edp','latency','energy'),default='edp')
    result.add_argument('--area-weight',type=float,default=0)
    result.add_argument('--power-weight',type=float,default=0)
    result.add_argument('--max-area',type=float)
    result.add_argument('--max-power',type=float)
    result.add_argument('--max-latency',type=float)
    for metric in ('power','area','latency'):
        result.add_argument('--soft-'+metric,action='store_true',help='treat entered limit as a soft goal')
    result.add_argument('--checkpoint',type=Path)
    result.add_argument('--resume',action='store_true')
    result.add_argument('--trace',action='store_true',help='write full region/service NDJSON (large for ResNet)')
    result.add_argument('--out',type=Path,default=Path(__file__).parent/'results'/'hybrid_run')
    return result


def prepare_plan(args,workload,evaluator):
    if args.mapping:
        plan = plan_from_dict(json.loads(args.mapping.read_text(encoding='utf-8')))
        if args.dataflow == 'auto':
            plan,flow = tune_dataflow(workload,plan,evaluator,objective(args))
            return plan,{'method':'supplied mapping','dataflow_search':flow}
        return plan,{'method':'supplied mapping'}
    if args.planner == 'beam':
        candidates = propose_plans(workload,evaluator.package.mapping_core_ids,evaluator.package.memory_ids,
                     evaluator.hardware,microbatch_sizes=tuple(int(value) for value in args.microbatch_candidates.split(',')),
                     max_group_layers=args.groups,max_candidates=args.planning_budget,objective=args.objective,
                     weight_policy=args.weights)
        candidates = tuple(replace(candidate,plan=replace(candidate.plan,collective=args.collective)) for candidate in candidates)
        if args.preference:
            reference = {key:1.0 for key in ('latency_s','energy_j','power_w','area_mm2')}
            ranked = [(objective(args).cost(evaluator.evaluate(candidate.plan).metrics,reference),candidate.plan) for candidate in candidates]
            best = min(ranked,key=lambda row:row[0])
            plan = best[1] if best[0] < float('inf') else None
            summary = {'method':'beam + preference oracle ranking','queries':len(ranked)}
        else:
            plan,summary = choose_plan(candidates,evaluator,objective=args.objective)
        if plan is None:
            raise ValueError('No feasible grouping proposal. Inspect capacity; reduce groups/microbatch or increase hardware buffers.')
        return finish_dataflow(args,workload,evaluator,plan,summary)
    common = dict(microbatch_size=args.microbatch,max_group_layers=args.groups,pipeline=not args.temporal)
    if args.planner in ('balanced','traffic_dp'):
        plan = balanced_mapping(workload,evaluator.package.mapping_core_ids,evaluator.package.memory_ids,
                                hardware=evaluator.hardware,weight_policy=args.weights,**common)
    else:
        plan = initial_mapping(workload,evaluator.package.mapping_core_ids,evaluator.package.memory_ids,**common)
        plan = replace(plan,layer_mappings=tuple(replace(item,weight_policy=args.weights) for item in plan.layer_mappings))
    plan = replace(plan,collective=args.collective)
    planning = {'method':args.planner,'globally_exact':False}
    if args.planner == 'traffic_dp':
        plan,planning = select_traffic_plan(workload,plan,evaluator,objective(args))
    return finish_dataflow(args,workload,evaluator,plan,planning)


def finish_dataflow(args,workload,evaluator,plan,planning):
    if args.dataflow == 'auto':
        plan,flow = tune_dataflow(workload,plan,evaluator,objective(args))
        planning['dataflow_search'] = flow
    else:
        plan = replace(plan,layer_mappings=tuple(replace(item,dataflow=args.dataflow) for item in plan.layer_mappings))
    return plan,planning


def metadata(evaluator):
    package = evaluator.package
    return {'core_clusters':{core:package.node_cluster[core.split('/')[0]] for core in package.mapping_core_ids},
            'core_positions':{core:(package.nodes[core.split('/')[0]].x_mm,
                                   package.nodes[core.split('/')[0]].y_mm) for core in package.mapping_core_ids},
            'memory_positions':{memory:package.positions[memory] for memory in package.memory_ids}}


def read_workload(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    tensors = tuple(TensorSpec(**{**item,'shape':tuple(item['shape'])}) for item in data['tensors'])
    layers = []
    for item in data['layers']:
        values = dict(item)
        for field in ('inputs','kernel','stride','padding','dilation','weight_shape','post_ops'):
            if field in values:
                values[field] = tuple(values[field])
        layers.append(LayerSpec(**values))
    workload = Workload(data['name'],tensors,tuple(layers),tuple(data['input_ids']),tuple(data['output_ids']),
                        data['batch_size'],data.get('source','explicit'))
    workload.validate()
    return workload


def objective(args, *, final=False):
    if args.preference:
        return preference_objective(args.preference,max_power=args.max_power,max_area=args.max_area,max_latency=args.max_latency,
                   strict_power=not args.soft_power,strict_area=not args.soft_area,strict_latency=not args.soft_latency,
                   relax_hard_limits=not final)
    return SearchObjective(latency_weight=float(args.objective != 'energy'),
                           energy_weight=float(args.objective != 'latency'),area_weight=args.area_weight,
                           power_weight=args.power_weight,max_area_mm2=None if args.soft_area else args.max_area,
                           max_power_w=None if args.soft_power else args.max_power,
                           max_latency_s=None if args.soft_latency else args.max_latency,
                           soft_area_mm2=args.max_area if args.soft_area else None,
                           soft_power_w=args.max_power if args.soft_power else None,
                           soft_latency_s=args.max_latency if args.soft_latency else None,relax_hard_limits=not final)


def run_one(args,workload,spec,folder,*,method=None,seed=None):
    start = time.monotonic()
    evaluator = HybridEvaluator(workload,spec,rapid_root=args.rapid_root,
                               max_inflight_microbatches=args.pipeline_window,trace=False)
    plan,planning = prepare_plan(args,workload,evaluator)
    result = None
    if method is not None:
        checkpoint = args.checkpoint or folder/'checkpoint.json'
        result = run_search(workload,plan,evaluator,evaluator.package.mapping_core_ids,evaluator.package.memory_ids,
                            method=method,seed=args.seed if seed is None else seed,budget=args.budget,
                            horizon=args.horizon,max_per_operator=args.max_per_operator,objective=objective(args),
                            checkpoint_path=checkpoint,resume=args.resume,evaluator_identity=evaluator.identity,
                            proposal=args.proposal,**metadata(evaluator))
        plan = result.best_plan
    evaluation = evaluator.evaluate(plan,retain_trace=True)
    save_evaluation(folder,evaluation,evaluator,plan,search=result,planning=planning,
                    arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},trace=args.trace)
    row = {'folder':str(folder.resolve()),'elapsed_s':time.monotonic()-start,
           'hardware_resources':resource_totals(spec),'method':method or 'evaluate',
           'seed':args.seed if seed is None else seed,'metrics':evaluation.metrics,
           'search_evaluations':None if result is None else result.evaluations}
    row['constraints'] = constraint_report(evaluation.metrics,objective(args,final=True))
    write_json(folder/'run.json',row)
    return row


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if min(args.batch,args.groups,args.microbatch,args.budget,args.planning_budget,args.pipeline_window) < 1:
        argument_parser.error('batch, grouping, window and budgets must be positive')
    if args.batch % args.microbatch:
        argument_parser.error('microbatch must divide batch')
    if args.temporal and args.planner == 'beam':
        argument_parser.error('--temporal uses the balanced or uniform planner; beam explores both modes')
    if args.resume and args.mode not in ('search','compare','architecture'):
        argument_parser.error('resume applies to search modes')
    if args.checkpoint and args.mode in ('compare','architecture'):
        argument_parser.error('multi-run modes use separate per-trial checkpoints; omit --checkpoint')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    workload = read_workload(args.workload) if args.workload else extract_model(args.model,batch_size=args.batch) if args.model!='toy' else make_toy_workload(
                   batch=args.batch,channels=args.toy_channels,spatial=args.toy_spatial)
    args.batch = workload.batch_size
    if args.batch % args.microbatch:
        argument_parser.error('microbatch must divide the supplied workload batch')
    reference = HardwareSpec.from_dict(json.loads(args.hardware.read_text(encoding='utf-8'))) if args.hardware else HardwareSpec()
    reference = replace(reference,prefetch_slots=args.prefetch_slots)
    candidates = enumerate_architectures(reference)
    spec = next(candidate.hardware for candidate in candidates if candidate.id==f'dies{args.dies}_{args.noc}')
    rows = []
    if args.mode == 'architecture':
        for candidate in candidates:
            row = run_one(args,workload,candidate.hardware,args.out/candidate.id,method=args.method)
            rows.append(row)
            write_json(args.out/'comparison.json',{'mode':'architecture','runs':rows,
                        'ranking':'raw common-unit objective across hardware; normalized per-run search costs cannot compare architectures'})
    elif args.mode == 'compare':
        for seed in (int(item) for item in args.seeds.split(',')):
            for method in ('rl','random','greedy','sa'):
                row = run_one(args,workload,spec,args.out/f'{method}_seed{seed}',method=method,seed=seed)
                rows.append(row)
                write_json(args.out/'comparison.json',{'mode':'compare','runs':rows,
                      'fairness':'same initial planner/resources/proposal and unique-query budget; independent trial caches'})
    else:
        rows.append(run_one(args,workload,spec,args.out,method=args.method if args.mode=='search' else None))
    def raw_cost(row):
        metrics = row['metrics']
        if not metrics['feasible']:
            return float('inf')
        reference = {key:1.0 for key in ('latency_s','energy_j','area_mm2','power_w')}
        return objective(args,final=True).cost(metrics,reference)
    feasible = [row for row in rows if raw_cost(row) < float('inf')]
    best = min(feasible,key=raw_cost) if feasible else None
    physical = [row for row in rows if row['constraints']['physical_feasible']]
    def gap_key(row):
        gaps = [limit['relative_excess'] for limit in row['constraints']['limits'] if limit['strict']]
        return max(gaps,default=0),sum(gaps)/max(1,len(gaps)),objective(args).cost(row['metrics'],{key:1 for key in ('energy_j','latency_s','area_mm2','power_w')})
    closest = min(physical,key=gap_key) if physical and best is None else None
    write_json(args.out/'experiment.json',{'runs':rows,'best_folder':None if best is None else best['folder'],
                                         'closest_folder':closest['folder'] if closest else None,
                                         'recommendation_status':'estimated_compliant' if best else 'closest_exceeds_strict_limits' if closest else 'no_physically_valid_candidate',
                                         'source':'native RapidChiplet plus analytical compute/NoC/memory/pipeline oracle'})
    for row in rows:
        metrics = row['metrics']
        print(f"{row['method']} seed={row['seed']}: feasible={metrics['feasible']} latency={metrics['latency_s']*1e3:.4f}ms "
              f"energy={metrics['energy_j']*1e3:.4f}mJ area={metrics['area_mm2']:.3f}mm² ({row['elapsed_s']:.1f}s)")
    print(f'Artifacts: {args.out}')
    return 0 if feasible else 2


if __name__ == '__main__':
    raise SystemExit(main())
