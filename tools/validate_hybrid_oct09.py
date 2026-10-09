"""Independent architecture workers, reproducible merge and saved-trace audit."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simplified_rapidchiplet'))
from hybrid_gui import run_suite, validate_config
from simple_rapidchiplet.hybrid.preferences import preference_objective, suite_cost
from simple_rapidchiplet.hybrid.report import write_json
from simple_rapidchiplet.hybrid.scheduler import Job, ScheduledJob, Schedule

JOB_ID = 'd09e10000001'
DEFAULT_FOLDER = ROOT/'simplified_rapidchiplet'/'results'/'hybrid_gui_oct09'/JOB_ID
CONFIG_PATH = ROOT/'simplified_rapidchiplet'/'configs'/'hybrid_oct09_suite.json'


def source_hashes():
    return {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT/'simplified_rapidchiplet'/'simple_rapidchiplet'/'hybrid').glob('*.py')}


def audit_trial(row, expected):
    folder = Path(row['folder'])
    metrics = row['metrics']
    if not metrics.get('feasible'):
        return {'folder':str(folder),'feasible':False,'error':row.get('error'),'ppa_claimed':False}
    if metrics['oracle_source_sha256']!=expected:
        raise ValueError(f'{folder}: estimator source changed')
    hw = json.loads((folder/'hardware.json').read_text(encoding='utf-8'))
    layout = json.loads((folder/'layout.json').read_text(encoding='utf-8'))
    workload = json.loads((folder/'workload.json').read_text(encoding='utf-8'))
    buffers = json.loads((folder/'buffers.json').read_text(encoding='utf-8'))
    assert metrics['total_macs']==workload['total_macs']
    assert not buffers['violations']
    assert all(value<=hw['unified_buffer_bytes'] for value in buffers['peak_by_die'].values())
    assert all(value<=buffers['allocated_scratch_capacity_bytes_per_core'] for value in buffers['concurrent_scratch_peak_by_core'].values())
    expected_dies = hw['cluster_rows']*hw['cluster_cols']*hw['compute_per_cluster']
    assert sum(node['role']=='compute' for node in layout['nodes'])==expected_dies
    for path in folder.glob('*.svg'):
        ET.parse(path)
    assert (folder/'die.svg').read_text(encoding='utf-8').count(' / core ')==hw['pe_count']
    events = []
    with (folder/'timeline.csv').open(encoding='utf-8',newline='') as stream:
        for item in csv.DictReader(stream):
            job = Job(item['id'],tuple(filter(None,item['dependencies'].split('|'))),
                      tuple(filter(None,item['resources'].split('|'))),float(item['finish_s'])-float(item['start_s']),
                      item['kind'],float(item['energy_j']))
            events.append(ScheduledJob(job,float(item['start_s']),float(item['finish_s'])))
    Schedule(tuple(events),metrics['latency_s'],{}).validate()
    assert math.isclose(max(event.finish_s for event in events),metrics['latency_s'],rel_tol=1e-12)
    assert math.isclose(sum(event.job.energy_j for event in events),metrics['dynamic_energy_j'],rel_tol=1e-10)
    assert math.isclose(sum(metrics['energy_components'].values()),metrics['energy_j'],rel_tol=1e-10)
    with (folder/'link_traffic.csv').open(encoding='utf-8',newline='') as stream:
        hop_bytes = sum(int(item['bytes']) for item in csv.DictReader(stream))
    assert hop_bytes==metrics['nop_hop_bytes']
    return {'folder':str(folder),'feasible':True,'source_verified':True,'event_count':len(events),
            'dependencies_and_calendars_verified':True,'energy_and_macs_verified':True,
            'svg_xml_and_actual_counts_verified':True,'ub_and_scratch_capacity_verified':True}


def merge(folder, config):
    rows = []
    models = None
    expected = source_hashes()
    for count in config['dies']:
        result = json.loads((folder/f'worker{count}'/'suite.json').read_text(encoding='utf-8'))
        assert result['config']['dies']==[count]
        for field in config:
            if field!='dies':
                assert result['config'][field]==config[field]
        if models is None:
            models = result['models']
        else:
            assert models==result['models']
        rows.extend(result['runs'])
    assert len(rows)==len(config['dies'])*len(config['noc'])*len(config['models'])
    objective = preference_objective(config['preference'])
    references = {model:{key:1.0 for key in ('latency_s','energy_j','power_w','area_mm2')} for model in config['models']}
    rankings = []
    for architecture in dict.fromkeys(row['architecture'] for row in rows):
        trials = [row for row in rows if row['architecture']==architecture]
        cost = suite_cost(trials,objective,references)
        rankings.append({'architecture':architecture,'cost':cost if math.isfinite(cost) else None,
                         'feasible_for_all_models':math.isfinite(cost)})
    legal = [row for row in rankings if row['feasible_for_all_models']]
    best = min(legal,key=lambda row:row['cost'])['architecture'] if legal else None
    result = {'config':config,'models':models,'runs':rows,'ranking':rankings,'best_architecture':best,
              'objective':'same common E/A/D units across architectures; equal-workload geometric mean; Power hard limits',
              'estimator_source_sha256':expected}
    write_json(folder/'suite.json',result)
    audit = [audit_trial(row,expected) for row in rows]
    write_json(folder/'validation.json',{'trials':audit,'passed':True,'feasible_count':sum(row['feasible'] for row in audit)})
    write_json(folder/'job.json',{'id':JOB_ID,'state':'complete','message':'五模型／三種架構驗證完成',
                                'completed':len(rows),'total':len(rows),'rows':rows,'result':result,'config':config})
    print(json.dumps({'best':best,'trials':len(rows),'feasible':sum(row['feasible'] for row in audit)},ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=int,choices=(4,8,16))
    parser.add_argument('--merge',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--folder',type=Path,default=DEFAULT_FOLDER)
    args = parser.parse_args()
    config = validate_config(json.loads(CONFIG_PATH.read_text(encoding='utf-8')))
    if args.worker:
        config['dies'] = [args.worker]
        run_suite(config,args.folder/f'worker{args.worker}',
                  lambda **items:print(items.get('message',''),flush=True) if 'message' in items else None,resume=args.resume)
    elif args.merge:
        merge(args.folder,config)
    else:
        parser.error('choose --worker or --merge')


if __name__=='__main__':
    main()
