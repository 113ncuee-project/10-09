"""Versioned cost16 full-model workers and independent saved-artifact audit."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simplified_rapidchiplet'))
from hybrid_gui import run_suite, validate_config
from simple_rapidchiplet.hybrid.hardware import HardwareSpec
from simple_rapidchiplet.hybrid.preferences import preference_objective, rank_architectures
from simple_rapidchiplet.hybrid.report import write_json
from validate_hybrid_oct09 import source_hashes, audit_trial

DEFAULT = ROOT/'simplified_rapidchiplet/results/hybrid_gui_cost16/c16100000001'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=int,choices=(4,8,16))
    parser.add_argument('--merge',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--folder',type=Path,default=DEFAULT)
    args = parser.parse_args()
    config = validate_config(json.loads((ROOT/'simplified_rapidchiplet/configs/hybrid_oct09_suite.json').read_text(encoding='utf-8')))
    if args.worker:
        config['dies']=[args.worker]
        run_suite(config,args.folder/f'worker{args.worker}',lambda **items:print(items['message'],flush=True) if 'message' in items else None,resume=args.resume)
        return
    if not args.merge:
        parser.error('choose --worker or --merge')
    rows,models = [],None
    hashes = source_hashes()
    for count in config['dies']:
        suite = json.loads((args.folder/f'worker{count}/suite.json').read_text(encoding='utf-8'))
        assert suite['config']=={**config,'dies':[count]}
        assert suite['cost_profile']=='literature16_v1'
        if models is not None:
            assert models==suite['models']
        models=suite['models']
        rows.extend(suite['runs'])
    assert len(rows)==15
    objective=preference_objective(config['preference'])
    refs={model:{key:1 for key in ('latency_s','energy_j','power_w','area_mm2')} for model in config['models']}
    ranking,recommendation=rank_architectures(rows,objective,refs)
    audit=[]
    for row in rows:
        result=audit_trial(row,hashes)
        assert result['feasible'],row
        network=json.loads((Path(row['folder'])/'network.json').read_text(encoding='utf-8'))
        native_root=Path(network['provenance']['rapid_root'])
        native_hashes={name:hashlib.sha256((native_root/name).read_bytes()).hexdigest()
                       for name in ('rapidchiplet.py','helpers.py','validation.py')}
        assert network['provenance']['native_source_sha256']==native_hashes
        result['native_source_verified']=True
        path=Path(row['folder'])/'hardware.json'
        hardware=HardwareSpec.from_dict(json.loads(path.read_text(encoding='utf-8')))
        assert hardware.cost_profile=='literature16_v1'
        assert hardware.compute_clock_hz==500e6
        assert hardware.unified_buffer_bytes*hardware.compute_count==128*1024**2
        assert hardware.prefetch_slots==2
        assert row['metrics']['hardware_cost_provenance']==json.loads(json.dumps(hardware.provenance))
        result.update(hardware_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cost_profile=hardware.cost_profile,
                      total_ub_bytes=hardware.unified_buffer_bytes*hardware.compute_count,
                      certified_physical_limits=row['metrics']['physical_limit_certified'])
        audit.append(result)
    result=dict(config=config,models=models,runs=rows,ranking=ranking,**recommendation,cost_profile='literature16_v1',
                estimator_source_sha256=hashes,interpretation='500MHz MAC reference; uncalibrated bank/PHY/other assumptions; fixed common E/A/D units; no global-optimality claim')
    write_json(args.folder/'suite.json',result)
    write_json(args.folder/'validation.json',dict(passed=True,trials=audit,feasible_count=len(audit)))
    write_json(args.folder/'recommendation.json',recommendation)
    write_json(args.folder/'job.json',dict(id=args.folder.name,state='complete',message='16 nm 成本基準／五模型三架構驗證完成',
                                         completed=len(rows),total=len(rows),rows=rows,result=result,config=config))
    print(json.dumps(dict(best=result['best_architecture'],trials=len(rows),passed=True),ensure_ascii=False))


if __name__=='__main__':
    main()
