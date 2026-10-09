"""Local GUI and multi-workload architecture DSE. Run --help for startup.

Stdlib server, one serialized worker, allowlisted configuration, no shell-based
commands or external submission. Results are persisted and independently usable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import math
import mimetypes
from pathlib import Path
import re
import threading
import time
from urllib.parse import unquote, urlsplit
import uuid

import hybrid_dse
from simple_rapidchiplet.hybrid.architecture import enumerate_architectures
from simple_rapidchiplet.hybrid.hardware import HardwareSpec
from simple_rapidchiplet.hybrid.models import MODEL_NAMES, extract_model
from simple_rapidchiplet.hybrid.preferences import PREFERENCES, preference_objective, suite_cost, constraint_report, rank_architectures
from simple_rapidchiplet.hybrid.report import clean_json, write_json
from simple_rapidchiplet.hybrid.workload import make_toy_workload

ROOT = Path(__file__).resolve().parent
STATIC = ROOT/'gui'


def validate_config(data):
    if not isinstance(data,dict):
        raise ValueError('設定必須是 JSON object')
    allowed = {'models','preference','batch','budget','seed','dies','noc','ub_mib','prefetch_slots',
               'dataflow','planner','groups','method','max_power','max_area','max_latency_ms',
               'strict_power','strict_area','strict_latency'}
    if set(data)-allowed:
        raise ValueError('未知設定欄位')
    config = {'models':['resnet18','vgg16'],'preference':'balanced','batch':1,'budget':8,'seed':7,
              'dies':[4,8,16],'noc':['multi_ring'],'ub_mib':8,'prefetch_slots':2,
              'dataflow':'auto','planner':'traffic_dp','groups':3,'method':'rl',
              'max_power':None,'max_area':None,'max_latency_ms':None,
              'strict_power':True,'strict_area':True,'strict_latency':True,**data}
    for key,options in (('models',('toy',*MODEL_NAMES)),('dies',(4,8,16)),('noc',('multi_ring','mesh'))):
        value = config[key]
        if not isinstance(value,list) or not value or len(value)!=len(set(value)) or any(item not in options for item in value):
            raise ValueError(f'無效選項：{key}')
    for key,low,high in (('batch',1,8),('budget',1,512),('seed',0,2**31-1),('groups',1,8),('ub_mib',2,32),('prefetch_slots',1,2)):
        value = config[key]
        if isinstance(value,bool) or not isinstance(value,int) or not low <= value <= high:
            raise ValueError(f'{key} 必須介於 {low} 與 {high}')
    if config['preference'] not in PREFERENCES or config['dataflow'] not in (*hybrid_dse.DATAFLOWS,'auto'):
        raise ValueError('無效偏好或 dataflow')
    if config['planner'] not in ('balanced','traffic_dp','beam') or config['method'] not in ('rl','greedy','sa','random'):
        raise ValueError('無效搜尋方法')
    for key in ('max_power','max_area','max_latency_ms'):
        value = config[key]
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0):
            raise ValueError('PPA 上限必須是有限正數')
    for key in ('strict_power','strict_area','strict_latency'):
        if not isinstance(config[key],bool):
            raise ValueError('嚴格遵守必須是 boolean')
    return config


def reusable_trial(folder, workload, hardware, args):
    """Reuse completed trials only when physical/source/settings evidence matches."""
    folder = Path(folder)
    try:
        row = json.loads((folder/'run.json').read_text(encoding='utf-8'))
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        saved_hardware = json.loads((folder/'hardware.json').read_text(encoding='utf-8'))
        from dataclasses import asdict
        source = {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (ROOT/'simple_rapidchiplet'/'hybrid').glob('*.py')}
        if row['metrics'].get('oracle_source_sha256')!=source or row['metrics'].get('workload_fingerprint')!=workload.fingerprint:
            return None
        if saved_hardware != asdict(hardware):
            return None
        for key in ('budget','seed','groups','planner','dataflow','prefetch_slots','preference','method',
                    'max_power','max_area','max_latency','soft_power','soft_area','soft_latency','pipeline_window','weights','collective'):
            if summary['arguments'].get(key)!=getattr(args,key):
                return None
        if not all((folder/name).is_file() for name in ('mapping.json','layout.html','architecture.svg','timeline.csv','layer_switching.json')):
            return None
        return row
    except (OSError,ValueError,KeyError):
        return None


def run_suite(config, folder, update=lambda **kwargs:None, *, resume=False):
    config = validate_config(config)
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if resume and (folder/'config.json').is_file() and json.loads((folder/'config.json').read_text(encoding='utf-8'))!=config:
        raise ValueError('saved suite config differs; use a new output folder')
    write_json(folder/'config.json',config)
    objective = preference_objective(config['preference'],max_power=config['max_power'],max_area=config['max_area'],
                    max_latency=None if config['max_latency_ms'] is None else config['max_latency_ms']/1000,
                    strict_power=config['strict_power'],strict_area=config['strict_area'],strict_latency=config['strict_latency'])
    reference = replace(HardwareSpec(),unified_buffer_bytes=config['ub_mib']*1024**2,prefetch_slots=config['prefetch_slots'])
    candidates = enumerate_architectures(reference,compute_counts=tuple(config['dies']),noc_topologies=tuple(config['noc']))
    workloads = {}
    for model in config['models']:
        update(message=f'抽取 {model} 真實模型 DAG')
        workloads[model] = make_toy_workload(batch=config['batch']) if model=='toy' else extract_model(model,config['batch'])
    rows, rankings = [], []
    common = {model:{key:1.0 for key in ('latency_s','energy_j','power_w','area_mm2')} for model in workloads}
    total = len(candidates)*len(workloads)
    for architecture in candidates:
        trials = []
        for model,workload in workloads.items():
            run_folder = folder/architecture.id/model
            update(completed=len(rows),total=total,message=f'{architecture.id} / {model}',current_folder=str(run_folder))
            args = hybrid_dse.parser().parse_args([])
            args.model = model
            args.batch = config['batch']
            args.budget = config['budget']
            args.groups = config['groups']
            args.planner = config['planner']
            args.dataflow = config['dataflow']
            args.prefetch_slots = config['prefetch_slots']
            args.preference = config['preference']
            args.max_power,args.max_area = config['max_power'],config['max_area']
            args.max_latency = None if config['max_latency_ms'] is None else config['max_latency_ms']/1000
            args.soft_power,args.soft_area,args.soft_latency = not config['strict_power'],not config['strict_area'],not config['strict_latency']
            args.seed,args.method,args.dies,args.noc = config['seed'],config['method'],architecture.hardware.compute_count,architecture.hardware.noc_topology
            args.out = run_folder
            started = time.monotonic()
            try:
                row = reusable_trial(run_folder,workload,architecture.hardware,args) if resume else None
                if row is None:
                    row = hybrid_dse.run_one(args,workload,architecture.hardware,run_folder,method=config['method'] if config['budget']>1 else None)
                else:
                    row['reused_completed_trial'] = True
                row.update(model=model,architecture=architecture.id)
                row['constraints'] = constraint_report(row['metrics'],objective)
                row['within_limits'] = row['constraints']['strict_compliant']
                row['all_targets_met'] = row['constraints']['all_targets_met']
                row['error'] = None
            except Exception as exc:
                # One infeasible model/architecture does not hide other trials.
                row = {'model':model,'architecture':architecture.id,'folder':str(run_folder.resolve()),
                       'elapsed_s':time.monotonic()-started,'metrics':{'feasible':False},'within_limits':False,
                       'error':f'{type(exc).__name__}: {exc}'}
                write_json(run_folder/'failure.json',row)
            rows.append(row)
            trials.append(row)
            write_json(folder/'trials.json',rows)
            update(completed=len(rows),total=total,rows=rows)
        cost = suite_cost(trials,objective,common)
        rankings.append({'architecture':architecture.id,'cost':cost if math.isfinite(cost) else None,
                         'feasible_for_all_models':math.isfinite(cost)})
    rankings,recommendation = rank_architectures(rows,objective,common)
    result = {'config':config,'models':{model:{'layers':len(w.layers),'macs':w.total_macs,'fingerprint':w.fingerprint} for model,w in workloads.items()},
              'runs':rows,'ranking':rankings,**recommendation,
              'objective':'gmean of E^wE A^wA D^wD; fixed common units. Checked strict limits apply to every model; unchecked limits are normalized soft-goal penalties.',
              'weights':PREFERENCES[config['preference']],
              'cost_profile':reference.cost_profile,
              'interpretation':'Gemini V-A inspired exponents; closest-candidate policy is our extension. Nominal analytical PPA, not real hardware limit certification.'}
    write_json(folder/'suite.json',result)
    write_json(folder/'recommendation.json',recommendation)
    update(completed=total,total=total,result=result,message='完成；估算符合嚴格限制' if recommendation['best_architecture'] else '完成；提供最接近方案與差距' if recommendation['recommended_architecture'] else '完成；沒有物理可行候選')
    return result


class App:
    def __init__(self,output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=True)
        self.jobs = {}
        self.lock = threading.Lock()
        self.worker = ThreadPoolExecutor(max_workers=1,thread_name_prefix='hybrid-dse')
    def submit(self,data):
        config = validate_config(data)
        with self.lock:
            if any(job['state'] in ('queued','running') for job in self.jobs.values()):
                raise ValueError('已有搜尋正在執行；請等待完成')
            ident = uuid.uuid4().hex[:12]
            self.jobs[ident] = {'id':ident,'state':'queued','message':'等待執行','completed':0,'total':0,'config':config,'rows':[]}
        folder = self.output/ident
        def update(**values):
            with self.lock:
                self.jobs[ident].update(clean_json(values))
                snapshot = dict(self.jobs[ident])
            write_json(folder/'job.json',snapshot)
        def execute():
            update(state='running')
            try:
                run_suite(config,folder,update)
                update(state='complete')
            except Exception as exc:
                update(state='failed',message=f'{type(exc).__name__}: {exc}')
        self.worker.submit(execute)
        return ident


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def json_response(self,value,status=200):
            payload = json.dumps(clean_json(value),ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == '/api/catalog':
                return self.json_response({'models':['toy',*MODEL_NAMES],'preferences':PREFERENCES})
            if path == '/api/jobs':
                with app.lock:
                    return self.json_response(list(app.jobs.values()))
            if path.startswith('/api/jobs/'):
                ident = path.rsplit('/',1)[-1]
                with app.lock:
                    job = dict(app.jobs[ident]) if ident in app.jobs else None
                if job is None and re.fullmatch('[0-9a-f]{12}',ident):
                    saved = app.output/ident/'job.json'
                    if saved.is_file():
                        job = json.loads(saved.read_text(encoding='utf-8'))
                        if job.get('state') in ('queued','running'):
                            job.update(state='interrupted',message='前次server已停止；已保存trial可用CLI resume-suite接續')
                return self.json_response(job if job else {'error':'找不到工作'},200 if job else 404)
            root = app.output if path.startswith('/artifacts/') else STATIC
            relative = path[len('/artifacts/'):] if root==app.output else 'index.html' if path=='/' else path.lstrip('/')
            target = (root/relative).resolve()
            if not target.is_relative_to(root) or not target.is_file() or target.suffix not in ('.html','.js','.css','.json','.svg','.csv','.ndjson'):
                return self.json_response({'error':'找不到檔案'},404)
            data = target.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type',mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_POST(self):
            origin = self.headers.get('Origin')
            if (self.path!='/api/jobs' or self.headers.get('X-Hybrid-Request')!='1'
                    or (origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'))):
                return self.json_response({'error':'不允許的請求'},403)
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 16384:
                    raise ValueError('請求大小不正確')
                data = json.loads(self.rfile.read(length))
                ident = app.submit(data)
            except (ValueError,TypeError) as exc:
                return self.json_response({'error':str(exc)},400)
            self.json_response({'id':ident},202)
        def log_message(self,format,*args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--out',type=Path,default=ROOT/'results'/'hybrid_gui_cost16')
    parser.add_argument('--suite-config',type=Path,help='headless reproducible JSON run; no server')
    parser.add_argument('--resume-suite',action='store_true',help='reuse completed trials only when source/settings match')
    args = parser.parse_args()
    if args.suite_config:
        result = run_suite(json.loads(args.suite_config.read_text(encoding='utf-8')),args.out,
                           lambda **values:print(values.get('message',''),flush=True) if 'message' in values else None,
                           resume=args.resume_suite)
        print('Best architecture:',result['best_architecture'])
        return 0 if result['best_architecture'] else 2
    if not 1024<=args.port<=65535:
        parser.error('port must be 1024..65535')
    app = App(args.out)
    server = ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(app))
    print(f'GUI: http://127.0.0.1:{args.port} (Ctrl+C to stop)',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.worker.shutdown(wait=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
