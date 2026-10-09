import json
import math
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET
import threading
import unittest
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simple_rapidchiplet.hybrid.models import MODEL_NAMES, extract_model
from simple_rapidchiplet.hybrid.hardware import HardwareSpec
from simple_rapidchiplet.hybrid.architecture import enumerate_architectures, resource_totals
from simple_rapidchiplet.hybrid.evaluator import HybridEvaluator
from simple_rapidchiplet.hybrid.mapping import compile_mapping, required_input_regions, TensorRegion, initial_mapping
from simple_rapidchiplet.hybrid.ondie import core_service
from simple_rapidchiplet.hybrid.ondie import _BankCache
import random
import subprocess
from simple_rapidchiplet.hybrid.planner import balanced_mapping
from simple_rapidchiplet.hybrid.preferences import preference_objective, suite_cost
from simple_rapidchiplet.hybrid.search import plan_from_dict
from simple_rapidchiplet.hybrid.traffic_planner import traffic_plans
from simple_rapidchiplet.hybrid.workload import make_toy_workload
from simple_rapidchiplet.hybrid.report import save_evaluation
from hybrid_gui import App, handler_for, validate_config, reusable_trial
import hybrid_dse


class ModelFrontendTests(unittest.TestCase):
    def test_fingerprint_stable_across_processes(self):
        code = "import json; from simple_rapidchiplet.hybrid.models import extract_model; print(json.dumps({n:extract_model(n).fingerprint for n in ('resnet18','mobilenet_v2')}))"
        output = subprocess.run([sys.executable,'-X','utf8','-c',code],cwd=Path(__file__).resolve().parents[1],
                                check=True,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(json.loads(output.stdout),{name:extract_model(name).fingerprint for name in ('resnet18','mobilenet_v2')})

    def test_real_model_families(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                workload = extract_model(name)
                workload.validate()
                self.assertEqual(workload.tensor_map[workload.input_ids[0]].shape,(1,3,224,224))
                self.assertEqual(workload.tensor_map[workload.output_ids[0]].shape,(1,1000,1,1))
                self.assertGreater(workload.total_macs,100_000_000)
                self.assertIn('weights=None',workload.source)
        resnet = extract_model('resnet18')
        self.assertEqual(resnet.layers[0].macs_per_sample,64*112*112*3*7*7)
        self.assertEqual(sum(layer.kind=='Add' for layer in resnet.layers),8)
        self.assertEqual(sum(layer.kind=='Conv2d' for layer in resnet.layers),20)
        self.assertEqual(sum(layer.kind=='Conv2d' for layer in extract_model('vgg16').layers),13)
        self.assertEqual(sum(layer.kind=='Linear' for layer in extract_model('vgg16').layers),3)
        self.assertTrue(any(layer.groups>1 for layer in extract_model('mobilenet_v2').layers))

    def test_batch_and_pool_attributes(self):
        one,two = extract_model('alexnet'),extract_model('alexnet',2)
        self.assertEqual(two.total_macs,2*one.total_macs)
        self.assertNotEqual(one.fingerprint,two.fingerprint)
        pool = next(layer for layer in one.layers if layer.kind=='MaxPool2d')
        self.assertEqual((pool.kernel,pool.stride,pool.padding),((3,3),(2,2),(0,0)))

    def test_spatial_flatten_exact_union(self):
        workload = extract_model('vgg16')
        layer = next(layer for layer in workload.layers if layer.kind=='Flatten')
        for start,end in ((0,25088),(13,24501),(42,112)):
            region = TensorRegion(((0,1),(start,end),(0,1),(0,1)))
            requirements = required_input_regions(workload,layer,region)
            self.assertEqual(sum(req.region.elements for req in requirements),end-start)
            self.assertTrue(all(a.region.intersection(b.region) is None for i,a in enumerate(requirements) for b in requirements[i+1:]))
        self.assertEqual(len(required_input_regions(workload,layer,TensorRegion(((0,1),(0,25088),(0,1),(0,1))))),1)


class PhysicalExtensionTests(unittest.TestCase):
    def test_indexed_bank_cache_matches_naive_lru(self):
        rng = random.Random(73)
        cache = _BankCache(128)
        entries, used, peak = [], 0, 0
        for step in range(800):
            start = rng.randrange(20)
            length = rng.randrange(1,12)
            region = TensorRegion(((0,1),(start,start+length),(0,rng.randrange(1,5)),(0,1)))
            tensor = 'a' if step%3 else 'w'
            bits = 8 if step%5 else 24
            amount = region.elements*bits//8
            if amount>128:
                with self.assertRaises(ValueError):
                    cache.ensure(tensor,region,bits)
                self.assertEqual((cache.bytes,cache.peak),(used,peak))
                continue
            hit = next((i for i,(name,old,precision,size) in enumerate(entries)
                        if name==tensor and bits==precision and old.intersection(region)==region),None)
            if hit is not None:
                entries.append(entries.pop(hit)); expected=False
            else:
                survivors=[]
                for entry in entries:
                    name,old,precision,size=entry
                    if name==tensor and precision==bits and old.intersection(region)==old:
                        used-=size
                    else:
                        survivors.append(entry)
                entries=survivors
                while used+amount>128 and entries:
                    used-=entries.pop(0)[3]
                entries.append((tensor,region,bits,amount));used+=amount;peak=max(peak,used);expected=True
            self.assertEqual(cache.ensure(tensor,region,bits),expected)
            self.assertEqual((cache.bytes,cache.peak),(used,peak))
            self.assertEqual(list(cache.entries.values()),entries)

    def test_paid_prefetch_and_bank_lifetime(self):
        workload = make_toy_workload(batch=4,channels=32,spatial=16)
        hw = HardwareSpec(cluster_rows=1,cluster_cols=1,compute_per_cluster=1,cores_per_die=1,
                          pe_count=4,ring_count=1,compute_clock_hz=1e8)
        baseline = HybridEvaluator(workload,hw)
        plan = balanced_mapping(workload,baseline.package.mapping_core_ids,baseline.package.memory_ids,hardware=hw,max_group_layers=1)
        serial = baseline.evaluate(plan,retain_trace=True)
        doubled = replace(hw,prefetch_slots=2)
        evaluator = HybridEvaluator(workload,doubled)
        overlapped = evaluator.evaluate(plan,retain_trace=True)
        self.assertTrue(overlapped.schedule.validate())
        self.assertTrue(overlapped.metrics['feasible'])
        self.assertGreater(overlapped.metrics['area_mm2'],serial.metrics['area_mm2'])
        self.assertGreater(overlapped.metrics['idle_power_w'],serial.metrics['idle_power_w'])
        # A second slot pays the same complete bank arrays and peripherals;
        # UB is not replicated by prefetch_slots.
        expected = hw.compute_count*(hw.compute_area_breakdown['l1_sram']+hw.compute_area_breakdown['l2_sram'])
        self.assertAlmostEqual(overlapped.metrics['area_mm2']-serial.metrics['area_mm2'],expected)
        per_core = {}
        for unit in overlapped.compiled.units:
            per_core.setdefault(unit.chiplet,[]).append(unit)
        overlap_found = False
        events = overlapped.schedule.event_map
        group_index = {g.id:i for i,g in enumerate(plan.groups)}
        layer_index = {l.id:i for i,l in enumerate(workload.layers)}
        for units in per_core.values():
            units.sort(key=lambda u:(group_index[u.group_id],u.microbatch,layer_index[u.layer_id],u.shard_index,u.tile_index))
            for index,unit in enumerate(units):
                read = events['read:'+unit.id]
                if index>=2:
                    self.assertGreaterEqual(read.start_s+1e-15,events[units[index-2].id].finish_s)
                if index:
                    overlap_found |= read.start_s < events[units[index-1].id].finish_s-1e-15
        self.assertTrue(overlap_found)
        reference = HardwareSpec(prefetch_slots=2)
        for variant in enumerate_architectures(reference):
            self.assertEqual(resource_totals(variant.hardware)['allocated_scratch_bytes'],resource_totals(reference)['allocated_scratch_bytes'])

    def test_dataflow_actual_accesses_and_roundtrip(self):
        workload = make_toy_workload(batch=1,channels=64,spatial=16)
        hw = HardwareSpec(cores_per_die=1,l1_activation_bytes=1024,unified_buffer_bytes=8*1024**2)
        evaluator = HybridEvaluator(workload,hw)
        plan = initial_mapping(workload,('compute:0/core:0','compute:1/core:0','compute:2/core:0'),evaluator.package.memory_ids)
        results = []
        for flow in ('output_stationary','activation_reuse','weight_reuse'):
            candidate = replace(plan,layer_mappings=tuple(replace(m,dataflow=flow,microtile_shape=(1,16,2,2),channel_tile=64) for m in plan.layer_mappings))
            restored = plan_from_dict(json.loads(json.dumps(candidate.to_dict())))
            self.assertEqual(candidate,restored)
            compiled = compile_mapping(workload,candidate)
            unit = next(u for u in compiled.units if u.layer_id=='conv0')
            service = core_service(unit,workload.layer_map[unit.layer_id],workload,hw)
            self.assertEqual(service.traffic_evidence['partial_sum_spill_bytes'],0)
            self.assertEqual(service.traffic_evidence['dataflow'],flow)
            self.assertEqual(sum(u.macs for u in compiled.units),workload.total_macs)
            results.append(service.noc_injected_bytes)
        # Dataflow traversal changes cache misses, with identical mathematical work.
        self.assertGreater(len(set(results)),1)

    def test_frontier_preserves_nonadjacent_residual(self):
        workload = make_toy_workload(batch=2)
        evaluator = HybridEvaluator(workload)
        base = balanced_mapping(workload,evaluator.package.mapping_core_ids,evaluator.package.memory_ids,max_group_layers=1)
        plans,evidence = traffic_plans(workload,base,evaluator,states_per_group=4,beam_width=4)
        self.assertTrue(evidence['residual_edges_preserved'])
        self.assertGreater(evidence['transitions'],0)
        for plan in plans:
            result = evaluator.evaluate(plan,retain_trace=True)
            self.assertTrue(result.metrics['feasible'])
            add = next(u for u in result.compiled.units if u.layer_id=='add')
            self.assertEqual(set(req.tensor_id for req in add.inputs),{'image','conv1'})

    def test_result_diagrams_use_actual_counts(self):
        for count in (4,8,16):
            spec = next(c.hardware for c in enumerate_architectures() if c.id==f'dies{count}_multi_ring')
            workload = make_toy_workload(batch=1)
            evaluator = HybridEvaluator(workload,spec)
            plan = balanced_mapping(workload,evaluator.package.mapping_core_ids,evaluator.package.memory_ids,hardware=spec)
            result = evaluator.evaluate(plan,retain_trace=True)
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                save_evaluation(folder,result,evaluator,plan)
                data = json.loads((folder/'layout.json').read_text(encoding='utf-8'))
                self.assertEqual(sum(n['role']=='compute' for n in data['nodes']),count)
                self.assertEqual((folder/'die.svg').read_text().count(' / core '),spec.pe_count)
                self.assertTrue((folder/'architecture.svg').is_file())
                ET.parse(folder/'architecture.svg')


class PreferenceAndApiTests(unittest.TestCase):
    def test_resume_rejects_changed_settings(self):
        args = hybrid_dse.parser().parse_args(['--budget','2','--preference','balanced'])
        workload = make_toy_workload(batch=args.batch)
        spec = HardwareSpec()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            hybrid_dse.run_one(args,workload,spec,folder,method='rl')
            self.assertIsNotNone(reusable_trial(folder,workload,spec,args))
            args.seed += 1
            self.assertIsNone(reusable_trial(folder,workload,spec,args))
            args.seed -= 1
            self.assertIsNone(reusable_trial(folder,workload,replace(spec,unified_buffer_bytes=4*1024**2),args))
            row = json.loads((folder/'run.json').read_text())
            row['metrics']['oracle_source_sha256']['hardware.py'] = 'changed'
            (folder/'run.json').write_text(json.dumps(row))
            self.assertIsNone(reusable_trial(folder,workload,spec,args))

    def test_shared_reference_and_preference_tradeoff(self):
        rows = [{'model':'a','metrics':{'feasible':True,'power_w':1,'energy_j':.002,'area_mm2':200,'latency_s':.002}},
                {'model':'b','metrics':{'feasible':True,'power_w':4,'energy_j':.032,'area_mm2':200,'latency_s':.008}}]
        refs = {name:{'power_w':1,'energy_j':1,'area_mm2':1,'latency_s':1} for name in ('a','b')}
        objective = preference_objective('balanced')
        self.assertAlmostEqual(suite_cost(rows,objective,refs),(.008*200*.004)**(1/3))
        self.assertEqual(suite_cost(rows,preference_objective('balanced',max_power=3),refs),math.inf)
        low_power = {'feasible':True,'power_w':1,'energy_j':.010,'area_mm2':200,'latency_s':.010}
        low_delay = {'feasible':True,'power_w':8,'energy_j':.016,'area_mm2':220,'latency_s':.002}
        self.assertLess(preference_objective('low_power').cost(low_power,refs['a']),preference_objective('low_power').cost(low_delay,refs['a']))
        self.assertGreater(preference_objective('low_latency').cost(low_power,refs['a']),preference_objective('low_latency').cost(low_delay,refs['a']))

    def test_input_validation(self):
        for data in ({'models':[]},{'models':['unknown']},{'budget':'8'},{'max_area':float('nan')},{'shell':'oops'},{'batch':True}):
            with self.subTest(data=data), self.assertRaises((ValueError,TypeError)):
                validate_config(data)

    def test_http_guard_and_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            app = App(directory)
            server = ThreadingHTTPServer(('127.0.0.1',0),handler_for(app))
            thread = threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            root = f'http://127.0.0.1:{server.server_port}'
            try:
                data = json.load(urlopen(root+'/api/catalog'))
                self.assertEqual(len(data['preferences']),4)
                self.assertIn('vgg16',data['models'])
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(root+'/api/jobs',data=b'{}',method='POST'))
                self.assertEqual(error.exception.code,403)
                with self.assertRaises(HTTPError) as error:
                    urlopen(root+'/artifacts/%2e%2e/README.md')
                self.assertEqual(error.exception.code,404)
            finally:
                server.shutdown();server.server_close();app.worker.shutdown()


if __name__=='__main__':
    unittest.main()
