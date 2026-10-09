import math
from dataclasses import replace
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from simple_rapidchiplet.hybrid.evaluator import HybridEvaluator, _transfer_service
from simple_rapidchiplet.hybrid.hardware import HardwareSpec
from simple_rapidchiplet.hybrid.mapping import initial_mapping, TensorRegion
from simple_rapidchiplet.hybrid.communication import Transfer
from simple_rapidchiplet.hybrid.scheduler import Job, schedule_jobs, peak_interval_bytes
from simple_rapidchiplet.hybrid.workload import make_toy_workload
from simple_rapidchiplet.hybrid.reuse import dram_residency


class SchedulerChecks(unittest.TestCase):
    def test_pipeline_fill_drain(self):
        jobs = []
        for mb in range(4):
            jobs.extend((Job(f'a{mb}',(),('stage_a',),2,priority=(mb,0)),
                         Job(f'b{mb}',(f'a{mb}',),('stage_b',),3,priority=(mb,1))))
        for policy in ('priority','earliest'):
            schedule = schedule_jobs(jobs,policy=policy)
            self.assertEqual(schedule.makespan_s,14)  # fill 2+3; three steady 3
            self.assertTrue(schedule.validate())
            self.assertEqual(schedule.resource_busy_s,{'stage_a':8,'stage_b':12})

    def test_inserts_independent_job_in_gap(self):
        schedule = schedule_jobs((Job('late',(),('r',),2,priority=(0,)),
                                  Job('producer',(),('other',),10,priority=(-1,)),
                                  Job('consumer',('producer',),('r',),2,priority=(-1,))))
        self.assertEqual(schedule.event_map['late'].start_s,0)
        self.assertEqual(schedule.event_map['consumer'].start_s,10)

    def test_unknown_and_cycle_rejected(self):
        with self.assertRaises(ValueError):
            schedule_jobs((Job('x',('missing',),(),1),))
        with self.assertRaises(ValueError):
            schedule_jobs((Job('x',('y',),(),1),Job('y',('x',),(),1)))

    def test_residency_release_before_allocate(self):
        self.assertEqual(peak_interval_bytes(((0,2,100),(2,3,150),(1,2,20))),150)


class HybridOracleChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workload = make_toy_workload(batch=4)
        cls.evaluator = HybridEvaluator(cls.workload,trace=False)
        cls.plan = initial_mapping(cls.workload,cls.evaluator.package.mapping_core_ids,
                                   cls.evaluator.package.memory_ids,max_group_layers=3)

    def test_energy_and_work_conserved(self):
        result = self.evaluator.evaluate(self.plan,retain_trace=True)
        self.assertTrue(result.metrics['feasible'])
        self.assertEqual(result.metrics['total_macs'],self.workload.total_macs)
        self.assertAlmostEqual(sum(result.metrics['energy_components'].values()),result.metrics['energy_j'])
        self.assertAlmostEqual(result.metrics['static_energy_j'],result.context.idle_power_w*result.schedule.makespan_s)
        self.assertAlmostEqual(result.metrics['energy_components']['mac_j'],
                               self.workload.total_macs*self.evaluator.hardware.mac_energy_pj*1e-12)
        self.assertGreater(result.metrics['area_mm2'],0)
        self.assertGreater(result.metrics['package_area_mm2'],result.metrics['area_mm2'])
        self.assertTrue(result.schedule.validate())
        self.assertTrue(0 <= result.metrics['compute_utilization'] <= 1)

    def test_same_core_macro_scratch_has_one_live_unit(self):
        result = self.evaluator.evaluate(self.plan,retain_trace=True)
        per_core = {}
        for unit in result.compiled.units:
            per_core.setdefault(unit.chiplet,[]).append((result.schedule.event_map[f'read:{unit.id}'].start_s,
                                                      result.schedule.event_map[unit.id].finish_s))
        for spans in per_core.values():
            spans.sort()
            self.assertTrue(all(a[1] <= b[0]+1e-15 for a,b in zip(spans,spans[1:])))

    def test_persistent_weights_have_one_first_load_and_live_storage(self):
        persistent = replace(self.plan,layer_mappings=tuple(replace(mapping,weight_policy='persistent')
                                                           for mapping in self.plan.layer_mappings))
        result = self.evaluator(persistent)
        reuse = result.metrics['weight_reuse']
        self.assertGreater(reuse['saved_weight_read_bytes'],0)
        self.assertEqual(reuse['cold_weight_read_bytes'],reuse['actual_weight_read_bytes']*4)
        self.assertTrue(result.metrics['feasible'])
        self.assertGreater(max(result.metrics['buffer_peak_bytes'].values()),0)

    def test_capacity_overflow_is_rejected(self):
        spec = replace(self.evaluator.hardware,unified_buffer_bytes=128)
        result = HybridEvaluator(self.workload,spec,trace=False)(self.plan)
        self.assertFalse(result.metrics['feasible'])
        self.assertTrue(result.metrics['violations'])
        result = self.evaluator.evaluate(self.plan,retain_trace=True)
        # Capacity sweep alone: shrinking a physical DRAM die also changes its
        # bump budget and must be rejected by the native adapter if impossible.
        memory = dram_residency(result.compiled,result.communication,result.schedule,
                                replace(self.evaluator.hardware,dram_capacity_bytes=32))
        self.assertTrue(memory['violations'])

    def test_multicast_common_prefix_and_full_local_copies(self):
        transfer = Transfer('shared','memory:0',('compute:0/core:0','compute:1/core:0'),
                            1000,(),(),'dram_input','input',TensorRegion(((0,1),(0,1),(0,1),(0,1000))),0)
        _,_,stats = _transfer_service(transfer,self.evaluator.hardware,self.evaluator.context)
        self.assertEqual(stats['nop_hop_bytes'],3000) # mem->IO once, two compute links
        self.assertEqual(stats['dram_bytes'],1000)
        self.assertEqual(stats['local_delivered_bytes'],2000)

    def test_compact_search_cache_can_regenerate_trace(self):
        evaluator = HybridEvaluator(self.workload,trace=False)
        self.assertIsNone(evaluator(self.plan).schedule)
        compact_metrics = dict(evaluator(self.plan).metrics)
        detailed = evaluator.evaluate(self.plan,retain_trace=True)
        self.assertEqual(compact_metrics,detailed.metrics)
        self.assertIsNotNone(detailed.schedule)
        self.assertTrue(detailed.to_dict()['events'])

    def test_precision_metadata_cannot_silently_change_int8_cost(self):
        wrong = replace(self.workload,tensors=tuple(replace(tensor,bits=16) for tensor in self.workload.tensors))
        with self.assertRaisesRegex(ValueError,'precision'):
            HybridEvaluator(wrong)


if __name__ == '__main__':
    unittest.main()
