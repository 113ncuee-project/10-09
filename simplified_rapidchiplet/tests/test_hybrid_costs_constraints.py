import json
import math
from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simple_rapidchiplet.hybrid.hardware import HardwareSpec
from simple_rapidchiplet.hybrid.costs import sram_bank, sram_access_energy, grs_port
from simple_rapidchiplet.hybrid.evaluator import HybridEvaluator
from simple_rapidchiplet.hybrid.planner import balanced_mapping
from simple_rapidchiplet.hybrid.workload import make_toy_workload
from simple_rapidchiplet.hybrid.preferences import preference_objective, constraint_report, rank_architectures
import hybrid_gui


class LiteratureCosts(unittest.TestCase):
    def test_mac_scalar_count_and_frequency(self):
        workload = make_toy_workload(batch=1)
        hw = HardwareSpec(cluster_rows=1,cluster_cols=1,compute_per_cluster=1)
        ev = HybridEvaluator(workload,hw)
        plan = balanced_mapping(workload,ev.package.mapping_core_ids,ev.package.memory_ids,hardware=hw)
        base = ev.evaluate(plan,retain_trace=True)
        self.assertEqual(hw.compute_clock_hz,500e6)
        self.assertEqual(hw.mac_area_mm2_per_unit,135.1e-6)
        self.assertAlmostEqual(base.metrics['energy_components']['mac_j'],workload.total_macs*.024e-12)
        doubled = HybridEvaluator(workload,replace(hw,mac_ops_per_mac=2)).evaluate(plan)
        self.assertAlmostEqual(doubled.metrics['energy_components']['mac_j'],workload.total_macs*.048e-12)
        fast = HybridEvaluator(workload,replace(hw,compute_clock_hz=1e9)).evaluate(plan)
        self.assertEqual(fast.metrics['energy_components']['mac_j'],base.metrics['energy_components']['mac_j'])
        self.assertFalse(fast.metrics['physical_limit_certified'])
        for service in base.core_services.values():
            ledger = service['traffic_evidence']['sram_access_ledger']
            self.assertEqual(sum(row['read_bits']+row['write_bits'] for row in ledger.values()),service['sram_bits'])
            self.assertAlmostEqual(sum(row['energy_j'] for row in ledger.values()),service['sram_energy_j'])

    def test_bank_anchor_and_actual_capacity(self):
        hw = HardwareSpec(l1_bandwidth_Bps=1e6,l1_weight_bytes=1024)
        self.assertAlmostEqual(sram_bank(hw,'WL1')['read_pj_per_bit'],.3)
        big = sram_bank(replace(hw,l1_weight_bytes=32768),'WL1')
        self.assertAlmostEqual(big['read_pj_per_bit'],.81)
        split = sram_bank(replace(hw,l1_weight_bytes=32768,l1_bandwidth_Bps=64e9),'WL1')
        self.assertEqual(split['banks'],8)
        self.assertEqual(split['bytes_per_bank'],4096)
        self.assertLess(split['read_pj_per_bit'],big['read_pj_per_bit'])
        self.assertGreater(split['area_mm2'],big['area_mm2'])

    def test_read_write_width_ports_are_paid(self):
        hw = HardwareSpec(l1_bandwidth_Bps=1e6,sram_write_energy_factor=1.2)
        _,ledger = sram_access_energy(hw,{'AL1':{'read_bits':8,'write_bits':16}})
        bank = ledger['AL1']['cost']
        self.assertAlmostEqual(ledger['AL1']['energy_j'],(8*bank['read_pj_per_bit']+16*bank['write_pj_per_bit'])*1e-12)
        wider = sram_bank(replace(hw,sram_word_bits=256),'AL1')
        ports = sram_bank(replace(hw,sram_rw_ports=2),'AL1')
        self.assertGreater(wider['read_pj_per_bit'],bank['read_pj_per_bit'])
        self.assertGreater(ports['area_mm2'],bank['area_mm2'])
        self.assertGreater(ports['read_pj_per_bit'],bank['read_pj_per_bit'])

    def test_grs_whole_bidirectional_macros_and_included_active_area(self):
        hw = HardwareSpec()
        port = grs_port(hw)
        self.assertEqual(port['endpoint_macros'],2)
        self.assertEqual(port['data_lanes'],16)
        self.assertEqual(port['clock_lanes'],2)
        self.assertEqual(port['power_ground_bumps'],22)
        self.assertEqual(port['total_bumps'],40)
        self.assertAlmostEqual(port['footprint_mm2'],.77518)
        self.assertAlmostEqual(port['active_circuitry_mm2'],.162812)
        self.assertEqual(port['per_direction_capacity_Bps'],25e9)
        self.assertEqual(grs_port(hw,25e9)['endpoint_macros'],2)
        self.assertEqual(grs_port(hw,25e9+1)['endpoint_macros'],4)
        self.assertEqual(hw.compute_area_breakdown['grs_phy'],port['footprint_mm2'])

    def test_old_json_is_not_silently_retagged(self):
        old = asdict(HardwareSpec())
        old.pop('cost_profile')
        old.update(mac_energy_pj=.192,sram_area_mm2_per_byte=.0964*8/1e6,mac_area_mm2_per_unit=57.9e-6)
        hw = HardwareSpec.from_dict(old)
        self.assertEqual(hw.cost_profile,'legacy_mixed_v0')
        self.assertIn('invalidated',hw.provenance['warning'])
        self.assertEqual(hw.sram_costs['AL1']['read_pj_per_bit'],.81)


class Constraints(unittest.TestCase):
    @staticmethod
    def metrics(power=8,area=100,latency=.1,energy=.8,feasible=True):
        return dict(power_w=power,area_mm2=area,latency_s=latency,energy_j=energy,feasible=feasible)

    def test_relaxed_search_prioritizes_compliance_and_rejects_physical_failure(self):
        objective = preference_objective(max_power=10,relax_hard_limits=True)
        reference = self.metrics()
        self.assertLess(objective.cost(self.metrics(energy=1000),reference),objective.cost(self.metrics(power=10.01,energy=.0001),reference))
        self.assertLess(objective.cost(self.metrics(power=11),reference),objective.cost(self.metrics(power=12),reference))
        self.assertEqual(objective.cost(self.metrics(feasible=False),reference),math.inf)
        self.assertEqual(preference_objective(max_power=10).cost(self.metrics(power=11),reference),math.inf)

    def test_soft_goal_is_reported_and_never_hard_rejected(self):
        obj = preference_objective(max_power=4,strict_power=False)
        metrics = self.metrics()
        report = constraint_report(metrics,obj)
        self.assertTrue(report['strict_compliant'])
        self.assertFalse(report['all_targets_met'])
        self.assertEqual(report['limits'][0]['excess_percent'],100)
        self.assertTrue(math.isfinite(obj.cost(metrics,metrics)))

    def test_multimodel_worst_violation_is_not_hidden_by_average(self):
        obj = preference_objective(max_power=10)
        rows = [dict(architecture='A',model='one',metrics=self.metrics(power=2)),
                dict(architecture='A',model='two',metrics=self.metrics(power=12)),
                dict(architecture='B',model='one',metrics=self.metrics(power=9)),
                dict(architecture='B',model='two',metrics=self.metrics(power=9))]
        refs = {model:self.metrics() for model in ('one','two')}
        _,result = rank_architectures(rows,obj,refs)
        self.assertEqual(result['best_architecture'],'B')
        self.assertEqual(result['recommended_architecture'],'B')
        rows[2]['metrics']['power_w']=11
        rows[3]['metrics']['power_w']=11
        _,result = rank_architectures(rows,obj,refs)
        self.assertIsNone(result['best_architecture'])
        self.assertEqual(result['recommended_architecture'],'B')
        self.assertEqual(result['recommendation_status'],'closest_exceeds_strict_limits')

    def test_invalid_or_incomplete_architecture_never_fallback(self):
        obj = preference_objective(max_power=1)
        refs = {model:self.metrics() for model in ('one','two')}
        rows = [dict(architecture='A',model='one',metrics=self.metrics()),
                dict(architecture='B',model='one',metrics=self.metrics(feasible=False)),
                dict(architecture='B',model='two',metrics=self.metrics())]
        _,result = rank_architectures(rows,obj,refs)
        self.assertIsNone(result['recommended_architecture'])
        self.assertFalse(constraint_report(self.metrics(energy=math.nan),obj)['physical_feasible'])

    def test_gui_strict_flags_and_actual_impossible_limit_fallback(self):
        config = hybrid_gui.validate_config(dict(models=['toy'],dies=[4],budget=2,max_area=1))
        self.assertTrue(config['strict_area'])
        with self.assertRaises(ValueError):
            hybrid_gui.validate_config(dict(strict_area='false'))
        with tempfile.TemporaryDirectory() as folder:
            result = hybrid_gui.run_suite(config,folder)
            self.assertIsNone(result['best_architecture'])
            self.assertEqual(result['recommended_architecture'],'dies4_multi_ring')
            self.assertTrue(result['runs'][0]['metrics']['feasible'])
            self.assertGreater(result['constraint_reports'][0]['limits'][0]['excess_percent'],0)
            self.assertTrue((Path(result['runs'][0]['folder'])/'architecture.svg').is_file())
            self.assertTrue((Path(folder)/'recommendation.json').is_file())
            self.assertEqual(result['cost_profile'],'literature16_v1')


if __name__=='__main__':
    unittest.main()
