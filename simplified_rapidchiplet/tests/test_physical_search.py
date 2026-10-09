import math
import unittest
from dataclasses import replace
from pathlib import Path

from simple_rapidchiplet.assignment_generator import legal_assignments
from simple_rapidchiplet.config import SearchConfig, load_config
from simple_rapidchiplet.physical_dse import PhysicalOracle, exhaustive_physical, physical_profile, q_learning_physical, score
from simple_rapidchiplet.physical_evaluator import evaluate_physical
from simple_rapidchiplet.toy_model import make_toy_graph

ROOT = Path(__file__).resolve().parents[1]


class PhysicalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = replace(load_config(ROOT/"configs/defaults.json"),max_chiplets=4)
        cls.graph = make_toy_graph()

    def test_native_same_flow_volume_and_bandwidth(self):
        result = evaluate_physical(self.graph,2,((0,1),(0,1)),self.cfg)
        self.assertEqual(result.metrics["backend"],"official_rapidchiplet")
        self.assertEqual(result.metrics["total_traffic_bytes"],64)
        event = result.communication_events[0]
        self.assertEqual(event["traffic_bytes"],64)
        self.assertEqual(event["bottleneck_link_bits"],32*8)
        self.assertEqual(event["bottleneck_bandwidth_bits_per_cycle"],1929)
        self.assertAlmostEqual(event["serialization_s"],32*8/1929/self.cfg.chiplet.frequency_hz)

    def test_single_chip_has_no_mesh_or_fake_external_flows(self):
        result = evaluate_physical(self.graph,1,((0,),(0,)),self.cfg)
        self.assertEqual(result.metrics["total_traffic_bytes"],0)
        self.assertEqual(result.metrics["total_latency_ns"],result.metrics["compute_latency_ns"])
        self.assertEqual(result.metrics["external_input_bytes"],64)
        self.assertEqual(result.metrics["weight_bytes"],128)

    def test_serial_timeline_disjoint_latency_components(self):
        result = evaluate_physical(self.graph,4,((0,1),(2,3)),self.cfg)
        m = result.metrics
        self.assertAlmostEqual(m["total_latency_ns"],m["compute_latency_ns"]+m["network_latency_ns"]+m["transition_latency_ns"])
        self.assertAlmostEqual(m["achieved_fps"],1e9/m["total_latency_ns"])
        self.assertGreater(result.operator_timings[1]["compute_start_ns"],result.operator_timings[0]["finish_ns"])
        self.assertEqual(result.operator_timings[-1]["finish_ns"],m["total_latency_ns"])
        used = [index for e in result.communication_events for index in e["flow_indices"]]
        self.assertEqual(sorted(used),list(range(len(result.derived.workload.flows))))

    def test_assignment_changes_hops_with_identical_injected_volume(self):
        near = evaluate_physical(self.graph,4,((0,),(1,)),self.cfg)
        far = evaluate_physical(self.graph,4,((0,),(3,)),self.cfg)
        self.assertEqual(near.metrics["total_traffic_bytes"],far.metrics["total_traffic_bytes"])
        self.assertEqual(far.metrics["hop_weighted_traffic_bytes"],2*near.metrics["hop_weighted_traffic_bytes"])
        self.assertGreater(far.metrics["total_latency_ns"],near.metrics["total_latency_ns"])
        self.assertEqual(sum(row["load_bytes"] for row in far.metrics["per_link_statistics"]),far.metrics["hop_weighted_traffic_bytes"])

    def test_idle_chiplets_still_count_in_area_power(self):
        small = evaluate_physical(self.graph,1,((0,),(0,)),self.cfg)
        large = evaluate_physical(self.graph,4,((0,),(0,)),self.cfg)
        self.assertEqual(small.metrics["total_latency_ns"],large.metrics["total_latency_ns"])
        self.assertGreater(large.metrics["total_power_w"],small.metrics["total_power_w"])
        self.assertGreater(large.metrics["total_area_mm2"],small.metrics["total_area_mm2"])

    def test_multiple_producers_share_one_consumer_kind_event(self):
        from test_physical_model import branch_graph
        graph = branch_graph(join_blocks=True)
        result = evaluate_physical(graph,4,((0,), (1,), (2,), (3,)),self.cfg)
        events = [event for event in result.communication_events if event["event_id"] == "add"]
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]["source_group_indices"],[1,2])
        self.assertEqual(events[0]["kind"],"transition")


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = replace(load_config(ROOT/"configs/defaults.json"),max_chiplets=2)
        cls.graph = make_toy_graph()
        cls.profile = physical_profile(cls.graph,cls.cfg)

    def test_assignments_bounded_and_order_sensitive(self):
        actions = legal_assignments(self.graph,0,4,(),(),(),2)
        self.assertLessEqual(len(actions),4*2)
        self.assertTrue(all(len(set(a.chiplets)) == len(a.chiplets) for a in actions))
        full = legal_assignments(self.graph,0,2,(),(),(),4)
        self.assertIn((0,1),[a.chiplets for a in full])
        self.assertIn((1,0),[a.chiplets for a in full])

    def test_cache_state_identity_includes_assignment_and_preference(self):
        oracle = PhysicalOracle(self.graph,self.cfg,self.profile)
        a = oracle.state(2,((0,1),))
        b = oracle.state(2,((1,0),))
        self.assertNotEqual(a,b)
        self.assertNotEqual(a.primary_ownership,b.primary_ownership)
        self.assertGreater(a.cumulative_latency_ns,0)
        changed = PhysicalOracle(self.graph,self.cfg,replace(self.profile, max_power_w=30))
        self.assertNotEqual(oracle.key(2,((0,),(0,))),changed.key(2,((0,),(0,))))
        oracle.evaluate(2,((0,),(0,)))
        oracle.evaluate(2,((0,),(0,)))
        self.assertEqual(len(oracle.cache),1)
        self.assertEqual(oracle.cache_hits,1)

    def test_rl_exhaustive_share_legal_space_and_reproducibility(self):
        exact = exhaustive_physical(PhysicalOracle(self.graph,self.cfg,self.profile))
        self.assertTrue(exact.complete)
        self.assertEqual(exact.unique_evaluations,17)
        first = q_learning_physical(PhysicalOracle(self.graph,self.cfg,self.profile),budget=17,seed=7,max_episodes=600)
        second = q_learning_physical(PhysicalOracle(self.graph,self.cfg,self.profile),budget=17,seed=7,max_episodes=600)
        self.assertEqual(first.history,second.history)
        self.assertGreater(first.q_entries,0)
        exact_designs = {(e.metrics["N_total"],tuple(map(tuple,e.metrics["assignments"]))) for e in exact.candidates}
        self.assertTrue(all((e.metrics["N_total"],tuple(map(tuple,e.metrics["assignments"]))) in exact_designs for e in first.candidates))
        self.assertAlmostEqual(score(first.best,self.profile)["weighted_cost"],score(exact.best,self.profile)["weighted_cost"])

    def test_constraints_prioritize_feasibility_and_custom_weights(self):
        result = evaluate_physical(self.graph,2,((0,1),(0,1)),self.cfg)
        feasible = score(result,self.profile)
        infeasible = score(result,replace(self.profile,max_power_w=1))
        self.assertTrue(feasible["feasible"])
        self.assertGreater(feasible["reward"],0)
        self.assertLess(infeasible["reward"],0)
        profile = physical_profile(self.graph,self.cfg,weights=(2,0,0))
        self.assertEqual(profile.weights,(1,0,0))
        self.assertEqual(profile.name,"custom")
        with self.assertRaises(ValueError):
            physical_profile(self.graph,self.cfg,"custom")

    def test_exhaustive_limit_never_claims_incomplete_optimum(self):
        with self.assertRaises(RuntimeError):
            exhaustive_physical(PhysicalOracle(self.graph,self.cfg,self.profile),max_evaluations=1)
        with self.assertRaises(ValueError):
            exhaustive_physical(PhysicalOracle(make_toy_graph(4),self.cfg,self.profile))


if __name__ == "__main__":
    unittest.main()
