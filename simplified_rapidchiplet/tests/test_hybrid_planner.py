"""Conserved architecture resources and DAG-aware planning invariants."""
from dataclasses import replace
import json
import unittest

from simple_rapidchiplet.hybrid.architecture import enumerate_architectures, resource_totals
from simple_rapidchiplet.hybrid.hardware import HardwareSpec, build_package
from simple_rapidchiplet.hybrid.mapping import compile_mapping, validate_plan
from simple_rapidchiplet.hybrid.planner import (PlanningCandidate, balanced_mapping,
    boundary_tensors, choose_plan, propose_plans)
from simple_rapidchiplet.hybrid.workload import make_toy_workload


class HybridPlannerTests(unittest.TestCase):
    def setUp(self):
        self.hardware = HardwareSpec(cluster_rows=1, cluster_cols=1, compute_per_cluster=2)
        self.package = build_package(self.hardware)
        self.workload = make_toy_workload(batch=4)

    def test_iso_resource_architectures_preserve_compute_storage_and_dram(self):
        candidates = enumerate_architectures()
        self.assertEqual(len(candidates), 6)
        base = resource_totals(HardwareSpec())
        for candidate in candidates:
            totals = resource_totals(candidate.spec)
            for resource in candidate.conserved_resources:
                self.assertAlmostEqual(totals[resource], base[resource])
            self.assertEqual(totals["pes"], 256)
            self.assertEqual(totals["mapping_cores"], 64)
            self.assertEqual(totals["unified_bytes"], 32 * 1024**2)
            self.assertEqual(totals["dram_bandwidth_Bps"], 75e9)
            self.assertEqual(len(candidate.package.mapping_core_ids), 64)
        areas = {candidate.spec.compute_area_mm2 for candidate in candidates}
        self.assertGreater(len(areas), 1)  # Hardware partitioning pays real cost.

    def test_incomplete_clusters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            enumerate_architectures(compute_counts=(6,))

    def test_proposals_do_not_invent_mesh_weight_replication(self):
        ring = balanced_mapping(self.workload, self.package.mapping_core_ids, self.package.memory_ids,
                                hardware=self.hardware)
        mesh = balanced_mapping(self.workload, self.package.mapping_core_ids, self.package.memory_ids,
                                hardware=replace(self.hardware, noc_topology="mesh"))
        self.assertEqual(ring.layer_mappings, mesh.layer_mappings)

    def test_actual_residual_frontier_is_deduplicated(self):
        inputs, outputs = boundary_tensors(self.workload, ("conv0", "conv1", "add"))
        self.assertEqual(inputs, ("image",))
        self.assertEqual(outputs, ("add",))
        inputs, outputs = boundary_tensors(self.workload, ("conv1", "add"))
        self.assertEqual(set(inputs), {"conv0", "image"})
        self.assertEqual(outputs, ("add",))
        inputs, outputs = boundary_tensors(self.workload, ("conv0",))
        self.assertEqual(inputs, ("image",))
        self.assertEqual(outputs, ("conv0",))

    def test_balanced_allocator_avoids_equal_vector_stage_allocation(self):
        plan = balanced_mapping(self.workload, self.package.mapping_core_ids, self.package.memory_ids,
                                hardware=self.hardware, max_group_layers=3, weight_policy="persistent")
        counts = {mapping.layer_id: len(mapping.chiplets) for mapping in plan.layer_mappings}
        self.assertGreater(counts["conv0"], counts["add"])
        self.assertEqual(plan.mapping_map["add"].weight_policy, "persistent")
        validate_plan(self.workload, plan, self.package.mapping_core_ids)
        compiled = compile_mapping(self.workload, plan, self.hardware)
        self.assertEqual(sum(unit.macs for unit in compiled.units), self.workload.total_macs)

    def test_beam_proposals_are_bounded_and_preserve_dag_and_microbatch(self):
        candidates = propose_plans(self.workload, self.package.mapping_core_ids, self.package.memory_ids,
                                  hardware=self.hardware, microbatch_sizes=(1, 2, 3, 4), beam_width=2,
                                  max_candidates=5, weight_policy="persistent")
        self.assertLessEqual(len(candidates), 5)
        self.assertGreater(len(candidates), 1)
        self.assertEqual({candidate.plan.microbatch_size for candidate in candidates}, {1, 2, 4})
        for candidate in candidates:
            validate_plan(self.workload, candidate.plan, self.package.mapping_core_ids)
            self.assertEqual(self.workload.batch_size % candidate.plan.microbatch_size, 0)
            self.assertGreater(candidate.boundary_bytes, 0)
            self.assertGreater(candidate.estimated_delay_s, 0)
            self.assertTrue(all(mapping.weight_policy == "persistent" for mapping in candidate.plan.layer_mappings))

    def test_final_ranking_uses_full_evaluator_and_json_has_no_infinity(self):
        first = balanced_mapping(self.workload, self.package.mapping_core_ids, self.package.memory_ids,
                                 hardware=self.hardware, max_group_layers=1)
        second = balanced_mapping(self.workload, self.package.mapping_core_ids, self.package.memory_ids,
                                  hardware=self.hardware, max_group_layers=3)
        invalid = replace(second, microbatch_size=2)
        candidates = (PlanningCandidate(first, 1, 1, 1), PlanningCandidate(second, 10, 10, 10),
                      PlanningCandidate(invalid, 0.1, 0.1, 1))
        def evaluator(plan):
            if plan.microbatch_size == 2:
                return {"latency_s": 0.01, "energy_j": 0.01, "feasible": False, "violations": ["capacity"]}
            return {"latency_s": 20 if len(plan.groups) == 3 else 2, "energy_j": 1, "feasible": True}
        chosen, summary = choose_plan(candidates, evaluator)
        self.assertEqual(chosen, second)
        self.assertFalse(summary["globally_exact"])
        self.assertEqual(summary["feasible_count"], 2)
        self.assertIsNone(summary["candidates"][2]["cost"])
        json.dumps(summary, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
