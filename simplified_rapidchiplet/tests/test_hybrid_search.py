"""Behavioral tests for legal Gemini actions, learned policy and continuation."""
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from simple_rapidchiplet.hybrid.actions import (MappingAction, apply_action, legal_actions,
                                               operator_mask, partition_options)
from simple_rapidchiplet.hybrid.mapping import (LayerGroup, LayerMapping, MappingPlan, Partition,
                                               compile_mapping, initial_mapping, validate_plan)
from simple_rapidchiplet.hybrid.search import (LinearActorCritic, SearchObjective, action_features,
                                              compare_searches, plan_fingerprint, run_search,
                                              state_features)
from simple_rapidchiplet.hybrid.workload import make_toy_workload


CORES = tuple(f"compute:{die}/core:{core}" for die in range(2) for core in range(4))
MEMORIES = ("memory:0", "memory:1")


def toy_oracle(plan):
    """Known imbalance/locality oracle, separate from production estimators."""
    mapped = plan.mapping_map
    service = max(9 / len(mapped["conv0"].chiplets), 1 / len(mapped["conv1"].chiplets))
    remote = sum(core.startswith("compute:1") for core in mapped["conv0"].chiplets)
    # Partition choices also matter: repeated activations are more costly with K.
    energy = 1 + 0.01 * remote + 0.002 * mapped["conv0"].partition.k
    return {"latency_s": service, "energy_j": energy, "area_mm2": 20,
            "power_w": energy / service, "feasible": True}


class GeminiActionsTests(unittest.TestCase):
    def setUp(self):
        self.workload = make_toy_workload(residual=False, batch=4)
        self.plan = initial_mapping(self.workload, CORES, MEMORIES, microbatch_size=2, max_group_layers=2)

    def test_all_five_actions_preserve_real_output_ownership(self):
        actions = legal_actions(self.workload, self.plan, MEMORIES, core_ids=CORES,
                                max_per_operator=6, rng=random.Random(19))
        self.assertEqual(set(action.operator for action in actions), {"OP1", "OP2", "OP3", "OP4", "OP5"})
        expected_macs = self.workload.total_macs
        for action in actions:
            changed = apply_action(self.plan, action)
            validate_plan(self.workload, changed, CORES)
            compiled = compile_mapping(self.workload, changed)
            self.assertEqual(sum(unit.macs for unit in compiled.units), expected_macs)
            for layer in self.workload.layers:
                units = [unit for unit in compiled.units if unit.layer_id == layer.id]
                self.assertEqual(sum(unit.output_region.elements for unit in units),
                                 self.workload.tensor_map[layer.output].elements)
            flattened = [core for item in changed.layer_mappings for core in item.chiplets]
            if action.variant == "gemini":
                self.assertEqual(set(flattened), set(CORES))
            else:
                self.assertLessEqual(set(flattened), set(CORES))
            self.assertEqual(len(flattened), len(set(flattened)))

    def test_partition_bounds_allow_uneven_shards_without_empty_shards(self):
        parts = partition_options(self.workload, "conv0", 3, 2)
        self.assertIn((1, 3, 1, 1), parts)
        self.assertFalse(partition_options(self.workload, "conv0", 257, 1))
        self.assertTrue(all(np.prod(part) == 3 and part[0] <= 2 for part in parts))

    def test_op4_requires_legal_new_cardinality_and_never_empties_layer(self):
        bad = MappingAction("OP4", "conv0", "conv1", 0, 0, (1, 4, 1, 1), (1, 4, 1, 1))
        with self.assertRaisesRegex(ValueError, "donor partition"):
            apply_action(self.plan, bad)
        single = replace(self.plan, layer_mappings=(
            replace(self.plan.mapping_map["conv0"], chiplets=(CORES[0],), partition=Partition()),
            replace(self.plan.mapping_map["conv1"], chiplets=CORES[1:], partition=Partition(1, 7, 1, 1))))
        with self.assertRaisesRegex(ValueError, "empty"):
            apply_action(single, MappingAction("OP4", "conv0", "conv1", 0, 0,
                                               (1, 1, 1, 1), (1, 8, 1, 1)))
        self.assertEqual(len(self.plan.mapping_map["conv0"].chiplets), 4)

    def test_cross_temporal_groups_cannot_swap_allocations(self):
        temporal = replace(self.plan, groups=(LayerGroup("a", ("conv0",)), LayerGroup("b", ("conv1",))))
        actions = legal_actions(self.workload, temporal, MEMORIES, core_ids=CORES)
        self.assertFalse(operator_mask(actions)["OP3"])
        self.assertFalse(any(action.operator == "OP4" and action.variant == "gemini" for action in actions))
        with self.assertRaisesRegex(ValueError, "one spatial group"):
            apply_action(temporal, MappingAction("OP3", "conv0", "conv1", 0, 0))

    def test_idle_pool_acquire_release_preserve_disjoint_exact_output_ownership(self):
        partial = replace(self.plan, layer_mappings=(
            replace(self.plan.mapping_map["conv0"], chiplets=CORES[:3], partition=Partition(1, 3, 1, 1)),
            replace(self.plan.mapping_map["conv1"], chiplets=CORES[4:7], partition=Partition(1, 3, 1, 1))))
        actions = legal_actions(self.workload, partial, MEMORIES, core_ids=CORES,
                                max_per_operator=6, rng=random.Random(23))
        variants = {action.variant for action in actions if action.operator == "OP4"}
        self.assertEqual(variants, {"gemini", "hybrid_idle_acquire", "hybrid_idle_release"})
        for action in (action for action in actions if action.variant != "gemini"):
            changed = apply_action(partial, action, core_ids=CORES)
            validate_plan(self.workload, changed, CORES)
            compiled = compile_mapping(self.workload, changed)
            self.assertEqual(sum(unit.macs for unit in compiled.units), self.workload.total_macs)
            for layer in self.workload.layers:
                self.assertEqual(sum(unit.output_region.elements for unit in compiled.units if unit.layer_id == layer.id),
                                 self.workload.tensor_map[layer.output].elements)
            original_count = len(partial.mapping_map[action.layer_id].chiplets)
            expected_delta = 1 if action.variant == "hybrid_idle_acquire" else -1
            self.assertEqual(len(changed.mapping_map[action.layer_id].chiplets), original_count + expected_delta)
            self.assertEqual(changed.mapping_map[action.layer_id].partition.product, original_count + expected_delta)
            self.assertEqual(MappingAction.from_dict(action.to_dict()), action)

    def test_idle_acquire_rejects_occupied_and_unknown_cores_and_release_cannot_empty(self):
        occupied = MappingAction("OP4", "conv0", partition=(1, 5, 1, 1),
                                 variant="hybrid_idle_acquire", idle_core=CORES[4])
        with self.assertRaisesRegex(ValueError, "unoccupied"):
            apply_action(self.plan, occupied, core_ids=CORES)
        with self.assertRaisesRegex(ValueError, "unknown idle"):
            apply_action(self.plan, replace(occupied, idle_core="compute:999/core:0"), core_ids=CORES)
        single = self.plan.replace_layer(replace(self.plan.mapping_map["conv0"],
                                                chiplets=(CORES[0],), partition=Partition()))
        with self.assertRaisesRegex(ValueError, "empty"):
            apply_action(single, MappingAction("OP4", "conv0", partition=(1, 1, 1, 1),
                                               variant="hybrid_idle_release", idle_core=CORES[0]))

    def test_temporal_layers_can_acquire_core_used_only_by_sequential_layer(self):
        temporal = replace(self.plan, groups=(LayerGroup("a", ("conv0", "conv1"), mode="temporal"),))
        action = MappingAction("OP4", "conv0", partition=(1, 5, 1, 1),
                               variant="hybrid_idle_acquire", idle_core=CORES[4])
        changed = apply_action(temporal, action, core_ids=CORES)
        self.assertTrue(validate_plan(self.workload, changed, CORES))

    def test_nearest_legal_resize_crosses_integer_cardinality_gap(self):
        cores = tuple(f"compute:{die}/core:{core}" for die in range(16) for core in range(4))
        locked = replace(self.plan, layer_mappings=(
            replace(self.plan.mapping_map["conv0"], chiplets=cores[:48], partition=Partition(1, 3, 4, 4)),
            replace(self.plan.mapping_map["conv1"], chiplets=cores[48:56], partition=Partition(1, 8, 1, 1))))
        self.assertFalse(partition_options(self.workload, "conv0", 47, locked.microbatch_size))
        self.assertFalse(partition_options(self.workload, "conv0", 46, locked.microbatch_size))
        actions = legal_actions(self.workload, locked, MEMORIES, core_ids=cores,
                                max_per_operator=6, rng=random.Random(29))
        release = next(action for action in actions if action.variant == "hybrid_legal_resize")
        self.assertEqual(release.layer_id, "conv0")
        self.assertEqual(release.core_delta, -3)
        changed = apply_action(locked, release, core_ids=cores)
        self.assertEqual(len(changed.mapping_map["conv0"].chiplets), 45)
        self.assertTrue(validate_plan(self.workload, changed, cores))
        acquire = MappingAction("OP4", "conv0", partition=(1, 3, 4, 4), variant="hybrid_legal_resize",
                                core_delta=3, resize_cores=release.resize_cores)
        restored = apply_action(changed, acquire, core_ids=cores)
        self.assertEqual(len(restored.mapping_map["conv0"].chiplets), 48)
        self.assertEqual(set(restored.mapping_map["conv0"].chiplets), set(locked.mapping_map["conv0"].chiplets))
        for candidate in (changed, restored):
            compiled = compile_mapping(self.workload, candidate)
            self.assertEqual(sum(unit.macs for unit in compiled.units), self.workload.total_macs)
            for layer in self.workload.layers:
                self.assertEqual(sum(unit.output_region.elements for unit in compiled.units if unit.layer_id == layer.id),
                                 self.workload.tensor_map[layer.output].elements)
        self.assertEqual(MappingAction.from_dict(acquire.to_dict()), acquire)
        with self.assertRaisesRegex(ValueError, "exact multi-core delta"):
            apply_action(changed, replace(acquire, core_delta=2), core_ids=cores)

    def test_internal_fd_and_no_weight_selectors_are_masked(self):
        actions = legal_actions(self.workload, self.plan, MEMORIES, max_per_operator=0)
        fields = {(action.layer_id, action.field) for action in actions if action.operator == "OP5"}
        self.assertIn(("conv0", "input_dram"), fields)
        self.assertIn(("conv1", "output_dram"), fields)
        self.assertNotIn(("conv1", "input_dram"), fields)
        self.assertNotIn(("conv0", "output_dram"), fields)
        residual = make_toy_workload(residual=True)
        plan = initial_mapping(residual, CORES, MEMORIES)
        residual_fields = {(action.layer_id, action.field) for action in legal_actions(residual, plan, MEMORIES,
                                  max_per_operator=0) if action.operator == "OP5"}
        self.assertNotIn(("add", "weight_dram"), residual_fields)


class LearnedSearchTests(unittest.TestCase):
    def setUp(self):
        self.workload = make_toy_workload(residual=False, batch=4)
        self.plan = initial_mapping(self.workload, CORES, MEMORIES, max_group_layers=2)

    def test_actor_reinforcement_changes_masked_action_probability(self):
        policy = LinearActorCritic(actor_features=2, state_size=1, entropy=0.0)
        features = np.eye(2)
        state = np.ones(1)
        for _ in range(30):
            probabilities = policy.probabilities(features)
            policy.update(features, 0, probabilities, state, state, 1.0, terminal=True)
        probabilities = policy.probabilities(features)
        self.assertGreater(probabilities[0], 0.6)
        self.assertEqual(policy.updates, 30)

    def test_features_include_context_and_geometry_and_have_correct_size(self):
        metrics = toy_oracle(self.plan)
        action = next(action for action in legal_actions(self.workload, self.plan, MEMORIES)
                      if action.operator == "OP4")
        before = state_features(self.workload, self.plan, metrics, metrics, 1, 0)
        after = state_features(self.workload, self.plan, metrics, metrics, 0.5, 0)
        features_a = action_features(self.workload, self.plan, action, before, len(CORES))
        features_b = action_features(self.workload, self.plan, action, after, len(CORES))
        self.assertEqual(len(features_a), len(LinearActorCritic().theta))
        self.assertFalse(np.array_equal(features_a, features_b))

    def test_idle_variants_have_distinct_context_sensitive_actor_features(self):
        metrics = toy_oracle(self.plan)
        state = state_features(self.workload, self.plan, metrics, metrics, 1, 0)
        release = next(action for action in legal_actions(self.workload, self.plan, MEMORIES,
                       core_ids=CORES, max_per_operator=6) if action.variant == "hybrid_idle_release")
        changed = apply_action(self.plan, release, core_ids=CORES)
        acquire = next(action for action in legal_actions(self.workload, changed, MEMORIES,
                       core_ids=CORES, max_per_operator=6) if action.variant == "hybrid_idle_acquire")
        release_features = action_features(self.workload, self.plan, release, state, len(CORES))
        acquire_features = action_features(self.workload, changed, acquire, state, len(CORES))
        self.assertEqual(len(release_features), 72)
        self.assertEqual(tuple(release_features[56:58]), (0, 1))
        self.assertEqual(tuple(acquire_features[56:58]), (1, 0))
        self.assertAlmostEqual(acquire_features[58], 1 / len(CORES))
        self.assertFalse(np.array_equal(release_features, acquire_features))

    def test_memory_proximity_uses_real_positions_and_never_identifier_numbers(self):
        metrics = toy_oracle(self.plan)
        state = state_features(self.workload, self.plan, metrics, metrics, 1, 0)
        near = MappingAction("OP5", "conv0", field="weight_dram", memory="memory:0")
        far = replace(near, memory="memory:1")
        positions = {core: (0.0 if core.startswith("compute:0/") else 10.0, 0.0) for core in CORES}
        memory_positions = {"memory:0": (-1.0, 0.0), "memory:1": (11.0, 0.0)}
        near_features = action_features(self.workload, self.plan, near, state, len(CORES),
                                        core_positions=positions, memory_positions=memory_positions)
        far_features = action_features(self.workload, self.plan, far, state, len(CORES),
                                       core_positions=positions, memory_positions=memory_positions)
        self.assertAlmostEqual(near_features[19], 1 / 12)
        self.assertAlmostEqual(far_features[19], 11 / 12)
        self.assertEqual(action_features(self.workload, self.plan, far, state, len(CORES))[19], 0)

    def test_rl_is_learned_and_physical_budget_is_respected(self):
        calls = []
        def oracle(plan):
            calls.append(plan_fingerprint(plan))
            return toy_oracle(plan)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "rl.json"
            result = run_search(self.workload, self.plan, oracle, CORES, MEMORIES,
                                budget=24, checkpoint_path=checkpoint, max_per_operator=5)
            saved = json.loads(checkpoint.read_text())
            self.assertEqual(len(calls), result.evaluations)
            self.assertLessEqual(result.evaluations, 24)
            self.assertGreater(result.policy_updates, 0)
            self.assertGreater(np.linalg.norm(saved["policy"]["theta"]), 0)
            self.assertLessEqual(result.best_cost, 1)
            self.assertTrue(all(row["operator_mask"][row["operator"]] for row in result.trace))

    def test_resume_restores_exact_learning_and_rng_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            full_path, resumed_path = Path(tmp) / "full.json", Path(tmp) / "resumed.json"
            kwargs = dict(seed=17, max_per_operator=5, horizon=8, evaluator_identity="toy-v1")
            full = run_search(self.workload, self.plan, toy_oracle, CORES, MEMORIES,
                              budget=31, checkpoint_path=full_path, **kwargs)
            run_search(self.workload, self.plan, toy_oracle, CORES, MEMORIES,
                       budget=11, checkpoint_path=resumed_path, **kwargs)
            resumed = run_search(self.workload, self.plan, toy_oracle, CORES, MEMORIES,
                                 budget=31, checkpoint_path=resumed_path, resume=True, **kwargs)
            self.assertEqual(full.trace, resumed.trace)
            self.assertEqual(full.best_plan, resumed.best_plan)
            self.assertEqual(full.evaluations, resumed.evaluations)
            self.assertEqual(json.loads(full_path.read_text())["policy"],
                             json.loads(resumed_path.read_text())["policy"])
            with self.assertRaisesRegex(ValueError, "identity"):
                run_search(self.workload, self.plan, toy_oracle, CORES, MEMORIES, budget=32,
                           checkpoint_path=resumed_path, resume=True,
                           **{**kwargs, "evaluator_identity": "changed-hardware"})

    def test_seed_matched_baselines_have_same_initial_reference_and_budget(self):
        results = compare_searches(self.workload, self.plan, toy_oracle, CORES, MEMORIES,
                                  budget=12, seeds=(3,), max_per_operator=3)
        self.assertEqual({result.method for result in results}, {"rl", "random", "greedy", "sa"})
        self.assertEqual({result.seed for result in results}, {3})
        for result in results:
            self.assertEqual(result.initial_metrics, results[0].initial_metrics)
            self.assertLessEqual(result.evaluations, 12)
            self.assertEqual(result.policy_updates > 0, result.method == "rl")

    def test_objective_reference_is_fixed_and_hard_limits_reject(self):
        objective = SearchObjective()
        reference = {"latency_s": 2, "energy_j": 3, "area_mm2": 20, "power_w": 1.5}
        candidate = {**reference, "latency_s": 1, "energy_j": 6}
        self.assertAlmostEqual(objective.cost(candidate, reference), 1)
        self.assertTrue(np.isinf(SearchObjective(max_area_mm2=19).cost(candidate, reference)))
        self.assertTrue(np.isinf(objective.cost({**candidate, "feasible": False}, reference)))

    def test_physical_resource_signals_reach_policy_without_reward_bonus(self):
        metrics = {**toy_oracle(self.plan), "compute_busy_s": 9.0,
                   "resource_busy_s": {"link:io:0>io:1": 1.5, "core:compute:0/core:0": 2.0}}
        state = state_features(self.workload, self.plan, metrics, metrics, 1, 0)
        self.assertAlmostEqual(state[4], 1.5 / metrics["latency_s"])
        self.assertAlmostEqual(state[6], 9.0 / (len(CORES) * metrics["latency_s"]))
        self.assertEqual(SearchObjective().cost(metrics, metrics), 1)


if __name__ == "__main__":
    unittest.main()
