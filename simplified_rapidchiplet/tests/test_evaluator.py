import unittest
from dataclasses import replace
from pathlib import Path

from simple_rapidchiplet.config import load_config
from simple_rapidchiplet.evaluator import (
    _score_results,
    evaluate_models,
    evaluate_one,
    make_ppa_reference,
    resolve_ppa_weights,
)
from simple_rapidchiplet.model import load_models, refine_model_blocks
from simple_rapidchiplet.block_extractor import extract_semantic_blocks
from simple_rapidchiplet.dag_scheduler import schedule_block_dag
from simple_rapidchiplet.mapping import mapping_action_is_feasible
from simple_rapidchiplet.model import ModelSpec, Stage
from simple_rapidchiplet.preference_dse import (
    DesignAction,
    DesignState,
    legal_actions,
    make_preference_profile,
    score_result,
)
from simple_rapidchiplet.topology import make_topology
from simple_rapidchiplet.workload import (
    brute_force_pipeline_workloads,
    build_pipeline_workload,
    pareto_dp_pipeline_workloads,
)


ROOT = Path(__file__).resolve().parents[1]


class EvaluatorTests(unittest.TestCase):
    def test_mesh_edge_count_for_four_chiplets(self):
        topology = make_topology("mesh", 4, chiplet_width_mm=8.6, spacing_mm=1.0)
        self.assertEqual(topology.node_count, 4)
        self.assertEqual(len(topology.edges), 4)

    def test_tree_edge_count_for_four_chiplets(self):
        topology = make_topology("tree", 4, chiplet_width_mm=8.6, spacing_mm=1.0)
        self.assertEqual(topology.node_count, 4)
        self.assertEqual(len(topology.edges), 3)

    def test_tree_uses_grid_placement_like_mesh(self):
        mesh = make_topology("mesh", 13, chiplet_width_mm=8.6, spacing_mm=1.0)
        tree = make_topology("tree", 13, chiplet_width_mm=8.6, spacing_mm=1.0)
        self.assertEqual(tree.positions, mesh.positions)
        self.assertEqual(len(tree.edges), tree.node_count - 1)
        self.assertEqual([edge.key() for edge in tree.edges[:3]], [(0, 1), (0, 2), (1, 3)])

    def test_default_chiplet_capacity_is_eight_by_eight_with_sixteen_available(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        self.assertEqual(cfg.chiplet.pe_rows, 8)
        self.assertEqual(cfg.chiplet.pe_cols, 8)
        self.assertEqual(cfg.max_chiplets, 16)

    def test_pareto_dp_generates_stage_partition_candidates(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["resnet18"]
        workloads = pareto_dp_pipeline_workloads(model, 2, cfg.chiplet.op_per_mac)
        self.assertGreater(len(workloads), 1)
        self.assertTrue(all(len(workload.ops_per_chiplet) == 2 for workload in workloads))
        self.assertTrue(all(workload.search_method == "pareto-dp" for workload in workloads))

    def test_pareto_dp_remains_valid_with_arbitrary_group_parallelism(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["resnet50"]
        workloads = pareto_dp_pipeline_workloads(model, 13, cfg.chiplet.op_per_mac)
        self.assertGreater(len(workloads), 1)
        self.assertTrue(all(sum(parts for _start, _end, parts in workload.group_specs) == 13 for workload in workloads))

    def test_pareto_dp_allows_uneven_stage_workloads(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["resnet50"]
        workloads = pareto_dp_pipeline_workloads(model, 4, cfg.chiplet.op_per_mac)
        self.assertTrue(
            any(len({round(ops, -6) for ops in workload.ops_per_chiplet}) > 1 for workload in workloads)
        )

    def test_pareto_dp_cost_modes_generate_valid_candidates(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["resnet50"]
        for cost_mode in ("both", "bottleneck", "traffic"):
            workloads = pareto_dp_pipeline_workloads(
                model,
                4,
                cfg.chiplet.op_per_mac,
                top_k=12,
                cost_mode=cost_mode,
            )
            self.assertGreater(len(workloads), 0)
            self.assertTrue(all(len(workload.ops_per_chiplet) == 4 for workload in workloads))

    def test_brute_force_generates_stage_cut_candidates(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["resnet18"]
        workloads = brute_force_pipeline_workloads(model, 4, cfg.chiplet.op_per_mac)
        plans = {workload.plan for workload in workloads}
        self.assertIn("conv1 + layer1 | layer2 | layer3 | layer4 + fc", plans)

    def test_brute_force_generates_single_stage_parallel_candidates(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["resnet18"]
        workloads = brute_force_pipeline_workloads(model, 16, cfg.chiplet.op_per_mac)
        self.assertTrue(any("layer3[1/5]" in workload.plan for workload in workloads))
        self.assertTrue(all(workload.search_method == "brute-force" for workload in workloads))

    def test_brute_force_generates_multi_block_parallel_candidates(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["resnet18"]
        workloads = brute_force_pipeline_workloads(model, 4, cfg.chiplet.op_per_mac)
        self.assertTrue(
            any(
                " + " in label and "[" in label
                for workload in workloads
                for label in workload.stage_labels
            )
        )

    def test_refined_model_exposes_a_finer_block_graph(self):
        model = load_models(ROOT / "configs" / "models.json")["resnet18"]
        refined = refine_model_blocks(model, split_factor=2)
        self.assertEqual(len(refined.blocks), 2 * len(model.blocks))
        self.assertAlmostEqual(sum(block.macs_g for block in refined.blocks), model.macs_g)
        self.assertEqual(refined.block_graph()["kind"], "block_dag")

    def test_resnet50_uses_native_semantic_blocks(self):
        model = load_models(ROOT / "configs" / "models.json")["resnet50"]
        extracted, report = extract_semantic_blocks(model, legacy_split_factor=2)
        self.assertEqual(model.block_source, "semantic")
        self.assertEqual(len(extracted.blocks), 18)
        self.assertEqual(extracted.blocks[0].block_type, "Stem")
        self.assertEqual(extracted.blocks[1].block_type, "Bottleneck")
        self.assertEqual(extracted.blocks[-1].block_type, "Head")
        self.assertTrue(all(block.branch_closed for block in extracted.blocks))
        self.assertTrue(all(len(block.operators) >= 2 for block in extracted.blocks))
        self.assertEqual(report.source, "semantic")
        self.assertEqual(extracted.block_graph()["kind"], "semantic_block_dag")
        self.assertAlmostEqual(sum(block.macs_g for block in extracted.blocks), model.macs_g, places=6)

    def test_model_catalog_covers_six_cnn_families(self):
        models = load_models(ROOT / "configs" / "models.json")
        expected = {
            "resnet18",
            "resnet50",
            "shufflenet_v2_x1_0",
            "shufflenet_v2_x2_0",
            "mobilenet_v2",
            "efficientnet_b0",
        }
        self.assertEqual(set(models), expected)
        for model in models.values():
            self.assertGreater(len(model.blocks), 0)
            self.assertAlmostEqual(
                sum(block.macs_g for block in model.blocks),
                model.macs_g,
                places=6,
                msg=model.name,
            )

        mobilenet, mobilenet_report = extract_semantic_blocks(models["mobilenet_v2"])
        efficientnet, efficientnet_report = extract_semantic_blocks(models["efficientnet_b0"])
        self.assertEqual(mobilenet_report.source, "semantic")
        self.assertEqual(efficientnet_report.source, "semantic")
        self.assertIn("InvertedResidual", {block.block_type for block in mobilenet.blocks})
        self.assertIn("MBConv", {block.block_type for block in efficientnet.blocks})
        self.assertTrue(all(block.branch_closed for block in mobilenet.blocks + efficientnet.blocks))

    def test_direct_workload_allows_any_group_parallelism(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["resnet18"]
        workload = build_pipeline_workload(
            model,
            cfg.chiplet.op_per_mac,
            ((0, 2, 2), (2, len(model.blocks), 2)),
        )
        self.assertEqual(len(workload.ops_per_chiplet), 4)
        self.assertIn(" + ", workload.stage_labels[0])

    def test_action_mask_allows_multi_block_parallel_action(self):
        actions = legal_actions(DesignState("latency"), block_count=6, max_chiplets=8)
        self.assertIn(DesignAction(group_length=3, chiplets=2), actions)

    def test_mapping_strategies_change_traffic_model(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["resnet18"]
        groups = ((0, 1, 1), (1, 2, 4), (2, len(model.blocks), 1))
        output_split = build_pipeline_workload(
            model,
            cfg.chiplet.op_per_mac,
            groups,
            mapping_strategies=("single", "output_channel", "single"),
        )
        input_split = build_pipeline_workload(
            model,
            cfg.chiplet.op_per_mac,
            groups,
            mapping_strategies=("single", "input_channel", "single"),
        )
        self.assertNotEqual(
            output_split.mapping_plan["extra_traffic_mb"],
            input_split.mapping_plan["extra_traffic_mb"],
        )
        self.assertIn("input_broadcast", str(output_split.mapping_plan))
        self.assertIn("partial_output_reduction", str(input_split.mapping_plan))

    def test_feasibility_rejects_parallel_head(self):
        model = load_models(ROOT / "configs" / "models.json")["resnet50"]
        last = len(model.blocks) - 1
        self.assertFalse(
            mapping_action_is_feasible(
                model,
                last,
                last + 1,
                2,
                "output_channel",
                sram_mb=16.0,
            )
        )

    def test_dag_scheduler_runs_ready_branches_concurrently(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = ModelSpec(
            name="branch_demo",
            dataset="test",
            input_shape="1x1",
            macs_g=0.004,
            params_m=0.1,
            block_source="semantic",
            stages=(
                Stage("a", 0.001, 0.1, (), "Branch", 0.1, 0.1, ("OpA",), True, ""),
                Stage("b", 0.001, 0.1, ("a",), "Branch", 0.1, 0.1, ("OpB",), True, ""),
                Stage("c", 0.001, 0.1, ("a",), "Branch", 0.1, 0.1, ("OpC",), True, ""),
                Stage("d", 0.001, 0.1, ("b", "c"), "Branch", 0.1, 0.1, ("OpD",), True, ""),
            ),
        )
        topology = make_topology("mesh", 4, cfg.chiplet.width_mm, cfg.chiplet.spacing_mm)
        result = schedule_block_dag(
            model,
            ((0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)),
            cfg,
            topology,
            ("single", "single", "single", "single"),
        )
        tasks = {task["name"]: task for task in result.to_dict()["tasks"]}
        self.assertTrue(result.feasible)
        self.assertEqual(tasks["b"]["start_cycles"], tasks["c"]["start_cycles"])
        self.assertGreaterEqual(result.concurrently_ready_pairs, 1)
        self.assertEqual(result.critical_path[0], "a")
        self.assertEqual(result.critical_path[-1], "d")

    def test_single_candidate_has_power_and_latency(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["shufflenet_v2_x1_0"]
        result = evaluate_one(model, "mesh", chiplet_count=4, cfg=cfg)
        self.assertGreater(result.total_power_w, 0)
        self.assertGreater(result.total_chiplet_power_w, 0)
        self.assertGreater(result.total_link_power_w, 0)
        self.assertGreater(result.total_area_mm2, 0)
        self.assertGreater(result.link_count, 0)
        self.assertGreater(result.avg_latency_ns, 0)
        self.assertEqual(len(result.connection_graph["nodes"]), result.selected_chiplets)

    def test_e2e_latency_is_canonical_and_rapid_latency_is_diagnostic(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        cfg = replace(
            cfg,
            rapidchiplet=replace(
                cfg.rapidchiplet,
                root=str(ROOT / "missing_rapidchiplet_root"),
            ),
        )
        model = load_models(ROOT / "configs" / "models.json")["resnet18"]
        workload = build_pipeline_workload(
            model,
            cfg.chiplet.op_per_mac,
            ((0, 3, 2), (3, 6, 2)),
            mapping_strategies=("output_channel", "output_channel"),
        )
        result = evaluate_one(
            model,
            "mesh",
            chiplet_count=4,
            cfg=cfg,
            workload=workload,
        )

        self.assertGreater(result.e2e_compute_latency_ns, 0)
        self.assertGreater(result.e2e_communication_latency_ns, 0)
        self.assertAlmostEqual(result.avg_latency_ns, result.e2e_latency_ns)
        self.assertAlmostEqual(
            result.e2e_latency_ns,
            result.e2e_compute_latency_ns + result.e2e_communication_latency_ns,
            places=6,
        )
        self.assertAlmostEqual(
            result.rapid_avg_latency_ns,
            result.avg_latency_cycles / cfg.chiplet.frequency_hz * 1e9,
            places=6,
        )
        self.assertEqual(
            result.e2e_latency_model,
            "batch1_sum_group_compute_plus_boundary_service",
        )
        self.assertEqual(
            result.e2e_path_latency_source,
            "local_proxy_per_flow",
        )

    def test_local_proxy_runs_when_rapidchiplet_root_is_missing(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        cfg = replace(
            cfg,
            rapidchiplet=replace(cfg.rapidchiplet, root=str(ROOT / "missing_rapidchiplet_root")),
        )
        model = load_models(ROOT / "configs" / "models.json")["shufflenet_v2_x1_0"]
        result = evaluate_one(model, "mesh", chiplet_count=4, cfg=cfg)
        self.assertGreater(result.total_power_w, 0)
        self.assertGreater(result.total_area_mm2, 0)
        self.assertGreater(result.avg_latency_ns, 0)

    def test_area_goal_does_not_force_all_chiplets(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["shufflenet_v2_x1_0"]
        summaries, _sweeps = evaluate_models([model], ["mesh"], cfg, ppa_goal="area")
        self.assertEqual(len(summaries), 1)
        self.assertLess(summaries[0].selected_chiplets, cfg.max_chiplets)
        self.assertGreater(summaries[0].unused_chiplets, 0)

    def test_shared_ppa_reference_gives_same_design_same_score(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["shufflenet_v2_x1_0"]
        result = evaluate_one(model, "mesh", chiplet_count=1, cfg=cfg)
        reference = make_ppa_reference([result])
        weights = resolve_ppa_weights("balanced")
        first = _score_results([result], weights, reference)[0]
        second = _score_results([result], weights, reference)[0]
        self.assertEqual(first.ppa_score, second.ppa_score)

    def test_preference_reward_uses_user_direction(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        model = load_models(ROOT / "configs" / "models.json")["shufflenet_v2_x1_0"]
        baseline = evaluate_one(model, "mesh", 1, cfg)
        low_latency = replace(
            baseline,
            avg_latency_ns=100.0,
            total_area_mm2=2_000.0,
            total_power_w=20.0,
        )
        low_area_power = replace(
            baseline,
            avg_latency_ns=200.0,
            total_area_mm2=1_000.0,
            total_power_w=10.0,
        )
        area_profile = make_preference_profile(
            "area",
            max_latency_ns=1_000.0,
            max_area_mm2=5_000.0,
            max_power_w=100.0,
            cfg=cfg,
        )
        latency_profile = make_preference_profile(
            "latency",
            max_latency_ns=1_000.0,
            max_area_mm2=5_000.0,
            max_power_w=100.0,
            cfg=cfg,
        )
        area_scores = (score_result(low_latency, area_profile), score_result(low_area_power, area_profile))
        latency_scores = (score_result(low_latency, latency_profile), score_result(low_area_power, latency_profile))
        self.assertGreater(area_scores[1].reward, area_scores[0].reward)
        self.assertGreater(latency_scores[0].reward, latency_scores[1].reward)

    def test_preference_profiles_use_fixed_simple_weights(self):
        cfg = load_config(ROOT / "configs" / "defaults.json")
        latency = make_preference_profile("latency", cfg=cfg)
        area = make_preference_profile("area", cfg=cfg)
        power = make_preference_profile("power", cfg=cfg)
        balanced = make_preference_profile("balanced", cfg=cfg)
        for actual, expected in zip(latency.weights, (0.8, 0.1, 0.1)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(area.weights, (0.1, 0.8, 0.1)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(power.weights, (0.1, 0.1, 0.8)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(balanced.weights, (1 / 3, 1 / 3, 1 / 3)):
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
