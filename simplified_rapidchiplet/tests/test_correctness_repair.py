"""Phase 1.5 regressions independent of future ownership heuristics."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from simple_rapidchiplet import rapidchiplet_engine as engine
from simple_rapidchiplet.config import load_config
from simple_rapidchiplet.e2e_latency import estimate_batch1_e2e_latency
from simple_rapidchiplet.evaluator import _score_results, evaluate_one, resolve_ppa_weights
from simple_rapidchiplet.flows import Flow, traffic_bits_by_pair
from simple_rapidchiplet.mapping import build_mapping_plan
from simple_rapidchiplet.model import ModelSpec, Stage
from simple_rapidchiplet.network_context import NetworkContext
from simple_rapidchiplet.preference_dse import (
    DesignState, EvaluationOracle, make_preference_profile, q_learning_search, score_result,
)
from simple_rapidchiplet.report import write_outputs
from simple_rapidchiplet.topology import make_topology
from simple_rapidchiplet.workload import build_pipeline_workload


ROOT = Path(__file__).resolve().parents[1]


class CorrectnessRepairTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/defaults.json")
        # Deliberately different tensor shapes: A=(1,4,2,2), B=(1,2,1,1).
        # These are toy inputs, not manually supplied ResNet operator shapes.
        self.producer_bytes = 1 * 4 * 2 * 2 * self.cfg.reference_model.activation_bytes_per_element
        self.consumer_bytes = 1 * 2 * 1 * 1 * self.cfg.reference_model.activation_bytes_per_element
        self.model = ModelSpec("repair_toy", "test", "4x2x2", 0.000002, 0.0, (
            Stage("A", 0.000001, self.producer_bytes / 1024**2, (), input_mb=16 / 1024**2),
            Stage("B", 0.000001, self.consumer_bytes / 1024**2, ("A",)),
        ))
        self.groups = ((0, 1, 2), (1, 2, 1))
        self.proxy_cfg = replace(self.cfg, rapidchiplet=replace(self.cfg.rapidchiplet, root=str(ROOT / "absent_engine")))

    def workload(self, strategy="output_channel"):
        return build_pipeline_workload(self.model, self.cfg.chiplet.op_per_mac,
                                       self.groups, mapping_strategies=(strategy, "single"))

    def test_boundary_uses_actual_producer_not_consumer_tensor(self):
        workload = self.workload()
        boundaries = [flow for flow in workload.flows if flow.kind == "group_boundary"]
        self.assertEqual({flow.tensor_id for flow in boundaries}, {"A:output"})
        self.assertEqual({(flow.src, flow.dst) for flow in boundaries}, {(0, 2), (1, 2)})
        self.assertEqual(sum(flow.bytes for flow in boundaries), self.producer_bytes)
        self.assertNotEqual(sum(flow.bytes for flow in boundaries), self.consumer_bytes)
        row = evaluate_one(self.model, "mesh", 3, self.proxy_cfg, self.workload())
        boundary = next(event for event in row.e2e_boundary_timings if event["kind"] == "group_boundary")
        self.assertEqual(boundary["traffic_bytes"], self.producer_bytes)
        self.assertEqual(boundary["tensor_ids"], ["A:output"])

    def test_mapping_strategy_candidates_do_not_alias_in_rl_cache(self):
        profile = make_preference_profile(cfg=self.proxy_cfg)
        oracle = EvaluationOracle(self.model, self.proxy_cfg, profile, 3)
        oc = DesignState("balanced", 2, 3, self.groups, self.model.name,
                         profile.constraint_key, ("output_channel", "single"))
        ic = replace(oc, mapping_strategies=("input_channel", "single"))
        oc_result, ic_result = oracle.evaluate(oc), oracle.evaluate(ic)
        self.assertIsNot(oc_result, ic_result)
        self.assertEqual(oracle.unique_evaluations, 2)
        self.assertEqual(oc_result.mapping_plan["groups"][0]["strategy"], "output_channel")
        self.assertEqual(ic_result.mapping_plan["groups"][0]["strategy"], "input_channel")
        self.assertNotEqual(oc_result.avg_latency_ns, ic_result.avg_latency_ns)
        self.assertIs(oracle.evaluate(oc), oc_result)

    def test_changed_hardware_cannot_return_old_cached_latency(self):
        profile = make_preference_profile(cfg=self.proxy_cfg)
        oracle = EvaluationOracle(self.model, self.proxy_cfg, profile, 3)
        state = DesignState("balanced", 2, 3, self.groups, self.model.name,
                            profile.constraint_key, ("output_channel", "single"))
        before = oracle.evaluate(state)
        oracle.cfg = replace(self.proxy_cfg, chiplet=replace(self.proxy_cfg.chiplet, utilization=0.5))
        after = oracle.evaluate(state)
        self.assertIsNot(before, after)
        self.assertGreater(after.e2e_compute_latency_ns, before.e2e_compute_latency_ns)
        self.assertEqual(oracle.unique_evaluations, 2)

    def test_q_learning_budget_counts_mapping_distinct_candidates(self):
        toy = replace(self.model, stages=(replace(self.model.blocks[0], depends_on=(), operators=("toy_conv",)),),
                      macs_g=self.model.blocks[0].macs_g)
        cfg = replace(self.proxy_cfg, max_chiplets=2)
        result = q_learning_search(toy, cfg, make_preference_profile(cfg=cfg),
                                   block_split=1, evaluation_budget=2,
                                   seed=123, max_episodes=100)
        # Phase 2 disables IC/spatial in default search.
        self.assertEqual(result.unique_evaluations, 2)

    def test_cache_fingerprints_official_input_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            native_root = Path(directory)
            cfg = replace(self.cfg, rapidchiplet=replace(self.cfg.rapidchiplet, root=directory,
                          chiplet_file="chiplets.json", packaging_file="packaging.json", technology_file="technology.json"))
            path = native_root / "packaging.json"
            path.write_text('{"bandwidth":1}', encoding="utf-8")
            oracle = EvaluationOracle(self.model, cfg, make_preference_profile(cfg=cfg), 3)
            state = DesignState("balanced", 2, 3, self.groups, mapping_strategies=("output_channel", "single"))
            before = oracle.cache_key(state)
            path.write_text('{"bandwidth":2}', encoding="utf-8")
            self.assertNotEqual(before, oracle.cache_key(state))

    def test_every_canonical_flow_is_counted_once_in_e2e(self):
        workload = self.workload()
        row = evaluate_one(self.model, "mesh", 3, self.proxy_cfg, workload)
        indices = [index for event in row.e2e_boundary_timings for index in event["flow_indices"]]
        self.assertEqual(sorted(indices), list(range(len(workload.flows))))
        self.assertEqual(sum(event["traffic_bytes"] for event in row.e2e_boundary_timings),
                         sum(flow.bytes for flow in workload.flows))
        self.assertEqual(sum(workload.traffic_bits_per_inference.values()) / 8,
                         sum(flow.bytes for flow in workload.flows))
        self.assertEqual(row.canonical_flows, tuple(flow.to_dict() for flow in workload.flows))

    def test_tensor_provenance_survives_pair_aggregation(self):
        flows = (Flow("main", 0, 1, 10, "event", 0, 1),
                 Flow("shortcut", 0, 1, 20, "event", 0, 1))
        self.assertEqual(traffic_bits_by_pair(flows), {(0, 1): 240})
        self.assertEqual([flow.tensor_id for flow in flows], ["main", "shortcut"])

    def test_scheduler_communication_estimate_does_not_change_search_metrics(self):
        workload = self.workload()
        before = evaluate_one(self.model, "mesh", 3, self.proxy_cfg, workload)
        with patch("simple_rapidchiplet.dag_scheduler._communication_cycles", return_value=1e12):
            after = evaluate_one(self.model, "mesh", 3, self.proxy_cfg, workload)
        self.assertNotEqual(before.schedule_latency_ns, after.schedule_latency_ns)
        self.assertEqual(before.e2e_latency_ns, after.e2e_latency_ns)
        self.assertEqual(before.achieved_fps, after.achieved_fps)
        self.assertEqual(before.architecture_feasible, after.architecture_feasible)
        profile = make_preference_profile(cfg=self.proxy_cfg)
        self.assertEqual(score_result(before, profile), score_result(after, profile))
        scored = _score_results([before, after], resolve_ppa_weights("balanced"))
        self.assertEqual(scored[0].ppa_score, scored[1].ppa_score)

    def test_serialization_uses_per_link_backend_bandwidth(self):
        workload = self.workload()
        topology = make_topology("mesh", 3, self.cfg.chiplet.width_mm, self.cfg.chiplet.spacing_mm)
        flows = tuple(flow for flow in workload.flows if flow.kind == "group_boundary")
        workload = replace(workload, flows=flows)
        context = NetworkContext("fixture", "backend", "backend", 100.0,
                                 {(0, 2): (0, 2), (1, 2): (1, 0, 2)},
                                 {(0, 2): 64.0, (1, 0): 8.0},
                                 {(0, 2): 2.0, (1, 2): 5.0})
        result = estimate_batch1_e2e_latency(self.model, workload,
                    build_mapping_plan(self.model, self.groups), topology, self.cfg, context)
        event = result.boundary_timings[0]
        # 1->0 carries 32 bytes at 8 bits/cycle: 32 cycles, longer than
        # 0->2's 64 bytes at 64 bits/cycle: 8 cycles.
        self.assertEqual(event["bottleneck_link"], [1, 0])
        self.assertAlmostEqual(event["serialization_s"], 0.32)
        self.assertAlmostEqual(event["path_latency_s"], 0.05)
        self.assertAlmostEqual(event["service_s"], 0.37)

    def test_single_chiplet_has_no_canonical_network_flows(self):
        workload = build_pipeline_workload(self.model, 2, ((0, 2, 1),))
        row = evaluate_one(self.model, "mesh", 1, self.proxy_cfg, workload)
        self.assertEqual(workload.flows, ())
        self.assertEqual(row.e2e_communication_latency_ns, 0)
        self.assertEqual(row.e2e_latency_ns, row.e2e_compute_latency_ns)

    def test_external_input_is_not_faked_as_chiplet_zero_broadcast(self):
        workload = build_pipeline_workload(self.model, 2, ((0, 2, 2),),
                                           mapping_strategies=("output_channel",))
        self.assertEqual(workload.flows, ())
        self.assertEqual(workload.traffic_bits_per_inference, {})

    def test_fps_is_inverse_e2e_and_cannot_change_reward_or_ppa(self):
        row = evaluate_one(self.model, "mesh", 3, self.proxy_cfg, self.workload())
        self.assertAlmostEqual(row.achieved_fps, 1e9 / row.e2e_latency_ns)
        altered = replace(row, achieved_fps=row.achieved_fps * 1000)
        scored = _score_results([row, altered], resolve_ppa_weights("balanced"))
        self.assertEqual(scored[0].ppa_score, scored[1].ppa_score)
        profile = make_preference_profile(cfg=self.cfg)
        self.assertEqual(score_result(row, profile), score_result(altered, profile))
        self.assertNotIn("fps", str(profile.constraint_key).lower())

    def test_obsolete_fps_constraints_rejected_by_api_and_config(self):
        with self.assertRaises(TypeError):
            make_preference_profile(min_fps=1)
        with self.assertRaises(TypeError):
            evaluate_one(self.model, "mesh", 3, self.cfg, target_fps=1)
        raw = json.loads((ROOT / "configs/defaults.json").read_text(encoding="utf-8"))
        raw["evaluation"]["target_fps"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obsolete.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_obsolete_fps_flags_rejected_before_search(self):
        for entry, flag in (("run.py", "--target-fps"), ("preference_dse.py", "--min-fps")):
            process = subprocess.run([sys.executable, str(ROOT / entry), flag, "1"],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn("unrecognized arguments", process.stderr)

    def test_report_has_diagnostic_fps_and_network_provenance(self):
        row = evaluate_one(self.model, "mesh", 3, self.proxy_cfg, self.workload())
        row = _score_results([row], resolve_ppa_weights("balanced"))[0]
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(all(write_outputs(directory, [row], [row]).values()))
            with (Path(directory) / "best.csv").open(encoding="utf-8", newline="") as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual(table[0]["network_backend"], "local_proxy")
            for key in ("target_fps", "fps_bonus", "fps_penalty"):
                self.assertNotIn(key, table[0])
            document = (Path(directory) / "report.html").read_text(encoding="utf-8")
            self.assertIn("FPS = 1 / serial Batch=1 E2E seconds", document)

    def test_reference_config_records_requested_baseline(self):
        reference = self.cfg.reference_model
        self.assertEqual(reference.input_shape, (1, 3, 224, 224))
        self.assertEqual(reference.num_classes, 1000)
        self.assertEqual(reference.activation_precision, "FP32")
        self.assertEqual(reference.weight_precision, "FP32")
        self.assertEqual(reference.activation_bytes_per_element, 4)
        self.assertEqual(reference.weight_bytes_per_element, 4)
        self.assertEqual((reference.initial_input_source, reference.weights_source), ("EXTERNAL", "EXTERNAL"))

    def test_official_adapter_injected_bytes_and_e2e_share_flows(self):
        if not (Path(self.cfg.rapidchiplet.root) / "rapidchiplet.py").is_file():
            self.skipTest("configured official RapidChiplet is unavailable")
        rc = engine._load_rapidchiplet_module(self.cfg.rapidchiplet.root)
        actual_call = rc.rapidchiplet
        captured = []
        def capture(inputs, intermediates, *args, **kwargs):
            outputs = actual_call(inputs, intermediates, *args, **kwargs)
            captured.append((dict(inputs["traffic_by_chiplet"]), dict(intermediates["link_bandwidths"])))
            return outputs
        workload = self.workload()
        with patch.object(rc, "rapidchiplet", side_effect=capture):
            row = evaluate_one(self.model, "mesh", 3, self.cfg, workload)
            # Official bandwidth is packaging-derived, not the proxy knob.
            changed = replace(self.cfg, network=replace(self.cfg.network, link_bandwidth_bits_per_cycle=1))
            second = evaluate_one(self.model, "mesh", 3, changed, workload)
        traffic, bandwidths = captured[0]
        self.assertEqual(traffic, traffic_bits_by_pair(workload.flows))
        self.assertEqual(sum(traffic.values()) / 8, sum(flow.bytes for flow in workload.flows))
        self.assertEqual(sum(event["traffic_bytes"] for event in row.e2e_boundary_timings), sum(traffic.values()) / 8)
        self.assertEqual(row.network_context["backend"], "official_rapidchiplet")
        self.assertEqual({link["bandwidth_bits_per_cycle"] for link in row.network_context["links"]}, set(bandwidths.values()))
        self.assertEqual(row.e2e_communication_latency_ns, second.e2e_communication_latency_ns)
        self.assertEqual(row.rapid_avg_latency_ns, second.rapid_avg_latency_ns)
        self.assertEqual(row.e2e_path_latency_source, "official_rapidchiplet_per_flow")


if __name__ == "__main__":
    unittest.main()
