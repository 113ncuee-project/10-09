import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from simple_rapidchiplet.config import load_config
from simple_rapidchiplet.physical_dse import PhysicalOracle, physical_profile, q_learning_physical
from simple_rapidchiplet.physical_report import write_physical_outputs
from simple_rapidchiplet.toy_model import make_toy_graph


class PhysicalReportTests(unittest.TestCase):
    def test_empty_flows_full_trace_and_strict_json(self):
        root = Path(__file__).resolve().parents[1]
        cfg = replace(load_config(root/"configs/defaults.json"),max_chiplets=1)
        graph = make_toy_graph()
        profile = physical_profile(graph,cfg)
        result = q_learning_physical(PhysicalOracle(graph,cfg,profile),budget=1)
        with tempfile.TemporaryDirectory() as directory:
            out = write_physical_outputs(directory,result,graph,cfg)
            for filename in ("best.csv","summary.csv","chiplet_results.csv","mapping_trace.csv",
                             "canonical_flows.csv","transition_traffic.csv","operator_trace.csv",
                             "per_link_statistics.csv","report.html","assumptions.json","model_metadata.json"):
                self.assertTrue((out/filename).exists())
            def reject(value):
                raise ValueError(value)
            for path in out.glob("*.json"):
                json.loads(path.read_text(encoding="utf-8"),parse_constant=reject)
            self.assertEqual(json.loads((out/"canonical_flows.json").read_text()),[])
            report = (out/"report.html").read_text(encoding="utf-8")
            self.assertIn("assignments=",report)
            self.assertIn("does not simulate queues",report)
            self.assertIn("source_fingerprints",(out/"run_manifest.json").read_text())
