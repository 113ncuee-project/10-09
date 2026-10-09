"""Physical contract tests for the independent hybrid implementation."""
from dataclasses import replace
from pathlib import Path
import unittest

from simple_rapidchiplet.hybrid.hardware import HardwareSpec, build_package, validate_package
from simple_rapidchiplet.hybrid.rapid_backend import DEFAULT_RAPID_ROOT, build_native_inputs, evaluate_package


class HybridHardwareTests(unittest.TestCase):
    def test_resources_are_physical_dies_and_mapping_slots(self):
        spec = HardwareSpec()
        package = build_package(spec)
        self.assertEqual((len(package.compute_ids), len(package.io_ids), len(package.memory_ids)), (16, 4, 4))
        self.assertEqual(len(package.mapping_core_ids), 64)
        self.assertEqual(spec.pe_count, spec.cores_per_die * spec.pes_per_core)
        self.assertEqual(spec.macs_per_second, spec.per_core_mac_rate * spec.cores_per_die)
        self.assertEqual(spec.total_buffer_bytes_per_die, 16 * (64 + 16 + 8) * 1024 + 88 * 1024 + 2 * 1024**2)
        self.assertAlmostEqual(spec.peak_tops, 65.536)  # 500MHz NN-Baton reference

    def test_added_compute_and_sram_have_cost(self):
        base = HardwareSpec()
        more_pe = replace(base, pe_count=32)
        more_sram = replace(base, unified_buffer_bytes=base.unified_buffer_bytes * 2)
        self.assertGreater(more_pe.compute_area_mm2, base.compute_area_mm2)
        self.assertGreater(more_pe.compute_idle_power_w, base.compute_idle_power_w)
        self.assertEqual(more_pe.macs_per_second, base.macs_per_second * 2)
        self.assertGreater(more_sram.compute_area_mm2, base.compute_area_mm2)
        self.assertGreater(more_sram.compute_idle_power_w, base.compute_idle_power_w)
        self.assertEqual(more_sram.macs_per_second, base.macs_per_second)
        self.assertGreater(replace(base, ring_count=8).compute_area_mm2, base.compute_area_mm2)

    def test_io_area_follows_actual_degree(self):
        spec = HardwareSpec(cluster_rows=3, cluster_cols=3)
        package = build_package(spec)
        center, corner = package.nodes["io:4"], package.nodes["io:0"]
        self.assertEqual(center.ports, 9)  # four leaves + four mesh + DDR
        self.assertEqual(corner.ports, 7)
        self.assertGreater(center.area_mm2, corner.area_mm2)
        native, _ = build_native_inputs(spec, package)
        self.assertEqual(len(native["chiplets"]["io:4"]["phys"]), 9)
        for desc in native["chiplets"].values():
            self.assertLessEqual(sum(phy["fraction_bump_area"] for phy in desc["phys"]), 1.0)

    def test_rejects_overlap_and_phy_reuse(self):
        package = build_package(HardwareSpec(cluster_rows=1, cluster_cols=1))
        nodes = dict(package.nodes)
        a, b = package.compute_ids[:2]
        nodes[b] = replace(nodes[b], x_mm=nodes[a].x_mm, y_mm=nodes[a].y_mm)
        positions = {i: (n.x_mm, n.y_mm) for i, n in nodes.items()}
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_package(replace(package, nodes=nodes, positions=positions))
        links = list(package.links)
        self.assertEqual(links[0].b, links[1].b)
        links[1] = replace(links[1], b_port=links[0].b_port)
        with self.assertRaisesRegex(ValueError, "PHY port reused"):
            validate_package(replace(package, links=tuple(links)))

    def test_rejects_stale_package_after_hardware_changes(self):
        spec = HardwareSpec(cluster_rows=1, cluster_cols=1)
        package = build_package(spec)
        with self.assertRaisesRegex(ValueError, "costs disagree"):
            build_native_inputs(replace(spec, unified_buffer_bytes=spec.unified_buffer_bytes * 2), package)

    def test_rejects_excessive_bump_rate(self):
        with self.assertRaisesRegex(ValueError, "bump array"):
            HardwareSpec(cluster_rows=1, cluster_cols=1, bump_pitch_mm=1.0)

    def test_rejects_unsupported_precision_and_fractional_pes(self):
        with self.assertRaisesRegex(ValueError, "INT8 only"):
            HardwareSpec(activation_bits=32)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            HardwareSpec(pe_count=16.0)
        with self.assertRaisesRegex(ValueError, "at most four"):
            HardwareSpec(compute_per_cluster=5)
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            HardwareSpec(mac_energy_pj=-1)


@unittest.skipUnless((Path(DEFAULT_RAPID_ROOT) / "rapidchiplet.py").is_file(), "local native RapidChiplet unavailable")
class HybridNativeBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = HardwareSpec()
        cls.package = build_package(cls.spec)
        cls.context = evaluate_package(cls.spec, cls.package)

    def test_actual_native_area_matches_hardware_contract(self):
        context = self.context
        expected = sum(n.area_mm2 for n in self.package.nodes.values())
        self.assertEqual(context.backend, "official_rapidchiplet")
        self.assertAlmostEqual(context.area_breakdown["total_silicon_mm2"], expected)
        self.assertGreaterEqual(context.area_breakdown["package_footprint_mm2"], expected)
        self.assertAlmostEqual(context.area_breakdown["compute_silicon_mm2"], self.spec.compute_count * self.spec.compute_area_mm2)
        self.assertAlmostEqual(context.native_power_summary["total_power"], sum(n.idle_power_w for n in self.package.nodes.values()))
        self.assertAlmostEqual(context.idle_power_w, context.native_power_summary["total_power"] + context.provenance["link_idle_power_w"])

    def test_xy_relay_and_memory_paths_are_legal(self):
        self.assertEqual(self.context.route("compute:0/core:0", "compute:15/core:3"),
                         ("compute:0", "io:0", "io:1", "io:3", "compute:15"))
        self.assertEqual(self.context.route("memory:0", "compute:0"), ("memory:0", "io:0", "compute:0"))
        self.assertEqual(self.context.route("compute:0/core:0", "compute:0/core:3"), ("compute:0",))
        for route in self.context.routes.values():
            self.assertEqual(len(route), len(set(route)))
            for endpoint in route[1:-1]:
                self.assertEqual(self.package.nodes[endpoint].role, "io")
            for pair in zip(route, route[1:]):
                self.assertIn(pair, self.context.link_bandwidth_Bps)

    def test_caps_bump_capacity_by_grs_and_ddr_rate(self):
        for edge in self.package.links:
            pair = (edge.a, edge.b)
            self.assertLessEqual(self.context.link_bandwidth_Bps[pair], self.context.native_link_bandwidth_Bps[pair])
            self.assertLessEqual(self.context.link_bandwidth_Bps[pair], edge.bandwidth_Bps)
            self.assertAlmostEqual(self.context.link_bandwidth_Bps[pair], edge.bandwidth_Bps)
        self.assertEqual(self.context.link_energy_pj_per_bit[("memory:0", "io:0")], 0.0)
        self.assertEqual(self.context.link_energy_pj_per_bit[("compute:0", "io:0")], self.spec.nop_energy_pj_per_bit)

    def test_native_path_latency_uses_clock_and_route(self):
        same_cluster = self.context.latency_ns("compute:0", "compute:1")
        cross_cluster = self.context.latency_ns("compute:0", "compute:15")
        self.assertEqual(self.context.latency_ns("compute:0/core:0", "compute:0/core:1"), 0.0)
        self.assertGreater(same_cluster, 0.0)
        self.assertGreater(cross_cluster, same_cluster)
        slower = replace(self.spec, nop_clock_hz=self.spec.nop_clock_hz / 2)
        context = evaluate_package(slower, traffic_pairs=(("compute:0", "compute:15"),))
        self.assertAlmostEqual(context.latency_ns("compute:0", "compute:15"), cross_cluster * 2)


if __name__ == "__main__":
    unittest.main()
