"""Hand-derived directed ring, multicast and stage service checks."""
from dataclasses import asdict, replace
import json
import unittest

from simple_rapidchiplet.hybrid.hardware import HardwareSpec, build_package
from simple_rapidchiplet.hybrid.mapping import TensorRegion, compile_mapping, initial_mapping
from simple_rapidchiplet.hybrid.ondie import _cached_core_service, build_on_die_fabric, core_service
from simple_rapidchiplet.hybrid.workload import LayerSpec, TensorSpec, Workload, make_toy_workload


class HybridOnDieTests(unittest.TestCase):
    def setUp(self):
        self.hardware = HardwareSpec(cluster_rows=1, cluster_cols=1, compute_per_cluster=1)
        self.workload = make_toy_workload(residual=False, batch=1, channels=8, spatial=8)
        package = build_package(self.hardware)
        plan = initial_mapping(self.workload, package.mapping_core_ids, package.memory_ids,
                               pipeline=False, max_group_layers=1)
        self.compiled = compile_mapping(self.workload, plan, self.hardware)
        self.unit = self.compiled.units[0]
        self.layer = self.workload.layer_map[self.unit.layer_id]

    def test_clockwise_ring_has_no_cross_ring_shortcut(self):
        fabric = build_on_die_fabric(self.hardware)
        self.assertEqual(fabric.route("l2", "pe:3"), ("l2", "pe:0", "pe:1", "pe:2", "pe:3"))
        self.assertEqual(fabric.route("pe:0", "l2"), ("pe:0", "pe:1", "pe:2", "pe:3", "l2"))
        with self.assertRaisesRegex(ValueError, "cannot cross rings"):
            fabric.route("pe:0", "pe:4")
        forward = fabric.route("pe:0", "l2")
        reinject = fabric.route("l2", "pe:4")
        self.assertGreater(len(forward) + len(reinject) - 2, 1)

    def test_integer_pe_output_ownership_and_halos(self):
        service = core_service(self.unit, self.layer, self.workload, self.hardware)
        records = service.traffic_evidence["pe_output_regions"]
        regions = [TensorRegion(tuple(tuple(interval) for interval in row["output_bounds"])) for row in records]
        self.assertEqual(sum(region.elements for region in regions), self.unit.output_region.elements)
        self.assertTrue(all(a.intersection(b) is None for index, a in enumerate(regions) for b in regions[index+1:]))
        shared_halos = [packet for packet in service.traffic_evidence["input_packets"]
                        if packet["tensor"] == "image" and len(packet["destinations"]) > 1]
        self.assertTrue(shared_halos)
        self.assertEqual(service.traffic_evidence["unique_input_weight_bytes"], self.unit.input_bytes + self.unit.weight_bytes)
        self.assertEqual(service.traffic_evidence["output_delivered_bytes"], self.unit.output_bytes)

    def test_shared_weight_on_ring_edges_is_counted_once(self):
        service = core_service(self.unit, self.layer, self.workload, self.hardware)
        packets = [packet for packet in service.traffic_evidence["input_packets"]
                   if packet["tensor"].startswith("weight:")]
        self.assertTrue(packets)
        for packet in packets:
            traversed = sum(len(route) - 1 for route in packet["routes"])
            self.assertLessEqual(len(packet["union_edges"]), traversed)
            self.assertEqual(packet["injected_copies"], 1)
        # Weight count is derived from multicast trees, not multiplied by PE count
        # at L2 injection. Outputs are still independently returned to L2.
        self.assertEqual(service.traffic_evidence["input_injected_bytes"], self.unit.input_bytes + self.unit.weight_bytes)

    def test_ring_and_mesh_use_the_same_logical_demands(self):
        ring = core_service(self.unit, self.layer, self.workload, self.hardware)
        mesh = core_service(self.unit, self.layer, self.workload, replace(self.hardware, noc_topology="mesh"))
        for field in ("unique_input_weight_bytes", "input_delivered_bytes", "output_delivered_bytes"):
            self.assertEqual(ring.traffic_evidence[field], mesh.traffic_evidence[field])
        self.assertEqual(ring.mac_cycles, mesh.mac_cycles)
        self.assertEqual(ring.mac_energy_j, mesh.mac_energy_j)
        # The mesh also multicasts once per common edge. There is no artificial
        # rule multiplying all mesh input packets by the number of PEs.
        self.assertEqual(ring.traffic_evidence["input_injected_bytes"], mesh.traffic_evidence["input_injected_bytes"])
        self.assertGreater(mesh.noc_hop_bytes, 0)

    def test_actual_link_loads_define_phase_latency_and_energy(self):
        service = core_service(self.unit, self.layer, self.workload, self.hardware)
        evidence = service.traffic_evidence
        total_hops = sum(row["bytes"] for row in evidence["input_link_loads"] + evidence["output_link_loads"])
        self.assertEqual(service.noc_hop_bytes, total_hops)
        self.assertAlmostEqual(service.noc_energy_j, total_hops * 8 * self.hardware.noc_energy_pj_per_bit * 1e-12)
        self.assertGreaterEqual(service.input_noc_s,
                                max(row["bytes"] for row in evidence["input_link_loads"]) / self.hardware.noc_bandwidth_Bps)
        self.assertAlmostEqual(service.service_s, max(service.input_buffer_s, service.input_noc_s) +
                               max(service.compute_s, service.l1_s) + max(service.output_buffer_s, service.output_noc_s))
        self.assertGreater(service.service_s, service.compute_s)
        json.dumps(service.to_dict())

    def test_shared_rings_are_shared_scheduler_resources(self):
        hardware = replace(self.hardware, ring_count=2)
        first = core_service(self.unit, self.layer, self.workload, hardware)
        second = core_service(replace(self.unit, chiplet="compute:0/core:1"), self.layer, self.workload, hardware)
        self.assertTrue(set(first.ring_resources) & set(second.ring_resources))
        third = core_service(replace(self.unit, chiplet="compute:0/core:2"), self.layer, self.workload, hardware)
        self.assertFalse(set(first.ring_resources) & set(third.ring_resources))

    def test_more_rings_require_real_reinjections(self):
        hardware = replace(self.hardware, cores_per_die=1, ring_count=4)
        package = build_package(hardware)
        plan = initial_mapping(self.workload, package.mapping_core_ids, package.memory_ids,
                               pipeline=False, max_group_layers=1)
        unit = compile_mapping(self.workload, plan, hardware).units[0]
        service = core_service(unit, self.workload.layer_map[unit.layer_id], self.workload, hardware)
        self.assertGreater(service.traffic_evidence["input_injected_bytes"],
                           service.traffic_evidence["unique_input_weight_bytes"])
        self.assertEqual(len(service.ring_resources), 4)
        self.assertTrue(any(packet["injected_copies"] > 1 for packet in service.traffic_evidence["input_packets"]))

    def test_named_bank_microtiles_preserve_macs_and_charge_ub_reload(self):
        hardware = replace(self.hardware, l1_weight_bytes=36, l1_activation_bytes=96,
                           l1_output_bytes=48, l2_weight_bytes=36,
                           l2_activation_bytes=192, l2_output_bytes=64)
        service = core_service(self.unit, self.layer, self.workload, hardware)
        evidence = service.traffic_evidence
        for bank, peak in evidence["bank_peak_bytes"].items():
            self.assertLessEqual(peak, evidence["bank_capacities_bytes"][bank])
        self.assertGreater(evidence["channel_microkernel_rounds"], 1)
        self.assertGreater(service.ub_read_bytes, self.unit.input_bytes + self.unit.weight_bytes)
        self.assertGreater(service.noc_injected_bytes,
                           self.unit.input_bytes + self.unit.weight_bytes + self.unit.output_bytes)
        self.assertEqual(evidence["output_delivered_bytes"], self.unit.output_bytes)
        self.assertEqual(evidence["partial_sum_spill_bytes"], 0)
        self.assertIn("fit OL1", evidence["partial_sum_policy"])
        self.assertAlmostEqual(service.input_noc_energy_j + service.output_noc_energy_j, service.noc_energy_j)
        self.assertEqual(service.mac_energy_j, self.unit.macs * hardware.mac_energy_pj * 1e-12)

    def test_another_bank_cannot_cover_too_small_activation_bank(self):
        hardware = replace(self.hardware, l1_activation_bytes=1, l2_activation_bytes=1,
                           unified_buffer_bytes=64 * 1024**2)
        with self.assertRaisesRegex(ValueError, "minimal internal microkernel"):
            core_service(self.unit, self.layer, self.workload, hardware)

    def test_uneven_five_positions_use_busiest_pe_l1_port(self):
        shape = (1, 16, 5, 1)
        layer = LayerSpec("conv", "Conv2d", ("image",), "out", weight_shape=(16, 16, 1, 1),
                          macs_per_sample=5 * 16 * 16)
        workload = Workload("five_positions", (TensorSpec("image", shape),
                            TensorSpec("out", shape, producer="conv")), (layer,), ("image",), ("out",))
        plan = initial_mapping(workload, ("compute:0/core:0",), ("memory:0",), pipeline=False)
        unit = compile_mapping(workload, plan, self.hardware).units[0]
        service = core_service(unit, layer, workload, self.hardware)
        bits = service.traffic_evidence["l1_compute_bits_by_pe"]
        # Three PEs own one output position, the fourth owns two. Each position
        # reads 16 activation bytes, 256 weight bytes and 96 accumulator bytes.
        self.assertEqual(sorted(bits.values()), [368 * 8] * 3 + [2 * 368 * 8])
        self.assertAlmostEqual(service.l1_s, 2 * 368 / self.hardware.l1_bandwidth_Bps)
        aggregate_average_s = sum(bits.values()) / 8 / (4 * self.hardware.l1_bandwidth_Bps)
        self.assertAlmostEqual(service.l1_s / aggregate_average_s, 2 / 1.25)
        self.assertGreater(service.l1_s, service.compute_s)
        self.assertEqual(service.mac_energy_j, workload.total_macs * self.hardware.mac_energy_pj * 1e-12)

    def test_batch_and_die_normalized_cache_matches_uncached_physics(self):
        hardware = replace(self.hardware, compute_per_cluster=2)
        workload = make_toy_workload(residual=False, batch=2, channels=8, spatial=8)
        plan = initial_mapping(workload, ("compute:0/core:0",), ("memory:0",), pipeline=False)
        units = [unit for unit in compile_mapping(workload, plan, hardware).units if unit.layer_id == "conv0"]
        layer = workload.layer_map["conv0"]
        first, second = units
        second = replace(second, chiplet="compute:1/core:0", id="other_mb", group_id="other_group",
                         dependencies=("a_scheduling_dependency",))
        core_service.cache_clear()
        reference = core_service(first, layer, workload, hardware)
        cached = core_service(second, layer, workload, hardware)
        self.assertEqual(core_service.cache_info().misses, 1)
        self.assertEqual(core_service.cache_info().hits, 1)
        raw = _cached_core_service.__wrapped__(second, layer, workload, hardware)
        for field in asdict(reference):
            if field not in ("traffic_evidence", "ring_resources"):
                self.assertEqual(getattr(reference, field), getattr(cached, field), field)
                self.assertEqual(getattr(raw, field), getattr(cached, field), field)
        self.assertEqual(cached.ring_resources, raw.ring_resources)
        self.assertTrue(all("compute:1" in resource for resource in cached.ring_resources))
        self.assertEqual(cached.traffic_evidence["batch_origin"], 1)
        self.assertEqual(cached.traffic_evidence["actual_output_bounds"], second.output_region.bounds)
        self.assertIn("unit-relative", cached.traffic_evidence["coordinate_frame"])
        for pe in cached.traffic_evidence["pe_output_regions"]:
            relative = pe["output_bounds"][0]
            restored = (relative[0] + 1, relative[1] + 1)
            self.assertEqual(restored, second.output_region.bounds[0])
        # A changed physical core slot, SRAM coefficient or C payload must miss.
        core_service(replace(second, chiplet="compute:1/core:1"), layer, workload, hardware)
        core_service(second, layer, workload, replace(hardware, sram_energy_pj_per_bit=0.9))
        self.assertEqual(core_service.cache_info().misses, 3)


if __name__ == "__main__":
    unittest.main()
