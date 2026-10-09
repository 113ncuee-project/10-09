"""Hand-computed ownership, traffic and pipeline contracts for the hybrid IR."""
from dataclasses import replace
import unittest

from simple_rapidchiplet.hybrid.communication import build_communication
from simple_rapidchiplet.hybrid.hardware import HardwareSpec
from simple_rapidchiplet.hybrid.mapping import (LayerGroup, LayerMapping, MappingPlan, Partition,
    TensorRegion, compile_mapping, initial_mapping, required_input_regions, validate_plan)
from simple_rapidchiplet.hybrid.workload import LayerSpec, TensorSpec, Workload, from_operator_graph, make_toy_workload


class HybridMappingTests(unittest.TestCase):
    def setUp(self):
        self.cores = tuple(f"compute:0/core:{index}" for index in range(4))

    def test_integer_shards_cover_macs_and_both_residual_operands(self):
        workload = make_toy_workload(batch=4, channels=7, spatial=5)
        plan = initial_mapping(workload, self.cores, microbatch_size=2)
        compiled = compile_mapping(workload, plan)
        self.assertEqual(sum(unit.macs for unit in compiled.units), workload.total_macs)
        for unit in compiled.units:
            if unit.layer_id == "add":
                self.assertEqual({req.tensor_id for req in unit.inputs}, {"image", "conv1"})
                self.assertTrue(unit.dependencies)
        for layer in workload.layers:
            regions = [unit.output_region for unit in compiled.units if unit.layer_id == layer.id]
            self.assertEqual(sum(region.elements for region in regions), workload.tensor_map[layer.output].elements)
            for left_index, left in enumerate(regions):
                self.assertTrue(all(left.intersection(right) is None for right in regions[left_index + 1:]))

    def test_halo_stride_padding_dilation_and_groups(self):
        tensors = (TensorSpec("x", (1, 8, 9, 9)), TensorSpec("y", (1, 12, 5, 5), producer="conv"))
        layer = LayerSpec("conv", "Conv2d", ("x",), "y", kernel=(3, 3), stride=(2, 2),
                          padding=(2, 2), dilation=(2, 2), groups=4, weight_shape=(12, 2, 3, 3),
                          macs_per_sample=12 * 5 * 5 * 2 * 9)
        workload = Workload("grouped", tensors, (layer,), ("x",), ("y",))
        workload.validate()
        output = TensorRegion(((0, 1), (3, 6), (1, 3), (0, 2)))
        request = required_input_regions(workload, layer, output)[0]
        self.assertEqual(request.region.bounds, ((0, 1), (2, 4), (0, 7), (0, 5)))

    def test_multicast_shared_input_and_spatial_weight_reuse(self):
        workload = make_toy_workload(residual=False, batch=1, channels=4, spatial=2)
        # Both layers stay in one temporal group so cross-layer K all-gather is visible.
        plan = initial_mapping(workload, self.cores, microbatch_size=1, max_group_layers=2, pipeline=False)
        compiled = compile_mapping(workload, plan)
        unicast = build_communication(workload, compiled, "unicast")
        multicast = build_communication(workload, compiled, "multicast")
        image_reads = [transfer for transfer in multicast.transfers if transfer.tensor == "image"]
        self.assertEqual(sum(transfer.bytes for transfer in image_reads), 16)
        self.assertEqual(len(image_reads), 1)
        self.assertEqual(len(image_reads[0].dsts), 4)
        self.assertGreater(unicast.dram_read_bytes, multicast.dram_read_bytes)
        # Four channel shards, each one channel of size 2x2; three other cores receive it.
        self.assertEqual(sum(transfer.bytes for transfer in unicast.transfers if transfer.kind == "activation"), 48)
        self.assertEqual(sum(transfer.bytes for transfer in multicast.transfers if transfer.kind == "activation"), 16)

    def test_boundary_read_follows_producer_write_and_explicit_dependency(self):
        workload = make_toy_workload(residual=False, batch=1, channels=4, spatial=2)
        plan = initial_mapping(workload, self.cores, ("memory:0", "memory:1"), max_group_layers=1)
        first = replace(plan.mapping_map["conv0"], output_dram="memory:1")
        second = replace(plan.mapping_map["conv1"], input_dram="memory:0")
        plan = plan.replace_layer(first).replace_layer(second)
        comm = build_communication(workload, compile_mapping(workload, plan))
        reads = [transfer for transfer in comm.transfers if transfer.kind == "inter_group_read"]
        self.assertTrue(reads)
        for transfer in reads:
            self.assertEqual(transfer.src, "memory:1")
            self.assertTrue(transfer.dependencies)
            self.assertTrue(all(comm.transfer_map[name].kind == "inter_group_write" for name in transfer.dependencies))
        self.assertEqual(comm.dram_write_bytes, 32)  # conv0 boundary + final conv1.

    def test_ring_chunks_have_real_forwarding_dependencies(self):
        workload = make_toy_workload(residual=False, batch=1, channels=4, spatial=2)
        plan = initial_mapping(workload, self.cores, pipeline=False)
        compiled = compile_mapping(workload, plan)
        comm = build_communication(workload, compiled, "ring")
        forwards = [transfer for transfer in comm.transfers if transfer.kind == "ring_forward"]
        self.assertEqual(sum(transfer.bytes for transfer in forwards), 48)
        with_dependencies = [transfer for transfer in forwards if transfer.dependencies]
        self.assertEqual(len(with_dependencies), 8)
        for transfer in with_dependencies:
            prior = comm.transfer_map[transfer.dependencies[0]]
            self.assertEqual(prior.region, transfer.region)
            self.assertEqual(prior.dsts, (transfer.src,))

    def test_temporal_channel_tile_changes_footprint_but_preserves_traffic(self):
        workload = make_toy_workload(residual=False, batch=1, channels=32, spatial=4)
        plan = initial_mapping(workload, self.cores, pipeline=False)
        full = compile_mapping(workload, plan)
        streamed = compile_mapping(workload, replace(plan, layer_mappings=tuple(
            replace(mapping, channel_tile=4) for mapping in plan.layer_mappings)))
        self.assertEqual(sum(unit.macs for unit in full.units), sum(unit.macs for unit in streamed.units))
        self.assertEqual(sum(unit.weight_bytes for unit in full.units), sum(unit.weight_bytes for unit in streamed.units))
        self.assertEqual(sum(unit.input_bytes for unit in full.units), sum(unit.input_bytes for unit in streamed.units))
        self.assertLess(max(unit.buffer_bytes for unit in streamed.units), max(unit.buffer_bytes for unit in full.units))

    def test_spatial_tiles_reload_weights_instead_of_free_persistent_cache(self):
        workload = make_toy_workload(residual=False, batch=1, channels=4, spatial=4)
        plan = initial_mapping(workload, self.cores, pipeline=False)
        tiled = plan.replace_layer(replace(plan.mapping_map["conv0"], tile_shape=(1, 0, 2, 2)))
        a = build_communication(workload, compile_mapping(workload, plan))
        b = build_communication(workload, compile_mapping(workload, tiled))
        read = lambda comm: sum(transfer.bytes for transfer in comm.transfers if transfer.tensor == "weights:conv0")
        self.assertEqual(read(b), 4 * read(a))

    def test_hardware_adaptive_tiling_obeys_core_quota_without_clipping_macs(self):
        workload = make_toy_workload(residual=False, batch=1, channels=32, spatial=64)
        plan = initial_mapping(workload, self.cores, pipeline=False)
        spec = replace(HardwareSpec(), l1_weight_bytes=1024, l1_activation_bytes=1024,
                       l1_output_bytes=1024, l2_weight_bytes=4096, l2_activation_bytes=4096,
                       l2_output_bytes=4096)
        compiled = compile_mapping(workload, plan, spec)
        quota = spec.pes_per_core * spec.l1_bytes_per_pe + spec.l2_bytes_per_die // spec.cores_per_die
        self.assertTrue(all(unit.buffer_bytes <= quota for unit in compiled.units))
        self.assertEqual(sum(unit.macs for unit in compiled.units), workload.total_macs)
        self.assertGreater(len(compiled.units), 8)

    def test_overlapping_spatial_stages_are_rejected(self):
        workload = make_toy_workload(residual=False, batch=1)
        mappings = tuple(LayerMapping(layer.id, (self.cores[0],)) for layer in workload.layers)
        plan = MappingPlan((LayerGroup("g", ("conv0", "conv1")),), mappings)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_plan(workload, plan)

    def test_interleaved_external_tensor_has_one_consistent_global_placement(self):
        workload = make_toy_workload(residual=False, batch=1, channels=2, spatial=8)
        first = LayerMapping("conv0", self.cores[:2], Partition(h=2), input_dram="interleave")
        second = LayerMapping("conv1", self.cores[2:], Partition(k=2))
        plan = MappingPlan((LayerGroup("g", ("conv0", "conv1")),), (first, second),
                           memory_ids=("memory:0", "memory:1"))
        comm = build_communication(workload, compile_mapping(workload, plan))
        for transfer in comm.transfers:
            if transfer.tensor == "image":
                lo, hi = transfer.region.bounds[2]
                self.assertTrue((transfer.src == "memory:0" and hi <= 4) or
                                (transfer.src == "memory:1" and lo >= 4))

    def test_postop_fusion_reconnects_edges_preserves_macs_and_vector_cost(self):
        from simple_rapidchiplet.toy_model import make_toy_graph
        from simple_rapidchiplet.tensor_metadata import OperationSpec as OldOp, TensorSpec as OldTensor
        graph = make_toy_graph(block_count=2, channels=4)
        bn = OldOp("bn", "BatchNorm2d", "block0", ("conv0",), "bn", ("conv0",), fused=True)
        relu = OldOp("relu", "ReLU", "block0", ("bn",), "relu", ("bn",), fused=True)
        last = replace(graph.operators[1], inputs=("relu",), dependencies=("relu",))
        graph = replace(graph, operators=(graph.operators[0], bn, relu, last),
                        tensors=(*graph.tensors[:2], OldTensor("bn", (1, 4, 2, 2), 4, "bn"),
                                 OldTensor("relu", (1, 4, 2, 2), 4, "relu"), graph.tensors[-1]))
        raw = from_operator_graph(graph, fuse_postops=False)
        fused = from_operator_graph(graph)
        self.assertEqual(len(raw.layers), 4)
        self.assertEqual(len(fused.layers), 2)
        self.assertEqual(raw.total_macs, fused.total_macs)
        self.assertEqual(fused.layers[0].output, "relu")
        self.assertEqual(fused.tensor_map["relu"].producer, "conv0")
        self.assertEqual(fused.layers[0].vector_ops_per_element, 3)
        self.assertEqual(fused.layers[1].inputs, ("relu",))


if __name__ == "__main__":
    unittest.main()
