import unittest
from pathlib import Path

from simple_rapidchiplet.config import load_config
from simple_rapidchiplet.mapping import build_mapping_plan
from simple_rapidchiplet.model import ModelSpec, Stage
from simple_rapidchiplet.preference_dse import DesignState, legal_actions
from simple_rapidchiplet.mapping import build_physical_mapping, integer_channel_ranges
from simple_rapidchiplet.tensor_metadata import OperationSpec, OperatorGraph, TensorSpec, build_blocks
from simple_rapidchiplet.traffic_model import derive_operator_workload, input_requirements

ROOT = Path(__file__).resolve().parents[1]


def conv_graph(channels=(4, 4), groups=(1, 1), same_block=False):
    tensors = [TensorSpec("x", (1, 4, 2, 2), 4, "", "EXTERNAL")]
    ops = []
    for index, (cout, group) in enumerate(zip(channels, groups)):
        previous = tensors[-1]
        name = f"conv{index}"
        block = "block0" if same_block else f"block{index}"
        ops.append(OperationSpec(name, "Conv2d", block, (previous.id,), name,
                   () if index == 0 else (previous.producer,), cin=previous.channels, cout=cout,
                   kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=group,
                   weight_shape=(cout, previous.channels//group, 1, 1),
                   parameter_bytes=cout*(previous.channels//group)*4,
                   macs=4*cout*(previous.channels//group)))
        tensors.append(TensorSpec(name, (1, cout, 2, 2), 4, name))
    kinds = tuple(dict.fromkeys((op.block_id, "Toy") for op in ops))
    graph = OperatorGraph("toy", tuple(tensors), tuple(ops), build_blocks(ops,tensors,kinds),
                          "x", tensors[-1].id, sum(op.parameter_bytes for op in ops), source="explicit toy fixture")
    graph.validate()
    return graph


class Phase2Tests(unittest.TestCase):
    def test_default_actions_exclude_ic_and_spatial(self):
        actions = legal_actions(DesignState("balanced"), 2, 4)
        self.assertEqual({action.mapping_strategy for action in actions}, {"single", "output_channel"})

    def test_mapping_name_never_scales_compute(self):
        model = ModelSpec("toy", "toy", "", 1, 0, (Stage("conv", 1, 1, operators=("Conv",)),))
        for strategy in ("output_channel", "input_channel", "spatial"):
            self.assertEqual(build_mapping_plan(model, ((0, 1, 3),), (strategy,)).groups[0].compute_efficiency, 1)


class ReferenceGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from simple_rapidchiplet.model_parser import extract_reference_graph
        cls.graph = extract_reference_graph(load_config(ROOT / "configs/defaults.json").reference_model)

    def test_actual_fx_hierarchy_has_18_semantic_blocks(self):
        self.assertEqual(len(self.graph.blocks), 18)
        self.assertEqual([block.kind for block in self.graph.blocks].count("Bottleneck"), 16)
        self.assertEqual(self.graph.tensor_map[self.graph.input_id].shape, (1, 3, 224, 224))
        self.assertEqual(self.graph.tensor_map[self.graph.output_id].shape, (1, 1000))

    def test_dependencies_include_add_and_projection(self):
        adds = [op for op in self.graph.operators if op.kind == "Add"]
        self.assertEqual(len(adds), 16)
        self.assertTrue(all(len(op.inputs) == 2 for op in adds))
        self.assertEqual(sum("downsample.0" in op.target for op in self.graph.operators), 4)

    def test_v15_stride_is_on_second_conv(self):
        op = next(op for op in self.graph.operators if op.target == "layer2.0.conv2")
        self.assertEqual(op.stride, (2, 2))
        self.assertEqual(next(op for op in self.graph.operators if op.target == "layer2.0.conv1").stride, (1, 1))

    def test_graph_count_and_precision_are_derived(self):
        self.assertEqual(sum(op.kind == "Conv2d" for op in self.graph.operators), 53)
        self.assertEqual(sum(op.macs for op in self.graph.operators), 4_089_184_256)
        self.assertEqual(self.graph.weight_bytes, 25_557_032 * 4)
        self.assertTrue(all(op.macs == 0 for op in self.graph.operators if op.fused))
        self.assertTrue(all(tensor.bytes_per_element == 4 for tensor in self.graph.tensors))

    def test_fingerprint_is_reproducible_in_another_process(self):
        import subprocess
        import sys
        code = "from simple_rapidchiplet.config import load_config; from simple_rapidchiplet.model_parser import extract_reference_graph; print(extract_reference_graph(load_config('configs/defaults.json').reference_model).fingerprint)"
        other = subprocess.check_output([sys.executable,"-c",code],cwd=ROOT,text=True).strip()
        self.assertEqual(self.graph.fingerprint,other)
        self.assertFalse(any("0x" in op.target for op in self.graph.operators))


class IntegerShardTests(unittest.TestCase):
    def test_nondivisible_k_has_natural_work_imbalance(self):
        graph = conv_graph(channels=(10,), groups=(1,))
        plan = build_physical_mapping(graph, 4, ((0, 1, 2, 3),))
        shards = plan.block_mappings[0].operator_shards[0][1]
        self.assertEqual([shard.channels for shard in shards], [3, 3, 2, 2])
        self.assertEqual([shard.macs for shard in shards], [48, 48, 32, 32])
        self.assertEqual(sum(shard.macs for shard in shards), graph.operators[0].macs)
        self.assertEqual(sum(shard.weight_bytes for shard in shards), graph.operators[0].parameter_bytes)

    def test_physical_ids_can_be_reused_across_blocks(self):
        plan = build_physical_mapping(conv_graph(), 4, ((0, 3), (0, 3)))
        self.assertEqual(plan.total_chiplets, 4)
        self.assertEqual([block.active_chiplet_count for block in plan.block_mappings], [2, 2])
        self.assertEqual(plan.block_mappings[0].active_chiplets, plan.block_mappings[1].active_chiplets)

    def test_invalid_or_empty_shards_are_rejected(self):
        for assignment in ((), (0, 0), (4,), (-1,), (0, 1, 2, 3, 4)):
            with self.assertRaises(ValueError):
                build_physical_mapping(conv_graph(channels=(4,), groups=(1,)), 4, (assignment,))
        self.assertEqual(integer_channel_ranges(4, 2), ((0, 2), (2, 4)))


def branch_graph(join_blocks=False):
    x = TensorSpec("x", (1,4,2,2), 4, "", "EXTERNAL")
    tensors = [x]
    ops = []
    description = (("producer", "a", "x"), ("main", "b", "producer"),
                   ("shortcut", "c" if join_blocks else "b", "x" if join_blocks else "producer"))
    for name,block,source in description:
        ops.append(OperationSpec(name,"Conv2d",block,(source,),name,
                   () if source == "x" else (source,),cin=4,cout=4,kernel=(1,1),
                   stride=(1,1),padding=(0,0),weight_shape=(4,4,1,1),parameter_bytes=64,macs=64))
        tensors.append(TensorSpec(name,(1,4,2,2),4,name))
    ops.append(OperationSpec("add","Add","d" if join_blocks else "b",("main","shortcut"),"add",("main","shortcut")))
    tensors.append(TensorSpec("add",(1,4,2,2),4,"add"))
    kinds = tuple(dict.fromkeys((op.block_id,"Toy") for op in ops))
    graph = OperatorGraph("residual_toy",tuple(tensors),tuple(ops),build_blocks(ops,tensors,kinds),"x","add",192,source="explicit toy fixture")
    graph.validate()
    return graph


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/defaults.json")

    def derive(self, graph, assignments, total=2):
        return derive_operator_workload(graph,build_physical_mapping(graph,total,assignments),self.cfg)

    def test_hand_calculated_two_chiplet_all_gather(self):
        derived = self.derive(conv_graph(),((0,1),(0,1)))
        flows = derived.workload.flows
        self.assertEqual([(flow.src,flow.dst,flow.bytes) for flow in flows],[(1,0,32),(0,1,32)])
        self.assertEqual([(flow.channel_start,flow.channel_end) for flow in flows],[(2,4),(0,2)])
        self.assertTrue(all(flow.tensor_id == "conv0" for flow in flows))

    def test_intra_block_communication_is_not_omitted(self):
        derived = self.derive(conv_graph(same_block=True),((0,1),))
        self.assertEqual(sum(flow.bytes for flow in derived.workload.flows),64)
        self.assertEqual({flow.kind for flow in derived.workload.flows},{"intra_block"})

    def test_grouped_and_depthwise_read_only_relevant_inputs(self):
        for group,cout in ((2,4),(4,4),(4,8)):
            graph = conv_graph(channels=(4,cout),groups=(1,group))
            derived = self.derive(graph,((0,1),(0,1)))
            self.assertEqual(derived.workload.flows,())
        graph = conv_graph(groups=(1,2))
        swapped = self.derive(graph,((0,1),(1,0)))
        self.assertEqual(sum(flow.bytes for flow in swapped.workload.flows),64)
        self.assertEqual(input_requirements(graph.operators[1],0,2,graph.tensors[1]),((0,2),))

    def test_residual_add_is_local_when_ownership_matches(self):
        graph = branch_graph(join_blocks=True)
        derived = self.derive(graph,((0,1),(0,1),(0,1),(0,1)))
        self.assertFalse(any(flow.consumer_operator == "add" for flow in derived.workload.flows))
        changed = self.derive(graph,((0,1),(0,1),(1,0),(0,1)))
        transfers = [flow for flow in changed.workload.flows if flow.consumer_operator == "add"]
        self.assertEqual(sum(flow.bytes for flow in transfers),64)
        self.assertEqual({flow.tensor_id for flow in transfers},{"shortcut"})

    def test_main_and_projection_deduplicate_same_resident_tensor(self):
        derived = self.derive(branch_graph(),((0,1),(0,1)))
        self.assertEqual(sum(flow.bytes for flow in derived.workload.flows),64)
        self.assertEqual({flow.consumer_operator for flow in derived.workload.flows},{"main"})
        self.assertFalse(any(flow.consumer_operator in ("shortcut","add") for flow in derived.workload.flows))

    def test_logical_fused_versions_are_distinct(self):
        graph = conv_graph()
        conv0,conv1 = graph.operators
        from dataclasses import replace
        bn = OperationSpec("bn","BatchNorm2d","block0",("conv0",),"bn",("conv0",),fused=True)
        relu = OperationSpec("relu","ReLU","block0",("bn",),"relu",("bn",),fused=True)
        second = replace(conv1,inputs=("relu",),dependencies=("relu",))
        third = replace(conv1,id="another",inputs=("conv0",),output="another",dependencies=("conv0",))
        tensors = (graph.tensors[0],graph.tensors[1],TensorSpec("bn",(1,4,2,2),4,"bn"),
                   TensorSpec("relu",(1,4,2,2),4,"relu"),graph.tensors[2],TensorSpec("another",(1,4,2,2),4,"another"))
        ops = (conv0,bn,relu,second,third)
        graph = replace(graph,tensors=tensors,operators=ops,blocks=build_blocks(ops,tensors,(("block0","Toy"),("block1","Toy"))),output_id="another")
        derived = self.derive(graph,((0,1),(0,1)))
        self.assertEqual({flow.tensor_id for flow in derived.workload.flows},{"conv0","relu"})
        self.assertEqual(sum(flow.bytes for flow in derived.workload.flows),128)
        self.assertFalse(any(flow.consumer_operator in ("bn","relu") for flow in derived.workload.flows))

    def test_single_chiplet_and_external_sources_have_no_mesh_traffic(self):
        derived = self.derive(branch_graph(),((0,),(0,)),total=1)
        self.assertEqual(derived.workload.flows,())
        self.assertTrue(derived.final_residency)
        self.assertFalse(derived.violations)

    def test_resident_data_is_bounded_by_sram_and_liveness(self):
        from dataclasses import replace
        cfg = replace(self.cfg,chiplet=replace(self.cfg.chiplet,sram_mb=0.000001))
        graph = conv_graph()
        result = derive_operator_workload(graph,build_physical_mapping(graph,2,((0,1),(0,1))),cfg)
        self.assertTrue(result.violations)
        self.assertEqual({item[0] for item in result.final_residency},{"conv1"})

    def test_flows_and_shards_are_deterministic(self):
        graph = conv_graph()
        self.assertEqual(self.derive(graph,((0,1),(0,1))),self.derive(graph,((0,1),(0,1))))


if __name__ == "__main__":
    unittest.main()
