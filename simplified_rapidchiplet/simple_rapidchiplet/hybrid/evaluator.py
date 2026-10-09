"""One physical oracle for hardware, pipeline mapping, transfers and energy."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path

from .communication import build_communication
from .reuse import apply_weight_reuse, dram_residency
from .hardware import HardwareSpec, build_package, physical_endpoint
from .mapping import compile_mapping, validate_plan
from .ondie import core_service
from .rapid_backend import DEFAULT_RAPID_ROOT, evaluate_package
from .scheduler import Job, peak_interval_bytes, schedule_jobs
from .costs import sram_access_energy


@dataclass(frozen=True)
class HybridEvaluation:
    metrics: dict
    compiled: object
    communication: object
    schedule: object
    context: object
    core_services: dict
    transfer_statistics: tuple[dict, ...]
    buffer_statistics: dict

    def to_dict(self):
        if self.compiled is None:
            return {"metrics": self.metrics, "network": self.context.to_dict(),
                    "buffer_statistics": self.buffer_statistics}
        return {"metrics": self.metrics, "mapping": self.compiled.plan.to_dict(),
                "workload": self.compiled.workload.to_dict(),
                "units": [asdict(unit) for unit in self.compiled.units],
                "transfers": [asdict(transfer) for transfer in self.communication.transfers],
                "events": [event.to_dict() for event in self.schedule.events],
                "network": self.context.to_dict(), "buffer_statistics": self.buffer_statistics,
                "core_services": self.core_services,
                "transfer_statistics": self.transfer_statistics}


def _transfer_service(transfer, hardware, context):
    loads = {}
    endpoints = tuple(dict.fromkeys(physical_endpoint(destination) for destination in transfer.dsts))
    paths = [context.route(transfer.src, destination) for destination in endpoints]
    # A region's multicast visits each directed physical edge once. Multiple
    # destination cores in one die share the NoP copy, then use the local rings.
    for path in paths:
        for edge in zip(path, path[1:]):
            loads[edge] = transfer.bytes
    resources, services = [], []
    nop_energy = 0.0
    for (a, b), amount in loads.items():
        resources.append(f"link:{a}>{b}")
        services.append(amount/context.link_bandwidth_Bps[(a, b)])
        nop_energy += amount*8*context.link_energy_pj_per_bit[(a, b)]*1e-12
    memory_endpoints = {endpoint for path in paths for endpoint in path if endpoint.startswith("memory:")}
    ios = {endpoint for path in paths for endpoint in path if endpoint.startswith("io:")}
    for memory in sorted(memory_endpoints):
        resources.append(f"{memory}:controller")
        services.append(transfer.bytes/hardware.dram_bandwidth_Bps)
    for io in sorted(ios):
        resources.append(f"{io}:crossbar")
        # One crossbar replication cycle can drive multiple physical ports;
        # aggregate capacity charges all outgoing copies at a branching IO.
        fanout = max(1, sum(a == io for a, b in loads))
        services.append(transfer.bytes*fanout/hardware.io_crossbar_bandwidth_Bps)
    compute_ends = {endpoint for path in paths for endpoint in (path[0], path[-1]) if endpoint.startswith("compute:")}
    local_copies = len({dst for dst in transfer.dsts if dst.startswith('compute:')})
    local_bits = transfer.bytes*8*(local_copies + int(transfer.src.startswith("compute:")))
    for die in sorted(compute_ends):
        resources.append(f"{die}:dma")
        copies = sum(physical_endpoint(destination) == die for destination in transfer.dsts)
        copies += int(physical_endpoint(transfer.src) == die)
        services.append(transfer.bytes*max(1, copies)/hardware.unified_bandwidth_Bps)
    # Core-to-core in one die bypasses package routing but uses the shared GLB
    # fabric. Its endpoint rings handle PE traffic in the subsequent core stage.
    local_energy = local_bits*hardware.noc_energy_pj_per_bit*1e-12
    sram_energy, sram_ledger = sram_access_energy(hardware,{'UB':{
        'read_bits':transfer.bytes*8*int(transfer.src.startswith('compute:')),
        'write_bits':transfer.bytes*8*local_copies}})
    dram_energy = len(memory_endpoints)*transfer.bytes*8*hardware.dram_energy_pj_per_bit*1e-12
    crossbar_energy = sum(transfer.bytes*8*max(1, sum(a == io for a, b in loads)) for io in ios)*hardware.io_crossbar_energy_pj_per_bit*1e-12
    path_s = max((context.latency_ns(transfer.src, endpoint) for endpoint in endpoints), default=0.0)*1e-9
    if memory_endpoints:
        path_s += hardware.dram_access_latency_ns*1e-9
    duration = max(services, default=0.0) + path_s
    return duration, tuple(dict.fromkeys(resources)), {
        "id": transfer.id, "kind": transfer.kind, "injected_bytes": transfer.bytes,
        "nop_hop_bytes": sum(loads.values()), "local_delivered_bytes": transfer.bytes*local_copies,
        "link_load_bytes": [{"source": a, "destination": b, "bytes": amount} for (a, b), amount in sorted(loads.items())],
        "dram_bytes": len(memory_endpoints)*transfer.bytes,
        "nop_energy_j": nop_energy, "dram_energy_j": dram_energy,
        "noc_energy_j": local_energy, "sram_energy_j": sram_energy,
        "sram_access_ledger": sram_ledger,
        "crossbar_energy_j": crossbar_energy,
        "energy_j": nop_energy+dram_energy+local_energy+sram_energy+crossbar_energy,
        "duration_s": duration, "path_s": path_s,
    }


class HybridEvaluator:
    def __init__(self, workload, hardware=None, *, rapid_root=DEFAULT_RAPID_ROOT,
                 max_inflight_microbatches=2, trace=True):
        self.workload = workload
        self.hardware = hardware or HardwareSpec()
        workload.validate()
        if any(tensor.bits != self.hardware.activation_bits for tensor in workload.tensors) or any(
                layer.weight_shape and layer.weight_bits != self.hardware.weight_bits for layer in workload.layers):
            raise ValueError("workload precision differs from the hardware INT8 cost profile")
        self.package = build_package(self.hardware)
        self.context = evaluate_package(self.hardware, self.package, rapid_root=rapid_root)
        if max_inflight_microbatches < 1:
            raise ValueError("pipeline window must be positive")
        self.max_inflight = max_inflight_microbatches
        self.trace = trace
        self.cache = {}
        self.cache_hits = 0
        self.evaluations = 0
        source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in Path(__file__).parent.glob("*.py")}
        self.source_hashes = source_hashes
        self.identity = hashlib.sha256(json.dumps({"workload":workload.fingerprint,
               "hardware":asdict(self.hardware), "native":self.context.provenance,
               "window":self.max_inflight, "implementation":source_hashes},sort_keys=True).encode()).hexdigest()

    def evaluate(self, plan, *, retain_trace=None):
        retain_trace = self.trace if retain_trace is None else retain_trace
        key = hashlib.sha256(json.dumps(plan.to_dict(),sort_keys=True).encode()).hexdigest()
        if key in self.cache and (not retain_trace or self.cache[key].compiled is not None):
            self.cache_hits += 1
            return self.cache[key]
        validate_plan(self.workload, plan, self.package.mapping_core_ids)
        if tuple(plan.memory_ids) != tuple(self.package.memory_ids):
            raise ValueError("mapping DRAM endpoints differ from the physical hardware")
        compiled = compile_mapping(self.workload, plan, self.hardware)
        communication = build_communication(self.workload, compiled, plan.collective)
        communication, reuse_statistics = apply_weight_reuse(compiled, communication)
        units = compiled.unit_map
        dependency_map = communication.dependency_map
        group_index = {group.id:index for index,group in enumerate(plan.groups)}
        layer_index = {layer.id:index for index,layer in enumerate(self.workload.layers)}
        jobs, services, transfer_stats = [], {}, []
        group_members = defaultdict(list)
        group_mb_members = defaultdict(list)
        previous_core_unit, core_predecessor = {}, {}
        for unit in sorted(compiled.units, key=lambda unit:(group_index[unit.group_id],unit.microbatch,
                                  layer_index[unit.layer_id],unit.shard_index,unit.tile_index)):
            if unit.chiplet in previous_core_unit:
                core_predecessor[unit.id] = previous_core_unit[unit.chiplet]
            previous_core_unit[unit.chiplet] = unit.id
        def group_dependencies(group_id, microbatch):
            index = group_index[group_id]
            result = [f"group_done:{plan.groups[index-1].id}"] if index else []
            if microbatch >= self.max_inflight:
                result.append(f"mb_done:{group_id}:{microbatch-self.max_inflight}")
            return result
        # Group/layer traversal keeps identical microbatch kernels adjacent and
        # avoids LRU thrashing; scheduling priorities retain the original order.
        for unit in sorted(compiled.units, key=lambda u:(group_index[u.group_id],layer_index[u.layer_id],u.microbatch,u.shard_index,u.tile_index)):
            layer = self.workload.layer_map[unit.layer_id]
            service = core_service(unit, layer, self.workload, self.hardware)
            services[unit.id] = service.to_dict() if retain_trace else {
                field.name:getattr(service,field.name) for field in fields(service)
                if field.name != "traffic_evidence"}
            die = unit.physical_die_id
            slot = int(unit.chiplet.rsplit(":",1)[1]) if "/core:" in unit.chiplet else 0
            ring = slot % self.hardware.ring_count
            priority = (group_index[unit.group_id],unit.microbatch,layer_index[unit.layer_id],unit.shard_index,unit.tile_index)
            dependencies = list(dependency_map.get(unit.id, unit.dependencies))
            dependencies.extend(group_dependencies(unit.group_id,unit.microbatch))
            predecessor = core_predecessor.get(unit.id)
            compute_dependencies = [f"read:{unit.id}"]
            if predecessor:
                if self.hardware.prefetch_slots == 1:
                    dependencies.append(predecessor)
                else:
                    # A bank may be reused only after the unit two positions
                    # earlier has drained. Compute remains serialized per core.
                    bank_owner = core_predecessor.get(predecessor)
                    if bank_owner:
                        dependencies.append(bank_owner)
                    compute_dependencies.append(f"compute:{predecessor}")
            input_s = getattr(service,"input_noc_s",service.noc_s*0.8)
            input_buffer_s = getattr(service,"input_buffer_s",service.buffer_s*0.8)
            output_s = getattr(service,"output_noc_s",service.noc_s*0.2)
            output_buffer_s = getattr(service,"output_buffer_s",service.buffer_s*0.2)
            l1_s = getattr(service,"l1_s",0.0)
            noc_resources = tuple(getattr(service,"ring_resources", (f"{die}:ring:{ring}",)))
            input_energy = getattr(service,"input_noc_energy_j", service.noc_energy_j*0.8)
            output_energy = getattr(service,"output_noc_energy_j", service.noc_energy_j*0.2)
            jobs.append(Job(f"read:{unit.id}",tuple(dict.fromkeys(dependencies)),
                        (f"{die}:l2_read",f"{die}:ub_read")+noc_resources,
                        max(input_s,input_buffer_s),"core_read",input_energy,priority,
                        {"unit_id":unit.id,"core":unit.chiplet}))
            jobs.append(Job(f"compute:{unit.id}",tuple(compute_dependencies),
                        (f"core:{unit.chiplet}",),max(service.compute_s,l1_s),"compute",
                        service.mac_energy_j+service.sram_energy_j,priority,{"unit_id":unit.id,"macs":unit.macs}))
            jobs.append(Job(unit.id,(f"compute:{unit.id}",),
                        (f"{die}:l2_write",f"{die}:ub_write")+noc_resources,
                        max(output_s,output_buffer_s),"core_write",output_energy,priority,
                        {"unit_id":unit.id,"core":unit.chiplet}))
            group_members[unit.group_id].append(unit.id)
            group_mb_members[(unit.group_id,unit.microbatch)].append(unit.id)
        for transfer in communication.transfers:
            duration, resources, stats = _transfer_service(transfer,self.hardware,self.context)
            transfer_stats.append(stats)
            source_units = [units[name] for name in transfer.producer_units]
            consumer_units = [units[name] for name in transfer.consumer_units]
            # Read jobs belong to the consuming group; writes to the producer.
            reference = consumer_units[0] if consumer_units else source_units[0] if source_units else None
            if reference is None:
                raise ValueError("transfer lacks a workload owner")
            dependencies = list(transfer.producer_units)+list(transfer.dependencies)
            dependencies.extend(group_dependencies(reference.group_id,transfer.microbatch))
            priority = (group_index[reference.group_id],transfer.microbatch,
                        layer_index[reference.layer_id],-1,0)
            jobs.append(Job(transfer.id,tuple(dict.fromkeys(dependencies)),resources,duration,
                            transfer.kind,stats["energy_j"],priority,{"transfer_id":transfer.id}))
            group_members[reference.group_id].append(transfer.id)
            group_mb_members[(reference.group_id,transfer.microbatch)].append(transfer.id)
        for group in plan.groups:
            for microbatch in range(compiled.microbatch_count):
                jobs.append(Job(f"mb_done:{group.id}:{microbatch}",tuple(group_mb_members[(group.id,microbatch)]),(),0,"barrier"))
            jobs.append(Job(f"group_done:{group.id}",tuple(group_members[group.id]),(),0,"barrier"))
        schedule = schedule_jobs(jobs)
        buffers = self._buffers(compiled,communication,schedule)
        memory = dram_residency(compiled,communication,schedule,self.hardware)
        buffers["dram"] = memory
        buffers["violations"].extend(memory["violations"])
        dynamic = sum(job.energy_j for job in jobs)
        static = self.context.idle_power_w*schedule.makespan_s
        energy = dynamic+static
        output_writes = [event for event in schedule.events if event.job.kind == "output_write"]
        first_finish = max((event.finish_s for event in output_writes if communication.transfer_map[event.job.id].microbatch == 0),default=schedule.makespan_s)
        area = dict(self.context.area_breakdown)
        violations = buffers["violations"]
        raw_macs = sum(unit.macs for unit in compiled.units)
        network_busy = max((duration for resource,duration in schedule.resource_busy_s.items()
                            if resource.startswith("link:")), default=0.0)
        energy_components = {
            "mac_j":sum(service["mac_energy_j"] for service in services.values()),
            "on_die_sram_j":sum(service["sram_energy_j"] for service in services.values()),
            "on_die_noc_j":sum(service["noc_energy_j"] for service in services.values()),
            "nop_j":sum(row["nop_energy_j"] for row in transfer_stats),
            "dram_j":sum(row["dram_energy_j"] for row in transfer_stats),
            "glb_fabric_j":sum(row["noc_energy_j"] for row in transfer_stats),
            "glb_sram_j":sum(row["sram_energy_j"] for row in transfer_stats),
            "io_crossbar_j":sum(row["crossbar_energy_j"] for row in transfer_stats),
            "idle_j":static,
        }
        if not math.isclose(sum(energy_components.values()),energy,rel_tol=1e-10,abs_tol=1e-18):
            raise ValueError("energy component ledger disagrees with scheduled jobs")
        metrics = {
            "model":self.workload.name,"workload_fingerprint":self.workload.fingerprint,
            "evaluator_identity":self.identity,"backend":self.context.backend,
            "oracle_source_sha256":self.source_hashes,
            "feasible":not violations,"violations":violations,
            "latency_s":schedule.makespan_s,"batch_latency_s":schedule.makespan_s,
            "first_result_latency_s":first_finish,"batch_size":self.workload.batch_size,
            "throughput_samples_s":self.workload.batch_size/schedule.makespan_s if schedule.makespan_s else 0,
            "energy_j":energy,"energy_per_sample_j":energy/self.workload.batch_size,
            "dynamic_energy_j":dynamic,"static_energy_j":static,
            "energy_components":energy_components,
            "power_w":energy/schedule.makespan_s if schedule.makespan_s else self.context.idle_power_w,
            "idle_power_w":self.context.idle_power_w,
            "area_mm2":area.get("total_silicon_mm2",area.get("total_chiplet_area_mm2",0)),
            "package_area_mm2":area.get("package_footprint_mm2",0),"area_breakdown":area,
            "edp_js":energy*schedule.makespan_s,"total_macs":raw_macs,
            "microbatch_size":plan.microbatch_size,"microbatch_count":compiled.microbatch_count,
            "max_inflight_microbatches":self.max_inflight,
            "prefetch_slots":self.hardware.prefetch_slots,
            "group_count":len(plan.groups),"unit_count":len(compiled.units),
            "transfer_count":len(communication.transfers),"logical_transfer_bytes":communication.total_bytes,
            "weight_reuse":reuse_statistics,
            "dram_read_bytes":communication.dram_read_bytes,"dram_write_bytes":communication.dram_write_bytes,
            "nop_hop_bytes":sum(row["nop_hop_bytes"] for row in transfer_stats),
            "noc_hop_bytes":sum(row["noc_hop_bytes"] for row in services.values()),
            "compute_utilization":raw_macs/(self.hardware.macs_per_second*self.hardware.compute_count*schedule.makespan_s) if schedule.makespan_s else 0,
            "network_busy_fraction":network_busy/schedule.makespan_s if schedule.makespan_s else 0,
            "compute_busy_s":sum(event.finish_s-event.start_s for event in schedule.events if event.job.kind == "compute"),
            "resource_busy_s":schedule.resource_busy_s,"buffer_peak_bytes":buffers["peak_by_die"],
            "calibration_status":self.hardware.calibration_status,
            "cost_profile":self.hardware.cost_profile,
            "hardware_cost_provenance":self.hardware.provenance,
            "sram_bank_costs":self.hardware.sram_costs,
            "physical_limit_certified":False,
            "timing_model":"resource calendar; read/compute/write stages; atomic analytical multicast route service; finite pipeline window",
        }
        result = HybridEvaluation(metrics,compiled,communication,schedule,self.context,services,
                                  tuple(transfer_stats),buffers)
        # Search needs scalar metrics, not hundreds of thousands of historical
        # tensors/events per candidate. Preserve full detail only on demand.
        self.cache[key] = result if retain_trace else HybridEvaluation(metrics,None,None,None,self.context,{},(),buffers)
        self.evaluations += 1
        return result if retain_trace else self.cache[key]

    __call__ = evaluate

    def _buffers(self,compiled,communication,schedule):
        events = schedule.event_map
        units = compiled.unit_map
        transfers = communication.transfer_map
        outgoing = defaultdict(list)
        local_consumers = defaultdict(list)
        forward_consumers = defaultdict(list)
        for unit in compiled.units:
            for producer in unit.dependencies:
                if producer in units and units[producer].chiplet == unit.chiplet:
                    local_consumers[producer].append(unit.id)
        for transfer in communication.transfers:
            for producer in transfer.producer_units:
                outgoing[producer].append(transfer.id)
            for dependency in transfer.dependencies:
                forward_consumers[(dependency,transfer.src)].append(transfer.id)
        intervals = defaultdict(list)
        # Exact produced tiles remain until their last outgoing transmission or
        # same-core local read. Remote consumers own separate received copies.
        for unit in compiled.units:
            local_users = local_consumers[unit.id]
            release = max([events[unit.id].finish_s]+
                          [events[name].finish_s for name in outgoing[unit.id]]+
                          [events[name].finish_s for name in local_users])
            intervals[unit.physical_die_id].append((events[unit.id].finish_s,release,unit.output_bytes))
            # Accumulator lives during compute; INT24 precision is physical.
            intervals[unit.physical_die_id].append((events[f"compute:{unit.id}"].start_s,
                    events[unit.id].finish_s,unit.output_region.elements*self.hardware.psum_bytes if unit.macs else unit.output_bytes))
        for transfer in communication.transfers:
            if transfer.dsts and all(dst.startswith("memory:") for dst in transfer.dsts):
                continue
            for destination in transfer.dsts:
                users = [name for name in transfer.consumer_units if units[name].chiplet == destination]
                forwarding = forward_consumers[(transfer.id,destination)]
                finish = max([events[transfer.id].finish_s]+
                             [events[name].finish_s for name in users]+
                             [events[name].finish_s for name in forwarding])
                intervals[physical_endpoint(destination)].append((events[transfer.id].finish_s,finish,transfer.bytes))
        peaks = {die:peak_interval_bytes(intervals[die]) for die in self.package.compute_ids}
        violations = [f"{die} unified buffer needs {amount} B > {self.hardware.unified_buffer_bytes} B"
                      for die,amount in peaks.items() if amount > self.hardware.unified_buffer_bytes]
        scratch = {core:0 for core in self.package.mapping_core_ids}
        capacity = self.hardware.pes_per_core*self.hardware.l1_bytes_per_pe + self.hardware.l2_bytes_per_die/self.hardware.cores_per_die
        for unit in compiled.units:
            scratch[unit.chiplet] = max(scratch[unit.chiplet],unit.buffer_bytes)
            if unit.buffer_bytes > capacity:
                violations.append(f"{unit.id} local tile needs {unit.buffer_bytes} B > core scratch {capacity:g} B")
        scratch_intervals = defaultdict(list)
        for unit in compiled.units:
            scratch_intervals[unit.chiplet].append((events[f'read:{unit.id}'].start_s,
                                                  events[unit.id].finish_s, unit.buffer_bytes))
        concurrent_scratch = {core:peak_interval_bytes(scratch_intervals[core]) for core in scratch}
        allocated = capacity * self.hardware.prefetch_slots
        for core, amount in concurrent_scratch.items():
            if amount > allocated:
                violations.append(f'{core} concurrent scratch needs {amount} B > allocated banks {allocated:g} B')
        return {"peak_by_die":peaks,"local_scratch_peak_by_core":scratch,
                "concurrent_scratch_peak_by_core":concurrent_scratch,
                "allocated_scratch_capacity_bytes_per_core":allocated,
                "unified_capacity_bytes_per_die":self.hardware.unified_buffer_bytes,
                "scratch_capacity_bytes_per_core":capacity,"violations":list(dict.fromkeys(violations)),
                "policy":"full received region in unified buffer until consuming unit completes (backs internal streaming); produced tile until last send/local consumer; temporal C tiling only reduces local scratch"}
