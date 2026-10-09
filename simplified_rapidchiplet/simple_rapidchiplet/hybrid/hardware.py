"""One resource contract for the INDM / Gemini hybrid architecture.

Paper values and uncalibrated component coefficients are deliberately separate.
Package endpoints are physical dies; ``mapping_core_ids`` are scheduling slots
inside a die and never contribute additional silicon area.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping
from .costs import sram_bank, grs_port, NN_BATON, GRS


@dataclass(frozen=True)
class HardwareSpec:
    cluster_rows: int = 2
    cluster_cols: int = 2
    compute_per_cluster: int = 4
    cores_per_die: int = 4
    pe_count: int = 16
    lanes: int = 16
    vector_size: int = 16
    compute_clock_hz: float = 500e6
    noc_clock_hz: float = 1e9
    nop_clock_hz: float = 1e9
    activation_bits: int = 8
    weight_bits: int = 8
    psum_bits: int = 24
    l1_weight_bytes: int = 64 * 1024
    l1_activation_bytes: int = 16 * 1024
    l1_output_bytes: int = 8 * 1024
    l2_weight_bytes: int = 64 * 1024
    l2_activation_bytes: int = 16 * 1024
    l2_output_bytes: int = 8 * 1024
    unified_buffer_bytes: int = 2 * 1024 * 1024
    prefetch_slots: int = 1  # Two slots replicate L1/L2 scratch, with paid silicon/leakage.
    nop_bandwidth_Bps: float = 12.5e9
    noc_bandwidth_Bps: float = 8.5e9
    dram_bandwidth_Bps: float = 18.75e9  # Per cluster; four clusters = 600 Gb/s.
    io_crossbar_bandwidth_Bps: float = 50e9
    l1_bandwidth_Bps: float = 64e9  # Per PE, per direction.
    l2_bandwidth_Bps: float = 64e9  # Shared die read/write service.
    unified_bandwidth_Bps: float = 64e9
    dram_capacity_bytes: int = 2 * 1024**3  # Per memory die; exploratory capacity.
    noc_topology: str = "multi_ring"
    ring_count: int = 4
    pe_router_latency_cycles: float = 1.0
    phy_latency_cycles: float = 12.0
    compute_router_latency_cycles: float = 3.0
    io_router_latency_cycles: float = 3.0
    memory_router_latency_cycles: float = 2.0
    wire_latency_cycles_per_mm: float = 0.25
    dram_access_latency_ns: float = 50.0
    spacing_mm: float = 0.15
    # 125um is an assumed pitch that fits 5x4 bumps in 686x565um, not a
    # measured pitch from GRS. The generic native budget is capped by real lanes.
    bump_pitch_mm: float = 0.125
    fraction_power_bumps: float = 11/20
    non_data_wires: int = 2
    cost_profile: str = 'literature16_v1'
    reference_process_nm: int = 16
    mac_reference_clock_hz: float = 500e6
    mac_reported_energy_pj_per_op: float = 0.024
    mac_ops_per_mac: float = 1.0  # Explicit convention; test 2 as uncertainty.
    mac_energy_pj: float = 0.024
    sram_energy_pj_per_bit: float = 0.81
    dram_energy_pj_per_bit: float = 8.75
    nop_energy_pj_per_bit: float = 1.17
    noc_energy_pj_per_bit: float = 0.4
    io_crossbar_energy_pj_per_bit: float = 0.1
    # Component costs are estimates. Changing resources changes these costs.
    sram_area_mm2_per_byte: float = 0.55 / 1e6
    sram_bank_overhead_mm2: float = 0.001
    sram_max_bank_bytes: int = 32*1024
    sram_word_bits: int = 128
    sram_rw_ports: int = 1
    sram_l1_anchor_pj_per_bit: float = 0.3
    sram_write_energy_factor: float = 1.0
    sram_width_energy_exponent: float = 0.15
    sram_extra_port_energy_factor: float = 0.3
    sram_energy_scale: float = 1.0
    sram_area_scale: float = 1.0
    mac_area_mm2_per_unit: float = 135.1 / 1e6
    grs_phy_area_mm2: float = 0.387590
    grs_active_area_mm2: float = 0.081406
    grs_data_lanes: int = 8
    grs_lane_rate_bps: float = 25e9
    interdie_router_area_mm2: float = 0.042
    cpu_area_mm2: float = 0.109
    pe_router_area_mm2: float = 0.006
    mesh_router_area_mm2: float = 0.012
    ring_injection_area_mm2: float = 0.002
    ddr_controller_phy_area_mm2: float = 10.5
    crossbar_area_mm2_per_port_squared: float = 0.001
    memory_die_area_mm2: float = 20.0
    compute_leakage_w_per_mm2: float = 0.02
    io_leakage_w_per_mm2: float = 0.008
    memory_leakage_w_per_mm2: float = 0.005
    link_idle_w_per_mm: float = 0.001
    calibration_status: str = "literature16_reference_uncalibrated"

    def __post_init__(self):
        if self.prefetch_slots not in (1, 2) or isinstance(self.prefetch_slots, bool):
            raise ValueError('prefetch_slots must be 1 or 2')
        positive_ints = (
            "cluster_rows", "cluster_cols", "compute_per_cluster", "cores_per_die",
            "pe_count", "lanes", "vector_size", "activation_bits", "weight_bits",
            "psum_bits", "l1_weight_bytes", "l1_activation_bytes", "l1_output_bytes",
            "l2_weight_bytes", "l2_activation_bytes", "l2_output_bytes",
            "unified_buffer_bytes", "dram_capacity_bytes", "ring_count",
            "sram_max_bank_bytes", "sram_word_bits", "sram_rw_ports", "grs_data_lanes",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.compute_per_cluster > 4:
            raise ValueError("INDM cluster supports at most four compute dies")
        if self.pe_count % self.cores_per_die:
            raise ValueError("pe_count must be divisible by cores_per_die")
        if self.ring_count > self.pe_count:
            raise ValueError("ring_count cannot exceed pe_count")
        if self.noc_topology not in ("multi_ring", "mesh"):
            raise ValueError("noc_topology must be multi_ring or mesh")
        if not 0 < self.fraction_power_bumps < 1:
            raise ValueError("fraction_power_bumps must leave both data and power bumps")
        if isinstance(self.non_data_wires, bool) or not isinstance(self.non_data_wires, int) or self.non_data_wires < 0:
            raise ValueError("non_data_wires must be a nonnegative integer")
        if self.activation_bits != 8 or self.weight_bits != 8 or self.psum_bits != 24:
            raise ValueError("this cost profile supports INT8 only with INT24 partial sums; supply a calibrated new profile for other precision")
        if self.cost_profile not in ('literature16_v1','legacy_mixed_v0'):
            raise ValueError('unknown cost_profile')
        if self.sram_word_bits % 8:
            raise ValueError('SRAM word width must be a multiple of 8 bits')
        if self.cost_profile == 'literature16_v1':
            if self.reference_process_nm != 16 or self.grs_data_lanes != 8 or self.grs_lane_rate_bps != 25e9:
                raise ValueError('literature16_v1 anchors require 16nm, eight 25Gb/s data lanes')
            if self.grs_active_area_mm2 > self.grs_phy_area_mm2:
                raise ValueError('active circuitry must fit inside the GRS footprint')
            if 5*self.bump_pitch_mm > .686 or 4*self.bump_pitch_mm > .565:
                raise ValueError('assumed GRS 5x4 bump array exceeds reference footprint')
            if not math.isclose(self.fraction_power_bumps,11/20):
                raise ValueError('GRS reference has eleven power/ground bumps out of twenty')
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("compute_clock_hz", "noc_clock_hz", "nop_clock_hz",
                     "nop_bandwidth_Bps", "noc_bandwidth_Bps", "dram_bandwidth_Bps",
                     "io_crossbar_bandwidth_Bps", "l1_bandwidth_Bps", "l2_bandwidth_Bps",
                     "unified_bandwidth_Bps", "bump_pitch_mm", "sram_area_mm2_per_byte",
                     "mac_area_mm2_per_unit", "grs_phy_area_mm2", "memory_die_area_mm2"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ('mac_ops_per_mac','sram_energy_scale','sram_area_scale','sram_write_energy_factor'):
            if getattr(self,name) <= 0:
                raise ValueError(f'{name} must be positive')

    @classmethod
    def from_dict(cls, data):
        """Never silently retag historical hardware.json as a new cost model."""
        data = dict(data)
        if 'cost_profile' not in data:
            data['cost_profile'] = 'legacy_mixed_v0'
            data['calibration_status'] = 'historical_mixed_uncalibrated'
        return cls(**data)

    @property
    def effective_mac_energy_pj(self):
        return self.mac_energy_pj*self.mac_ops_per_mac

    @property
    def sram_costs(self):
        return {name:sram_bank(self,name) for name in ('WL1','AL1','OL1','WL2','AL2','OL2','UB')}

    @property
    def cluster_count(self) -> int:
        return self.cluster_rows * self.cluster_cols

    @property
    def compute_count(self) -> int:
        return self.cluster_count * self.compute_per_cluster

    @property
    def core_count(self) -> int:
        return self.compute_count * self.cores_per_die

    @property
    def pes_per_core(self) -> int:
        return self.pe_count // self.cores_per_die

    @property
    def macs_per_cycle(self) -> int:
        return self.pe_count * self.lanes * self.vector_size

    @property
    def macs_per_second(self) -> float:
        return self.macs_per_cycle * self.compute_clock_hz

    @property
    def per_core_mac_rate(self) -> float:
        return self.macs_per_second / self.cores_per_die

    @property
    def peak_tops(self) -> float:
        return 2 * self.macs_per_second * self.compute_count / 1e12

    @property
    def activation_bytes(self) -> int:
        return (self.activation_bits + 7) // 8

    @property
    def weight_bytes(self) -> int:
        return (self.weight_bits + 7) // 8

    @property
    def psum_bytes(self) -> int:
        return (self.psum_bits + 7) // 8

    @property
    def l1_bytes_per_pe(self) -> int:
        return self.l1_weight_bytes + self.l1_activation_bytes + self.l1_output_bytes

    @property
    def l2_bytes_per_die(self) -> int:
        return self.l2_weight_bytes + self.l2_activation_bytes + self.l2_output_bytes

    @property
    def total_buffer_bytes_per_die(self) -> int:
        return self.prefetch_slots * (self.pe_count * self.l1_bytes_per_pe + self.l2_bytes_per_die) + self.unified_buffer_bytes

    @property
    def compute_area_breakdown(self) -> dict[str, float]:
        noc_scale = self.noc_bandwidth_Bps / 8.5e9
        noc_area = (self.pe_count * self.pe_router_area_mm2 + self.ring_count * self.ring_injection_area_mm2
                    if self.noc_topology == "multi_ring"
                    else (self.pe_count + 3) * self.mesh_router_area_mm2)
        return {
            "l1_sram": self.prefetch_slots * self.pe_count * sum(sram_bank(self,n)['area_mm2'] for n in ('WL1','AL1','OL1')),
            "l2_sram": self.prefetch_slots * sum(sram_bank(self,n)['area_mm2'] for n in ('WL2','AL2','OL2')),
            "unified_sram": sram_bank(self,'UB')['area_mm2'],
            "int8_macs": self.macs_per_cycle * self.mac_area_mm2_per_unit,
            "on_die_network": noc_area * noc_scale,
            "grs_phy": (grs_port(self)['footprint_mm2'] if self.cost_profile == 'literature16_v1'
                        else self.grs_phy_area_mm2 * self.nop_bandwidth_Bps / 12.5e9),
            "interdie_router": self.interdie_router_area_mm2,
            "control_cpu": self.cpu_area_mm2,
        }

    @property
    def compute_area_mm2(self) -> float:
        return sum(self.compute_area_breakdown.values())

    @property
    def compute_idle_power_w(self) -> float:
        return self.compute_area_mm2 * self.compute_leakage_w_per_mm2

    def io_area_breakdown(self, grs_ports: int) -> dict[str, float]:
        if grs_ports <= 0:
            raise ValueError("an I/O die must have at least one GRS port")
        return {
            "ddr_phy_controller": self.ddr_controller_phy_area_mm2 * self.dram_bandwidth_Bps / 18.75e9,
            "grs_phys": grs_ports * (grs_port(self)['footprint_mm2'] if self.cost_profile == 'literature16_v1'
                                     else self.grs_phy_area_mm2 * self.nop_bandwidth_Bps / 12.5e9),
            "crossbar": (grs_ports + 1)**2 * self.crossbar_area_mm2_per_port_squared * self.io_crossbar_bandwidth_Bps / 50e9,
            "interdie_router": self.interdie_router_area_mm2,
        }

    @property
    def memory_area_mm2(self) -> float:
        return self.memory_die_area_mm2 * self.dram_capacity_bytes / (2 * 1024**3)

    @property
    def provenance(self) -> dict[str, object]:
        if self.cost_profile == 'legacy_mixed_v0':
            return {'status':self.calibration_status,'cost_profile':self.cost_profile,
                    'warning':'historical mixed 12/16nm flat coefficients; MAC x8 derivation invalidated; retain for archival comparison only',
                    'effective_mac_energy_pj':self.effective_mac_energy_pj,
                    'constraint_scope':'uncalibrated historical estimate, not certified'}
        return {
            "status": self.calibration_status,
            "architecture": "INDM Fig.3 cluster mesh and W/A/O hierarchy + Gemini-style core/pipeline residency extension",
            "physical_rates": "INDM Table IV; DRAM rate divided over four cloud clusters",
            "memory_capacities": "INDM Table V/Eq.2; 2MiB unified SRAM is hybrid extension",
            "cost_profile": self.cost_profile,
            "MAC": {'source':NN_BATON,'reference':'Sec.V-A, 135.1 um2 and 0.024 pJ/op, 500MHz; UMC28 synthesis scaled to 16nm',
                    'counting':'program counts scalar MAC; assume one reported op per scalar MAC, not an explicit author definition; 2op/MAC sensitivity required',
                    'ops_per_mac':self.mac_ops_per_mac,'effective_pj_per_mac':self.effective_mac_energy_pj,
                    'voltage':'source MAC voltage unavailable; constant per-MAC energy assumes unchanged voltage',
                    'target_clock_hz':self.compute_clock_hz,'reference_clock_hz':self.mac_reference_clock_hz,
                    'timing':'default compute clock follows 500MHz reference; other clocks/voltages are exploratory, not timing signoff; no automatic frequency energy multiplier'},
            "SRAM": {'source':NN_BATON,'anchors':'Table I: 1KiB .30, 32KiB .81 pJ/bit; width/ports unspecified',
                     'model':'log-size energy interpolation per physical bank; explicit read/write, width and port heuristics; rough Fig.10 area slope plus per-bank overhead',
                     'qualification':'not a calibrated macro table; Fig.10 trend and Table I anchors cannot establish exact costs for these banks'},
            "GRS": grs_port(self),
            "energy": "GRS 1.17 pJ/bit covers Tx+Rx once per hop. DRAM 8.75 reference. NoC .4 and crossbar .1 are assumptions.",
            "area": "16nm reference MAC and rough NN-Baton SRAM; GRS whole footprint includes active circuitry; control/router values are adopted estimates; DDR/crossbar/memory scaling are assumptions",
            "leakage": "explicit exploratory W/mm2 coefficients; no fabricated TDP claim",
            "mapping_core": "four scheduling cores per die, PE allocation divided equally, shared L2/unified resources",
            "prefetch": "1/2 independently capacity-backed L1/L2 scratch slots; duplicated SRAM costs paid; shared NoC/UB/ports unchanged",
            "assumptions": ['NoC and crossbar energy','die leakage and temperature/voltage','memory die area/capacity','DDR controller/PHY linear scaling','router/CPU area applicability','bank conflict-free aggregate bandwidth','NoC/NoP timing and any compute clock differing from 500MHz'],
            "constraint_scope": "nominal analytical estimate only; no physical area/power/timing signoff or silicon-limit certification",
        }

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "compute_count": self.compute_count,
                "core_count": self.core_count, "macs_per_second": self.macs_per_second,
                "per_core_mac_rate": self.per_core_mac_rate, "peak_tops": self.peak_tops,
                "compute_area_breakdown": self.compute_area_breakdown,
                "sram_costs": self.sram_costs, "provenance": self.provenance}


@dataclass(frozen=True)
class PackageNode:
    id: str
    role: str
    cluster: int
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    ports: int
    idle_power_w: float
    area_components: Mapping[str, float] = field(default_factory=dict)

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.height_mm


@dataclass(frozen=True)
class PackageLink:
    a: str
    b: str
    a_port: int
    b_port: int
    kind: str
    bandwidth_Bps: float
    energy_pj_per_bit: float


@dataclass(frozen=True)
class PackageGraph:
    nodes: Mapping[str, PackageNode]
    links: tuple[PackageLink, ...]
    compute_ids: tuple[str, ...]
    io_ids: tuple[str, ...]
    memory_ids: tuple[str, ...]
    mapping_core_ids: tuple[str, ...]
    node_cluster: Mapping[str, int]
    positions: Mapping[str, tuple[float, float]]
    cluster_coordinates: Mapping[int, tuple[int, int]]

    def to_dict(self) -> dict[str, object]:
        return {"nodes": [asdict(n) for n in self.nodes.values()],
                "links": [asdict(e) for e in self.links],
                "compute_ids": self.compute_ids, "io_ids": self.io_ids,
                "memory_ids": self.memory_ids, "mapping_core_ids": self.mapping_core_ids,
                "cluster_coordinates": {str(k): list(v) for k, v in self.cluster_coordinates.items()}}


def physical_endpoint(endpoint: str) -> str:
    return endpoint.split("/", 1)[0]


def build_package(spec: HardwareSpec) -> PackageGraph:
    """Build a legal IO mesh, four-or-fewer compute leaves and DRAM per cluster.

    Wide cluster cells reserve separate bands for compute, IO and memory dies.
    Actual resource-derived dimensions determine pitch. No arbitrary width
    override can make a larger native die overlap a neighboring placement.
    """
    compute_ids = tuple(f"compute:{i}" for i in range(spec.compute_count))
    io_ids = tuple(f"io:{i}" for i in range(spec.cluster_count))
    memory_ids = tuple(f"memory:{i}" for i in range(spec.cluster_count))
    coords = {r * spec.cluster_cols + c: (r, c)
              for r in range(spec.cluster_rows) for c in range(spec.cluster_cols)}
    edge_specs: list[tuple[str, str, str]] = []
    for cluster, (row, col) in coords.items():
        io_id = io_ids[cluster]
        for local in range(spec.compute_per_cluster):
            edge_specs.append((compute_ids[cluster * spec.compute_per_cluster + local], io_id, "compute_io"))
        edge_specs.append((memory_ids[cluster], io_id, "dram_io"))
        if col + 1 < spec.cluster_cols:
            edge_specs.append((io_id, io_ids[cluster + 1], "io_mesh"))
        if row + 1 < spec.cluster_rows:
            edge_specs.append((io_id, io_ids[cluster + spec.cluster_cols], "io_mesh"))
    degrees = {endpoint: 0 for endpoint in (*compute_ids, *io_ids, *memory_ids)}
    for a, b, _ in edge_specs:
        degrees[a] += 1
        degrees[b] += 1
    compute_side = math.sqrt(spec.compute_area_mm2)
    memory_side = math.sqrt(spec.memory_area_mm2)
    io_components = {i: spec.io_area_breakdown(degrees[i] - 1) for i in io_ids}
    io_side = {i: math.sqrt(sum(parts.values())) for i, parts in io_components.items()}
    gap = spec.spacing_mm
    max_io_side = max(io_side.values())
    compute_band = 2 * compute_side + gap
    cell_width = max(compute_band, max_io_side, memory_side) + 2 * gap
    cell_height = compute_band + max_io_side + memory_side + 4 * gap
    nodes = {}
    for cluster, (row, col) in coords.items():
        base_x, base_y = col * cell_width, row * cell_height
        for local in range(spec.compute_per_cluster):
            ident = compute_ids[cluster * spec.compute_per_cluster + local]
            x = base_x + gap + (local % 2) * (compute_side + gap)
            y = base_y + gap + (local // 2) * (compute_side + gap)
            nodes[ident] = PackageNode(ident, "compute", cluster, x, y, compute_side, compute_side,
                                      degrees[ident], spec.compute_idle_power_w, spec.compute_area_breakdown)
        ident = io_ids[cluster]
        side = io_side[ident]
        x, y = base_x + (cell_width - side) / 2, base_y + compute_band + 2 * gap
        nodes[ident] = PackageNode(ident, "io", cluster, x, y, side, side, degrees[ident],
                                  side**2 * spec.io_leakage_w_per_mm2, io_components[ident])
        ident = memory_ids[cluster]
        x, y = base_x + (cell_width - memory_side) / 2, base_y + compute_band + max_io_side + 3 * gap
        nodes[ident] = PackageNode(ident, "memory", cluster, x, y, memory_side, memory_side,
                                  degrees[ident], spec.memory_area_mm2 * spec.memory_leakage_w_per_mm2,
                                  {"memory_array": spec.memory_area_mm2})
    next_port = {ident: 0 for ident in nodes}
    links = []
    for a, b, kind in edge_specs:
        links.append(PackageLink(a, b, next_port[a], next_port[b], kind,
                                 spec.dram_bandwidth_Bps if kind == "dram_io" else spec.nop_bandwidth_Bps,
                                 0.0 if kind == "dram_io" else spec.nop_energy_pj_per_bit))
        next_port[a] += 1
        next_port[b] += 1
    graph = PackageGraph(nodes, tuple(links), compute_ids, io_ids, memory_ids,
                         tuple(f"{die}/core:{core}" for die in compute_ids for core in range(spec.cores_per_die)),
                         {i: n.cluster for i, n in nodes.items()},
                         {i: (n.x_mm, n.y_mm) for i, n in nodes.items()}, coords)
    validate_package(graph, spec)
    return graph


def validate_package(package: PackageGraph, spec: HardwareSpec | None = None) -> None:
    """Fail closed on physical overlap, port reuse, role/relay and connectivity."""
    nodes = package.nodes
    if not nodes:
        raise ValueError("package must contain nodes")
    listed = (*package.compute_ids, *package.io_ids, *package.memory_ids)
    if len(set(listed)) != len(listed) or set(listed) != set(nodes):
        raise ValueError("package role lists must uniquely cover all nodes")
    for ident, node in nodes.items():
        if ident != node.id or node.role not in ("compute", "io", "memory"):
            raise ValueError("invalid package node identity or role")
        if any(not math.isfinite(v) for v in (node.x_mm, node.y_mm, node.width_mm, node.height_mm, node.idle_power_w)):
            raise ValueError("package geometry and power must be finite")
        if min(node.width_mm, node.height_mm) <= 0 or node.idle_power_w < 0:
            raise ValueError("invalid dimensions or idle power")
        if not isinstance(node.ports, int) or node.ports < 1:
            raise ValueError("physical node port count must be a positive integer")
        if package.node_cluster.get(ident) != node.cluster:
            raise ValueError("node_cluster differs from physical node cluster")
        if node.cluster not in package.cluster_coordinates:
            raise ValueError("physical node refers to an absent cluster")
        if package.positions.get(ident) != (node.x_mm, node.y_mm):
            raise ValueError("positions disagree with physical die placement")
    for role, ids in (("compute", package.compute_ids), ("io", package.io_ids), ("memory", package.memory_ids)):
        if any(nodes[ident].role != role for ident in ids):
            raise ValueError("endpoint role list disagrees with physical node role")
    records = list(nodes.values())
    for index, a in enumerate(records):
        for b in records[index + 1:]:
            overlap_x = min(a.x_mm + a.width_mm, b.x_mm + b.width_mm) - max(a.x_mm, b.x_mm)
            overlap_y = min(a.y_mm + a.height_mm, b.y_mm + b.height_mm) - max(a.y_mm, b.y_mm)
            if overlap_x > 1e-10 and overlap_y > 1e-10:
                raise ValueError(f"overlapping physical dies: {a.id}, {b.id}")
    used_ports: set[tuple[str, int]] = set()
    used_edges = set()
    adjacency = {i: set() for i in nodes}
    for edge in package.links:
        if edge.a == edge.b or edge.a not in nodes or edge.b not in nodes:
            raise ValueError("invalid link endpoints")
        pair = frozenset((edge.a, edge.b))
        if pair in used_edges:
            raise ValueError("parallel physical links require an explicit multigraph model")
        used_edges.add(pair)
        if not math.isfinite(edge.bandwidth_Bps) or edge.bandwidth_Bps <= 0:
            raise ValueError("link bandwidth must be finite and positive")
        for ident, port in ((edge.a, edge.a_port), (edge.b, edge.b_port)):
            if not isinstance(port, int) or port < 0 or port >= nodes[ident].ports:
                raise ValueError("link refers to an unallocated PHY port")
            if (ident, port) in used_ports:
                raise ValueError(f"PHY port reused: {ident}:{port}")
            used_ports.add((ident, port))
        if "io" not in (nodes[edge.a].role, nodes[edge.b].role):
            raise ValueError("cluster architecture only connects endpoints through I/O dies")
        a, b = nodes[edge.a], nodes[edge.b]
        if a.role != "io" or b.role != "io":
            if a.cluster != b.cluster:
                raise ValueError("compute/memory leaf must connect to its own cluster")
        else:
            ar, ac = package.cluster_coordinates[a.cluster]
            br, bc = package.cluster_coordinates[b.cluster]
            if abs(ar - br) + abs(ac - bc) != 1:
                raise ValueError("I/O mesh links must connect neighboring clusters")
        adjacency[edge.a].add(edge.b)
        adjacency[edge.b].add(edge.a)
    for ident, node in nodes.items():
        if node.role != "io" and len(adjacency[ident]) != 1:
            raise ValueError("compute and memory nodes must be leaves")
        if len(adjacency[ident]) != node.ports:
            raise ValueError("allocated port count differs from physical degree")
    seen, frontier = set(), [next(iter(nodes))]
    while frontier:
        current = frontier.pop()
        if current not in seen:
            seen.add(current)
            frontier.extend(adjacency[current] - seen)
    if seen != set(nodes):
        raise ValueError("package network is disconnected")
    if spec is not None:
        if (len(package.compute_ids), len(package.io_ids), len(package.memory_ids)) != (
                spec.compute_count, spec.cluster_count, spec.cluster_count):
            raise ValueError("package node counts disagree with HardwareSpec")
        expected_coords = {r * spec.cluster_cols + c: (r, c)
                           for r in range(spec.cluster_rows) for c in range(spec.cluster_cols)}
        if dict(package.cluster_coordinates) != expected_coords:
            raise ValueError("package cluster coordinates disagree with HardwareSpec")
        expected_cores = tuple(f"{die}/core:{core}" for die in package.compute_ids for core in range(spec.cores_per_die))
        if package.mapping_core_ids != expected_cores:
            raise ValueError("mapping core resource count disagrees with HardwareSpec")
        for cluster in expected_coords:
            counts = {role: sum(n.cluster == cluster and n.role == role for n in nodes.values())
                      for role in ("compute", "io", "memory")}
            if counts != {"compute": spec.compute_per_cluster, "io": 1, "memory": 1}:
                raise ValueError("cluster population disagrees with HardwareSpec")
        for node in nodes.values():
            if node.role == "compute":
                expected_area, expected_power = spec.compute_area_mm2, spec.compute_idle_power_w
            elif node.role == "io":
                expected_area = sum(spec.io_area_breakdown(node.ports - 1).values())
                expected_power = expected_area * spec.io_leakage_w_per_mm2
            else:
                expected_area = spec.memory_area_mm2
                expected_power = expected_area * spec.memory_leakage_w_per_mm2
            if not math.isclose(node.area_mm2, expected_area, rel_tol=1e-10) or not math.isclose(
                    node.idle_power_w, expected_power, rel_tol=1e-10, abs_tol=1e-15):
                raise ValueError("package physical costs disagree with HardwareSpec; rebuild resource-derived package")
        for edge in package.links:
            roles = {nodes[edge.a].role, nodes[edge.b].role}
            expected_kind = "dram_io" if "memory" in roles else "compute_io" if "compute" in roles else "io_mesh"
            expected_bw = spec.dram_bandwidth_Bps if expected_kind == "dram_io" else spec.nop_bandwidth_Bps
            expected_energy = 0.0 if expected_kind == "dram_io" else spec.nop_energy_pj_per_bit
            if edge.kind != expected_kind or not math.isclose(edge.bandwidth_Bps, expected_bw) or not math.isclose(
                    edge.energy_pj_per_bit, expected_energy):
                raise ValueError("physical link resources disagree with HardwareSpec")
