"""Heterogeneous cluster mesh adapter for the native RapidChiplet engine.

Native PHY/package path latencies and bump bandwidth ceilings are preserved.
Explicit signaling rates cap those ceilings. The inputs' power is idle/leakage;
workload energy is accounted by the hybrid evaluator, never counted twice here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping

from .hardware import HardwareSpec, PackageGraph, PackageNode, build_package, physical_endpoint, validate_package
from .costs import grs_port


DEFAULT_RAPID_ROOT = os.environ.get("RAPIDCHIPLET_ROOT", str(Path(__file__).resolve().parents[3] / "external" / "rapidchiplet"))


@dataclass(frozen=True)
class NativeNetworkContext:
    backend: str
    routes: Mapping[tuple[str, str], tuple[str, ...]]
    link_bandwidth_Bps: Mapping[tuple[str, str], float]
    path_latency_ns: Mapping[tuple[str, str], float]
    area_breakdown: Mapping[str, float]
    idle_power_w: float
    node_idle_power_w: Mapping[str, float]
    native_link_bandwidth_Bps: Mapping[tuple[str, str], float]
    link_length_mm: Mapping[tuple[str, str], float]
    link_energy_pj_per_bit: Mapping[tuple[str, str], float]
    provenance: Mapping[str, object]
    native_area_summary: Mapping[str, float]
    native_power_summary: Mapping[str, float]

    def route(self, src: str, dst: str) -> tuple[str, ...]:
        return self.routes[(physical_endpoint(src), physical_endpoint(dst))]

    def latency_ns(self, src: str, dst: str) -> float:
        return self.path_latency_ns[(physical_endpoint(src), physical_endpoint(dst))]

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend, "area_breakdown": dict(self.area_breakdown),
            "idle_power_w": self.idle_power_w, "provenance": dict(self.provenance),
            "native_area_summary": dict(self.native_area_summary),
            "native_power_summary": dict(self.native_power_summary),
            "node_idle_power_w": dict(self.node_idle_power_w),
            "links": [{"source": a, "destination": b,
                       "bandwidth_Bps": self.link_bandwidth_Bps[(a, b)],
                       "native_bump_ceiling_Bps": self.native_link_bandwidth_Bps[(a, b)],
                       "length_mm": self.link_length_mm[(a, b)],
                       "energy_pj_per_bit": self.link_energy_pj_per_bit[(a, b)]}
                      for a, b in sorted(self.link_bandwidth_Bps)],
        }


def _route_for_pair(package: PackageGraph, spec: HardwareSpec, src: str, dst: str) -> tuple[str, ...]:
    if src == dst:
        return (src,)
    src_cluster, dst_cluster = package.node_cluster[src], package.node_cluster[dst]
    source_io = f"io:{src_cluster}"
    destination_io = f"io:{dst_cluster}"
    path = [src]
    if src != source_io:
        path.append(source_io)
    row, col = package.cluster_coordinates[src_cluster]
    dst_row, dst_col = package.cluster_coordinates[dst_cluster]
    # X first (columns), then Y (rows); cluster mesh routing is fixed per design.
    while col != dst_col:
        col += 1 if dst_col > col else -1
        path.append(f"io:{row * spec.cluster_cols + col}")
    while row != dst_row:
        row += 1 if dst_row > row else -1
        path.append(f"io:{row * spec.cluster_cols + col}")
    if path[-1] != destination_io:
        raise ValueError("invalid cluster coordinates for deterministic XY route")
    if dst != destination_io:
        path.append(dst)
    return tuple(path)


def _phy_coordinates(node: PackageNode, port: int) -> tuple[float, float]:
    # All physical ports have distinct positions on the perimeter. Directional
    # ports are an implementation choice; allocation uniqueness is authoritative.
    fraction = (port + 0.5) / node.ports
    perimeter = 2 * (node.width_mm + node.height_mm)
    offset = fraction * perimeter
    if offset <= node.width_mm:
        return offset, 0.0
    offset -= node.width_mm
    if offset <= node.height_mm:
        return node.width_mm, offset
    offset -= node.height_mm
    if offset <= node.width_mm:
        return node.width_mm - offset, node.height_mm
    return 0.0, node.height_mm - (offset - node.width_mm)


def build_native_inputs(spec: HardwareSpec, package: PackageGraph | None = None) -> tuple[dict, dict[str, int]]:
    """Return complete in-memory native inputs and the explicit endpoint ID map."""
    package = build_package(spec) if package is None else package
    validate_package(package, spec)
    node_ids = {ident: index for index, ident in enumerate(package.nodes)}
    chiplets = {}
    placement = []
    port_rates: dict[tuple[str, int], float] = {}
    grs_ports = set()
    for edge in package.links:
        port_rates[(edge.a, edge.a_port)] = edge.bandwidth_Bps
        port_rates[(edge.b, edge.b_port)] = edge.bandwidth_Bps
        if edge.kind != 'dram_io':
            grs_ports.update(((edge.a,edge.a_port),(edge.b,edge.b_port)))
    for ident, node in package.nodes.items():
        phys = []
        for port in range(node.ports):
            wire_rate = spec.grs_lane_rate_bps if spec.cost_profile == 'literature16_v1' else spec.nop_clock_hz
            data_wires = math.ceil(2 * port_rates[(ident, port)] * 8 / wire_rate)
            # Native compute_link_bandwidths halves a shared bidirectional wire
            # budget. Two guard wires prevent floating floor losing requested BW.
            fraction = (data_wires + spec.non_data_wires + 2) * spec.bump_pitch_mm**2 / (node.area_mm2 * (1 - spec.fraction_power_bumps))
            if spec.cost_profile == 'literature16_v1' and (ident,port) in grs_ports:
                fraction = grs_port(spec,port_rates[(ident,port)])['footprint_mm2']/node.area_mm2
            x, y = _phy_coordinates(node, port)
            phys.append({"x": x, "y": y, "fraction_bump_area": fraction})
        if sum(phy["fraction_bump_area"] for phy in phys) > 1.0 + 1e-10:
            raise ValueError(f"PHY data-wire allocation exceeds die bump budget: {ident}")
        internal_latency = {"compute": spec.compute_router_latency_cycles,
                            "io": spec.io_router_latency_cycles,
                            "memory": spec.memory_router_latency_cycles}[node.role]
        if internal_latency <= 0:
            raise ValueError("native internal latency must be positive")
        chiplets[ident] = {
            "dimensions": {"x": node.width_mm, "y": node.height_mm},
            "type": node.role, "phys": phys, "fraction_power_bumps": spec.fraction_power_bumps,
            "technology": f"{node.role}_reference", "power": node.idle_power_w,
            "relay": node.role == "io", "internal_latency": internal_latency,
            # Injection units represent mapping cores, not multiplier/PE count.
            "unit_count": spec.cores_per_die if node.role == "compute" else 1,
        }
        placement.append({"name": ident, "position": {"x": node.x_mm, "y": node.y_mm}, "rotation": 0})
    topology = []
    for edge in package.links:
        topology.append({"ep1": {"type": "chiplet", "outer_id": node_ids[edge.a], "inner_id": edge.a_port},
                         "ep2": {"type": "chiplet", "outer_id": node_ids[edge.b], "inner_id": edge.b_port},
                         "color": "green" if edge.kind == "io_mesh" else "gray"})
    routes = {(src, dst): _route_for_pair(package, spec, src, dst)
              for src in package.nodes for dst in package.nodes}
    edges = {(edge.a, edge.b) for edge in package.links} | {(edge.b, edge.a) for edge in package.links}
    table = {}
    for src in package.nodes:
        row = {}
        for dst in package.nodes:
            path = routes[(src, dst)]
            if any(pair not in edges for pair in zip(path, path[1:])):
                raise ValueError("routing table uses an absent physical link")
            if any(package.nodes[node].role != "io" for node in path[1:-1]):
                raise ValueError("routing table relays through a compute/memory leaf")
            row[("chiplet", node_ids[dst])] = ("chiplet", node_ids[path[1]]) if len(path) > 1 else None
        table[("chiplet", node_ids[src])] = row
    technologies = {f"{role}_reference": {
        "phy_latency": spec.phy_latency_cycles,
        "wafer_radius": 150.0, "wafer_cost": 1000.0, "defect_density": 0.01,
    } for role in ("compute", "io", "memory")}
    # The hybrid's idle link model is added explicitly to native idle die power.
    # Dynamic DDR/NoP energy is exclusively handled by the operation evaluator.
    packaging = {"link_routing": "manhattan", "link_latency_type": "function",
                 "link_latency": f"lambda x: {spec.wire_latency_cycles_per_mm!r} * x",
                 "link_power_type": "constant", "link_power": 0.0,
                 "packaging_yield": 0.9, "bump_pitch": spec.bump_pitch_mm,
                 "non_data_wires": spec.non_data_wires, "is_active": False,
                 "latency_irouter": 0.0, "power_irouter": 0.0,
                 "has_interposer": False}
    inputs = {
        "design": {"design_name": "hybrid_cluster_mesh", **{key: "in-memory" for key in (
            "technologies", "chiplets", "placement", "topology", "packaging",
            "routing_table", "traffic_by_chiplet")}},
        "verbose": False, "validate": False, "chiplets": chiplets,
        "placement": {"chiplets": placement, "interposer_routers": []},
        "topology": topology, "technologies": technologies, "packaging": packaging,
        "routing_table": {"type": "default", "table": table}, "traffic_by_chiplet": {},
    }
    return inputs, node_ids


def _load_native(root: str | Path):
    path = Path(root).resolve()
    if not (path / "rapidchiplet.py").is_file():
        raise FileNotFoundError(f"native RapidChiplet engine unavailable at {path}")
    loaded = sys.modules.get("rapidchiplet")
    if loaded is not None and Path(loaded.__file__).resolve().parent != path:
        raise RuntimeError("a different native RapidChiplet root is already imported; use a separate process")
    for name in ("helpers", "booksim_wrapper", "validation"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None) and Path(module.__file__).resolve().parent != path:
            raise RuntimeError(f"native module {name} collides with an already imported module")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.import_module("rapidchiplet")


def evaluate_package(
    spec: HardwareSpec,
    package: PackageGraph | None = None,
    rapid_root: str | Path = DEFAULT_RAPID_ROOT,
    traffic_pairs: Iterable[tuple[str, str]] | Mapping[tuple[str, str], float] | None = None,
) -> NativeNetworkContext:
    """Evaluate physical area/idle power and all-pair package route properties.

    No local proxy is substituted. ``traffic_pairs`` can limit path-latency
    queries, but routes always cover all physical nodes, including same-die
    zero-length paths. Capacities are bytes/second, path times are nanoseconds.
    """
    package = build_package(spec) if package is None else package
    inputs, node_ids = build_native_inputs(spec, package)
    rc = _load_native(rapid_root)
    intermediates = {}
    enabled = {name: name in ("area_summary", "power_summary", "link_summary") for name in rc.metrics}
    outputs = rc.rapidchiplet(inputs, intermediates, enabled, "hybrid_cluster_mesh", verbose=False, validate=False)
    inverse = {index: ident for ident, index in node_ids.items()}
    wire_rate = spec.grs_lane_rate_bps if spec.cost_profile == 'literature16_v1' else spec.nop_clock_hz
    native_bandwidth = {(inverse[a[1]], inverse[b[1]]): float(rate) * wire_rate / 8
                        for (a, b), rate in intermediates["link_bandwidths"].items()}
    link_lengths = {(inverse[a[1]], inverse[b[1]]): float(length)
                    for (a, b), length in intermediates["link_lengths"].items()}
    bandwidth = {}
    link_energy = {}
    link_idle = 0.0
    for edge in package.links:
        for pair in ((edge.a, edge.b), (edge.b, edge.a)):
            bandwidth[pair] = min(native_bandwidth[pair], edge.bandwidth_Bps)
            if spec.cost_profile == 'literature16_v1' and edge.kind != 'dram_io':
                bandwidth[pair] = min(bandwidth[pair],grs_port(spec,edge.bandwidth_Bps)['per_direction_capacity_Bps'])
            link_energy[pair] = edge.energy_pj_per_bit
            if bandwidth[pair] <= 0:
                raise ValueError("native PHY bump budget produces a nonpositive link capacity")
        link_idle += spec.link_idle_w_per_mm * link_lengths[(edge.a, edge.b)] * edge.bandwidth_Bps / 12.5e9
    routes = {(src, dst): _route_for_pair(package, spec, src, dst)
              for src in package.nodes for dst in package.nodes}
    if traffic_pairs is None:
        pairs = tuple(routes)
    else:
        pairs = tuple(dict.fromkeys((physical_endpoint(src), physical_endpoint(dst)) for src, dst in traffic_pairs))
    path_latencies = {(ident, ident): 0.0 for ident in package.nodes}
    for src, dst in pairs:
        if src not in node_ids or dst not in node_ids:
            raise ValueError("traffic endpoint outside physical package")
        if src == dst:
            continue
        traffic = {(node_ids[src], node_ids[dst]): 1.0}
        cycles = float(rc.compute_latency({**inputs, "traffic_by_chiplet": traffic}, intermediates)["avg"])
        path_latencies[(src, dst)] = cycles / spec.nop_clock_hz * 1e9
    area = outputs["area_summary"]
    power = outputs["power_summary"]
    role_area = {role: sum(node.area_mm2 for node in package.nodes.values() if node.role == role)
                 for role in ("compute", "io", "memory")}
    area_breakdown = {
        "compute_silicon_mm2": role_area["compute"], "io_silicon_mm2": role_area["io"],
        "memory_silicon_mm2": role_area["memory"], "total_silicon_mm2": float(area["total_chiplet_area"]),
        "package_footprint_mm2": float(area["total_interposer_area"]),
        "max_compute_die_mm2": max(package.nodes[i].area_mm2 for i in package.compute_ids),
        "max_io_die_mm2": max(package.nodes[i].area_mm2 for i in package.io_ids),
        "package_width_mm": float(area["chip_width"]), "package_height_mm": float(area["chip_height"]),
    }
    if not math.isclose(sum(role_area.values()), area_breakdown["total_silicon_mm2"], rel_tol=1e-10):
        raise ValueError("native silicon area disagrees with resource-derived die dimensions")
    root_path = Path(rapid_root).resolve()
    provenance = {
        **spec.provenance, "rapid_root": str(root_path),
        "native_source_sha256": {name: hashlib.sha256((root_path / name).read_bytes()).hexdigest()
                                 for name in ("rapidchiplet.py", "helpers.py", "validation.py")},
        "bandwidth_source": "native generic bump ceiling, serial wire-rate conversion separate from router clock, capped by whole-macro GRS lanes and requested rate; DDR remains an assumed generic link",
        "wire_rate_bps": wire_rate,
        "GRS_port": grs_port(spec),
        "bump_model_scope": "GRS footprint paid once at each endpoint; active area included. Assumed pitch fits reference 5x4 array. Native continuous bump ceiling is an analytical bound, not a placed pin assignment; explicit lanes cap it. DDR bumps/clocking uncalibrated.",
        "path_latency_source": "native RapidChiplet per-flow PHY/router/wire formula, NoP clock conversion",
        "routing": "deterministic cluster XY (columns then rows); only physical I/O dies relay",
        "power_semantics": "native die power = leakage/idle only; link idle added once; dynamic inference energy separate",
        "link_idle_power_w": link_idle,
        "controller_service": "DRAM latency and bandwidth are separate resources in hybrid evaluator",
        "ddr_phy_latency": "generic native PHY latency placeholder also used on DDR package connection; DDR controller delay separate, not calibrated DDR timing",
    }
    return NativeNetworkContext(
        "official_rapidchiplet", routes, bandwidth, path_latencies, area_breakdown,
        float(power["total_power"]) + link_idle,
        {ident: node.idle_power_w for ident, node in package.nodes.items()}, native_bandwidth,
        link_lengths, link_energy, provenance, dict(area), dict(power),
    )
