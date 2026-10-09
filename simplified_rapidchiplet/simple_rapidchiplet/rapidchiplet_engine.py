from __future__ import annotations

import importlib
import hashlib
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import EvalConfig
from .network_context import NetworkContext
from .rapid_proxy import compute_latency, compute_throughput
from .routing import shortest_paths_for_pairs
from .topology import Topology
from .workload import WorkloadPartition


_LOCAL_PROXY_WARNED = False


@dataclass(frozen=True)
class RapidChipletMetrics:
    avg_latency_cycles: float
    avg_latency_ns: float
    min_latency_cycles: float
    max_latency_cycles: float
    network_limited_fps: float
    aggregate_throughput_bits_per_cycle: float
    total_area_mm2: float
    total_chiplet_area_mm2: float
    chip_width_mm: float
    chip_height_mm: float
    total_power_w: float
    total_chiplet_power_w: float
    total_interposer_power_w: float
    total_link_power_w: float
    link_count: int
    min_link_length_mm: float
    avg_link_length_mm: float
    max_link_length_mm: float
    bottleneck_link: tuple[int, int] | None
    graph: dict[str, Any]
    network_context: NetworkContext


def evaluate_with_rapidchiplet(
    topology: Topology,
    workload: WorkloadPartition,
    cfg: EvalConfig,
    design_name: str,
) -> RapidChipletMetrics:
    global _LOCAL_PROXY_WARNED
    for flow in workload.flows:
        if flow.src >= topology.node_count or flow.dst >= topology.node_count:
            raise ValueError("canonical flow endpoint is outside the physical topology")
    try:
        rc = _load_rapidchiplet_module(cfg.rapidchiplet.root)
    except (FileNotFoundError, ModuleNotFoundError, ImportError) as exc:
        if not _LOCAL_PROXY_WARNED:
            warnings.warn(
                f"Using local RapidChiplet proxy because the official engine is unavailable: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _LOCAL_PROXY_WARNED = True
        return _evaluate_with_local_proxy(topology, workload, cfg)

    packaging = _read_rapidchiplet_json(cfg.rapidchiplet.root, cfg.rapidchiplet.packaging_file)
    technologies = _read_rapidchiplet_json(cfg.rapidchiplet.root, cfg.rapidchiplet.technology_file)
    chiplets = _build_chiplets(cfg)
    placement = _build_placement(topology, cfg)
    rc_topology = _build_topology(topology)
    routing_table = _build_splif_routing_table(topology.node_count, rc_topology)
    traffic_by_chiplet = dict(workload.traffic_bits_per_inference)

    inputs = {
        "design": {
            "design_name": design_name,
            "technologies": "in-memory",
            "chiplets": "in-memory",
            "placement": "in-memory",
            "topology": "in-memory",
            "packaging": "in-memory",
            "routing_table": "in-memory",
            "traffic_by_chiplet": "in-memory",
        },
        "chiplets": chiplets,
        "placement": placement,
        "topology": rc_topology,
        "packaging": packaging,
        "technologies": technologies,
        "routing_table": routing_table,
        "traffic_by_chiplet": traffic_by_chiplet,
    }
    intermediates: dict[str, Any] = {}
    do_compute = {metric: False for metric in rc.metrics}
    for metric in ("area_summary", "power_summary"):
        do_compute[metric] = True
    if rc_topology:
        do_compute["link_summary"] = True
    if traffic_by_chiplet:
        do_compute["latency"] = True
        do_compute["throughput"] = True

    outputs = rc.rapidchiplet(inputs, intermediates, do_compute, design_name, verbose=False, validate=False)
    area = outputs["area_summary"]
    power = outputs["power_summary"]
    link_lengths = intermediates.get("link_lengths", {})
    link_bandwidths = intermediates.get("link_bandwidths", {})
    link_power_w = _compute_link_power(packaging, link_lengths)
    latency = outputs.get("latency", {})
    throughput = outputs.get("throughput", {})

    avg_latency_cycles = float(latency.get("avg", 0.0))
    aggregate_throughput = float(throughput.get("aggregate_throughput", math.inf))
    total_traffic = workload.total_traffic_bits_per_inference
    network_limited_fps = (
        aggregate_throughput * cfg.chiplet.frequency_hz / total_traffic
        if total_traffic > 0 and math.isfinite(aggregate_throughput)
        else math.inf
    )
    link_lengths_once = [length for link, length in link_lengths.items() if link[0] < link[1]]
    bottleneck_link = _find_bottleneck_link(traffic_by_chiplet, routing_table, link_bandwidths)
    routes = _routes_for_traffic(traffic_by_chiplet, routing_table, topology.node_count)
    # Query the official per-flow formula using its actual PHY/packaging
    # intermediates. Do not reconstruct it from local center distances.
    path_latencies = {
        pair: float(rc.compute_latency(
            {**inputs, "traffic_by_chiplet": {pair: bits}}, intermediates
        )["avg"])
        for pair, bits in traffic_by_chiplet.items()
    }
    context = NetworkContext(
        backend="official_rapidchiplet",
        bandwidth_source="rapidchiplet_chiplet_packaging_inputs",
        path_latency_source="official_rapidchiplet_per_flow",
        frequency_hz=cfg.chiplet.frequency_hz,
        routes=routes,
        link_bandwidths_bits_per_cycle={
            (left[1], right[1]): float(bandwidth)
            for (left, right), bandwidth in link_bandwidths.items()
        },
        path_latencies_cycles=path_latencies,
    )

    return RapidChipletMetrics(
        avg_latency_cycles=avg_latency_cycles,
        avg_latency_ns=avg_latency_cycles / cfg.chiplet.frequency_hz * 1e9,
        min_latency_cycles=float(latency.get("min", 0.0)),
        max_latency_cycles=float(latency.get("max", 0.0)),
        network_limited_fps=network_limited_fps,
        aggregate_throughput_bits_per_cycle=aggregate_throughput,
        total_area_mm2=float(area["total_interposer_area"]),
        total_chiplet_area_mm2=float(area["total_chiplet_area"]),
        chip_width_mm=float(area["chip_width"]),
        chip_height_mm=float(area["chip_height"]),
        total_power_w=float(power["total_power"]) + link_power_w,
        total_chiplet_power_w=float(power["total_chiplet_power"]),
        total_interposer_power_w=float(power["total_interposer_power"]),
        total_link_power_w=link_power_w,
        link_count=len(rc_topology),
        min_link_length_mm=min(link_lengths_once, default=0.0),
        avg_link_length_mm=sum(link_lengths_once) / len(link_lengths_once) if link_lengths_once else 0.0,
        max_link_length_mm=max(link_lengths_once, default=0.0),
        bottleneck_link=bottleneck_link,
        graph=_build_graph(topology, workload, rc_topology, link_lengths),
        network_context=context,
    )


def _evaluate_with_local_proxy(
    topology: Topology,
    workload: WorkloadPartition,
    cfg: EvalConfig,
) -> RapidChipletMetrics:
    routes = shortest_paths_for_pairs(topology, workload.traffic_bits_per_inference)
    latency = compute_latency(
        topology=topology,
        routes=routes,
        traffic_bits_per_inference=workload.traffic_bits_per_inference,
        chiplet=cfg.chiplet,
        network=cfg.network,
    )
    throughput = compute_throughput(
        topology=topology,
        routes=routes,
        traffic_bits_per_inference=workload.traffic_bits_per_inference,
        chiplet=cfg.chiplet,
        network=cfg.network,
    )

    link_lengths_once = [edge.length_mm for edge in topology.edges]
    total_chiplet_area = topology.node_count * cfg.chiplet.width_mm * cfg.chiplet.width_mm
    total_area = _local_package_area(topology, cfg)
    total_chiplet_power = _local_chiplet_power(workload, cfg)
    total_link_power = _local_link_power(topology, workload, throughput.saturation_fps, cfg)
    aggregate_throughput_cycles = (
        throughput.aggregate_throughput_bits_per_second / cfg.chiplet.frequency_hz
        if math.isfinite(throughput.aggregate_throughput_bits_per_second)
        else math.inf
    )
    context = NetworkContext(
        backend="local_proxy",
        bandwidth_source="config.network_proxy",
        path_latency_source="local_proxy_per_flow",
        frequency_hz=cfg.chiplet.frequency_hz,
        routes=routes,
        link_bandwidths_bits_per_cycle={
            pair: cfg.network.link_bandwidth_bits_per_cycle
            for edge in topology.edges for pair in ((edge.a, edge.b), (edge.b, edge.a))
        },
        path_latencies_cycles={
            pair: compute_latency(topology, routes, {pair: bits}, cfg.chiplet, cfg.network).avg_cycles
            for pair, bits in workload.traffic_bits_per_inference.items()
        },
    )

    return RapidChipletMetrics(
        avg_latency_cycles=latency.avg_cycles,
        avg_latency_ns=latency.avg_ns,
        min_latency_cycles=latency.min_cycles,
        max_latency_cycles=latency.max_cycles,
        network_limited_fps=throughput.saturation_fps,
        aggregate_throughput_bits_per_cycle=aggregate_throughput_cycles,
        total_area_mm2=total_area,
        total_chiplet_area_mm2=total_chiplet_area,
        chip_width_mm=_local_package_width(topology, cfg),
        chip_height_mm=_local_package_height(topology, cfg),
        total_power_w=total_chiplet_power + total_link_power,
        total_chiplet_power_w=total_chiplet_power,
        total_interposer_power_w=0.0,
        total_link_power_w=total_link_power,
        link_count=len(topology.edges),
        min_link_length_mm=min(link_lengths_once, default=0.0),
        avg_link_length_mm=sum(link_lengths_once) / len(link_lengths_once) if link_lengths_once else 0.0,
        max_link_length_mm=max(link_lengths_once, default=0.0),
        bottleneck_link=throughput.bottleneck_link,
        graph=_build_local_graph(topology, workload),
        network_context=context,
    )


def rapid_input_fingerprint(cfg: EvalConfig) -> tuple[tuple[str, str], ...]:
    """Cache identity includes changing official model files, not only paths."""
    root = Path(cfg.rapidchiplet.root)
    names = (cfg.rapidchiplet.chiplet_file, cfg.rapidchiplet.packaging_file,
             cfg.rapidchiplet.technology_file, "rapidchiplet.py", "helpers.py")
    return tuple(
        (name, hashlib.sha256((root / name).read_bytes()).hexdigest()
         if (root / name).is_file() else "missing")
        for name in names
    )


def _routes_for_traffic(traffic, routing_table, node_count):
    routes = {}
    table = routing_table["table"]
    for src, dst in traffic:
        current, destination = ("chiplet", src), ("chiplet", dst)
        path = [src]
        while current != destination:
            current = tuple(table[current][destination])
            if current[0] != "chiplet" or current[1] in path or len(path) >= node_count:
                raise ValueError("invalid compute-chiplet route")
            path.append(current[1])
        routes[(src, dst)] = tuple(path)
    return routes


def _load_rapidchiplet_module(root: str):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"RapidChiplet root not found: {root_path}. Update configs/defaults.json rapidchiplet.root."
        )
    root_str = str(root_path)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module("rapidchiplet")


def _read_rapidchiplet_json(root: str, relative_path: str) -> Any:
    path = Path(root) / relative_path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_chiplets(cfg: EvalConfig) -> dict[str, Any]:
    chiplets = _read_rapidchiplet_json(cfg.rapidchiplet.root, cfg.rapidchiplet.chiplet_file)
    if cfg.rapidchiplet.chiplet_name not in chiplets:
        known = ", ".join(sorted(chiplets))
        raise KeyError(f"Unknown RapidChiplet chiplet '{cfg.rapidchiplet.chiplet_name}'. Known chiplets: {known}")
    return chiplets


def _build_placement(topology: Topology, cfg: EvalConfig) -> dict[str, Any]:
    return {
        "chiplets": [
            {
                "position": {"x": topology.positions[node][0], "y": topology.positions[node][1]},
                "rotation": 0,
                "name": cfg.rapidchiplet.chiplet_name,
            }
            for node in range(topology.node_count)
        ],
        "interposer_routers": [],
    }


def _build_topology(topology: Topology) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for edge in topology.edges:
        phy_a, phy_b = _endpoint_phys(topology, edge.a, edge.b)
        links.append(
            {
                "ep1": {"type": "chiplet", "outer_id": edge.a, "inner_id": phy_a},
                "ep2": {"type": "chiplet", "outer_id": edge.b, "inner_id": phy_b},
                "color": "#000099",
            }
        )
    return links


def _endpoint_phys(topology: Topology, a: int, b: int) -> tuple[int, int]:
    ax, ay = topology.positions[a]
    bx, by = topology.positions[b]
    dx = bx - ax
    dy = by - ay
    if abs(dx) >= abs(dy):
        return (2, 0) if dx >= 0 else (0, 2)
    return (1, 3) if dy >= 0 else (3, 1)


def _build_splif_routing_table(node_count: int, links: list[dict[str, Any]]) -> dict[str, Any]:
    adj: dict[tuple[str, int], list[tuple[str, int]]] = {
        ("chiplet", node): [] for node in range(node_count)
    }
    for link in links:
        a = (link["ep1"]["type"], link["ep1"]["outer_id"])
        b = (link["ep2"]["type"], link["ep2"]["outer_id"])
        adj[a].append(b)
        adj[b].append(a)
    for node in adj:
        adj[node].sort(key=_node_sort_key)

    table: dict[tuple[str, int], dict[tuple[str, int], tuple[str, int] | None]] = {}
    for src_id in range(node_count):
        src = ("chiplet", src_id)
        table[src] = {}
        for dst_id in range(node_count):
            dst = ("chiplet", dst_id)
            if src == dst:
                table[src][dst] = None
            else:
                path = _shortest_path_lowest_id_first(src, dst, adj)
                table[src][dst] = path[1]
    return {"type": "default", "table": table}


def _shortest_path_lowest_id_first(
    src: tuple[str, int],
    dst: tuple[str, int],
    adj: dict[tuple[str, int], list[tuple[str, int]]],
) -> tuple[tuple[str, int], ...]:
    queue: list[tuple[tuple[str, int], ...]] = [(src,)]
    seen = {src}
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == dst:
            return path
        for nxt in adj[node]:
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append(path + (nxt,))
    raise ValueError(f"No RapidChiplet route from {src} to {dst}")


def _node_sort_key(node: tuple[str, int]) -> tuple[int, int]:
    return (0 if node[0] == "chiplet" else 1, node[1])


def _compute_link_power(packaging: dict[str, Any], link_lengths: dict[Any, float]) -> float:
    # Mirrors RapidChiplet's compute_power_summary link-power block. The upstream
    # function computes this value internally but does not expose it in outputs.
    if packaging["link_power_type"] == "constant":
        return len(link_lengths) * float(packaging["link_power"]) / 2
    link_power_fn = eval(packaging["link_power"])
    return sum(link_power_fn(length) for length in link_lengths.values()) / 2


def _local_package_width(topology: Topology, cfg: EvalConfig) -> float:
    xs = [position[0] for position in topology.positions.values()]
    if not xs:
        return 0.0
    return max(xs) - min(xs) + cfg.chiplet.width_mm


def _local_package_height(topology: Topology, cfg: EvalConfig) -> float:
    ys = [position[1] for position in topology.positions.values()]
    if not ys:
        return 0.0
    return max(ys) - min(ys) + cfg.chiplet.width_mm


def _local_package_area(topology: Topology, cfg: EvalConfig) -> float:
    return _local_package_width(topology, cfg) * _local_package_height(topology, cfg)


def _local_chiplet_power(workload: WorkloadPartition, cfg: EvalConfig) -> float:
    max_ops = max(workload.ops_per_chiplet, default=0.0)
    total = 0.0
    for ops in workload.ops_per_chiplet:
        relative_load = 0.0 if max_ops <= 0 else min(1.0, ops / max_ops)
        total += cfg.power.chiplet_static_w
        total += cfg.power.chiplet_peak_dynamic_w * cfg.chiplet.utilization * relative_load
        total += 4 * cfg.power.phy_w
    return total


def _local_link_power(
    topology: Topology,
    workload: WorkloadPartition,
    saturation_fps: float,
    cfg: EvalConfig,
) -> float:
    static_w = sum(edge.length_mm * cfg.power.link_static_w_per_mm for edge in topology.edges)
    reference_activity_hz = cfg.power.proxy_reference_activity_hz
    if math.isfinite(saturation_fps):
        reference_activity_hz = min(reference_activity_hz, saturation_fps)
    traffic_energy_j = (
        workload.total_traffic_bits_per_inference
        * cfg.power.link_dynamic_pj_per_bit
        * 1e-12
    )
    return static_w + traffic_energy_j * reference_activity_hz


def _build_local_graph(topology: Topology, workload: WorkloadPartition) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": node,
                "x": topology.positions[node][0],
                "y": topology.positions[node][1],
                "stage": workload.stage_labels[node] if node < len(workload.stage_labels) else "",
                "ops": workload.ops_per_chiplet[node] if node < len(workload.ops_per_chiplet) else 0.0,
            }
            for node in range(topology.node_count)
        ],
        "physical_links": [
            {"source": edge.a, "target": edge.b, "length_mm": edge.length_mm}
            for edge in topology.edges
        ],
        "traffic_edges": [
            {"source": src, "target": dst, "mb": bits / 8 / 1024 / 1024}
            for (src, dst), bits in sorted(workload.traffic_bits_per_inference.items())
        ],
    }


def _find_bottleneck_link(
    traffic_by_chiplet: dict[tuple[int, int], float],
    routing_table: dict[str, Any],
    link_bandwidths: dict[Any, float],
) -> tuple[int, int] | None:
    if not traffic_by_chiplet or not link_bandwidths:
        return None

    table = routing_table["table"]
    routing_type = routing_table["type"]
    link_loads = {link: 0.0 for link in link_bandwidths}

    for (src_id, dst_id), traffic_bits in traffic_by_chiplet.items():
        previous_node: tuple[str, int] | str = "-1"
        current_node = ("chiplet", src_id)
        dst_node = ("chiplet", dst_id)
        while current_node != dst_node:
            if routing_type == "default":
                next_node = tuple(table[current_node][dst_node])
            elif routing_type == "extended":
                next_node = tuple(table[current_node][dst_node][previous_node])
            else:
                raise ValueError(f"Unsupported RapidChiplet routing table type: {routing_type}")
            link_loads[(current_node, next_node)] += traffic_bits
            previous_node = current_node
            current_node = next_node

    loaded_links = [
        (link_bandwidths[link] / load, link)
        for link, load in link_loads.items()
        if load > 0
    ]
    if not loaded_links:
        return None

    _throughput, ((src_type, src_id), (dst_type, dst_id)) = min(
        loaded_links,
        key=lambda item: (item[0], item[1]),
    )
    if src_type != "chiplet" or dst_type != "chiplet":
        return None
    return (src_id, dst_id)


def _build_graph(
    topology: Topology,
    workload: WorkloadPartition,
    links: list[dict[str, Any]],
    link_lengths: dict[Any, float],
) -> dict[str, Any]:
    physical_links = []
    for link in links:
        source = link["ep1"]["outer_id"]
        target = link["ep2"]["outer_id"]
        length = link_lengths.get((("chiplet", source), ("chiplet", target)), 0.0)
        physical_links.append({"source": source, "target": target, "length_mm": length})
    return {
        "nodes": [
            {
                "id": node,
                "x": topology.positions[node][0],
                "y": topology.positions[node][1],
                "stage": workload.stage_labels[node] if node < len(workload.stage_labels) else "",
                "ops": workload.ops_per_chiplet[node] if node < len(workload.ops_per_chiplet) else 0.0,
            }
            for node in range(topology.node_count)
        ],
        "physical_links": physical_links,
        "traffic_edges": [
            {"source": src, "target": dst, "mb": bits / 8 / 1024 / 1024}
            for (src, dst), bits in sorted(workload.traffic_bits_per_inference.items())
        ],
    }
