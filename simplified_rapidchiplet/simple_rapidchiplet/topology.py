from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Edge:
    a: int
    b: int
    length_mm: float

    def key(self) -> tuple[int, int]:
        return tuple(sorted((self.a, self.b)))


@dataclass(frozen=True)
class Topology:
    kind: str
    node_count: int
    positions: dict[int, tuple[float, float]]
    edges: tuple[Edge, ...]

    def neighbors(self) -> dict[int, list[tuple[int, float]]]:
        adj = {node: [] for node in range(self.node_count)}
        for edge in self.edges:
            adj[edge.a].append((edge.b, edge.length_mm))
            adj[edge.b].append((edge.a, edge.length_mm))
        for node in adj:
            adj[node].sort(key=lambda item: item[0])
        return adj

    def edge_lengths(self) -> dict[tuple[int, int], float]:
        return {edge.key(): edge.length_mm for edge in self.edges}


def make_topology(kind: str, node_count: int, chiplet_width_mm: float, spacing_mm: float) -> Topology:
    if node_count < 1:
        raise ValueError("node_count must be >= 1")
    if kind == "mesh":
        return make_mesh(node_count, chiplet_width_mm, spacing_mm)
    if kind == "tree":
        return make_tree(node_count, chiplet_width_mm, spacing_mm)
    raise ValueError(f"Unsupported topology '{kind}'. Use mesh or tree.")


def make_mesh(node_count: int, chiplet_width_mm: float, spacing_mm: float) -> Topology:
    positions, cols, rows = _grid_positions(node_count, chiplet_width_mm, spacing_mm)

    edges: list[Edge] = []
    for node in range(node_count):
        row = node // cols
        col = node % cols
        right = row * cols + col + 1
        down = (row + 1) * cols + col
        if col + 1 < cols and right < node_count:
            edges.append(_edge(node, right, positions))
        if row + 1 < rows and down < node_count:
            edges.append(_edge(node, down, positions))

    return Topology(kind="mesh", node_count=node_count, positions=positions, edges=tuple(edges))


def make_tree(node_count: int, chiplet_width_mm: float, spacing_mm: float) -> Topology:
    positions, _cols, _rows = _grid_positions(node_count, chiplet_width_mm, spacing_mm)
    edges = [_edge((node - 1) // 2, node, positions) for node in range(1, node_count)]

    return Topology(kind="tree", node_count=node_count, positions=positions, edges=tuple(edges))


def _grid_positions(
    node_count: int,
    chiplet_width_mm: float,
    spacing_mm: float,
) -> tuple[dict[int, tuple[float, float]], int, int]:
    pitch = chiplet_width_mm + spacing_mm
    cols = math.ceil(math.sqrt(node_count))
    rows = math.ceil(node_count / cols)
    positions: dict[int, tuple[float, float]] = {}
    for node in range(node_count):
        row = node // cols
        col = node % cols
        positions[node] = (col * pitch, row * pitch)
    return positions, cols, rows


def node_degrees(edges: Iterable[Edge], node_count: int) -> dict[int, int]:
    degrees = {node: 0 for node in range(node_count)}
    for edge in edges:
        degrees[edge.a] += 1
        degrees[edge.b] += 1
    return degrees


def _edge(a: int, b: int, positions: dict[int, tuple[float, float]]) -> Edge:
    ax, ay = positions[a]
    bx, by = positions[b]
    return Edge(a=a, b=b, length_mm=math.hypot(ax - bx, ay - by))
