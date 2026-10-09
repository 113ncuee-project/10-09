from __future__ import annotations

import heapq

from .topology import Topology


def shortest_path_lowest_id_first(topology: Topology) -> dict[tuple[int, int], tuple[int, ...]]:
    """RapidChiplet-style splif routing: shortest path, lowest-ID tie break."""

    adj = topology.neighbors()
    routes: dict[tuple[int, int], tuple[int, ...]] = {}
    for src in range(topology.node_count):
        for dst in range(topology.node_count):
            if src == dst:
                routes[(src, dst)] = (src,)
            else:
                routes[(src, dst)] = _shortest_path(src, dst, adj)
    return routes


def shortest_paths_for_pairs(
    topology: Topology,
    pairs: set[tuple[int, int]] | dict[tuple[int, int], object],
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Compute splif routes only for the traffic pairs used by this workload."""

    adj = topology.neighbors()
    routes: dict[tuple[int, int], tuple[int, ...]] = {}
    for src, dst in pairs:
        routes[(src, dst)] = (src,) if src == dst else _shortest_path(src, dst, adj)
    return routes


def _shortest_path(
    src: int,
    dst: int,
    adj: dict[int, list[tuple[int, float]]],
) -> tuple[int, ...]:
    heap: list[tuple[int, tuple[int, ...], int]] = [(0, (src,), src)]
    best: dict[int, tuple[int, tuple[int, ...]]] = {src: (0, (src,))}

    while heap:
        dist, path, node = heapq.heappop(heap)
        if node == dst:
            return path
        if best[node] < (dist, path):
            continue
        for nxt, _length in adj[node]:
            candidate = (dist + 1, path + (nxt,))
            if nxt not in best or candidate < best[nxt]:
                best[nxt] = candidate
                heapq.heappush(heap, (candidate[0], candidate[1], nxt))

    raise ValueError(f"No route from {src} to {dst}")
