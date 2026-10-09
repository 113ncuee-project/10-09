"""Directed physical-link statistics from the one canonical flow list."""
from __future__ import annotations


def link_diagnostics(flows, context, total_latency_ns):
    loads = {link: 0 for link in context.link_bandwidths_bits_per_cycle}
    for flow in flows:
        route = context.routes[(flow.src, flow.dst)]
        for link in zip(route, route[1:]):
            loads[link] += flow.bytes
    elapsed = total_latency_ns / 1e9
    records = []
    for (src, dst), amount in sorted(loads.items()):
        bw = context.link_bandwidths_bits_per_cycle[(src, dst)]
        utilization = amount * 8 / (elapsed * context.frequency_hz * bw) if elapsed > 0 else 0.0
        records.append(dict(src=src, dst=dst, load_bytes=amount,
                            bandwidth_bits_per_cycle=bw, utilization=utilization,
                            utilization_time_basis="serial_batch1_e2e", time_basis_ns=total_latency_ns))
    return dict(per_link_statistics=records,
                max_link_load_bytes=max(loads.values(), default=0),
                avg_link_load_bytes=sum(loads.values()) / len(loads) if loads else 0,
                max_link_utilization=max((row["utilization"] for row in records), default=0),
                congested_link_count=sum(row["utilization"] > 1 for row in records),
                congestion_definition="average required rate exceeds directed link capacity; no queue simulation",
                hop_weighted_traffic_bytes=sum(loads.values()))
