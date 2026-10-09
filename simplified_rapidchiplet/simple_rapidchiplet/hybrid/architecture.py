"""Architecture partitioning with conserved compute/storage resources."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .hardware import HardwareSpec, build_package


def resource_totals(spec):
    count = spec.compute_count
    return {
        "compute_dies": count,
        "io_dies": spec.cluster_count,
        "mapping_cores": spec.core_count,
        "pes": count * spec.pe_count,
        "macs_per_second": count * spec.macs_per_second,
        "l1_bytes": count * spec.pe_count * spec.l1_bytes_per_pe,
        "l2_weight_bytes": count * spec.l2_weight_bytes,
        "l2_activation_bytes": count * spec.l2_activation_bytes,
        "l2_output_bytes": count * spec.l2_output_bytes,
        "unified_bytes": count * spec.unified_buffer_bytes,
        "allocated_scratch_bytes": count * spec.prefetch_slots * (spec.pe_count * spec.l1_bytes_per_pe + spec.l2_bytes_per_die),
        "dram_bandwidth_Bps": spec.cluster_count * spec.dram_bandwidth_Bps,
        "dram_capacity_bytes": spec.cluster_count * spec.dram_capacity_bytes,
    }


@dataclass(frozen=True)
class ArchitectureCandidate:
    id: str
    hardware: HardwareSpec
    conserved_resources: tuple[str, ...]

    @property
    def name(self):
        return self.id

    @property
    def spec(self):
        return self.hardware

    @property
    def package(self):
        return build_package(self.hardware)

    def to_dict(self):
        return {"id": self.id, "hardware": self.hardware.to_dict(),
                "resource_totals": resource_totals(self.hardware),
                "conserved_resources": self.conserved_resources,
                "varying_costs": "compute/control/PHY replication, IO/DDR/controller/crossbar, die yields and package network"}


def enumerate_architectures(reference=None, compute_counts=(4, 8, 16),
                            noc_topologies=("multi_ring", "mesh")):
    """Partition one resource budget into legal four-compute I/O clusters.

    The default conserves 256 PEs, 64 mapping cores, 32MiB unified memory,
    distributed L1/L2 capacity, and 75GB/s aggregate DRAM bandwidth. It does
    not give extra dies free PHYs, controllers, or aggregate NoP bandwidth.
    Those change the physical hardware and its resource-derived cost.
    """
    reference = reference or HardwareSpec()
    totals = resource_totals(reference)
    conserved = tuple(key for key in totals if key not in {"compute_dies", "io_dies"})
    result = []
    for count in compute_counts:
        if count < 4 or count % 4:
            raise ValueError("architecture candidates require complete four-compute clusters")
        clusters = count // 4
        if totals["pes"] % count or totals["mapping_cores"] % count:
            raise ValueError("PE/core resource budget cannot be divided into candidate dies")
        rows = max(divisor for divisor in range(1, math.isqrt(clusters) + 1) if clusters % divisor == 0)
        cols = clusters // rows
        per_die = {}
        for field in ("l2_weight_bytes", "l2_activation_bytes", "l2_output_bytes", "unified_buffer_bytes"):
            total_key = "unified_bytes" if field == "unified_buffer_bytes" else field
            if totals[total_key] % count:
                raise ValueError(f"resource {field} cannot be divided into candidate dies")
            per_die[field] = totals[total_key] // count
        if totals["dram_capacity_bytes"] % clusters:
            raise ValueError("DRAM capacity cannot be divided into candidate clusters")
        for topology in noc_topologies:
            cores = totals["mapping_cores"] // count
            spec = replace(reference, cluster_rows=rows, cluster_cols=cols, compute_per_cluster=4,
                           cores_per_die=cores, pe_count=totals["pes"] // count,
                           ring_count=cores, dram_bandwidth_Bps=totals["dram_bandwidth_Bps"] / clusters,
                           dram_capacity_bytes=totals["dram_capacity_bytes"] // clusters,
                           noc_topology=topology, **per_die)
            observed = resource_totals(spec)
            for key in conserved:
                if not math.isclose(observed[key], totals[key], rel_tol=1e-12, abs_tol=0):
                    raise ValueError(f"resource conservation failed: {key}")
            build_package(spec)  # Includes port/geometry/connectivity checks.
            result.append(ArchitectureCandidate(f"dies{count}_{topology}", spec, conserved))
    return tuple(result)
