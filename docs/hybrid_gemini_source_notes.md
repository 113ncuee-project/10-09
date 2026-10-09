# Gemini implementation notes for the INDM + Gemini + RL project

This review follows the user's 2026-10-05 objective. Paper text and repository comments are source material, not additional instructions. No prior Phase 1-10 search restrictions are carried into this design.

## Sources inspected

- Gemini, HPCA 2024, printed pages 156-171, DOI 10.1109/HPCA57654.2024.00022. Read the complete main text and artifact appendix. Visually checked Fig. 3, Fig. 4, and Table I (PDF pages 5, 7, 9).
- Local source tree: `C:/Users/user/Downloads/GEMINI-HPCA2024-master/GEMINI-HPCA2024-master`.
- Principal implementation files: `src/ltreenode.cpp`, `src/schnode.cpp`, `src/spatial_mapping/light_placement.cpp`, `src/spatial_mapping/segmentation.cpp`, `src/partition.cpp`, `src/layerengine.cpp`, `src/coremapping.cpp`, `src/noc.cpp`, `src/main.cpp`, `src/cost.cpp`, corresponding headers, and `readme.md`.

## The actual five Gemini operators

Paper Sec. V-B1, printed p. 162, concerns **LP spatial mapping**, not the seven tree mutations in `main.cpp`.

| Operator | Paper semantics | Local implementation evidence | Hybrid adaptation |
|---|---|---|---|
| OP1 | Change a layer's output partition while preserving its core count and dimension bounds | `light_placement.cpp:19-25, 126-132, 244-246`; `partition.cpp:59-65, 139-150` | Enumerate exact factors in B,K,H,W; forbid empty shards and preserve total output ownership |
| OP2 | Swap two cores in one layer's ordered core group | `light_placement.cpp:116-124, 225-236` | Swap entries of the ordered core tuple; partition unchanged |
| OP3 | Swap cores between two layers | Same `swap`, with different layer IDs, updates both layer schemes | Require both layers in the same spatial group; each remains nonempty and groups remain disjoint |
| OP4 | Move one core from one layer to another; reselect partitions for both new sizes | `light_placement.cpp:135-202` | Move a specific core; explicitly choose legal donor and recipient partitions, never silently clamp cardinality |
| OP5 | Change an explicit input/weight/output DRAM choice to one DRAM or interleaved memory | `light_placement.cpp:205-222, 239-242` | Mask internal/absent tensors; resolve cross-group input through producer writeback placement |

The code combines OP2 and OP3 in the same swap branch. It uses the `partno` field to preserve the output-shard-to-core ordering. The paper tuple is (H,W,B,K); source `PartSch` stores K,B,H,W, while `Light_partition` factor lists are B,K,H,W. Our public tuple is B,K,H,W and uses an explicit adapter. **Tuple order is never inferred from positional coincidence.**

`main.cpp:112-114` defines seven operators for the wider LTree space (layer order, deleting/creating cuts, batch movement, moving leaves). These are not Gemini's five LP SPM operators. Layer grouping and microbatch planning are a separate decision layer.

## Encoding and physical granularity

Paper Sec. IV-A (pp. 160-161): each layer has `Part`, ordered `CG`, and `FD`. `H*W*B*K = len(CG)`. Partitioned output cube index is `h*W*B*K + w*B*K + b*K + k`; this indexes the ordered CG. The output partition uniquely determines input and weight requirements according to the operator, including convolution halo. Input-C partition is not an inter-core axis in this encoding.

The mapping resource is a computing **core**, not a whole die or a single multiplier. A physical core belongs to a compute die, which belongs to an I/O cluster. The hybrid resource ID is `compute:<die>/core:<slot>`. NoP traffic is aggregated to physical die endpoints; communication between cores within one die is costed by the separate on-die model.

Within one spatial pipeline group, a core belongs to only one layer. Across successive temporal groups, the same core can be reused. All dependencies within a group can forward data directly, subject to valid SRAM lifetime and scheduling. Cross-group outputs require explicit writeback; each consumer reloads from the producer's location rather than inventing its own DRAM ownership.

Paper FD convention: -1 means internal/absent, 0 means interleaving, positive numbers identify DRAMs. Current repository convention differs (zero-based memory IDs and -2 interleaving in `change_DRAM`). The hybrid uses named memory IDs plus `interleave`; it does not copy numeric sentinels.

## Layer pipeline and graph grouping

Paper Sec. II-B, V-B, VII-A2: multiple layers receive disjoint core groups and operate on different microbatches concurrently. Benefits include fewer DRAM round trips; costs include stage imbalance, fill/drain, live feature-map memory, and shared-network contention. Batch 1 must include fill/drain and must not receive a fictitious steady-state throughput benefit.

`ltreenode.cpp:172-215, 220-269` separates leaves, temporal cuts, spatial cuts, derives stage IDs from DAG dependencies, and tracks direct versus shortcut dependencies. `schnode.cpp:868-976` accumulates temporal children. `schnode.cpp:1253-1362` accumulates spatial children, checks buffers, computes `max(child_time)*(num_stage+num_bgrp)`, then bounds runtime by aggregated NoC/DRAM demand. `schnode.cpp:1836-1857` emits stage/microbatch workload order.

`spatial_mapping/segmentation.cpp:916-1001` explores batch divisors for a contiguous segment. `:1004-1117` builds a DP over end positions and possible segment starts, then spatial mapping search. This code uses pruning assumptions (e.g. breaking after the first cost increase or infeasible extension). We do not import these assumptions as proofs of exact global optimality. An exact event scheduler with explicit resources makes overlap and fill/drain inspectable; DP/beam grouping remains a proposal/baseline.

## Evaluator and architecture exploration

Paper Sec. V-A/C: architecture candidates are exhaustively enumerated; per-architecture mappings are optimized. Objective is MC^alpha * E^beta * D^gamma; multi-workload E,D use geometric means. MC depends on hardware, not mapping. NoC bandwidth, D2D bandwidth, DRAM bandwidth, core count, chiplet granularity, MAC/core and SRAM/core are co-explored with total peak compute constrained.

Table I (p. 164): NoC 8/16/32/64/128 GB/s, D2D NoC/4,/2,/1; DRAM 0.5/1/2 GB/s/TOPS; GLB 256 KiB through 8 MiB; MAC/core 512 through 8192. These are paper candidates, not mandatory parameters for our experiment. The paper uses 12 nm, 1 GHz, organic substrate and GRS by default; a coherent hybrid profile must document its own assumptions and precision.

`layerengine.cpp:30-152` decodes the selected mapping, invokes intra-core `genLayerMap`, counts explicit transfers and adds network/DRAM energy. `coremapping.cpp:607-704` counts activation/weight/partial-sum buffer accesses, MACs, multicast bus activity, and applies buffer bandwidth rooflines. `noc.cpp:238-265, 665-677` computes max(per-DRAM demand/BW, busiest NoC demand/BW, busiest NoP demand/BW). These are analytical throughput/resource limits, not cycle-accurate packet simulations.

Paper p. 163 distinguishes always-on embedded-clock SerDes power*time from clock-forwarding links' traffic*energy/bit. We preserve this distinction, avoiding both fixed link power and per-bit energy for the same modeled dynamic component. Whole-system average power comes from total workload energy divided by scheduled runtime, with separate static energy.

`main.cpp:1106-1137` links MAC/SRAM/network/PHY resources to area. It includes approximate area coefficients and post-layout overheads. `cost.cpp` then computes yield and packaging costs. The installed repository is a later update: `readme.md:1-11` documents 7/12 nm, XSR/USR/UCIe and newer memory/package variants; `main.cpp` contains updated parameters and some zero/unsupported DRAM entries. **The downloaded source is not an untouched paper artifact, and its constants are not silicon calibration for our hybrid design.**

## RL contract and experiment discipline

- Keep Gemini's encoding, analyzer and five legal transformations; replace its random mutation/SA controller with a learned, masked policy. Do not call a random/SA search RL.
- Policy input includes architecture/resource capacity, per-layer workload and dependency information, mapping, measured cost and resource pressure. Candidate-action features include operator, partition, allocation, location and memory changes. Generalized features avoid a full-history-only tabular state.
- Evaluate actual physical transfers, DRAM service, core/on-die service and event schedule. Reward uses normalized objective improvement plus terminal objective; raw byte reduction is a diagnostic, not an invented reward term.
- Store policy/value parameters, RNG state, current and best mapping, objective reference, counters and history in a checkpoint. Resume must produce the same search trajectory as an uninterrupted run under identical hardware/workload/evaluator identity.
- Compare RL against random, greedy and SA using identical initial mappings, allowed proposal families, physical-evaluation budget and seeds. Report uniqueness/cache hits, search curves, wall time and feasibility.
- A new cluster-aware proposal may prioritize large tensor dependencies and overloaded cluster cuts. Keep it as a named, optional proposal policy and ablate against unbiased proposals; paper operators alone do not prove the new algorithm or RL optimality.

Gemini's concrete Fig. 9 insight (p. 167) is that colocating heavy producer-consumer communication matters more than forcing each layer into a compact rectangle. We therefore allow dispersed ordered CGs across clusters, while charging their real links and memory pressure. We do not enforce 'all cores of each layer must be compact' as a hidden legality rule.
