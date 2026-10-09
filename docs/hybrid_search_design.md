# Learned Gemini-operator search for the hybrid accelerator

The implementation is in `simplified_rapidchiplet/simple_rapidchiplet/hybrid/actions.py` and `search.py`. It is a new controller for the current INDM + Gemini objective, not a continuation of the earlier K-only Q-learning search.

## Mapping and actions

Each layer owns an ordered tuple of mapping cores. A core ID such as `compute:2/core:1` resolves to one physical compute die and an on-die service slot. A spatial layer group gives disjoint core sets to simultaneous layers; temporal groups may reuse a core. The output partition is B,K,H,W and its product must exactly equal the ordered core count. All factors stay within output extents and microbatch size; uneven integer shards remain legal and are charged by the oracle.

The actions implement the paper's five spatial operators: repartition, within-layer swap, inter-layer swap, transfer one core and repartition both layers, and external DRAM placement/interleaving. Internal activations and absent weights have their OP5 field masked. Across a temporal-group boundary, input data follows the producer's writeback placement. Grouping, microbatch selection, tiling and collective choice are separate planning dimensions.

`legal_actions` produces a bounded seeded pool per operator. The default is 24 candidates per family (maximum 120). All candidates satisfy structural legality. Capacity and live-memory feasibility are evaluated by the physical oracle because those depend on tiling, schedule and hardware resources. A rejected physical candidate receives a recorded infeasibility outcome and does not modify the current mapping.

OP4 cannot empty its donor, and both new partition products match the new core counts. Its partition-pair proposals sample from legal factor combinations; destination insertion order is explicit. Bounded proposals preserve repeated reachability but do not establish exhaustive-search optimality.

The balanced initializer can leave cores unused. To make those resources searchable, OP4 also has explicitly recorded `hybrid_idle_acquire` and `hybrid_idle_release` variants. Acquisition adds one idle physical core at a chosen position in the ordered core group; release returns one selected core to the idle pool. Both choose an exact legal new partition and release cannot empty a layer. Spatial-group occupation determines idle status; sequential layers in a temporal group may reuse a core. These are extensions to Gemini, shared by all four search methods. Each available variant receives a representative in a sufficiently large bounded OP4 pool, and actor features distinguish acquisition, release, signed core delta and idle fraction.

Strict integer factors can disconnect one-core transitions on small tensors: with B=1 and K,H,W at most 8, a legal 48-core mapping cannot release to 47 or 46 cores. The separately recorded `hybrid_legal_resize` extension jumps to the nearest legal count in either direction; this example releases three selected cores to reach 45. Every resize records `core_delta` and the exact `resize_cores`, chooses a legal integer partition and preserves spatial disjointness. This extends the original operators and repairs the cardinality barrier without permitting empty output shards. It still does not establish global optimality.

## Actual learning

`LinearActorCritic` uses a masked softmax policy and a learned temporal-difference value baseline. It has 72 actor weights and 10 value weights. Candidate features include the operator, target/recipient compute demand, dependency data, allocation, partition change, core locality, physical location, DRAM selector, idle-pool variant and state/operator interaction terms. The interactions are necessary: adding a state-only constant to every softmax candidate would cancel and would not give a context-sensitive policy.

The critic observes fixed-reference latency and energy, current objective, runtime communication fraction, stage imbalance, compute utilization, grouping, microbatch and episode progress. When a dedicated communication-time metric is absent, busiest link, I/O crossbar or DMA occupancy provides a congestion observation. When utilization is absent, total compute busy time divided by used-core capacity and makespan provides the observation. These are state features, not extra reward terms. The actor receives actual cluster IDs and die positions when provided by `PackageGraph`; no fixed `die_index // 4` cluster assumption is used.

The update is a policy-gradient actor update with TD advantage, a small entropy term and norm clipping, plus the value-baseline update. Tests verify that positive reinforcement changes action probability and that production search updates nonzero actor weights. This is a learned policy; its usefulness must still be established by measured comparisons. A short run is not a claim that RL outperforms SA or random search.

## Objective and reward

The default physical objective is end-to-end batch latency times total batch energy. The total workload batch stays fixed when microbatch size changes. Optional area/power/latency hard limits can reject candidates. Optional multiplicative exponents can include area and power. Reference scales come from the initial feasible mapping and remain fixed for every candidate.

Per-step reward is `log(previous_cost / candidate_cost)`. A terminal horizon step also receives `-log(candidate_cost / initial_cost)`. Infeasible candidates receive a finite negative outcome. There is no manual OC/IC efficiency coefficient and no byte-count bonus in the reward. Traffic remains an explanatory diagnostic and, for the optional proposal ablation, a proposal signal.

`SearchResult.best_cost` is normalized to the initial mapping of that search. Architecture co-exploration must rank hardware with raw batch EDP or a common reference shared by all hardware candidates. Comparing each architecture's improvement ratio would favor a poor baseline instead of a physically better design.

## Budget and baselines

`budget` limits cumulative calls to the physical evaluator for unique mapping plans. The initial evaluation consumes one query. A memoized repeated mapping is a cache hit, separately counted. Search traces record physical evaluations, requests, chosen operator, masked operator families, candidate cost, best-so-far cost, acceptance, feasibility, probability and entropy. A separate request ceiling prevents an infinite cache-only loop if the finite proposal space is revisited.

`compare_searches` uses identical initial plans, seeds, proposal families, objective and physical-evaluation budgets, with independent caches for RL, random, greedy and SA. RL and random accept feasible proposals to explore the state space; greedy accepts an improving member of a small sampled neighborhood; SA additionally accepts some worsening proposals under a fixed cooling rule. Best-so-far mappings are always retained. Reporting should include raw PPA/energy, search curves and elapsed time in addition to the best objective.

The optional `cluster_guided` proposal variant requires actual cluster metadata and retains a uniform minority of proposals in every family. It is an independently named heuristic ablation, not part of Gemini's original algorithm or the default RL reward. Results must identify its use.

The current policy uses package die positions for mapping cores, so it does not explicitly encode every on-die ring position. DRAM proximity uses actual core and memory-controller positions when supplied; it is zero when geometry is absent, and never derives a distance from identifier numbers. Exact memory placement, routes and congestion still come from the evaluator. These limit policy generalization and make the cluster-guided proposal a coarse heuristic, rather than a reproduction of INDM's full mapper.

## Resuming work

Checkpoints contain policy/value weights, RNG state, mapping, best mapping, objective reference, oracle cache, evaluation/request counters, episode position and the trace. Identity covers workload, initial plan, hardware/evaluator fingerprint, resources, cluster/geometry metadata, objective and search settings. A changed identity rejects resume.

`budget` on resume is a cumulative ceiling, allowing a saved 64-query run to continue to 128 queries. Horizon and cooling parameters remain unchanged. A behavioral test compares a 31-query uninterrupted RL run against an 11-query checkpoint followed by a resumed 31-query run and verifies identical trace, best mapping and policy parameters. Wall-clock timing is intentionally excluded from exact equality.

Checkpoints enable continuation when the process can run again; they do not control account usage limits or make a sleeping machine execute. The parent task can register an authorized heartbeat to resume saved work after a rate-limit reset. A local work plan must remain recoverable in the checkpoint and user-visible progress files.

## Verified checks

- Eighteen behavioral tests cover exact compiled MAC/output ownership across all five operators and idle/resize variants, integer cardinality barriers, empty-shard masks, temporal allocation restrictions, external-flow masks, real memory geometry, actor updates, objective hard limits, physical query accounting, equal-reference baselines, physical resource state observations and exact RL continuation.
- A short real `HybridEvaluator` residual-workload check compared eight uninterrupted evaluations with four evaluations followed by a resumed eight-evaluation ceiling. Trace, best mapping, physical metrics and actor/value parameters matched exactly. This is continuation correctness evidence, not a performance comparison.
- Real ResNet-50 v1.5 smoke: 73 fused operators, 25 initial groups, 64 mapping cores. All five families produced 24 candidates; proposal generation took approximately 0.257 seconds on this host. Timing is a local diagnostic, not a performance benchmark.
- Full fidelity depends on the separately implemented mapping compiler, on-die model, memory service, physical RapidChiplet adapter and event scheduler. Structural search tests do not calibrate silicon accuracy.

## Completed frozen-source experiments

`simplified_rapidchiplet/results/hybrid_search_validation_final` contains the complete physical-oracle comparison: three seeds, four search methods, two proposal strategies, and cumulative 64/128-query curves. At 128 queries, mean relative EDP is 0.54877 for uniform RL and 0.42059 for guided RL; greedy has the lowest measured mean for both proposal strategies (0.37318 and 0.32802). These measurements do not support claiming that short-budget RL beats greedy.

The separate `hybrid_search_validation_rl512` directory continues only the six RL trials to 512 queries. Mean relative EDP reaches 0.36763 for uniform and 0.32285 for guided proposals. No same-budget 512-query baseline comparison was run. Both experiments use frozen source identity `558f718b91136b09fe68028921bb45e3c6deb565a066e7440244bdbdff8114c7`, a balanced initializer, cold weights, real physical services, and explicitly uncalibrated coefficients. `summary.json/csv/md`, `learning_curves.csv` and `milestones.csv` provide statistics and plot inputs; exact real-oracle continuation evidence is in `hybrid_search_validation_final/verification_resume.json`.

Traffic diagnostics must accompany the objective: in uniform seed8, the 128→512 improvement changes core counts from 48/3/10 to 35/2/3; NoC hop bytes fall 197632→165824 and NoP hop bytes fall 73536→69664, while logical transfer and DRAM-read bytes stay constant. The final NoP volume remains above the initial mapping's 66144 hop bytes. Much of this small workload's energy improvement follows its lower makespan and static idle energy, so improved EDP is not evidence that every form of interdie traffic decreased.
