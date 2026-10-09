"""Preference-conditioned tabular Q-learning over 18 semantic decisions."""
from __future__ import annotations

import json
import math
import random
import hashlib
from pathlib import Path
from dataclasses import asdict, dataclass, replace

from .assignment_generator import legal_assignments
from .config import SearchConfig
from .physical_evaluator import evaluate_physical
from .preference_dse import PreferenceProfile, make_preference_profile
from .rapidchiplet_engine import rapid_input_fingerprint


@dataclass(frozen=True)
class PhysicalState:
    model_fingerprint: str
    total_chiplets: int
    next_block: int
    assignments: tuple[tuple[int, ...], ...]
    previous_active_chiplets: tuple[int, ...]
    live_residency: tuple
    primary_ownership: tuple
    next_tensor_shapes: tuple
    next_operator_macs: tuple
    cumulative_latency_ns: float
    preference_key: str
    evaluator_identity: str


class PhysicalOracle:
    def __init__(self, graph, cfg, profile, *, fixed_assignment=False, fixed_active_count=None):
        self.graph, self.cfg = graph, cfg
        self.search = cfg.search or SearchConfig()
        self.profile = profile
        self.fixed_assignment, self.fixed_active_count = fixed_assignment, fixed_active_count
        self.identity = json.dumps({"model":graph.fingerprint, "config":asdict(cfg),
                                   "native":rapid_input_fingerprint(cfg), "preference":asdict(profile),
                                   "evaluator_sources":{path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in sorted(Path(__file__).parent.glob("*.py"))},
                                   "fixed_assignment":fixed_assignment, "fixed_active_count":fixed_active_count}, sort_keys=True)
        self.cache, self.prefix_cache = {}, {}
        self.cache_hits = 0

    def key(self, n, assignments):
        return self.identity, n, tuple(tuple(ids) for ids in assignments)

    def evaluate(self, n, assignments):
        key = self.key(n, assignments)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        value = evaluate_physical(self.graph, n, assignments, self.cfg)
        self.cache[key] = value
        return value

    def state(self, n, assignments=()):
        assignments = tuple(tuple(ids) for ids in assignments)
        index = len(assignments)
        latency, residency, primary = 0.0, (), ()
        if assignments:
            key = self.key(n, assignments)
            if key not in self.prefix_cache:
                prefix = evaluate_physical(self.graph, n, assignments, self.cfg, prefix=True)
                self.prefix_cache[key] = (prefix.metrics["total_latency_ns"], prefix.derived.final_residency,
                                          prefix.derived.final_primary)
            latency, residency, primary = self.prefix_cache[key]
        block = self.graph.blocks[index] if index < len(self.graph.blocks) else None
        shapes = tuple((name, self.graph.tensor_map[name].shape) for name in block.inputs+block.outputs) if block else ()
        macs = tuple((name, self.graph.operation_map[name].macs) for name in block.operators) if block else ()
        return PhysicalState(self.graph.fingerprint,n,index,assignments,assignments[-1] if assignments else (),
                             residency,primary,shapes,macs,latency,json.dumps(asdict(self.profile),sort_keys=True),self.identity)

    def actions(self, state):
        if state.next_block >= len(self.graph.blocks):
            return ()
        return legal_assignments(self.graph,state.next_block,state.total_chiplets,state.previous_active_chiplets,
                                 state.live_residency,state.primary_ownership,self.search.assignment_top_k,
                                 fixed_assignment=self.fixed_assignment,fixed_active_count=self.fixed_active_count)


def physical_profile(graph, cfg, name="balanced", *, weights=None, **limits):
    if name == "custom" and weights is None:
        raise ValueError("custom preference requires explicit L/A/P weights")
    profile = make_preference_profile("balanced" if name == "custom" else name, cfg=cfg, **limits)
    if weights is not None:
        if any(not math.isfinite(x) or x < 0 for x in weights) or sum(weights) <= 0:
            raise ValueError("custom weights must be finite, nonnegative, and sum positive")
        profile = replace(profile, name="custom",latency_weight=weights[0]/sum(weights), area_weight=weights[1]/sum(weights),power_weight=weights[2]/sum(weights))
    reference = evaluate_physical(graph,cfg.max_chiplets,((0,),)*len(graph.blocks),cfg)
    single_compute = sum(op.macs for op in graph.operators)*cfg.chiplet.op_per_mac/(cfg.chiplet.peak_ops_per_second*cfg.chiplet.utilization)*1e9
    return replace(profile,latency_scale_ns=max(single_compute,1e-12),
                   area_scale_mm2=reference.metrics["total_area_mm2"],power_scale_w=reference.metrics["total_power_w"])


def score(evaluation, profile):
    m = evaluation.metrics
    metrics = (m["total_latency_ns"],m["total_area_mm2"],m["total_power_w"])
    limits = (profile.max_latency_ns,profile.max_area_mm2,profile.max_power_w)
    scales = (profile.latency_scale_ns,profile.area_scale_mm2,profile.power_scale_w)
    violation = sum(max(0,value/limit-1) for value,limit in zip(metrics,limits) if math.isfinite(limit))
    if not m["architecture_feasible"]:
        violation += 1
    cost = sum(weight*value/(limit if math.isfinite(limit) else scale)
               for weight,value,limit,scale in zip(profile.weights,metrics,limits,scales))
    feasible = m["architecture_feasible"] and violation == 0
    # Strict feasibility priority with a bounded monotone transform of L/A/P.
    reward = 1/(1+cost) if feasible else -violation
    return dict(feasible=feasible, weighted_cost=cost, normalized_violation=violation, reward=reward)


def rank(evaluation, profile):
    s = score(evaluation,profile)
    return (not s["feasible"], s["normalized_violation"] if not s["feasible"] else 0,
            s["weighted_cost"],evaluation.metrics["N_total"],tuple(map(tuple,evaluation.metrics["assignments"])))


@dataclass
class PhysicalSearchResult:
    best: object
    candidates: list
    profile: object
    seed: int
    episodes: int
    history: list
    q_entries: int
    unique_evaluations: int
    prefix_evaluations: int
    cache_hits: int
    method: str = "Q-learning"
    complete: bool = False


def q_learning_physical(oracle, *, budget=64, seed=20260915, max_episodes=None):
    if budget < 1:
        raise ValueError("evaluation budget must be positive")
    rng, q, history = random.Random(seed), {}, []
    settings = oracle.search
    episodes = 0
    cap = max_episodes if max_episodes is not None else budget*20
    if cap < 1:
        raise ValueError("episode limit must be positive")
    architecture_actions = tuple(range(1,oracle.cfg.max_chiplets+1))
    root = ("architecture",oracle.identity)
    while len(oracle.cache) < budget and episodes < cap:
        # Uniform legal starts cover each hardware size; subsequent
        # episodes use the same epsilon-greedy Q policy at every decision.
        if episodes < len(architecture_actions):
            n = architecture_actions[episodes]
        elif rng.random() < settings.exploration:
            n = rng.choice(architecture_actions)
        else:
            values = [q.get((root,n),0.0) for n in architecture_actions]
            maximum = max(values)
            n = rng.choice([n for n,v in zip(architecture_actions,values) if v == maximum])
        trajectory = [(root,n)]
        state = oracle.state(n)
        while state.next_block < len(oracle.graph.blocks):
            actions = oracle.actions(state)
            if not actions:
                raise ValueError("search mode leaves a state without legal actions")
            if episodes < len(architecture_actions):
                action = min(actions,key=lambda a:(-len(a.chiplets),a.boundary_hop_byte_lower_bound,a.chiplets))
            elif rng.random() < settings.exploration:
                action = rng.choice(actions)
            else:
                values = [q.get((state,a.chiplets),0.0) for a in actions]
                maximum = max(values)
                action = rng.choice([a for a,v in zip(actions,values) if v == maximum])
            trajectory.append((state,action.chiplets))
            assignments = state.assignments+(action.chiplets,)
            if len(assignments) == len(oracle.graph.blocks):
                break
            state = oracle.state(n,assignments)
        evaluation = oracle.evaluate(n,assignments)
        breakdown = score(evaluation,oracle.profile)
        # Backward TD backup; terminal reward is solely feasibility and L/A/P.
        for index in reversed(range(len(trajectory))):
            state_key, action_key = trajectory[index]
            if index == len(trajectory)-1:
                target = breakdown["reward"]
            else:
                next_state = trajectory[index+1][0]
                next_actions = oracle.actions(next_state)
                target = settings.discount*max(q.get((next_state,a.chiplets),0.0) for a in next_actions)
            key = state_key,action_key
            q[key] = q.get(key,0.0)+settings.learning_rate*(target-q.get(key,0.0))
        episodes += 1
        history.append(dict(episode=episodes, unique_evaluations=len(oracle.cache), N_total=n,
                            assignments=assignments, **breakdown))
    candidates = list(oracle.cache.values())
    best = min(candidates,key=lambda e:rank(e,oracle.profile))
    return PhysicalSearchResult(best,candidates,oracle.profile,seed,episodes,history,len(q),len(candidates),len(oracle.prefix_cache),oracle.cache_hits)


def exhaustive_physical(oracle, *, max_evaluations=None):
    if oracle.cfg.max_chiplets > 4 or len(oracle.graph.blocks) > 3:
        raise ValueError("exhaustive validation is limited to toy graphs: <=4 chiplets, <=3 blocks")
    cap = max_evaluations or oracle.search.exhaustive_max_evaluations
    history = []
    def visit(state):
        if state.next_block == len(oracle.graph.blocks):
            if len(oracle.cache) >= cap:
                raise RuntimeError("exhaustive bound reached; no global-optimum claim is valid")
            value = oracle.evaluate(state.total_chiplets,state.assignments)
            history.append(dict(index=len(history),**score(value,oracle.profile)))
            return
        for action in oracle.actions(state):
            assignments = state.assignments+(action.chiplets,)
            if len(assignments) == len(oracle.graph.blocks):
                visit(replace(state,next_block=len(assignments),assignments=assignments))
            else:
                visit(oracle.state(state.total_chiplets,assignments))
    for n in range(1,oracle.cfg.max_chiplets+1):
        visit(oracle.state(n))
    candidates = list(oracle.cache.values())
    return PhysicalSearchResult(min(candidates,key=lambda e:rank(e,oracle.profile)),candidates,oracle.profile,0,0,history,0,len(candidates),len(oracle.prefix_cache),oracle.cache_hits,"exhaustive bounded candidate space",True)
