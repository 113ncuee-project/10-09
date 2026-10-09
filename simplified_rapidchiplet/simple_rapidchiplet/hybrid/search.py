"""Checkpointable learned mapping search and fair random/greedy/SA baselines.

The physical oracle is injected: evaluator(MappingPlan) -> metrics dictionary
or an object with a metrics dictionary. No search-dependent hardware estimator
or communication efficiency multiplier is defined in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from collections.abc import Mapping

import numpy as np

from .actions import OPERATORS, apply_action, legal_actions, operator_mask, partition_tuple
from .mapping import LayerGroup, LayerMapping, MappingPlan, Partition


ACTION_FEATURE_SIZE = 72
CHECKPOINT_SCHEMA = 3


@dataclass(frozen=True)
class SearchObjective:
    """Fixed-reference multiplicative objective; EDP is the default.

    Scales come from the initial valid mapping and never from candidates.
    Hard limits are physical feasibility conditions rather than reward bonuses.
    """
    latency_weight: float = 1.0
    energy_weight: float = 1.0
    area_weight: float = 0.0
    power_weight: float = 0.0
    max_area_mm2: float | None = None
    max_power_w: float | None = None
    max_latency_s: float | None = None
    soft_area_mm2: float | None = None
    soft_power_w: float | None = None
    soft_latency_s: float | None = None
    relax_hard_limits: bool = False

    def __post_init__(self):
        weights = (self.latency_weight, self.energy_weight, self.area_weight, self.power_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not any(weights):
            raise ValueError("objective weights must be nonnegative with at least one positive weight")
        if any(limit is not None and (not math.isfinite(limit) or limit <= 0)
               for limit in (self.max_area_mm2, self.max_power_w, self.max_latency_s,
                             self.soft_area_mm2,self.soft_power_w,self.soft_latency_s)):
            raise ValueError("objective hard limits must be positive")

    def cost(self, metrics, reference):
        if metrics.get("feasible", metrics.get("architecture_feasible", True)) is False:
            return math.inf
        gaps = []
        for key, limit in (("area_mm2", self.max_area_mm2), ("power_w", self.max_power_w),
                           ("latency_s", self.max_latency_s)):
            if limit is not None:
                value = metrics.get(key)
                if value is None or not math.isfinite(float(value)) or float(value)<=0:
                    return math.inf
                gaps.append(max(0.0,float(value)/limit-1))
        if any(gaps) and not self.relax_hard_limits:
            return math.inf
        log_cost = 0.0
        for key, weight in (("latency_s", self.latency_weight), ("energy_j", self.energy_weight),
                            ("area_mm2", self.area_weight), ("power_w", self.power_weight)):
            if not weight:
                continue
            value, base = metrics.get(key), reference.get(key)
            if value is None or base is None or not math.isfinite(float(value)) or float(value) <= 0 or float(base) <= 0:
                return math.inf
            log_cost += weight * math.log(float(value) / float(base))
        # Optional goals affect search without rejecting valid hardware. This
        # normalized penalty is our extension to Gemini's exponent objective.
        for key,limit in (('area_mm2',self.soft_area_mm2),('power_w',self.soft_power_w),('latency_s',self.soft_latency_s)):
            if limit is not None:
                value = metrics.get(key)
                if value is None or not math.isfinite(float(value)) or float(value)<=0:
                    return math.inf
                log_cost += math.log(max(1.0,float(value)/limit))
        cost = math.exp(max(-700.0, min(log_cost, 700.0)))
        if self.relax_hard_limits and gaps:
            bounded = cost/(1+cost)
            # Every compliant candidate beats every noncompliant candidate,
            # yet search can start above an impossible limit and make progress.
            return 1+max(gaps)+sum(gaps)/len(gaps)+1e-6*bounded if any(gaps) else bounded
        return cost


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    raise TypeError(f"nonserializable checkpoint value {type(value).__name__}")


def plan_fingerprint(plan):
    return hashlib.sha256(json.dumps(asdict(plan), sort_keys=True).encode()).hexdigest()


def plan_from_dict(data):
    groups = tuple(LayerGroup(item["id"], tuple(item["layer_ids"]), item.get("mode", "spatial"))
                   for item in data["groups"])
    mappings = []
    for item in data["layer_mappings"]:
        copied = dict(item)
        copied["chiplets"] = tuple(copied["chiplets"])
        copied["partition"] = Partition(**copied["partition"])
        copied["tile_shape"] = tuple(copied.get("tile_shape", (0, 0, 0, 0)))
        copied["microtile_shape"] = tuple(copied.get("microtile_shape", (0, 0, 0, 0)))
        mappings.append(LayerMapping(**copied))
    return MappingPlan(groups, tuple(mappings), data.get("microbatch_size", 1),
                       data.get("collective", "multicast"), tuple(data.get("memory_ids", ("memory:0",))))


def _metrics(result):
    value = result if isinstance(result, Mapping) else getattr(result, "metrics", None)
    if value is None and hasattr(result, "to_dict"):
        value = result.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("hybrid evaluator must return metrics or an object with .metrics")
    return _jsonable(value)


def _number(metrics, key, fallback=0.0):
    value = metrics.get(key, fallback)
    return fallback if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) else float(value)


def _log_ratio(value, reference):
    return float(np.clip(math.log(max(value, 1e-30) / max(reference, 1e-30)), -3.0, 3.0))


def state_features(workload, plan, metrics, reference, current_cost, progress):
    total_macs = max(sum(layer.macs_per_sample for layer in workload.layers), 1)
    allocations = [max(workload.layer_map[item.layer_id].macs_per_sample, 1) / len(item.chiplets)
                   for item in plan.layer_mappings]
    largest = max(allocations, default=1.0)
    average = sum(allocations) / max(len(allocations), 1)
    communication = _number(metrics, "communication_latency_s", _number(metrics, "network_latency_s"))
    latency = _number(metrics, "latency_s", 1.0)
    busy = metrics.get("resource_busy_s", {})
    if not communication and isinstance(busy, Mapping):
        communication = max((float(value) for name, value in busy.items()
                             if name.startswith("link:") or name.endswith(":crossbar") or name.endswith(":dma")),
                            default=0.0)
    core_ids = {core for item in plan.layer_mappings for core in item.chiplets}
    compute_utilization = _number(metrics, "compute_utilization", _number(metrics, "core_utilization", -1.0))
    if compute_utilization < 0:
        compute_utilization = _number(metrics, "compute_busy_s") / max(len(core_ids) * latency, 1e-30)
    return np.asarray([
        1.0, _log_ratio(current_cost, 1.0),
        _log_ratio(latency, _number(reference, "latency_s", 1.0)),
        _log_ratio(_number(metrics, "energy_j", 1.0), _number(reference, "energy_j", 1.0)),
        min(communication / max(latency, 1e-30), 1.0),
        min(largest / max(average, 1e-30), 4.0) / 4,
        min(compute_utilization, 1.0),
        min(len(plan.groups) / max(len(workload.layers), 1), 1.0),
        min(plan.microbatch_size / max(workload.batch_size, 1), 1.0),
        min(max(float(progress), 0.0), 1.0),
    ], dtype=np.float64)


def _core_locality(cores):
    dies = [core.split("/")[0] for core in cores]
    return len(set(dies)) / max(len(dies), 1)


def action_features(workload, plan, action, state, core_count, core_clusters=None, core_positions=None,
                    memory_positions=None):
    """Generalized local features plus state/operator interactions.

    Core IDs enter through location/locality, not a full-history lookup table.
    A caller may replace the oracle while retaining this real learned policy.
    """
    mapping = plan.mapping_map
    target = mapping[action.layer_id]
    other = mapping.get(action.other_layer_id)
    layer = workload.layer_map[action.layer_id]
    total_macs = max(sum(item.macs_per_sample for item in workload.layers), 1)
    total_bytes = max(sum(tensor.bytes for tensor in workload.tensors), 1)
    demand = sum(workload.tensor_map[name].bytes for name in layer.inputs) + layer.weight_bytes
    target_parts = np.asarray(partition_tuple(target.partition), dtype=np.float64)
    changed_parts = np.asarray(action.partition or partition_tuple(target.partition), dtype=np.float64)
    delta = np.clip(np.log(changed_parts / target_parts) / math.log(2), -3.0, 3.0) / 3
    one_hot = np.asarray([float(action.operator == operator) for operator in OPERATORS])
    core_a = target.chiplets[min(action.index, len(target.chiplets) - 1)]
    if action.variant == "hybrid_idle_acquire" or (action.variant == "hybrid_legal_resize" and action.core_delta > 0):
        core_b = action.idle_core or action.resize_cores[0]
    elif action.operator == "OP2":
        core_b = target.chiplets[min(action.other_index, len(target.chiplets) - 1)]
    elif other is not None:
        core_b = other.chiplets[min(action.other_index, len(other.chiplets) - 1)]
    else:
        core_b = core_a
    other_macs = workload.layer_map[other.layer_id].macs_per_sample if other is not None else 0
    same_die = float(core_a.split("/")[0] == core_b.split("/")[0])
    distance = 0.0
    same_cluster = float(core_clusters[core_a] == core_clusters[core_b]) if core_clusters is not None else same_die
    if core_positions is not None:
        xs, ys = zip(*core_positions.values())
        diameter = max(max(xs) - min(xs) + max(ys) - min(ys), 1e-12)
        left, right = core_positions[core_a], core_positions[core_b]
        distance = (abs(left[0] - right[0]) + abs(left[1] - right[1])) / diameter
    memory_distance = 0.0
    if action.operator == "OP5" and core_positions is not None and memory_positions is not None:
        destinations = tuple(memory_positions.values()) if action.memory == "interleave" else (memory_positions[action.memory],)
        xs, ys = zip(*tuple(core_positions.values()), *tuple(memory_positions.values()))
        diameter = max(max(xs) - min(xs) + max(ys) - min(ys), 1e-12)
        memory_distance = sum(abs(core_positions[core][0] - destination[0]) + abs(core_positions[core][1] - destination[1])
                              for core in target.chiplets for destination in destinations)
        memory_distance /= max(len(target.chiplets) * len(destinations) * diameter, 1e-12)
    local = np.asarray([
        layer.macs_per_sample / total_macs,
        demand / total_bytes,
        len(target.chiplets) / max(core_count, 1),
        other_macs / total_macs,
        len(other.chiplets) / max(core_count, 1) if other is not None else 0.0,
        same_cluster, distance, _core_locality(target.chiplets),
        _core_locality(other.chiplets) if other is not None else 0.0,
        float(action.operator == "OP4"),
        float(action.field == "input_dram"), float(action.field == "weight_dram"),
        float(action.field == "output_dram"), float(action.memory == "interleave"), memory_distance,
        action.index / max(len(target.chiplets), 1),
        action.other_index / max(len(other.chiplets) if other is not None else len(target.chiplets), 1),
    ], dtype=np.float64)
    # State-only terms would cancel in softmax. Interaction terms make the
    # operator preference conditional on runtime, energy and stage imbalance.
    interaction = np.outer(state[1:7], one_hot).ravel()
    allocation_delta = action.core_delta
    if not allocation_delta and action.operator == "OP4":
        allocation_delta = 1 if action.variant == "hybrid_idle_acquire" else -1
    variant = np.asarray([float(action.variant == "hybrid_idle_acquire" or
                                (action.variant == "hybrid_legal_resize" and allocation_delta > 0)),
                          float(action.variant == "hybrid_idle_release" or
                                (action.variant == "hybrid_legal_resize" and allocation_delta < 0))])
    group = plan.group_map[action.layer_id]
    occupied_layers = group.layer_ids if group.mode == "spatial" else (action.layer_id,)
    occupied = {core for layer_id in occupied_layers for core in mapping[layer_id].chiplets}
    idle_fraction = max(core_count - len(occupied), 0) / max(core_count, 1)
    idle = np.asarray([*variant, idle_fraction, allocation_delta / max(core_count, 1)])
    variant_interaction = np.outer(state[1:7], variant).ravel()
    return np.clip(np.concatenate((one_hot, local, delta, interaction, idle, variant_interaction)), -3.0, 3.0)


class LinearActorCritic:
    """Masked softmax actor with a TD value baseline, implemented in NumPy."""
    def __init__(self, actor_features=ACTION_FEATURE_SIZE, state_size=10, actor_lr=0.03, value_lr=0.03,
                 discount=0.95, entropy=0.005):
        self.theta = np.zeros(actor_features, dtype=np.float64)
        self.value_weights = np.zeros(state_size, dtype=np.float64)
        self.actor_lr, self.value_lr = actor_lr, value_lr
        self.discount, self.entropy = discount, entropy
        self.updates = 0

    def probabilities(self, features):
        scores = features @ self.theta
        scores -= np.max(scores)
        weights = np.exp(scores)
        return weights / np.sum(weights)

    def value(self, state):
        return float(np.dot(state, self.value_weights))

    def update(self, features, chosen, probabilities, state, next_state, reward, terminal):
        target = reward + (0.0 if terminal else self.discount * self.value(next_state))
        advantage = float(np.clip(target - self.value(state), -4.0, 4.0))
        expected = probabilities @ features
        gradient = advantage * (features[chosen] - expected)
        log_prob = np.log(np.maximum(probabilities, 1e-30))
        entropy = -float(np.dot(probabilities, log_prob))
        gradient += self.entropy * ((probabilities * (-log_prob - entropy)) @ features)
        norm = float(np.linalg.norm(gradient))
        self.theta += self.actor_lr * gradient / max(norm / 5, 1.0)
        self.value_weights += self.value_lr * advantage * state / max(float(np.linalg.norm(state)) / 5, 1.0)
        self.updates += 1
        return advantage

    def to_dict(self):
        return {"theta": self.theta.tolist(), "value_weights": self.value_weights.tolist(),
                "actor_lr": self.actor_lr, "value_lr": self.value_lr, "discount": self.discount,
                "entropy": self.entropy, "updates": self.updates}

    @classmethod
    def from_dict(cls, data):
        obj = cls(len(data["theta"]), len(data["value_weights"]), data["actor_lr"], data["value_lr"],
                  data["discount"], data["entropy"])
        obj.theta[:] = data["theta"]
        obj.value_weights[:] = data["value_weights"]
        obj.updates = data["updates"]
        return obj


@dataclass
class SearchResult:
    method: str
    seed: int
    best_plan: MappingPlan
    best_metrics: dict
    initial_metrics: dict
    best_cost: float
    evaluations: int
    requests: int
    policy_updates: int
    trace: list[dict]
    status: str
    elapsed_s: float
    checkpoint_path: str | None = None

    def to_dict(self):
        return _jsonable({**asdict(self), "best_plan": self.best_plan.to_dict(),
                          "cache_hits": self.requests - self.evaluations})


class _Oracle:
    def __init__(self, evaluator, cache=None, evaluations=0, requests=0):
        self.evaluator, self.cache = evaluator, cache or {}
        self.evaluations, self.requests = evaluations, requests

    def get(self, plan, budget):
        self.requests += 1
        key = plan_fingerprint(plan)
        if key in self.cache:
            return self.cache[key], True
        if self.evaluations >= budget:
            self.requests -= 1
            return None, False
        try:
            metrics = _metrics(self.evaluator(plan))
        except ValueError as error:
            metrics = {"feasible": False, "reason": str(error)}
        self.cache[key] = metrics
        self.evaluations += 1
        return metrics, False


def _rng_tuple(value):
    return tuple(_rng_tuple(item) if isinstance(item, list) else item for item in value)


def _identity(workload, initial_plan, core_ids, memory_ids, objective, evaluator_identity,
              core_clusters, core_positions, memory_positions):
    data = {"workload": workload.fingerprint, "initial_plan": plan_fingerprint(initial_plan),
            "cores": list(core_ids), "memories": list(memory_ids), "objective": asdict(objective),
            "evaluator": evaluator_identity, "core_clusters": core_clusters,
            "core_positions": core_positions, "memory_positions": memory_positions}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _save(path, data):
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(data), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)


def run_search(workload, initial_plan, evaluator, core_ids, memory_ids=None, *, method="rl", seed=7,
               budget=128, horizon=16, max_per_operator=24, objective=None, checkpoint_path=None,
               resume=False, evaluator_identity="unspecified", checkpoint_every=8,
               proposal="uniform", core_clusters=None, core_positions=None, memory_positions=None):
    """Run a finite physical-evaluation budget with serializable continuation.

    ``resume`` preserves the policy, RNG, mapping, cache and counters. ``budget``
    is the cumulative physical-evaluation ceiling, so a resumed run may increase
    it. The training episode horizon and SA cooling reference stay fixed.
    ``proposal=cluster_guided`` enables a separately named ablation that uses
    action locality/demand features to prioritize a bounded proposal pool.
    """
    if method not in {"rl", "random", "greedy", "sa"}:
        raise ValueError("method must be rl, random, greedy, or sa")
    if proposal not in {"uniform", "cluster_guided"}:
        raise ValueError("unknown proposal strategy")
    if budget < 1 or horizon < 1 or max_per_operator < 1 or checkpoint_every < 1:
        raise ValueError("search budgets/horizon/proposal bounds must be positive")
    objective = objective or SearchObjective()
    memory_ids = tuple(memory_ids or initial_plan.memory_ids)
    core_ids = tuple(core_ids)
    if core_clusters is not None and set(core_clusters) != set(core_ids):
        raise ValueError("cluster metadata must cover every mapping core")
    if core_positions is not None and set(core_positions) != set(core_ids):
        raise ValueError("position metadata must cover every mapping core")
    if memory_positions is not None and set(memory_positions) != set(memory_ids):
        raise ValueError("memory position metadata must cover every memory controller")
    if proposal == "cluster_guided" and core_clusters is None:
        raise ValueError("cluster-guided proposals require actual hardware cluster metadata")
    identity = _identity(workload, initial_plan, core_ids, memory_ids, objective, evaluator_identity,
                         core_clusters, core_positions, memory_positions)
    settings = {"method": method, "seed": seed, "horizon": horizon,
                "max_per_operator": max_per_operator, "proposal": proposal}
    started = time.perf_counter()
    rng = random.Random(seed)
    policy = LinearActorCritic()
    trace, steps, episode_step, elapsed_before = [], 0, 0, 0.0
    oracle = _Oracle(evaluator)
    current_plan = best_plan = initial_plan
    if resume:
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise ValueError("resume requires an existing checkpoint")
        saved = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        if saved.get("schema") != CHECKPOINT_SCHEMA or saved["identity"] != identity or saved["settings"] != settings:
            raise ValueError("checkpoint workload/hardware/objective/search identity differs")
        rng.setstate(_rng_tuple(saved["rng_state"]))
        policy = LinearActorCritic.from_dict(saved["policy"])
        oracle = _Oracle(evaluator, saved["cache"], saved["evaluations"], saved["requests"])
        current_plan, best_plan = plan_from_dict(saved["current_plan"]), plan_from_dict(saved["best_plan"])
        current_metrics, best_metrics, reference = saved["current_metrics"], saved["best_metrics"], saved["reference"]
        current_cost, best_cost = saved["current_cost"], saved["best_cost"]
        trace, steps, episode_step = saved["trace"], saved["steps"], saved["episode_step"]
        elapsed_before = saved.get("elapsed_s", 0.0)
    else:
        reference, _ = oracle.get(initial_plan, budget)
        current_metrics = best_metrics = reference
        current_cost = best_cost = objective.cost(reference, reference)
        if not math.isfinite(current_cost):
            raise ValueError("initial mapping must have valid positive metrics and satisfy hard limits")
    status = "evaluation_budget_exhausted"

    def save():
        _save(checkpoint_path, {
            "schema": CHECKPOINT_SCHEMA, "identity": identity, "settings": settings,
            "rng_state": rng.getstate(), "policy": policy.to_dict(), "cache": oracle.cache,
            "evaluations": oracle.evaluations, "requests": oracle.requests,
            "current_plan": current_plan.to_dict(), "best_plan": best_plan.to_dict(),
            "current_metrics": current_metrics, "best_metrics": best_metrics, "reference": reference,
            "current_cost": current_cost, "best_cost": best_cost, "steps": steps,
            "episode_step": episode_step, "trace": trace,
            "elapsed_s": elapsed_before + time.perf_counter() - started,
            "evaluator_identity": evaluator_identity,
        })

    # Cache-only loops can revisit a finite space forever. This separate bound
    # is explicitly reported and does not consume imaginary physical queries.
    request_ceiling = max(budget * 20, 100)
    while oracle.evaluations < budget and oracle.requests < request_ceiling:
        actions = legal_actions(workload, current_plan, memory_ids, core_ids=core_ids,
                                max_per_operator=max_per_operator, rng=rng)
        if not actions:
            status = "no_legal_action"
            break
        state = state_features(workload, current_plan, current_metrics, reference, current_cost,
                               episode_step / horizon)
        features = np.asarray([action_features(workload, current_plan, action, state, len(core_ids),
                                              core_clusters, core_positions, memory_positions)
                               for action in actions])
        if proposal == "cluster_guided":
            # Optional proposal ablation, not a reward term or a Gemini claim.
            # Preserve every operator family and a uniform minority so remote
            # or dispersed placements remain reachable.
            locality = features[:, 10] - features[:, 11] + features[:, 6] - features[:, 19]
            keep = []
            for operator in OPERATORS:
                indices = [i for i, action in enumerate(actions) if action.operator == operator]
                if not indices:
                    continue
                ordered = sorted(indices, key=lambda i: (-locality[i], i))
                keep.extend(ordered[:max(1, len(ordered) // 2)])
                keep.extend(rng.sample(ordered[max(1, len(ordered) // 2):],
                                       min(2, len(ordered[max(1, len(ordered) // 2):]))))
            indices = sorted(set(keep))
            actions, features = tuple(actions[i] for i in indices), features[indices]
        probabilities = policy.probabilities(features) if method == "rl" else np.full(len(actions), 1 / len(actions))
        if method == "rl":
            chosen = rng.choices(range(len(actions)), weights=probabilities.tolist(), k=1)[0]
            evaluated_indices = [chosen]
        elif method == "greedy":
            evaluated_indices = rng.sample(range(len(actions)), min(8, len(actions), budget - oracle.evaluations))
        else:
            evaluated_indices = [rng.randrange(len(actions))]
        scored = []
        for index in evaluated_indices:
            candidate = apply_action(current_plan, actions[index], core_ids=core_ids)
            metrics, cache_hit = oracle.get(candidate, budget)
            if metrics is not None:
                scored.append((objective.cost(metrics, reference), index, candidate, metrics, cache_hit))
        if not scored:
            break
        candidate_cost, chosen, candidate_plan, candidate_metrics, cache_hit = min(scored, key=lambda item: (item[0], item[1]))
        old_cost = current_cost
        terminal = episode_step + 1 >= horizon
        feasible = math.isfinite(candidate_cost)
        reward = math.log(old_cost) - math.log(max(candidate_cost, 1e-300)) if feasible else -2.0
        if terminal:
            reward += -math.log(max(candidate_cost if feasible else current_cost, 1e-300))
        accepted = feasible
        if method == "greedy":
            accepted = feasible and candidate_cost < current_cost
        elif method == "sa" and feasible and candidate_cost > current_cost:
            # Fixed horizon-based cooling continues identically after resume.
            temperature = max(0.01, 0.2 * (0.95 ** (steps / horizon)))
            accepted = rng.random() < math.exp(-math.log(candidate_cost / current_cost) / temperature)
        next_plan = candidate_plan if accepted else current_plan
        next_metrics = candidate_metrics if accepted else current_metrics
        next_cost = candidate_cost if accepted else current_cost
        next_state = state_features(workload, next_plan, next_metrics, reference, next_cost,
                                    ((episode_step + 1) % horizon) / horizon)
        advantage = policy.update(features, chosen, probabilities, state, next_state, reward, terminal) if method == "rl" else 0.0
        if feasible and candidate_cost < best_cost:
            best_plan, best_metrics, best_cost = candidate_plan, candidate_metrics, candidate_cost
        current_plan, current_metrics, current_cost = next_plan, next_metrics, next_cost
        steps += 1
        episode_step = (episode_step + 1) % horizon
        trace.append({"step": steps, "evaluations": oracle.evaluations, "requests": oracle.requests,
                      "operator": actions[chosen].operator, "action": actions[chosen].to_dict(),
                      "operator_mask": operator_mask(actions), "candidate_count": len(actions),
                      "candidate_cost": candidate_cost if feasible else None,
                      "current_cost": current_cost, "best_cost": best_cost, "reward": reward,
                      "advantage": advantage, "accepted": accepted, "feasible": feasible,
                      "cache_hit": cache_hit, "policy_entropy": -float(np.dot(probabilities, np.log(np.maximum(probabilities, 1e-30)))),
                      "chosen_probability": float(probabilities[chosen]), "terminal": terminal})
        if steps % checkpoint_every == 0:
            save()
    if oracle.requests >= request_ceiling and oracle.evaluations < budget:
        status = "request_bound_reached"
    save()
    return SearchResult(method, seed, best_plan, best_metrics, reference, best_cost, oracle.evaluations,
                        oracle.requests, policy.updates, trace, status,
                        elapsed_before + time.perf_counter() - started,
                        str(Path(checkpoint_path).resolve()) if checkpoint_path else None)


def compare_searches(workload, initial_plan, evaluator, core_ids, memory_ids=None, *, budget=128,
                     seeds=(7, 8, 9), methods=("rl", "random", "greedy", "sa"), **kwargs):
    """Seed-matched baselines, with independent caches and identical budgets."""
    if "checkpoint_path" in kwargs or kwargs.get("resume"):
        raise ValueError("comparison checkpoints require separate method/seed paths")
    return tuple(run_search(workload, initial_plan, evaluator, core_ids, memory_ids,
                            method=method, seed=seed, budget=budget, **kwargs)
                 for seed in seeds for method in methods)
