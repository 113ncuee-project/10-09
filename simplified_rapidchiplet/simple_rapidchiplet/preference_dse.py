"""Preference-aware direct-action DSE over a fine-grained CNN block graph.

This module is intentionally independent from the legacy exhaustive/DP sweep.
It turns a model into smaller pipeline blocks, lets the agent assign any
contiguous block group to any positive number of chiplets, and learns a
preference-conditioned policy with terminal PPA/constraint feedback.
"""

from __future__ import annotations

import csv
import json
import math
import random
from functools import lru_cache
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from .config import EvalConfig
from .evaluator import EvaluationResult, evaluate_one
from .block_extractor import extract_semantic_blocks
from .mapping import PARALLEL_MAPPING_STRATEGIES, mapping_action_is_feasible
from .rapidchiplet_engine import rapid_input_fingerprint
from .model import ModelSpec
from .workload import build_pipeline_workload


PreferenceName = Literal["balanced", "latency", "area", "power"]


def _json_safe(value: object) -> object:
    """Convert non-finite floats to JSON null for portable result files."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class PreferenceProfile:
    """User constraints and the direction of the PPA trade-off."""

    name: PreferenceName
    latency_weight: float
    area_weight: float
    power_weight: float
    max_latency_ns: float = math.inf
    max_area_mm2: float = math.inf
    max_power_w: float = math.inf
    # Used only when a user does not provide a hard limit.  It keeps the
    # reward scale stable while still making lower latency/area/power better.
    latency_scale_ns: float = 1_000_000.0
    area_scale_mm2: float = 1_000.0
    power_scale_w: float = 16.0

    def __post_init__(self) -> None:
        weights = (self.latency_weight, self.area_weight, self.power_weight)
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("preference weights must be finite and non-negative")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("preference weights must sum to one")
        for name in ("max_latency_ns", "max_area_mm2", "max_power_w"):
            value = getattr(self, name)
            if not math.isfinite(value) and value != math.inf:
                raise ValueError(f"{name} must be finite or +inf")
            if value <= 0.0 and value != math.inf:
                raise ValueError(f"{name} must be positive when provided")
        if min(self.latency_scale_ns, self.area_scale_mm2, self.power_scale_w) <= 0.0:
            raise ValueError("reward scales must be positive")

    @property
    def weights(self) -> tuple[float, float, float]:
        return self.latency_weight, self.area_weight, self.power_weight

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def constraint_key(self) -> tuple[tuple[str, float], ...]:
        return (
            ("max_latency_ns", self.max_latency_ns),
            ("max_area_mm2", self.max_area_mm2),
            ("max_power_w", self.max_power_w),
        )


def make_preference_profile(
    name: PreferenceName = "balanced",
    *,
    max_latency_ns: float = math.inf,
    max_area_mm2: float = math.inf,
    max_power_w: float = math.inf,
    cfg: EvalConfig | None = None,
) -> PreferenceProfile:
    """Create one user preference profile.

    The hard limits are optional.  If a limit is omitted, the corresponding
    metric remains a soft preference and is normalised by a stable fallback
    scale.  Supplying a limit turns that metric into a hard constraint.
    """

    if name == "balanced":
        weights = (1 / 3, 1 / 3, 1 / 3)
    else:
        weights = {
            "latency": (0.8, 0.1, 0.1),
            "area": (0.1, 0.8, 0.1),
            "power": (0.1, 0.1, 0.8),
        }[name]
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("preference weights must have a positive sum")
    normalised = tuple(value / total for value in weights)

    area_scale = 1_000.0
    power_scale = 16.0
    if cfg is not None:
        area_scale = max(
            1.0,
            cfg.max_chiplets * cfg.chiplet.width_mm * cfg.chiplet.width_mm,
        )
        power_scale = max(
            1.0,
            cfg.max_chiplets
            * (
                cfg.power.chiplet_static_w
                + cfg.power.chiplet_peak_dynamic_w * cfg.chiplet.utilization
                + 4.0 * cfg.power.phy_w
            ),
        )

    return PreferenceProfile(
        name=name,
        latency_weight=normalised[0],
        area_weight=normalised[1],
        power_weight=normalised[2],
        max_latency_ns=max_latency_ns,
        max_area_mm2=max_area_mm2,
        max_power_w=max_power_w,
        area_scale_mm2=area_scale,
        power_scale_w=power_scale,
    )


@dataclass(frozen=True, order=True)
class DesignState:
    """Immutable RL state for sequential block-group construction."""

    preference: str
    next_block: int = 0
    used_chiplets: int = 0
    groups: tuple[tuple[int, int, int], ...] = ()
    model: str = ""
    constraints: tuple[tuple[str, float], ...] = ()
    mapping_strategies: tuple[str, ...] = ()


@dataclass(frozen=True, order=True)
class DesignAction:
    """Assign a contiguous block group and choose its intra-block mapping."""

    group_length: int
    chiplets: int
    mapping_strategy: str = "output_channel"

    @property
    def label(self) -> str:
        return (
            f"assign(length={self.group_length}, chiplets={self.chiplets}, "
            f"mapping={self.mapping_strategy})"
        )


def legal_actions(
    state: DesignState,
    block_count: int,
    max_chiplets: int,
    model: ModelSpec | None = None,
    sram_mb: float = math.inf,
) -> tuple[DesignAction, ...]:
    """Return all legal split/merge/parallelisation actions.

    Unlike the legacy workload search, there is deliberately no condition on
    ``group_length``.  A group containing one block or many blocks can use any
    positive number of chiplets, as long as the complete design remains
    finishable under the chiplet budget.
    """

    if block_count < 1 or max_chiplets < 1:
        raise ValueError("block_count and max_chiplets must be positive")
    if not 0 <= state.next_block <= block_count:
        raise ValueError("state.next_block is outside the block graph")
    if not 0 <= state.used_chiplets <= max_chiplets:
        raise ValueError("state.used_chiplets is outside the chiplet budget")
    if state.next_block == block_count:
        return ()

    remaining_blocks = block_count - state.next_block
    remaining_chiplets = max_chiplets - state.used_chiplets
    actions: list[DesignAction] = []
    @lru_cache(maxsize=None)
    def minimum_feasible_suffix(start: int) -> int | None:
        if start == block_count:
            return 0
        best: int | None = None
        for end in range(start + 1, block_count + 1):
            for parts in range(1, max_chiplets + 1):
                strategies = ("single",) if parts == 1 else PARALLEL_MAPPING_STRATEGIES
                if any(
                    mapping_action_is_feasible(
                        model,
                        start,
                        end,
                        parts,
                        strategy,
                        sram_mb=sram_mb,
                    )
                    for strategy in strategies
                ):
                    suffix = minimum_feasible_suffix(end)
                    if suffix is not None and (best is None or parts + suffix < best):
                        best = parts + suffix
        return best

    for group_length in range(1, remaining_blocks + 1):
        blocks_after = remaining_blocks - group_length
        reserve = 1 if blocks_after else 0
        max_parts = remaining_chiplets - reserve
        for parts in range(1, max_parts + 1):
            strategies = ("single",) if parts == 1 else PARALLEL_MAPPING_STRATEGIES
            for strategy in strategies:
                if model is not None:
                    if not mapping_action_is_feasible(
                        model,
                        state.next_block,
                        state.next_block + group_length,
                        parts,
                        strategy,
                        sram_mb=sram_mb,
                    ):
                        continue
                    suffix_minimum = minimum_feasible_suffix(state.next_block + group_length)
                    if suffix_minimum is None or parts + suffix_minimum > remaining_chiplets:
                        continue
                actions.append(DesignAction(group_length, parts, strategy))
    return tuple(actions)


def apply_action(
    state: DesignState,
    action: DesignAction,
    block_count: int,
    max_chiplets: int,
    model: ModelSpec | None = None,
    sram_mb: float = math.inf,
) -> DesignState:
    if action not in legal_actions(state, block_count, max_chiplets, model, sram_mb):
        raise ValueError(f"illegal action: {action.label}")
    start = state.next_block
    end = start + action.group_length
    return DesignState(
        preference=state.preference,
        next_block=end,
        used_chiplets=state.used_chiplets + action.chiplets,
        groups=state.groups + ((start, end, action.chiplets),),
        model=state.model,
        constraints=state.constraints,
        mapping_strategies=state.mapping_strategies + (action.mapping_strategy,),
    )


def is_complete(state: DesignState, block_count: int) -> bool:
    return state.next_block == block_count and bool(state.groups)


def random_complete_state(
    preference: PreferenceProfile,
    block_count: int,
    max_chiplets: int,
    rng: random.Random,
    model_name: str = "",
    model: ModelSpec | None = None,
    sram_mb: float = math.inf,
) -> DesignState:
    state = DesignState(
        preference=preference.name,
        model=model_name,
        constraints=preference.constraint_key,
    )
    while not is_complete(state, block_count):
        actions = legal_actions(state, block_count, max_chiplets, model, sram_mb)
        if not actions:
            raise RuntimeError("action mask produced a dead-end state")
        state = apply_action(
            state,
            rng.choice(actions),
            block_count,
            max_chiplets,
            model,
            sram_mb,
        )
    return state


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    feasible: bool
    weighted_cost: float
    total_violation: float
    latency_ratio: float
    area_ratio: float
    power_ratio: float
    violated_constraints: tuple[str, ...]
    architecture_feasible: bool
    architecture_violations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _soft_ratio(value: float, limit: float, fallback_scale: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        return math.inf
    return value / limit if math.isfinite(limit) else value / fallback_scale


def score_result(
    result: EvaluationResult,
    profile: PreferenceProfile,
) -> RewardBreakdown:
    """Apply preference direction and hard user constraints to one design."""

    latency_ratio = _soft_ratio(
        result.avg_latency_ns, profile.max_latency_ns, profile.latency_scale_ns
    )
    area_ratio = _soft_ratio(
        result.total_area_mm2, profile.max_area_mm2, profile.area_scale_mm2
    )
    power_ratio = _soft_ratio(
        result.total_power_w, profile.max_power_w, profile.power_scale_w
    )
    violations: list[tuple[str, float]] = []
    if math.isfinite(profile.max_latency_ns):
        violations.append(("latency", max(0.0, latency_ratio - 1.0)))
    if math.isfinite(profile.max_area_mm2):
        violations.append(("area", max(0.0, area_ratio - 1.0)))
    if math.isfinite(profile.max_power_w):
        violations.append(("power", max(0.0, power_ratio - 1.0)))

    architecture_feasible = bool(getattr(result, "architecture_feasible", True))
    architecture_violations = tuple(getattr(result, "architecture_violations", ()))
    if not architecture_feasible:
        # Architecture invalidity is a hard feasibility failure, independent
        # of the user's soft PPA direction.
        violations.append(("architecture", 1.0 + min(1.0, len(architecture_violations) / 10.0)))

    total_violation = sum(value for _name, value in violations)
    feasible = total_violation == 0.0
    weighted_cost = (
        profile.latency_weight * latency_ratio
        + profile.area_weight * area_ratio
        + profile.power_weight * power_ratio
    )
    reward = (
        1.0 - weighted_cost
        if feasible
        else -1.0 - min(total_violation, 1.0)
    )
    return RewardBreakdown(
        reward=reward,
        feasible=feasible,
        weighted_cost=weighted_cost,
        total_violation=total_violation,
        latency_ratio=latency_ratio,
        area_ratio=area_ratio,
        power_ratio=power_ratio,
        violated_constraints=tuple(name for name, value in violations if value > 0.0),
        architecture_feasible=architecture_feasible,
        architecture_violations=architecture_violations,
    )


@dataclass
class EvaluationOracle:
    model: ModelSpec
    cfg: EvalConfig
    profile: PreferenceProfile
    max_chiplets: int
    _cache: dict[tuple[object, ...], EvaluationResult] = field(default_factory=dict)
    requests: int = 0

    def cache_key(self, state: DesignState) -> tuple[object, ...]:
        return (self.model, self.cfg, self.profile, self.max_chiplets,
                state, rapid_input_fingerprint(self.cfg))

    def evaluate(self, state: DesignState) -> EvaluationResult:
        if not is_complete(state, len(self.model.blocks)):
            raise ValueError("only complete designs can be evaluated")
        key = self.cache_key(state)
        self.requests += 1
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        workload = build_pipeline_workload(
            self.model,
            self.cfg.chiplet.op_per_mac,
            state.groups,
            search_method="preference-q-learning",
            mapping_strategies=state.mapping_strategies,
        )
        result = evaluate_one(
            self.model,
            "mesh",
            state.used_chiplets,
            self.cfg,
            workload=workload,
            ppa_goal=self.profile.name,
        )
        self._cache[key] = result
        return result

    @property
    def unique_evaluations(self) -> int:
        return len(self._cache)


@dataclass(frozen=True)
class PreferenceSearchResult:
    model: str
    block_source: str
    extraction_report: dict[str, object]
    preference: dict[str, object]
    seed: int
    block_split: int
    evaluation_budget: int
    unique_evaluations: int
    requests: int
    episodes: int
    final_epsilon: float
    best_reward: float
    best_breakdown: RewardBreakdown
    best_state: DesignState
    best_candidate: EvaluationResult
    history: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "block_source": self.block_source,
            "extraction_report": self.extraction_report,
            "preference": self.preference,
            "seed": self.seed,
            "block_split": self.block_split,
            "evaluation_budget": self.evaluation_budget,
            "unique_evaluations": self.unique_evaluations,
            "requests": self.requests,
            "episodes": self.episodes,
            "final_epsilon": self.final_epsilon,
            "best_reward": self.best_reward,
            "best_breakdown": self.best_breakdown.to_dict(),
            "best_state": asdict(self.best_state),
            "best_candidate": self.best_candidate.to_dict(),
            "history": list(self.history),
        }


def q_learning_search(
    model: ModelSpec,
    cfg: EvalConfig,
    profile: PreferenceProfile,
    *,
    block_split: int = 2,
    evaluation_budget: int = 64,
    seed: int = 1,
    max_episodes: int | None = None,
    learning_rate: float = 0.2,
    discount: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
) -> PreferenceSearchResult:
    """Run terminal-reward tabular Q-learning for one model and preference."""

    if evaluation_budget < 1:
        raise ValueError("evaluation_budget must be positive")
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")
    if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
    if not 0.0 <= discount < 1.0:
        raise ValueError("discount must be in [0, 1)")

    extracted_model, extraction_report = extract_semantic_blocks(
        model,
        legacy_split_factor=block_split,
    )
    block_count = len(extracted_model.blocks)
    oracle = EvaluationOracle(extracted_model, cfg, profile, cfg.max_chiplets)
    rng = random.Random(seed)
    q_values: dict[DesignState, dict[DesignAction, float]] = {}
    episodes_without_new = 0
    episodes = 0
    epsilon = epsilon_start
    history: list[dict[str, object]] = []
    best_state: DesignState | None = None
    best_candidate: EvaluationResult | None = None
    best_breakdown: RewardBreakdown | None = None
    episode_cap = max_episodes or max(1_000, evaluation_budget * 250)

    def q_for(state: DesignState, action: DesignAction) -> float:
        return q_values.setdefault(state, {}).setdefault(action, 0.0)

    def best_next_value(state: DesignState) -> float:
        actions = legal_actions(
            state,
            block_count,
            cfg.max_chiplets,
            extracted_model,
            cfg.chiplet.sram_mb,
        )
        return max((q_for(state, action) for action in actions), default=0.0)

    def better(candidate: EvaluationResult, breakdown: RewardBreakdown) -> bool:
        nonlocal best_candidate, best_breakdown
        if best_candidate is None or best_breakdown is None:
            return True
        return (
            breakdown.reward,
            -breakdown.total_violation,
            -breakdown.weighted_cost,
            -candidate.avg_latency_ns,
            -candidate.total_area_mm2,
            -candidate.total_power_w,
        ) > (
            best_breakdown.reward,
            -best_breakdown.total_violation,
            -best_breakdown.weighted_cost,
            -best_candidate.avg_latency_ns,
            -best_candidate.total_area_mm2,
            -best_candidate.total_power_w,
        )

    while oracle.unique_evaluations < evaluation_budget and episodes < episode_cap:
        episodes += 1
        state = DesignState(
            preference=profile.name,
            model=model.name,
            constraints=profile.constraint_key,
        )
        trajectory: list[tuple[DesignState, DesignAction, DesignState]] = []
        force_random = episodes_without_new >= 100
        while not is_complete(state, block_count):
            actions = legal_actions(
                state,
                block_count,
                cfg.max_chiplets,
                extracted_model,
                cfg.chiplet.sram_mb,
            )
            if not actions:
                raise RuntimeError("Q-learning reached an illegal/dead-end state")
            if force_random or rng.random() < epsilon:
                action = rng.choice(actions)
            else:
                values = [q_for(state, candidate) for candidate in actions]
                best_value = max(values)
                choices = [
                    candidate
                    for candidate, value in zip(actions, values)
                    if math.isclose(value, best_value, abs_tol=1e-15)
                ]
                action = rng.choice(choices)
            next_state = apply_action(
                state,
                action,
                block_count,
                cfg.max_chiplets,
                extracted_model,
                cfg.chiplet.sram_mb,
            )
            trajectory.append((state, action, next_state))
            state = next_state

        is_new = oracle.cache_key(state) not in oracle._cache
        candidate = oracle.evaluate(state)
        breakdown = score_result(candidate, profile)
        if is_new:
            episodes_without_new = 0
        else:
            episodes_without_new += 1

        if better(candidate, breakdown):
            best_state = state
            best_candidate = candidate
            best_breakdown = breakdown
        history.append(
            {
                "episode": episodes,
                "evaluations": oracle.unique_evaluations,
                "reward": breakdown.reward,
                "feasible": breakdown.feasible,
                "groups": [list(group) for group in state.groups],
                "mapping_strategies": list(state.mapping_strategies),
                "best_reward": best_breakdown.reward if best_breakdown else breakdown.reward,
            }
        )

        for index in range(len(trajectory) - 1, -1, -1):
            previous, action, following = trajectory[index]
            target = (
                breakdown.reward
                if index == len(trajectory) - 1
                else discount * best_next_value(following)
            )
            current = q_for(previous, action)
            q_values[previous][action] = current + learning_rate * (target - current)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Once the Q policy repeatedly revisits designs, explicitly sample a
        # random legal trajectory so a finite evaluation budget can still be
        # filled without requiring an arbitrarily large episode cap.
        if episodes_without_new >= 100 and oracle.unique_evaluations < evaluation_budget:
            random_state = random_complete_state(
                profile,
                block_count,
                cfg.max_chiplets,
                rng,
                model_name=model.name,
                model=extracted_model,
                sram_mb=cfg.chiplet.sram_mb,
            )
            if oracle.cache_key(random_state) not in oracle._cache:
                random_candidate = oracle.evaluate(random_state)
                random_breakdown = score_result(random_candidate, profile)
                if better(random_candidate, random_breakdown):
                    best_state = random_state
                    best_candidate = random_candidate
                    best_breakdown = random_breakdown
                episodes_without_new = 0

    if best_state is None or best_candidate is None or best_breakdown is None:
        raise RuntimeError("Q-learning did not evaluate any complete design")
    return PreferenceSearchResult(
        model=model.name,
        block_source=extracted_model.block_source,
        extraction_report=extraction_report.to_dict(),
        preference=profile.to_dict(),
        seed=seed,
        block_split=block_split,
        evaluation_budget=evaluation_budget,
        unique_evaluations=oracle.unique_evaluations,
        requests=oracle.requests,
        episodes=episodes,
        final_epsilon=epsilon,
        best_reward=best_breakdown.reward,
        best_breakdown=best_breakdown,
        best_state=best_state,
        best_candidate=best_candidate,
        history=tuple(history),
    )


def write_search_outputs(
    output_dir: str | Path,
    results: Iterable[PreferenceSearchResult],
) -> dict[str, Path]:
    """Write machine-readable results for the new preference-aware run."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    values = list(results)
    json_path = directory / "preference_search_results.json"
    json_path.write_text(
        json.dumps(
            _json_safe([result.to_dict() for result in values]),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summaries = []
    for result in values:
        candidate = result.best_candidate
        summaries.append(
            {
                "model": result.model,
                "preference": result.preference["name"],
                "reward": result.best_reward,
                "feasible": result.best_breakdown.feasible,
                "violated_constraints": ",".join(result.best_breakdown.violated_constraints),
                "evaluations": result.unique_evaluations,
                "episodes": result.episodes,
                "blocks": len(result.best_candidate.block_graph.get("nodes", [])),
                "chiplets": candidate.selected_chiplets,
                "fps": candidate.achieved_fps,
                "latency_ns": candidate.avg_latency_ns,
                "area_mm2": candidate.total_area_mm2,
                "power_w": candidate.total_power_w,
                "workload_plan": candidate.workload_plan,
            }
        )
    csv_path = directory / "preference_search_summary.csv"
    if summaries:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "version": "preference-semantic-block-dse-phase-1.5",
                    "canonical_flows": True,
                    "fps_policy": "inverse_e2e_diagnostic_only",
                    "results": len(values),
                    "method": "tabular_q_learning_terminal_preference_reward",
                    "semantic_block_extraction": True,
                    "arbitrary_group_parallelization": True,
                    "intra_block_mapping": [
                        "single",
                        "output_channel",
                    ],
                    "feasibility_checker": True,
                    "dependency_aware_scheduler": True,
                    "outputs": [json_path.name, csv_path.name],
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path, "manifest": manifest_path}
