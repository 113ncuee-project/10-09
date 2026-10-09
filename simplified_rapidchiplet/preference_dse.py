"""Run fine-grained block-level, preference-aware Chiplet DSE."""

from __future__ import annotations

import argparse
from pathlib import Path

from simple_rapidchiplet.config import load_config
from simple_rapidchiplet.model import load_models, select_models
from simple_rapidchiplet.preference_dse import (
    make_preference_profile,
    q_learning_search,
    write_search_outputs,
)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fine-grained block graph + preference-aware Q-learning DSE."
    )
    parser.add_argument("--config", default=str(root / "configs" / "defaults.json"))
    parser.add_argument("--models-file", default=str(root / "configs" / "models.json"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument(
        "--preference",
        choices=["balanced", "latency", "area", "power"],
        default="balanced",
    )
    parser.add_argument("--max-latency-ns", type=float, default=float("inf"))
    parser.add_argument("--max-area-mm2", type=float, default=float("inf"))
    parser.add_argument("--max-power-w", type=float, default=float("inf"))
    parser.add_argument(
        "--block-split",
        type=int,
        default=1,
        help=(
            "Legacy fallback only: split stage-only configs into inferred blocks. "
            "Explicit semantic blocks are never split."
        ),
    )
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--out", default=str(root / "results" / "preference_dse"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    models = select_models(load_models(args.models_file), args.models)
    profile = make_preference_profile(
        args.preference,
        max_latency_ns=args.max_latency_ns,
        max_area_mm2=args.max_area_mm2,
        max_power_w=args.max_power_w,
        cfg=cfg,
    )

    results = []
    for model_index, model in enumerate(models):
        result = q_learning_search(
            model,
            cfg,
            profile,
            block_split=args.block_split,
            evaluation_budget=args.budget,
            seed=args.seed + model_index,
            max_episodes=args.episodes,
        )
        results.append(result)
        candidate = result.best_candidate
        print(
            f"{model.name}: preference={profile.name} "
            f"blocks={len(candidate.block_graph.get('nodes', []))} "
            f"evaluations={result.unique_evaluations} "
            f"reward={result.best_reward:.6f} "
            f"feasible={result.best_breakdown.feasible} "
            f"chiplets={candidate.selected_chiplets} "
            f"fps={candidate.achieved_fps:.3f} "
            f"latency={candidate.avg_latency_ns:.3f}ns "
            f"area={candidate.total_area_mm2:.3f}mm2 "
            f"power={candidate.total_power_w:.3f}W"
        )
        print(f"  plan: {candidate.workload_plan}")

    written = write_search_outputs(args.out, results)
    for path in written.values():
        print(f"Wrote {path.resolve()}")


if __name__ == "__main__":
    import sys
    if "--legacy" in sys.argv:
        sys.argv.remove("--legacy")
        main()
    else:
        from simple_rapidchiplet.physical_cli import main as physical_main
        physical_main()
