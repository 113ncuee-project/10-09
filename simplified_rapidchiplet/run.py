from __future__ import annotations

import argparse
from pathlib import Path

from simple_rapidchiplet.config import load_config
from simple_rapidchiplet.evaluator import PPA_WEIGHT_PRESETS, evaluate_models, select_best_by_model
from simple_rapidchiplet.evaluator import EvaluationResult
from simple_rapidchiplet.model import load_models, select_models
from simple_rapidchiplet.report import write_outputs


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run simplified RapidChiplet-style DNN chiplet evaluation.")
    parser.add_argument("--config", default=str(root / "configs" / "defaults.json"))
    parser.add_argument("--models-file", default=str(root / "configs" / "models.json"))
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model names. Default: run every model in configs/models.json.",
    )
    parser.add_argument(
        "--topologies",
        nargs="*",
        default=["mesh"],
        choices=["mesh"],
        help="Topology to evaluate. The current problem definition is mesh-only.",
    )
    parser.add_argument(
        "--dp-top-k",
        type=int,
        default=12,
        help="Number of Pareto DP candidates kept per state. Use 0 to keep the full Pareto frontier. Default: 12.",
    )
    parser.add_argument(
        "--workload-search",
        default="pareto-dp",
        choices=["brute-force", "pareto-dp"],
        help="Workload partition search method. Default: pareto-dp.",
    )
    parser.add_argument(
        "--ppa-goal",
        default="balanced",
        choices=sorted(PPA_WEIGHT_PRESETS),
        help="PPA objective used to pick the summary row. Default: balanced.",
    )
    parser.add_argument(
        "--ppa-weights",
        nargs=3,
        type=float,
        metavar=("LATENCY", "AREA", "POWER"),
        help="Override --ppa-goal with custom latency/area/power weights.",
    )
    parser.add_argument("--out", default=str(root / "results" / "latest"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    models = select_models(load_models(args.models_file), args.models)
    ppa_weights = tuple(args.ppa_weights) if args.ppa_weights is not None else None
    summaries, sweeps = evaluate_models(
        models,
        args.topologies,
        cfg,
        ppa_goal=args.ppa_goal,
        ppa_weights=ppa_weights,
        dp_top_k=args.dp_top_k,
        workload_search=args.workload_search,
    )
    written = write_outputs(args.out, summaries, sweeps)

    out_path = Path(args.out).resolve()
    best_rows = select_best_by_model(sweeps)
    print(f"Evaluated {len(sweeps)} RapidChiplet candidates")
    for filename in ("best.csv", "summary.csv", "chiplet_results.csv", "report.html"):
        if written.get(filename):
            print(f"Wrote {out_path / filename}")
    print("Overall best per model:")
    for row in best_rows:
        _print_result(row)
    print("Best per topology:")
    for row in summaries:
        _print_result(row)


def _print_result(row: EvaluationResult) -> None:
    status = "VALID" if row.architecture_feasible else "INVALID"
    print(
        f"{status:4} {row.model:22} {row.topology:4} "
        f"chiplets={row.selected_chiplets:2d}/{row.available_chiplets:2d} "
        f"unused={row.unused_chiplets:2d} fps={row.achieved_fps:9.2f} "
        f"area={row.total_area_mm2:8.2f}mm^2 power={row.total_power_w:8.2f}W "
        f"latency={row.avg_latency_ns:8.2f}ns "
        f"score={row.ppa_score:6.3f} search={row.workload_search}"
    )
    print(f"     partition={row.workload_plan}")
    print(f"     mapping={_format_mapping(row)}")


def _format_mapping(row: EvaluationResult) -> str:
    graph = row.connection_graph
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    if not nodes:
        return "-"
    return " | ".join(
        f"C{int(node['id']) + 1}: {node.get('stage', '')}"
        for node in sorted(nodes, key=lambda item: int(item["id"]))
    )


if __name__ == "__main__":
    import sys
    if "--legacy" in sys.argv:
        sys.argv.remove("--legacy")
        main()
    else:
        from simple_rapidchiplet.physical_cli import main as physical_main
        physical_main()
