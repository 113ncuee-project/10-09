"""Default entry point; legacy analytical models require explicit --legacy."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .model_parser import extract_reference_graph
from .physical_dse import PhysicalOracle, exhaustive_physical, physical_profile, q_learning_physical, score
from .physical_report import summary_row, write_physical_outputs
from .toy_model import make_toy_graph


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="RL communication-aware K mapping, actual operator graph, official mesh evaluation")
    parser.add_argument("--config",default=str(root/"configs/defaults.json"))
    parser.add_argument("--models",nargs="+",default=["resnet50"],choices=["resnet50","toy"])
    parser.add_argument("--preference","--ppa-goal",dest="preference",choices=["balanced","latency","area","power","custom"],default="balanced")
    parser.add_argument("--ppa-weights",nargs=3,type=float,metavar=("LATENCY","AREA","POWER"))
    parser.add_argument("--max-latency-ns",type=float,default=float("inf"))
    parser.add_argument("--max-area-mm2",type=float,default=float("inf"))
    parser.add_argument("--max-power-w",type=float,default=float("inf"))
    parser.add_argument("--max-chiplets",type=int,help="Override physical hardware count search limit")
    parser.add_argument("--assignment-top-k",type=int,help="Bound ordered assignments per active count")
    parser.add_argument("--budget",type=int,default=64)
    parser.add_argument("--episodes",type=int)
    parser.add_argument("--seed",type=int,default=20260915)
    parser.add_argument("--exhaustive-check",action="store_true",help="Toy-only exhaustive comparison in the identical bounded action space")
    parser.add_argument("--ablation",action="store_true",help="Compare fixed count/assignment, dynamic count, communication assignment, and RL")
    parser.add_argument("--out",default=str(root/"results/communication_aware"))
    args = parser.parse_args()
    if args.preference == "custom" and args.ppa_weights is None:
        parser.error("--preference custom requires --ppa-weights L A P")
    cfg = load_config(args.config)
    if args.max_chiplets is not None:
        if args.max_chiplets < 1:
            parser.error("--max-chiplets must be positive")
        cfg = replace(cfg,max_chiplets=args.max_chiplets)
    if args.assignment_top_k is not None:
        cfg = replace(cfg,search=replace(cfg.search,assignment_top_k=args.assignment_top_k))
    if args.exhaustive_check and (args.models != ["toy"] or cfg.max_chiplets > 4):
        parser.error("--exhaustive-check requires --models toy --max-chiplets <=4")
    for name in args.models:
        graph = make_toy_graph() if name == "toy" else extract_reference_graph(cfg.reference_model)
        profile = physical_profile(graph,cfg,args.preference,weights=args.ppa_weights,
                    max_latency_ns=args.max_latency_ns,max_area_mm2=args.max_area_mm2,max_power_w=args.max_power_w)
        oracle = PhysicalOracle(graph,cfg,profile)
        result = q_learning_physical(oracle,budget=args.budget,seed=args.seed,max_episodes=args.episodes)
        comparison = None
        if args.exhaustive_check:
            exact = exhaustive_physical(PhysicalOracle(graph,cfg,profile))
            rl_score, exact_score = score(result.best,profile),score(exact.best,profile)
            comparison = dict(scope="identical bounded ordered candidate space",exhaustive_complete=exact.complete,
                              exhaustive_evaluations=exact.unique_evaluations,rl_evaluations=result.unique_evaluations,
                              rl_feasible=rl_score["feasible"],exhaustive_feasible=exact_score["feasible"],
                              rl_cost=rl_score["weighted_cost"],exhaustive_cost=exact_score["weighted_cost"],
                              rl_best=summary_row(result.best,profile),exhaustive_best=summary_row(exact.best,profile),
                              cost_gap_percent=(100*(rl_score["weighted_cost"]-exact_score["weighted_cost"])/exact_score["weighted_cost"]
                                  if rl_score["feasible"] and exact_score["feasible"] else None))
        ablation = None
        if args.ablation:
            ablation = []
            for mode, options, episodes in (
                ("fixed_count_fixed_assignment",dict(fixed_assignment=True,fixed_active_count=1),args.episodes),
                ("dynamic_count_fixed_assignment",dict(fixed_assignment=True),args.episodes),
                ("dynamic_count_communication_assignment",{},args.episodes)):
                baseline = PhysicalOracle(graph,cfg,profile,**options)
                if mode == "dynamic_count_communication_assignment":
                    baseline.search = replace(baseline.search,exploration=1.0)
                run = q_learning_physical(baseline,budget=args.budget,seed=args.seed,max_episodes=episodes)
                ablation.append(dict(mode=mode,seed=args.seed,unique_evaluations=run.unique_evaluations,
                                     **summary_row(run.best,profile)))
            ablation.append(dict(mode="Q_learning",seed=args.seed,unique_evaluations=result.unique_evaluations,
                                 **summary_row(result.best,profile)))
        output = Path(args.out)/(name if len(args.models)>1 else "")
        out = write_physical_outputs(output,result,graph,cfg,comparison=comparison,ablation=ablation)
        m = result.best.metrics
        print(f"{name}: backend={m['backend']} blocks={len(graph.blocks)} evaluations={result.unique_evaluations} "
              f"feasible={score(result.best,profile)['feasible']} N_total={m['N_total']} "
              f"latency={m['total_latency_ns']/1e6:.6f}ms area={m['total_area_mm2']:.3f}mm2 power={m['total_power_w']:.3f}W")
        print(f"Report: {(out/'report.html').resolve()}")


if __name__ == "__main__":
    main()
