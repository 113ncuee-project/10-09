"""Portable CSV/JSON traces and a self-contained mapping inspection report."""
from __future__ import annotations

import csv
import hashlib
import html
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path

from .physical_dse import score
from .physical_evaluator import ASSUMPTIONS
from .preference_dse import _json_safe
from .rapidchiplet_engine import rapid_input_fingerprint


def source_fingerprints():
    root = Path(__file__).parent
    return {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.glob("*.py"))}


def write_json(path, data):
    path.write_text(json.dumps(_json_safe(data),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",encoding="utf-8-sig",newline="") as output:
        writer = csv.DictWriter(output,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(_json_safe(value),ensure_ascii=False) if isinstance(value,(dict,list,tuple))
                             else value for key,value in row.items()})


def summary_row(evaluation, profile):
    return {key:value for key,value in evaluation.metrics.items() if key not in ("connection_graph","per_link_statistics")} | score(evaluation,profile)


def write_physical_outputs(directory, result, graph, cfg, *, comparison=None, ablation=None):
    out = Path(directory)
    out.mkdir(parents=True,exist_ok=True)
    best = result.best
    row = summary_row(best,result.profile)
    for name in ("best.csv","summary.csv"):
        write_csv(out/name,[row])
    write_csv(out/"chiplet_results.csv",[summary_row(value,result.profile) for value in result.candidates])
    mapping_rows = []
    for block in best.mapping_plan["blocks"]:
        mapping_rows.append({"block_id":block["block_id"],"N_total":best.metrics["N_total"],
                             "N_active":block["active_chiplet_count"],"partition_axis":block["partition_axis"],
                             "active_chiplets":block["active_chiplets"],"input_sources":block["input_sources"],
                             "output_owners":block["output_owners"],"operator_shards":block["operator_shards"]})
    write_csv(out/"mapping_trace.csv",mapping_rows)
    flow_rows = [{**flow.to_dict(),"route":best.network_context.routes[(flow.src,flow.dst)]}
                 for flow in best.derived.workload.flows]
    flow_fields = list(flow_rows[0]) if flow_rows else ["tensor_id","src","dst","bytes","kind","producer_operator","consumer_operator","channel_start","channel_end","route"]
    write_csv(out/"canonical_flows.csv",flow_rows,flow_fields)
    write_csv(out/"transition_traffic.csv",[flow for flow in flow_rows if flow["kind"] == "transition"],flow_fields)
    write_csv(out/"operator_trace.csv",best.operator_timings)
    write_csv(out/"per_link_statistics.csv",best.metrics["per_link_statistics"],
              ["src","dst","load_bytes","bandwidth_bits_per_cycle","utilization","utilization_time_basis","time_basis_ns"])
    write_csv(out/"search_history.csv",result.history)
    write_json(out/"canonical_flows.json",[flow.to_dict() for flow in best.derived.workload.flows])
    write_json(out/"mapping_plan.json",best.mapping_plan)
    write_json(out/"model_metadata.json",graph.to_dict())
    write_json(out/"config_snapshot.json",asdict(cfg))
    write_json(out/"assumptions.json",{**ASSUMPTIONS,"hardware":asdict(cfg.chiplet),
               "reference_model":asdict(cfg.reference_model),"network_context":best.network_context.to_dict(),
               "preferences":asdict(result.profile),"FPS":"output only: 1e9 / total_latency_ns",
               "network_latency_ns":"intra-block service only; transition_latency_ns is a disjoint component",
               "candidate_pruning":"top K per active count; first-operator missing input bytes times Manhattan hop lower bound",
               "exhaustive_scope":"bounded ordered candidate space shared with RL; no full permutation global optimum claim"})
    versions = {name:importlib.metadata.version(name) for name in ("torch","torchvision","numpy","Pillow")}
    manifest = dict(schema_version=2,model=graph.model_name,model_fingerprint=graph.fingerprint,
                    source_fingerprints=source_fingerprints(),native_input_fingerprints=rapid_input_fingerprint(cfg),
                    python=platform.python_version(),packages=versions,seed=result.seed,
                    method=result.method,episodes=result.episodes,unique_evaluations=result.unique_evaluations,
                    prefix_evaluations=result.prefix_evaluations,cache_hits=result.cache_hits,q_entries=result.q_entries,
                    search_settings=asdict(cfg.search) if cfg.search else {},profile=asdict(result.profile),
                    exact_space_exhausted=result.complete,backend=best.network_context.backend)
    write_json(out/"run_manifest.json",manifest)
    write_json(out/"search_result.json",dict(manifest=manifest,best=best.to_dict(),best_score=score(best,result.profile),
                candidates=[summary_row(value,result.profile) for value in result.candidates],history=result.history))
    write_json(out/"operator_trace.json",best.operator_timings)
    write_json(out/"communication_events.json",best.communication_events)
    write_json(out/"ownership_trace.json",best.derived.ownership_trace)
    if comparison is not None:
        write_json(out/"exhaustive_vs_rl.json",comparison)
        write_csv(out/"exhaustive_vs_rl.csv",[comparison])
    if ablation is not None:
        write_json(out/"ablation.json",ablation)
        write_csv(out/"ablation.csv",ablation)
    render_report(out/"report.html",result,graph,comparison,ablation)
    return out


def render_report(path, result, graph, comparison, ablation):
    best, profile = result.best, result.profile
    m = best.metrics
    def escape(value):
        return html.escape(str(value))
    def table(rows, columns):
        return '<table><thead><tr>'+''.join(f'<th>{escape(c)}</th>' for c in columns)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join(f'<td>{escape(row.get(c,""))}</td>' for c in columns)+'</tr>' for row in rows)+'</tbody></table>'
    blocks = []
    for block in best.mapping_plan["blocks"]:
        ops = [op for op in best.operator_timings if op["block_id"] == block["block_id"]]
        blocks.append(dict(block=block["block_id"],N_active=block["active_chiplet_count"],chiplets=block["active_chiplets"],
                           compute_ms=round(sum(op["compute_latency_ns"] for op in ops)/1e6,5),
                           communication_ms=round(sum(op["communication_latency_ns"] for op in ops)/1e6,5)))
    geometry = m["connection_graph"]
    max_x = max((node["x"] for node in geometry["nodes"]),default=0)
    max_y = max((node["y"] for node in geometry["nodes"]),default=0)
    coords = {node["id"]:(45+node["x"]*35/(max_x or 1)*8,45+node["y"]*35/(max_y or 1)*8) for node in geometry["nodes"]}
    links = ''.join(f'<line x1="{coords[link["source"]][0]}" y1="{coords[link["source"]][1]}" x2="{coords[link["target"]][0]}" y2="{coords[link["target"]][1]}" stroke="#94a3b8" stroke-width="4"/>' for link in geometry["physical_links"])
    nodes = ''.join(f'<g><circle id="chip{node}" cx="{x}" cy="{y}" r="24" fill="#cbd5e1"/><text x="{x}" y="{y+5}" text-anchor="middle">C{node}</text></g>' for node,(x,y) in coords.items())
    selector = '<select id="block">'+''.join(f'<option value="{index}">{escape(row["block"])}</option>' for index,row in enumerate(blocks))+'</select>'
    mapping_js = json.dumps([row["chiplets"] for row in blocks])
    assumptions = '<ul>'+''.join(f'<li><strong>{escape(key)}</strong>: {escape(value)}</li>' for key,value in ASSUMPTIONS.items())+'</ul>'
    comparisons = '<h2>Exhaustive / RL</h2>'+table([comparison],list(comparison)) if comparison else ''
    ablations = '<h2>Ablation</h2>'+table(ablation,["mode","seed","unique_evaluations","feasible","weighted_cost","N_total","total_latency_ns","total_area_mm2","total_power_w"]) if ablation else ''
    document = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Communication-aware Chiplet DSE</title>
<style>body{{margin:32px auto;max-width:1100px;padding:0 24px;font:16px/1.6 system-ui;color:#17243b;background:#f8fafc}}h1{{font-size:30px}}h2{{margin-top:36px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:white;border:1px solid #dce4ee;padding:20px;min-width:220px;border-radius:12px}}.value{{font-size:28px;color:#0f766e}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}td,th{{padding:9px;border-bottom:1px solid #dde4ec;text-align:left}}th{{background:#e2e8f0}}svg{{max-width:420px;background:white;border:1px solid #ddd;border-radius:12px}}a{{color:#0f766e}}li{{margin-bottom:10px}}select{{font:inherit;padding:8px}}code{{background:#e2e8f0;padding:2px 5px}}</style>
<h1>Communication-aware Chiplet DSE</h1><p>{escape(graph.model_name)} · {len(graph.blocks)} semantic blocks · {len(graph.operators)} operators · {escape(m["backend"])} · {escape(result.method)} · seed {result.seed}</p>
<div class="cards"><div class="card">Batch=1 latency<div class="value">{m["total_latency_ns"]/1e6:.4f} ms</div></div><div class="card">Area / power<div class="value">{m["total_area_mm2"]:.2f} mm² / {m["total_power_w"]:.2f} W</div></div><div class="card">Physical chiplets / output FPS<div class="value">{m["N_total"]} / {m["achieved_fps"]:.3f}</div></div></div>
<p>Feasible: <strong>{score(best,profile)["feasible"]}</strong> · weighted L/A/P cost: {score(best,profile)["weighted_cost"]:.6f} · {result.unique_evaluations} unique complete evaluations · {result.prefix_evaluations} prefix evaluations · {result.episodes} episodes.</p>
<p>Serial latency = compute {m["compute_latency_ns"]/1e6:.5f} ms + intra-block communication {m["network_latency_ns"]/1e6:.5f} ms + cross-block communication {m["transition_latency_ns"]/1e6:.5f} ms. Residual branches follow the same serial timeline.</p>
<p>Mesh traffic: {m["total_traffic_bytes"]/1024**2:.3f} MiB; hop-weighted traffic: {m["hop_weighted_traffic_bytes"]/1024**2:.3f} MiB·hop; maximum average link utilization: {m["max_link_utilization"]:.4%}. Utilization uses the complete serial inference time; this model does not simulate queues.</p>
<h2>Chiplet assignment</h2><p>Select a block to see its active physical chiplets. Ordered IDs map ascending channel shards.</p>{selector}<p id="active"></p><svg viewBox="0 0 380 380" role="img" aria-label="Physical mesh and active chiplets">{links}{nodes}</svg>
{table(blocks,["block","N_active","chiplets","compute_ms","communication_ms"])}
<h2>Traces and reproducibility</h2><p>{' · '.join(f'<a href="{name}">{name}</a>' for name in ("best.csv","chiplet_results.csv","mapping_trace.csv","canonical_flows.csv","transition_traffic.csv","operator_trace.csv","per_link_statistics.csv","assumptions.json","model_metadata.json","run_manifest.json"))}</p>
{comparisons}{ablations}<h2>Model scope and assumptions</h2>{assumptions}
<p>Architecture violations: {escape(m["architecture_violations"])}</p>
<script>const assignments={mapping_js};const select=document.getElementById('block');function update(){{const ids=assignments[Number(select.value)];for(let i=0;i<{m["N_total"]};i++)document.getElementById('chip'+i).setAttribute('fill',ids.includes(i)?'#5eead4':'#cbd5e1');document.getElementById('active').textContent='N_active = '+ids.length+'; ordered chiplets: '+ids.join(', ');}}select.addEventListener('change',update);update();</script></html>'''
    path.write_text(document,encoding="utf-8")
