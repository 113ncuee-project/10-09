from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

from .evaluator import EvaluationResult, select_best_by_model


CSV_FIELDS = [
    "model",
    "topology",
    "total_chiplets",
    "used_chiplets",
    "unused_chiplets",
    "status",
    "achieved_fps",
    "area_mm2",
    "power_w",
    "latency_ns",
    "rapid_ici_latency_ns",
    "latency_policy",
    "network_backend",
    "injected_bytes",
    "rapidchiplet_throughput_bits_per_cycle",
    "ppa_goal",
    "ppa_score",
    "workload_plan",
]


def write_outputs(out_dir: str | Path, summaries: list[EvaluationResult], sweeps: list[EvaluationResult]) -> dict[str, bool]:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    best_rows = select_best_by_model(sweeps)
    written = {
        "report.html": _try_write(path / "report.html", lambda: _write_html(path / "report.html", summaries, best_rows)),
        "best.json": _try_write(path / "best.json", lambda: _write_json(path / "best.json", best_rows)),
        "best.csv": _try_write(path / "best.csv", lambda: _write_csv(path / "best.csv", best_rows)),
        "summary.json": _try_write(path / "summary.json", lambda: _write_json(path / "summary.json", summaries)),
        "summary.csv": _try_write(path / "summary.csv", lambda: _write_csv(path / "summary.csv", summaries)),
        "chiplet_results.csv": _try_write(
            path / "chiplet_results.csv",
            lambda: _write_csv(path / "chiplet_results.csv", sweeps),
        ),
    }
    return written


def _try_write(path: Path, writer) -> bool:
    try:
        writer()
        return True
    except PermissionError:
        print(f"Skipped locked output file: {path}")
        return False


def _write_json(path: Path, rows: list[EvaluationResult]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump([_json_safe(row.to_dict()) for row in rows], f, indent=2, allow_nan=False)


def _write_csv(path: Path, rows: list[EvaluationResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row))


def _csv_row(row: EvaluationResult) -> dict[str, object]:
    return {
        "model": row.model,
        "topology": row.topology,
        "total_chiplets": row.available_chiplets,
        "used_chiplets": row.selected_chiplets,
        "unused_chiplets": row.unused_chiplets,
        "status": "VALID" if row.architecture_feasible else "INVALID",
        "achieved_fps": _csv_float(row.achieved_fps),
        "area_mm2": _csv_float(row.total_area_mm2),
        "power_w": _csv_float(row.total_power_w),
        "latency_ns": _csv_float(row.avg_latency_ns),
        "rapid_ici_latency_ns": _csv_float(row.rapid_avg_latency_ns),
        "latency_policy": row.latency_policy,
        "network_backend": row.network_context.get("backend", ""),
        "injected_bytes": _csv_float(sum(float(flow["bytes"]) for flow in row.canonical_flows)),
        "rapidchiplet_throughput_bits_per_cycle": _csv_float(row.aggregate_throughput_bits_per_cycle),
        "ppa_goal": row.ppa_goal,
        "ppa_score": _csv_float(row.ppa_score),
        "workload_plan": row.workload_plan,
    }


def _csv_float(value: float) -> str:
    if not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.4f}"


def _write_html(path: Path, rows: list[EvaluationResult], best_rows: list[EvaluationResult]) -> None:
    max_perf = max((r.achieved_fps for r in rows if math.isfinite(r.achieved_fps)), default=1.0)
    max_power = max((r.total_power_w for r in rows if math.isfinite(r.total_power_w)), default=1.0)
    best_max_perf = max((r.achieved_fps for r in best_rows if math.isfinite(r.achieved_fps)), default=1.0)
    best_max_power = max((r.total_power_w for r in best_rows if math.isfinite(r.total_power_w)), default=1.0)
    best_table_rows = "\n".join(_html_row(row, best_max_perf, best_max_power) for row in best_rows)
    table_rows = "\n".join(_html_row(row, max_perf, max_power) for row in rows)
    connection_cards = "\n".join(_connection_card(row, index) for index, row in enumerate(rows))
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Simplified RapidChiplet Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #657280;
      --line: #d8dee6;
      --ok: #117a65;
      --bad: #a93226;
      --perf: #2266aa;
      --power: #b86b1b;
      --bg: #f7f8fa;
      --traffic: #c6422f;
      --link: #8a97a6;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: white;
    }}
    header {{
      padding: 24px 32px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--bg);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    main {{
      padding: 24px 32px 36px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
      margin-bottom: 24px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }}
    th {{
      color: #26323f;
      background: #eef2f6;
      font-weight: 700;
    }}
    .status {{
      font-weight: 700;
    }}
    .ok {{
      color: var(--ok);
    }}
    .bad {{
      color: var(--bad);
    }}
    .bar {{
      display: inline-block;
      width: 120px;
      height: 9px;
      border: 1px solid var(--line);
      background: #fff;
      margin-left: 8px;
      vertical-align: middle;
    }}
    .bar span {{
      display: block;
      height: 9px;
    }}
    .perf span {{
      background: var(--perf);
    }}
    .power span {{
      background: var(--power);
    }}
    h2 {{
      margin: 26px 0 12px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .connections {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .partition-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
      align-items: start;
      margin-bottom: 24px;
    }}
    .partition-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }}
    .partition-card h3 {{
      margin: 0 0 6px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .partition-card p {{
      margin-bottom: 10px;
      font-size: 13px;
    }}
    .partition-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(94px, 1fr));
      gap: 8px;
    }}
    .partition-chiplet {{
      min-height: 82px;
      border: 1px solid var(--line);
      border-top: 4px solid var(--stage-color);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfd;
    }}
    .partition-chiplet strong {{
      display: block;
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .partition-chiplet .stage {{
      min-height: 28px;
      color: var(--ink);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .partition-chiplet .metric {{
      color: var(--muted);
      font-size: 11px;
      margin-top: 4px;
    }}
    .load-bar {{
      width: 100%;
      height: 5px;
      margin-top: 7px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .load-bar span {{
      display: block;
      height: 5px;
      background: var(--stage-color);
    }}
    .connection-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }}
    .connection-card h3 {{
      margin: 0 0 6px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .connection-card p {{
      margin-bottom: 10px;
      font-size: 13px;
    }}
    .topology-svg {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--line);
      background: #fbfcfd;
    }}
    .topology-link {{
      stroke: var(--link);
      stroke-width: 2;
    }}
    .traffic-link {{
      stroke: var(--traffic);
      stroke-dasharray: 5 4;
      fill: none;
    }}
    .chiplet-node {{
      fill: #eef5ff;
      stroke: #2f6fad;
      stroke-width: 1.5;
    }}
    .chiplet-label {{
      fill: var(--ink);
      font-size: 11px;
      text-anchor: middle;
      dominant-baseline: central;
      font-weight: 700;
    }}
    .stage-label {{
      fill: var(--muted);
      font-size: 10px;
      text-anchor: middle;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 22px;
      height: 0;
      margin-right: 6px;
      vertical-align: middle;
      border-top: 2px solid var(--link);
    }}
    .legend .traffic::before {{
      border-top: 2px dashed var(--traffic);
    }}
    @media (max-width: 900px) {{
      main {{
        padding: 16px;
        overflow-x: auto;
      }}
      header {{
        padding: 18px 16px 10px;
      }}
      .connections {{
        grid-template-columns: 1fr;
      }}
      .partition-cards {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Simplified RapidChiplet Report</h1>
    <p>Legacy pipeline report. Summary rows use latency/area/power only. FPS = 1 / serial Batch=1 E2E seconds is a diagnostic. Canonical flows and effective network parameters are saved in JSON. Mapping-specific compute efficiency has been removed; exact operator ownership is available through the default communication-aware entry point.</p>
  </header>
  <main>
    <h2>Overall Best</h2>
    <table>
      <thead>
        {_table_header()}
      </thead>
      <tbody>
        {best_table_rows}
      </tbody>
    </table>
    <h2>Best Partition Views</h2>
    <div class="partition-cards">
      {"".join(_partition_card(row) for row in best_rows)}
    </div>
    <h2>Best Per Topology</h2>
    <table>
      <thead>
        {_table_header()}
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
    <h2>Best Logical Topology Views</h2>
    <div class="connections">
      {connection_cards}
    </div>
  </main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def _table_header() -> str:
    return """<tr>
          <th>Model</th>
          <th>Topology</th>
          <th>Status</th>
          <th>Chiplets</th>
          <th>FPS</th>
          <th>Area mm^2</th>
          <th>Power W</th>
          <th>PPA Score</th>
          <th>E2E latency ns</th>
          <th>Bottleneck Link</th>
          <th>Search</th>
          <th>Workload Plan</th>
        </tr>"""


def _html_row(row: EvaluationResult, max_perf: float, max_power: float) -> str:
    status = "valid" if row.architecture_feasible else "invalid"
    status_class = "ok" if row.architecture_feasible else "bad"
    perf_width = _pct(row.achieved_fps, max_perf)
    power_width = _pct(row.total_power_w, max_power)
    return f"""<tr>
  <td>{html.escape(row.model)}</td>
  <td>{html.escape(row.topology)}</td>
  <td class="status {status_class}">{status}</td>
  <td>{row.selected_chiplets}/{row.available_chiplets} <span style="color:#657280">(unused {row.unused_chiplets})</span></td>
  <td>{row.achieved_fps:.1f}<span class="bar perf"><span style="width:{perf_width}%"></span></span></td>
  <td>{row.total_area_mm2:.2f}</td>
  <td>{row.total_power_w:.2f}<span class="bar power"><span style="width:{power_width}%"></span></span></td>
  <td>{row.ppa_score:.3f} <span style="color:#657280">({html.escape(row.ppa_goal)})</span></td>
  <td>{row.avg_latency_ns:.2f}</td>
  <td>{html.escape(row.bottleneck_link)}</td>
  <td>{html.escape(row.workload_search)}</td>
  <td>{html.escape(row.workload_plan)}</td>
</tr>"""


def _partition_card(row: EvaluationResult) -> str:
    graph = row.connection_graph
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    traffic_edges = graph.get("traffic_edges", []) if isinstance(graph, dict) else []
    if not nodes:
        return ""

    traffic_by_source = {int(edge["source"]): float(edge["mb"]) for edge in traffic_edges}
    max_ops = max((float(node.get("ops", 0.0)) for node in nodes), default=1.0)
    node_tiles = []
    for node in sorted(nodes, key=lambda item: int(item["id"])):
        node_id = int(node["id"])
        stage = str(node.get("stage", ""))
        ops_g = float(node.get("ops", 0.0)) / 1e9
        ops_width = _pct(float(node.get("ops", 0.0)), max_ops)
        traffic_mb = traffic_by_source.get(node_id)
        traffic_label = "out -" if traffic_mb is None else f"out {traffic_mb:.2f} MB"
        color = _partition_color(_stage_key(stage))
        node_tiles.append(
            f"""<div class="partition-chiplet" style="--stage-color:{color}">
  <strong>C{node_id + 1}</strong>
  <div class="stage">{html.escape(stage)}</div>
  <div class="metric">{ops_g:.2f} GOP, {traffic_label}</div>
  <div class="load-bar"><span style="width:{ops_width}%"></span></div>
</div>"""
        )

    return f"""<section class="partition-card">
  <h3>{html.escape(row.model)} / {html.escape(row.topology)}</h3>
  <p>{row.selected_chiplets} active chiplets, PPA score {row.ppa_score:.3f}, {html.escape(row.ppa_goal)} goal.</p>
  <div class="partition-strip">
    {"".join(node_tiles)}
  </div>
</section>"""


def _connection_card(row: EvaluationResult, index: int) -> str:
    graph = row.connection_graph
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    if not nodes:
        return ""

    return f"""<section class="connection-card">
  <h3>{html.escape(row.model)} / {html.escape(row.topology)}</h3>
  <p>{row.selected_chiplets} active chiplets, {row.link_count} RapidChiplet links, package {row.chip_width_mm:.2f} x {row.chip_height_mm:.2f} mm, traffic {row.total_traffic_mb_per_inference:.2f} MB/inference.</p>
  {_graph_svg(row, index)}
  <div class="legend"><span>topology link</span><span class="traffic">workload traffic</span></div>
</section>"""


def _graph_svg(row: EvaluationResult, index: int) -> str:
    graph = row.connection_graph
    nodes = graph.get("nodes", [])
    physical_links = graph.get("physical_links", [])
    traffic_edges = graph.get("traffic_edges", [])
    width = 760
    height = 360
    margin = 54
    positions = _display_positions(row, nodes, physical_links, width, height, margin)
    max_traffic = max((float(edge["mb"]) for edge in traffic_edges), default=1.0)
    marker_id = f"arrow-{index}"
    link_markup = []
    for link in physical_links:
        source = int(link["source"])
        target = int(link["target"])
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        length = float(link.get("length_mm", 0.0))
        link_markup.append(
            f'<line class="topology-link" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"><title>C{source + 1}->C{target + 1}, RapidChiplet link length {length:.2f} mm</title></line>'
        )

    traffic_markup = []
    for edge in traffic_edges:
        source = int(edge["source"])
        target = int(edge["target"])
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        mb = float(edge["mb"])
        bend = 18 if source <= target else -18
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 - bend
        stroke_width = 1.6 + 2.8 * (mb / max_traffic if max_traffic > 0 else 0)
        traffic_markup.append(
            f'<path class="traffic-link" d="M {x1:.1f} {y1:.1f} Q {mx:.1f} {my:.1f} {x2:.1f} {y2:.1f}" stroke-width="{stroke_width:.2f}" marker-end="url(#{marker_id})"><title>C{source + 1}->C{target + 1}, {mb:.2f} MB/inference</title></path>'
        )

    node_markup = []
    node_size = 36
    for node in nodes:
        node_id = int(node["id"])
        x, y = positions[node_id]
        stage = str(node.get("stage", ""))
        ops_g = float(node.get("ops", 0.0)) / 1e9
        label = _short_label(stage)
        node_markup.append(
            f"""<g>
  <rect class="chiplet-node" x="{x - node_size / 2:.1f}" y="{y - node_size / 2:.1f}" width="{node_size}" height="{node_size}" rx="5">
    <title>chiplet C{node_id + 1}: {html.escape(stage)}, {ops_g:.2f} GOP/inference</title>
  </rect>
  <text class="chiplet-label" x="{x:.1f}" y="{y - 2:.1f}">C{node_id + 1}</text>
  <text class="stage-label" x="{x:.1f}" y="{y + node_size / 2 + 14:.1f}">{html.escape(label)}</text>
</g>"""
        )

    return f"""<svg class="topology-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(row.model)} {html.escape(row.topology)} chiplet connection graph">
  <defs>
    <marker id="{marker_id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#c6422f"></path>
    </marker>
  </defs>
  {"".join(link_markup)}
  {"".join(traffic_markup)}
  {"".join(node_markup)}
</svg>"""


def _display_positions(
    row: EvaluationResult,
    nodes: list[dict[str, object]],
    links: list[dict[str, object]],
    width: int,
    height: int,
    margin: int,
) -> dict[int, tuple[float, float]]:
    if row.topology == "tree" and len(nodes) > 1:
        return _tree_display_positions(nodes, links, width, height, margin)
    return _scaled_display_positions(nodes, width, height, margin)


def _scaled_display_positions(
    nodes: list[dict[str, object]],
    width: int,
    height: int,
    margin: int,
) -> dict[int, tuple[float, float]]:
    xs = [float(node["x"]) for node in nodes]
    ys = [float(node["y"]) for node in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    def sx(value: float) -> float:
        if max_x == min_x:
            return width / 2
        return margin + (value - min_x) / span_x * (width - margin * 2)

    def sy(value: float) -> float:
        if max_y == min_y:
            return height / 2
        return margin + (value - min_y) / span_y * (height - margin * 2)

    return {int(node["id"]): (sx(float(node["x"])), sy(float(node["y"]))) for node in nodes}


def _tree_display_positions(
    nodes: list[dict[str, object]],
    links: list[dict[str, object]],
    width: int,
    height: int,
    margin: int,
) -> dict[int, tuple[float, float]]:
    node_ids = sorted(int(node["id"]) for node in nodes)
    root = node_ids[0]
    adj = {node_id: [] for node_id in node_ids}
    for link in links:
        source = int(link["source"])
        target = int(link["target"])
        if source in adj and target in adj:
            adj[source].append(target)
            adj[target].append(source)
    for node_id in adj:
        adj[node_id].sort()

    children = {node_id: [] for node_id in node_ids}
    depth = {root: 0}
    parent = {root: None}
    queue = [root]
    for node_id in queue:
        for neighbor in adj[node_id]:
            if neighbor in parent:
                continue
            parent[neighbor] = node_id
            depth[neighbor] = depth[node_id] + 1
            children[node_id].append(neighbor)
            queue.append(neighbor)

    for node_id in node_ids:
        if node_id not in parent:
            parent[node_id] = None
            depth[node_id] = 0

    leaf_slots: dict[int, float] = {}
    next_slot = 0

    def assign_slot(node_id: int) -> float:
        nonlocal next_slot
        if not children[node_id]:
            leaf_slots[node_id] = float(next_slot)
            next_slot += 1
            return leaf_slots[node_id]
        child_slots = [assign_slot(child) for child in children[node_id]]
        leaf_slots[node_id] = sum(child_slots) / len(child_slots)
        return leaf_slots[node_id]

    assign_slot(root)
    for node_id in node_ids:
        if node_id not in leaf_slots:
            leaf_slots[node_id] = float(next_slot)
            next_slot += 1

    max_slot = max(leaf_slots.values(), default=0.0)
    max_depth = max(depth.values(), default=0)

    def sx(slot: float) -> float:
        if max_slot <= 0:
            return width / 2
        return margin + slot / max_slot * (width - margin * 2)

    def sy(level: int) -> float:
        if max_depth <= 0:
            return height / 2
        return margin + level / max_depth * (height - margin * 2)

    return {node_id: (sx(leaf_slots[node_id]), sy(depth[node_id])) for node_id in node_ids}


def _short_label(value: str) -> str:
    if len(value) <= 18:
        return value
    return value[:15] + "..."


def _stage_key(value: str) -> str:
    return value.split("[", 1)[0]


def _partition_color(value: str) -> str:
    palette = ["#2f6fad", "#117a65", "#b86b1b", "#7d4aa8", "#a94442", "#4d7f87"]
    return palette[sum(ord(char) for char in value) % len(palette)]


def _pct(value: float, maximum: float) -> int:
    if not math.isfinite(value) or maximum <= 0:
        return 0
    return max(1, min(100, int(round(value / maximum * 100))))


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
