"""Actual package geometry + parameter-derived die/PE diagrams and inspection."""
from dataclasses import asdict
import html
import json
import math
import re
from pathlib import Path

from .ondie import build_on_die_fabric


def layout_data(evaluator, plan, evaluation):
    hw = evaluator.hardware
    assignments = {die:[] for die in evaluator.package.compute_ids}
    for item in plan.layer_mappings:
        for die in dict.fromkeys(core.split('/')[0] for core in item.chiplets):
            assignments[die].append({'layer':item.layer_id,'cores':[core for core in item.chiplets if core.startswith(die+'/')],
                                    'partition':item.partition.as_tuple(),'dataflow':item.dataflow,
                                    'tile':item.tile_shape,'microtile':item.microtile_shape})
    return {'hardware':hw.to_dict(),'metrics':evaluation.metrics,
            'nodes':[asdict(node) for node in evaluator.package.nodes.values()],
            'links':[asdict(link) for link in evaluator.package.links],
            'assignments':assignments,'fabric':asdict(build_on_die_fabric(hw)),
            'source_identity':evaluator.identity,'model':evaluator.workload.name,
            'interpretation':'Package positions/dimensions are actual model inputs; die/PE inset is a resource schematic, not a routed physical floorplan.'}


def die_svg(hw, *, title='Compute die', width=740):
    # Every PE and ring derives from the evaluated hardware, including 4/8-die
    # granularity variants with 64/32 PEs. No stock 16-PE image is substituted.
    fabric = build_on_die_fabric(hw)
    columns = math.ceil(math.sqrt(hw.pe_count))
    rows = hw.ring_count if hw.noc_topology=='multi_ring' else math.ceil(hw.pe_count/columns)
    height = 175+rows*68
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" rx="14" fill="#f2f7fc" stroke="#24374d"/>',
             f'<text x="18" y="28" font-family="sans-serif" font-size="18">{html.escape(title)}: {hw.pe_count} PEs / {hw.cores_per_die} cores</text>']
    label = f'D2D → interdie router → UB {hw.unified_buffer_bytes/1024**2:g} MiB → W/A/O L2'
    parts += [f'<text x="18" y="58" font-family="sans-serif" font-size="13">{label}</text>',
              f'<text x="18" y="80" font-family="sans-serif" font-size="13">L2/slot W={hw.l2_weight_bytes//1024} A={hw.l2_activation_bytes//1024} O={hw.l2_output_bytes//1024} KiB; {hw.prefetch_slots} scratch slots</text>']
    palette = ['#d9eaf8','#d9eddd','#fff0c9','#f0dff2','#d9edef','#e6e1fd']
    cell = (width-36)/columns
    centers = {}
    for pe in range(hw.pe_count):
        row,col = divmod(pe,columns)
        ring = fabric.ring_for_pe(f'pe:{pe}') if hw.noc_topology=='multi_ring' else None
        if ring is not None:
            members = fabric.rings[ring][1:]
            col = members.index(f'pe:{pe}')
            cell = (width-160)/len(members)
            x,y = 155+col*cell,112+ring*68
        else:
            x,y = 18+col*cell,112+row*68
        centers[f'pe:{pe}'] = (x+(cell-8)/2,y+27)
        fill = palette[(ring or 0)%len(palette)]
        parts.append(f'<rect x="{x}" y="{y}" width="{cell-8}" height="54" rx="5" fill="{fill}" stroke="#536577"/>')
        parts.append(f'<text x="{x+7}" y="{y+22}" font-family="sans-serif" font-size="12">PE {pe} / core {pe//hw.pes_per_core}</text>')
        parts.append(f'<text x="{x+7}" y="{y+42}" font-family="sans-serif" font-size="11">{("Ring "+str(ring)) if ring is not None else "XY mesh"}</text>')
    parts.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#48759f"/></marker></defs>')
    if hw.noc_topology=='multi_ring':
        parts.append(f'<rect x="18" y="112" width="118" height="{rows*68-14}" fill="#dce7ee" stroke="#536577"/>')
        for i,label in enumerate(('UB / DMA','WL2','AL2','OL2')):
            parts.append(f'<text x="30" y="{134+i*16}" font-family="sans-serif" font-size="12">{label}</text>')
        for ring in fabric.rings:
            nodes = ring[1:]
            y = centers[nodes[0]][1]
            start,end = centers[nodes[0]],centers[nodes[-1]]
            parts.append(f'<path d="M136 {y} L{start[0]-(cell-8)/2} {y}" fill="none" stroke="#48759f" marker-end="url(#arrow)"/>')
            for a,b in zip(nodes,nodes[1:]):
                left,right = centers[a],centers[b]
                parts.append(f'<path d="M{left[0]+(cell-8)/2} {y} L{right[0]-(cell-8)/2} {y}" fill="none" stroke="#48759f" marker-end="url(#arrow)"/>')
            parts.append(f'<path d="M{end[0]} {y+27} L{end[0]} {y+33} L136 {y+33}" fill="none" stroke="#48759f" marker-end="url(#arrow)"/>')
    else:
        # Exact mesh adjacency, behind labels; resource connections are the
        # physical fabric used by the oracle (not a ring-efficiency drawing).
        for a,b in fabric.edges:
            if a!='l2' and b!='l2' and a<b:
                left,right = centers[a],centers[b]
                parts.append(f'<line x1="{left[0]}" y1="{left[1]}" x2="{right[0]}" y2="{right[1]}" stroke="#48759f" opacity=".3"/>')
    footer = f'{hw.ring_count} directed L2→PE→L2 rings' if hw.noc_topology=='multi_ring' else f'{hw.pe_count} PE XY mesh + L2 port'
    parts.append(f'<text x="18" y="{height-18}" font-family="sans-serif" font-size="13">{footer}; {hw.noc_bandwidth_Bps/1e9:g} GB/s/link</text></svg>')
    return ''.join(parts)


def pe_svg(hw):
    width,height = 410,max(310,130+hw.lanes*22)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#fffbef" stroke="#24374d" rx="12"/>',
             f'<text x="16" y="28" font-family="sans-serif" font-size="18">PE: {hw.lanes} lanes × {hw.vector_size} vector MACs</text>',
             f'<text x="16" y="54" font-family="sans-serif" font-size="13">INT{hw.activation_bits} × INT{hw.weight_bits} → INT{hw.psum_bits} local ACC</text>',
             f'<text x="16" y="76" font-family="sans-serif" font-size="13">WL1/AL1/OL1 {hw.l1_weight_bytes//1024}/{hw.l1_activation_bytes//1024}/{hw.l1_output_bytes//1024} KiB/slot</text>']
    for lane in range(hw.lanes):
        y = 96+lane*22
        parts.append(f'<rect x="18" y="{y}" width="372" height="19" fill="#d9eaf8" stroke="#859bac"/>')
        parts.append(f'<text x="26" y="{y+14}" font-family="sans-serif" font-size="11">Lane {lane}: {hw.vector_size} × MAC → ACC → OL1</text>')
    parts.append(f'<text x="16" y="{height-16}" font-family="sans-serif" font-size="13">{hw.compute_clock_hz/1e9:g} GHz; {hw.prefetch_slots} capacity-backed scratch slots</text></svg>')
    return ''.join(parts)


def save_layout(folder,evaluator,plan,evaluation):
    from .report import write_json
    folder = Path(folder)
    data = layout_data(evaluator,plan,evaluation)
    write_json(folder/'layout.json',data)
    die = die_svg(evaluator.hardware)
    pe = pe_svg(evaluator.hardware)
    (folder/'die.svg').write_text(die,encoding='utf-8')
    (folder/'pe.svg').write_text(pe,encoding='utf-8')
    package = (folder/'package.svg').read_text(encoding='utf-8')
    # SVG nested viewBoxes preserve geometry. Insets are explicitly schematics.
    heights = [float(part.split('viewBox="0 0 ')[1].split('"')[0].split()[1]) for part in (die,pe)]
    total_height = max(680,heights[0],heights[1])+130
    def nested(svg,x,y,w,h):
        head,body = svg.split('>',1)
        head = re.sub(r'\s(?:x|y|width|height)="[^"]*"','',head)
        return head.replace('<svg ',f'<svg x="{x}" y="{y}" width="{w}" height="{h}" ',1)+'>'+body
    arch = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1810 {total_height}"><rect width="100%" height="100%" fill="white"/><text x="25" y="35" font-size="24" font-family="sans-serif">{html.escape(evaluator.workload.name)}: {evaluator.hardware.compute_count} compute + {evaluator.hardware.cluster_count} IO + {evaluator.hardware.cluster_count} DRAM</text>'
    arch += nested(package,20,75,565,total_height-145)+nested(die,615,75,740,heights[0])+nested(pe,1380,75,410,heights[1])
    arch += f'<text x="25" y="{total_height-20}" font-size="16" font-family="sans-serif">Package: evaluated placement (mm). Die and PE: resource schematics. No RTL/floorplan calibration.</text></svg>'
    (folder/'architecture.svg').write_text(arch,encoding='utf-8')
    payload = json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>晶片布局與 mapping</title>
<style>body{font:15px system-ui;background:#f5f7fc;color:#233548;margin:20px}h1{font-size:22px}.grid{display:grid;grid-template-columns:minmax(380px,1fr) minmax(420px,1fr);gap:20px}.card{background:white;padding:18px;border-radius:12px}svg{max-width:100%;height:auto}[data-node]{cursor:pointer}pre{white-space:pre-wrap;font:13px system-ui;max-height:330px;overflow:auto}select{padding:8px;max-width:100%}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1 id="heading"></h1><p>點選實際 compute die，檢查該 die 的 cores、layer、Part 與片內 dataflow；切換 layer 可查看部署範圍。</p><label>Layer <select id="layer"><option value="">全部</option></select></label><div class="grid"><section class="card" id="package">PACKAGE_SVG</section><section class="card"><h2 id="selected"></h2><div>DIE_SVG</div><pre id="details"></pre></section></div><section class="card"><h2>PE 資源</h2>PE_SVG</section><p id="note"></p>
<script>const data=PAYLOAD;let selected=data.nodes.find(n=>n.role==='compute').id;const layer=document.getElementById('layer');const names=[...new Set(Object.values(data.assignments).flat().map(x=>x.layer))];for(const n of names){const o=document.createElement('option');o.value=n;o.textContent=n;layer.append(o)}function render(){document.getElementById('heading').textContent=`${data.model}: ${data.hardware.compute_count} compute / ${data.hardware.cluster_rows}×${data.hardware.cluster_cols} IO mesh`;document.getElementById('selected').textContent=selected;const rows=data.assignments[selected].filter(x=>!layer.value||x.layer===layer.value);document.getElementById('details').textContent=JSON.stringify({physical:data.nodes.find(n=>n.id===selected),mapping:rows},null,2);for(const el of document.querySelectorAll('[data-node]')){const id=el.dataset.node;const active=!layer.value||(data.assignments[id]||[]).some(x=>x.layer===layer.value);el.style.opacity=active?1:.22;el.style.strokeWidth=id===selected?'3':'1'}}for(const el of document.querySelectorAll('[data-node]'))el.addEventListener('click',()=>{if(el.dataset.node.startsWith('compute:')){selected=el.dataset.node;render()}});layer.addEventListener('change',render);document.getElementById('note').textContent=data.interpretation;render();</script></html>'''
    page = page.replace('PACKAGE_SVG',package).replace('DIE_SVG',die).replace('PE_SVG',pe).replace('PAYLOAD',payload)
    (folder/'layout.html').write_text(page,encoding='utf-8')
