"""Fixed-mapping cost sensitivity grounded in each actual hardware.json.

Pure-energy/leakage coefficients rescale exact saved component ledgers (no
timing/geometry dependence in this oracle). Area/clock changes run the full
native/compute/network/buffer evaluator. No reoptimization claim is made.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simplified_rapidchiplet'))
import hybrid_dse
from simple_rapidchiplet.hybrid.hardware import HardwareSpec
from simple_rapidchiplet.hybrid.evaluator import HybridEvaluator
from simple_rapidchiplet.hybrid.search import plan_from_dict
from simple_rapidchiplet.hybrid.preferences import preference_objective, rank_architectures, constraint_report
from simple_rapidchiplet.hybrid.report import write_json
from validate_hybrid_oct09 import source_hashes

FOLDER=ROOT/'simplified_rapidchiplet/results/hybrid_gui_cost16/c16100000001'
FULL={
    'sram_area_075':{'sram_area_scale':.75},
    'sram_area_125':{'sram_area_scale':1.25},
    'clock_1GHz_unverified':{'compute_clock_hz':1e9},
    'assumptions_high':{'sram_area_scale':1.25,'sram_energy_scale':1.5,'mac_ops_per_mac':2,
                        'noc_energy_pj_per_bit':.6,'io_crossbar_energy_pj_per_bit':.15,
                        'compute_leakage_w_per_mm2':.04,'io_leakage_w_per_mm2':.016,'memory_leakage_w_per_mm2':.01,
                        'memory_die_area_mm2':30,'ddr_controller_phy_area_mm2':12.6,
                        'dram_energy_pj_per_bit':10.5,'nop_energy_pj_per_bit':1.404},
}
ENERGY={
    'mac_2ops':({'mac_j':2},{'mac_ops_per_mac':2}),
    'sram_energy_050':({'on_die_sram_j':.5,'glb_sram_j':.5},{'sram_energy_scale':.5}),
    'sram_energy_150':({'on_die_sram_j':1.5,'glb_sram_j':1.5},{'sram_energy_scale':1.5}),
    'leakage_050':({'idle_j':.5},{'compute_leakage_w_per_mm2':.01,'io_leakage_w_per_mm2':.004,'memory_leakage_w_per_mm2':.0025,'link_idle_w_per_mm':.0005}),
    'leakage_200':({'idle_j':2},{'compute_leakage_w_per_mm2':.04,'io_leakage_w_per_mm2':.016,'memory_leakage_w_per_mm2':.01,'link_idle_w_per_mm':.002}),
    'fabric_energy_050':({'on_die_noc_j':.5,'glb_fabric_j':.5,'io_crossbar_j':.5},{'noc_energy_pj_per_bit':.2,'io_crossbar_energy_pj_per_bit':.05}),
    'fabric_energy_150':({'on_die_noc_j':1.5,'glb_fabric_j':1.5,'io_crossbar_j':1.5},{'noc_energy_pj_per_bit':.6,'io_crossbar_energy_pj_per_bit':.15}),
    'grs_energy_080':({'nop_j':.8},{'nop_energy_pj_per_bit':.936}),
    'grs_energy_120':({'nop_j':1.2},{'nop_energy_pj_per_bit':1.404}),
    'dram_energy_080':({'dram_j':.8},{'dram_energy_pj_per_bit':7.0}),
    'dram_energy_120':({'dram_j':1.2},{'dram_energy_pj_per_bit':10.5}),
}


def baseline():
    suite=json.loads((FOLDER/'suite.json').read_text(encoding='utf-8'))
    hashes=source_hashes()
    for row in suite['runs']:
        assert row['metrics']['oracle_source_sha256']==hashes
    return suite


def run_full(name):
    suite=baseline()
    rows=[]
    for row in suite['runs']:
        parent=Path(row['folder'])
        data=json.loads((parent/'hardware.json').read_text(encoding='utf-8'))
        hardware=replace(HardwareSpec.from_dict(data),**FULL[name])
        mapping_path=parent/'mapping.json'
        plan=plan_from_dict(json.loads(mapping_path.read_text(encoding='utf-8')))
        workload=hybrid_dse.read_workload(parent/'workload.json')
        ev=HybridEvaluator(workload,hardware)
        metrics=ev.evaluate(plan).metrics
        assert metrics['total_macs']==row['metrics']['total_macs']
        folder=FOLDER/'sensitivity'/name/row['architecture']/row['model']
        write_json(folder/'hardware.json',asdict(hardware))
        result=dict(model=row['model'],architecture=row['architecture'],metrics=metrics,
                    hardware_json=str((folder/'hardware.json').resolve()),
                    baseline_hardware_sha256=hashlib.sha256((parent/'hardware.json').read_bytes()).hexdigest(),
                    fixed_mapping_sha256=hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
                    mode='full native/physical oracle, same saved mapping; not reoptimized')
        write_json(folder/'summary.json',result)
        rows.append(result)
        write_json(FOLDER/'sensitivity'/name/'rows.json',rows)
        print(name,row['architecture'],row['model'],metrics['feasible'],flush=True)
    return rows


def energy_rows(suite,name):
    scales,overrides=ENERGY[name]
    rows=[]
    for row in suite['runs']:
        parent=Path(row['folder'])
        hardware=HardwareSpec.from_dict(json.loads((parent/'hardware.json').read_text(encoding='utf-8')))
        scenario_hardware=replace(hardware,**overrides)
        metrics=dict(row['metrics'])
        components={key:value*scales.get(key,1) for key,value in metrics['energy_components'].items()}
        energy=sum(components.values())
        metrics.update(energy_components=components,energy_j=energy,energy_per_sample_j=energy/metrics['batch_size'],
                       static_energy_j=components['idle_j'],dynamic_energy_j=energy-components['idle_j'],
                       idle_power_w=metrics['idle_power_w']*scales.get('idle_j',1),power_w=energy/metrics['latency_s'],
                       edp_js=energy*metrics['latency_s'],sensitivity_derivation='exact coefficient rescaling of fixed schedule component ledger')
        # This transformed record is not a separately queried evaluator identity.
        metrics['derived_from_evaluator_identity']=metrics.pop('evaluator_identity',None)
        metrics['hardware_cost_provenance']=scenario_hardware.provenance
        metrics['sram_bank_costs']=scenario_hardware.sram_costs
        folder=FOLDER/'sensitivity'/name/row['architecture']/row['model']
        assert scenario_hardware.compute_area_mm2==hardware.compute_area_mm2
        write_json(folder/'hardware.json',asdict(scenario_hardware))
        rows.append(dict(model=row['model'],architecture=row['architecture'],metrics=metrics,
                         hardware_json=str((folder/'hardware.json').resolve()),
                         baseline_hardware_sha256=hashlib.sha256((parent/'hardware.json').read_bytes()).hexdigest(),
                         fixed_mapping_sha256=hashlib.sha256((parent/'mapping.json').read_bytes()).hexdigest(),
                         mode='exact fixed-schedule component ledger rescaling; no new timing/geometry oracle query'))
    write_json(FOLDER/'sensitivity'/name/'rows.json',rows)
    return rows


def merge():
    suite=baseline()
    sets={'nominal':suite['runs']}
    for name in ENERGY:
        sets[name]=energy_rows(suite,name)
    for name in FULL:
        sets[name]=json.loads((FOLDER/'sensitivity'/name/'rows.json').read_text(encoding='utf-8'))
        assert len(sets[name])==len(suite['runs'])
    refs={model:{key:1 for key in ('area_mm2','power_w','energy_j','latency_s')} for model in suite['models']}
    objectives={'unlimited':preference_objective(),
                'example_user_limits':preference_objective(max_power=16,max_area=800,max_latency=.5),
                # A boundary probe, not a user request or chosen success target.
                'boundary_probe':preference_objective(max_power=16,max_area=260,max_latency=.04)}
    summaries=[]
    for name,rows in sets.items():
        for label,objective in objectives.items():
            ranking,recommendation=rank_architectures(rows,objective,refs)
            summaries.append(dict(scenario=name,limits=label,ranking=ranking,**recommendation,
                                  compliant_trial_count=sum(constraint_report(row['metrics'],objective)['strict_compliant'] for row in rows),
                                  physical_trial_count=sum(row['metrics']['feasible'] for row in rows)))
    result=dict(scenarios=FULL,coefficient_scenarios=ENERGY,assumptions='chosen exploratory ranges, not statistical confidence intervals',
                methodology='same saved mapping/hardware per model; cost-only ledger transformations are exact for this oracle. Area or clock changes rerun native geometry/timing/physical buffers. Search is not rerun; no robust global-optimum claim.',
                source_sha256=source_hashes(),summaries=summaries)
    write_json(FOLDER/'sensitivity.json',result)
    for row in summaries:
        if row['limits']!='unlimited':
            print(row['scenario'],row['limits'],row['compliant_trial_count'],row['recommendation_status'],row['recommended_architecture'])


def main():
    global FOLDER
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=tuple(FULL))
    parser.add_argument('--merge',action='store_true')
    parser.add_argument('--folder',type=Path,default=FOLDER,help='source-matched fresh baseline folder')
    args=parser.parse_args()
    FOLDER=args.folder.resolve()
    run_full(args.scenario) if args.scenario else merge() if args.merge else parser.error('choose --scenario or --merge')


if __name__=='__main__':
    main()
