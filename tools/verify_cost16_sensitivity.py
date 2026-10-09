"""Independent full-oracle spot check of cost-only ledger transformations."""
import json
import argparse
import math
from pathlib import Path
from dataclasses import replace
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'simplified_rapidchiplet'))
import hybrid_dse
from simple_rapidchiplet.hybrid.hardware import HardwareSpec
from simple_rapidchiplet.hybrid.evaluator import HybridEvaluator
from simple_rapidchiplet.hybrid.search import plan_from_dict
from simple_rapidchiplet.hybrid.report import write_json
from sensitivity_hybrid_cost16 import FOLDER, ENERGY


def main():
    global FOLDER
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=FOLDER)
    FOLDER=parser.parse_args().folder.resolve()
    parent=FOLDER/'worker16/dies16_multi_ring/resnet18'
    hw=HardwareSpec.from_dict(json.loads((parent/'hardware.json').read_text(encoding='utf-8')))
    plan=plan_from_dict(json.loads((parent/'mapping.json').read_text(encoding='utf-8')))
    workload=hybrid_dse.read_workload(parent/'workload.json')
    checks=[]
    for name in ('mac_2ops','sram_energy_150','leakage_200'):
        actual=HybridEvaluator(workload,replace(hw,**ENERGY[name][1])).evaluate(plan).metrics
        derived=next(row['metrics'] for row in json.loads((FOLDER/f'sensitivity/{name}/rows.json').read_text(encoding='utf-8'))
                     if row['architecture']=='dies16_multi_ring' and row['model']=='resnet18')
        for key in ('latency_s','energy_j','area_mm2','power_w','idle_power_w','dynamic_energy_j','static_energy_j'):
            assert math.isclose(actual[key],derived[key],rel_tol=1e-12,abs_tol=1e-15),(name,key,actual[key],derived[key])
        for key in actual['energy_components']:
            assert math.isclose(actual['energy_components'][key],derived['energy_components'][key],rel_tol=1e-12,abs_tol=1e-15)
        checks.append(dict(scenario=name,passed=True,model='resnet18',architecture='dies16_multi_ring',
                           full_oracle_identity=actual['evaluator_identity']))
        print(name,'matches full oracle',flush=True)
    write_json(FOLDER/'sensitivity_verification.json',dict(passed=True,checks=checks))


if __name__=='__main__':
    main()
