"""Gemini V-A exponent objective adapted to GUI Energy/Area/Delay preferences.

Area is an explicit proxy for fabrication cost; no monetary/yield estimator is
claimed. Each workload receives equal weight via a geometric mean, using fixed
common references across hardware (never each hardware's relative improvement).
"""
import math
from .search import SearchObjective

PREFERENCES = {
    'balanced': {'label': '均衡', 'energy': 1/3, 'area': 1/3, 'latency': 1/3},
    'low_power': {'label': '低功耗', 'energy': .8, 'area': .1, 'latency': .1},
    'small_area': {'label': '小面積', 'energy': .1, 'area': .8, 'latency': .1},
    'low_latency': {'label': '低延遲', 'energy': .1, 'area': .1, 'latency': .8},
}

def preference_objective(name='balanced', *, max_power=None, max_area=None, max_latency=None,
                         strict_power=True,strict_area=True,strict_latency=True,relax_hard_limits=False):
    weights = PREFERENCES[name]
    return SearchObjective(latency_weight=weights['latency'], energy_weight=weights['energy'],
        power_weight=0, area_weight=weights['area'], max_power_w=max_power if strict_power else None,
        max_area_mm2=max_area if strict_area else None, max_latency_s=max_latency if strict_latency else None,
        soft_power_w=None if strict_power else max_power, soft_area_mm2=None if strict_area else max_area,
        soft_latency_s=None if strict_latency else max_latency,relax_hard_limits=relax_hard_limits)


def constraint_report(metrics, objective):
    rows = []
    physical = metrics.get('feasible',False) is True and all(
        isinstance(metrics.get(key),(int,float)) and math.isfinite(metrics[key]) and metrics[key]>0
        for key in ('power_w','area_mm2','latency_s','energy_j'))
    for key,hard,soft in (('power_w',objective.max_power_w,objective.soft_power_w),
                          ('area_mm2',objective.max_area_mm2,objective.soft_area_mm2),
                          ('latency_s',objective.max_latency_s,objective.soft_latency_s)):
        limit = hard if hard is not None else soft
        if limit is None:
            continue
        value = metrics.get(key)
        valid = isinstance(value,(int,float)) and math.isfinite(value) and value>0
        gap = max(0,value/limit-1) if valid else math.inf
        rows.append(dict(metric=key,actual=value,limit=limit,strict=hard is not None,
                         within_target=gap==0,relative_excess=gap if valid else None,
                         excess_percent=gap*100 if valid else None))
    return dict(physical_feasible=physical,
                strict_compliant=physical and all(row['within_target'] for row in rows if row['strict']),
                all_targets_met=physical and all(row['within_target'] for row in rows),
                limits=rows,scope='nominal analytical PPA; no physical signoff')


def rank_architectures(rows, objective, references):
    """All selected workloads must pass; never use an average to hide a miss.

    If none pass strict limits, minimize maximum normalized strict excess,
    then mean excess, soft-goal excess, and preference cost. This closest-policy
    is our explicit extension, not an algorithm specified by Gemini.
    """
    from dataclasses import replace
    grouped = {}
    for row in rows:
        grouped.setdefault(row['architecture'],[]).append(row)
    expected = set(references)
    ranking = []
    unconstrained = replace(objective,max_area_mm2=None,max_power_w=None,max_latency_s=None,
                            relax_hard_limits=False)
    for name,trials in grouped.items():
        reports = [constraint_report(row['metrics'],objective) for row in trials]
        valid = {row['model'] for row in trials}==expected and all(report['physical_feasible'] for report in reports)
        pass_strict = valid and all(report['strict_compliant'] for report in reports)
        strict_gaps,soft_gaps = [],[]
        for report in reports:
            for limit in report['limits']:
                (strict_gaps if limit['strict'] else soft_gaps).append(limit['relative_excess'] if limit['relative_excess'] is not None else math.inf)
        cost = suite_cost(trials,objective,references) if pass_strict else math.inf
        pref = suite_cost(trials,unconstrained,references) if valid else math.inf
        ranking.append(dict(architecture=name,cost=cost if math.isfinite(cost) else None,
                            feasible_for_all_models=pass_strict,physical_feasible_for_all_models=valid,
                            preference_cost=pref if math.isfinite(pref) else None,
                            worst_strict_excess=max(strict_gaps,default=0) if valid else None,
                            mean_strict_excess=sum(strict_gaps)/max(1,len(strict_gaps)) if valid else None,
                            worst_soft_excess=max(soft_gaps,default=0) if valid else None))
    eligible = [row for row in ranking if row['feasible_for_all_models']]
    best = min(eligible,key=lambda row:(row['cost'],row['architecture'])) if eligible else None
    physical = [row for row in ranking if row['physical_feasible_for_all_models']]
    closest = (min(physical,key=lambda row:(row['worst_strict_excess'],row['mean_strict_excess'],
               row['worst_soft_excess'],row['preference_cost'],row['architecture'])) if physical and not best else None)
    recommended = best or closest
    return ranking, dict(best_architecture=best['architecture'] if best else None,
                         recommended_architecture=recommended['architecture'] if recommended else None,
                         recommendation_status='estimated_compliant' if best else 'closest_exceeds_strict_limits' if closest else 'no_physically_valid_candidate',
                         closest_policy='minimax normalized strict excess across all models, then mean excess, soft excess, preference',
                         constraint_reports=[dict(model=row['model'],**constraint_report(row['metrics'],objective))
                            for row in rows if recommended and row['architecture']==recommended['architecture']])

def suite_cost(rows, objective, references):
    costs = [objective.cost(row['metrics'], references[row['model']]) for row in rows]
    if not costs or any(not math.isfinite(cost) or cost <= 0 for cost in costs):
        return math.inf
    return math.exp(sum(math.log(cost) for cost in costs) / len(costs))
