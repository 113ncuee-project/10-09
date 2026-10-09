"""Layer-switching traffic evidence derived from the evaluated region plan."""
from collections import defaultdict


def layer_switch_evidence(evaluation):
    compiled,communication = evaluation.compiled,evaluation.communication
    if compiled is None:
        return []
    workload,plan = compiled.workload,compiled.plan
    units = compiled.unit_map
    stats = {row['id']:row for row in evaluation.transfer_statistics}
    rows = {}
    for layer in workload.layers:
        for tensor_id in layer.inputs:
            tensor = workload.tensor_map[tensor_id]
            if not tensor.producer:
                continue
            producer = tensor.producer
            a,b = plan.mapping_map[producer],plan.mapping_map[layer.id]
            key = producer,layer.id,tensor_id
            rows[key] = {'producer':producer,'consumer':layer.id,'tensor':tensor_id,
                         'source_part_BKHW':a.partition.as_tuple(),'destination_part_BKHW':b.partition.as_tuple(),
                         'same_group':plan.group_map[producer].id==plan.group_map[layer.id].id,
                         'requested_bytes':0,'injected_bytes':0,'delivered_bytes':0,'nop_hop_bytes':0,
                         'transfer_count':0,'fanout_max':0,'source_cores':set(),'destination_cores':set()}
    for unit in compiled.units:
        for req in unit.inputs:
            tensor = workload.tensor_map[req.tensor_id]
            key = tensor.producer,unit.layer_id,tensor.id
            if key in rows:
                rows[key]['requested_bytes'] += req.region.elements*tensor.bytes_per_element
    for transfer in communication.transfers:
        if transfer.tensor not in workload.tensor_map or not transfer.consumer_units:
            continue
        tensor = workload.tensor_map[transfer.tensor]
        for consumer in dict.fromkeys(units[name].layer_id for name in transfer.consumer_units):
            key = tensor.producer,consumer,transfer.tensor
            if key not in rows:
                continue
            row = rows[key]
            # Per-edge attribution: a transfer shared by distinct consumer
            # layers appears in each edge; totals below are not additive.
            destinations = {units[name].chiplet for name in transfer.consumer_units if units[name].layer_id==consumer}
            row['injected_bytes'] += transfer.bytes
            row['delivered_bytes'] += transfer.bytes*len(destinations)
            row['nop_hop_bytes'] += stats[transfer.id]['nop_hop_bytes']
            row['transfer_count'] += 1
            row['fanout_max'] = max(row['fanout_max'],len(destinations))
            row['source_cores'].add(transfer.src)
            row['destination_cores'].update(destinations)
    result = []
    for row in rows.values():
        a,b = row['source_part_BKHW'],row['destination_part_BKHW']
        if not row['same_group']:
            pattern = 'DRAM materialization / gather and redistribution'
        elif row['transfer_count']==0:
            pattern = 'local reuse'
        elif a[1]>1 and workload.layer_map[row['consumer']].kind=='Conv2d' and workload.layer_map[row['consumer']].groups==1:
            pattern = 'output-K shards → input-C gather (region transfers, no free collective)'
        elif row['fanout_max']>1:
            pattern = 'multicast / shared input regions'
        else:
            pattern = 'unicast / halo or partition redistribution'
        result.append({**row,'pattern':pattern,'source_cores':sorted(row['source_cores']),
                       'destination_cores':sorted(row['destination_cores']),
                       'accounting':'shared transfer is attributed to each consuming layer; edge rows are not additive totals'})
    return result
