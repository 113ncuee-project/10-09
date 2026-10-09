"""Logical tensor versions and channel-slice residency until last consumer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OwnedSlice:
    chiplet: int
    start: int
    end: int


def merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def missing_ranges(start, end, available):
    result, cursor = [], start
    for left, right in merge_ranges(available):
        if right <= cursor or left >= end:
            continue
        if left > cursor:
            result.append((cursor, min(left, end)))
        cursor = max(cursor, right)
    if cursor < end:
        result.append((cursor, end))
    return tuple(result)


class TensorOwnership:
    def __init__(self, tensors, total_chiplets):
        self.tensors = tensors
        self.total_chiplets = total_chiplets
        self.primary = {}
        self.available = {}

    def produce(self, tensor_id, slices):
        tensor = self.tensors[tensor_id]
        ordered = sorted(slices, key=lambda owner: owner.start)
        cursor = 0
        for owner in ordered:
            if owner.start != cursor or owner.end <= owner.start:
                raise ValueError("primary ownership must exactly partition channels")
            cursor = owner.end
        if cursor != tensor.channels:
            raise ValueError("incomplete channel ownership")
        self.primary[tensor_id] = tuple(ordered)
        self.available[tensor_id] = {}
        for owner in ordered:
            self.receive(tensor_id, owner.chiplet, owner.start, owner.end)

    def receive(self, tensor_id, node, start, end):
        nodes = self.available.setdefault(tensor_id, {})
        nodes[node] = merge_ranges((*nodes.get(node, ()), (start, end)))

    def required_transfers(self, tensor_id, destination, ranges):
        tensor = self.tensors[tensor_id]
        transfers = []
        for start, end in merge_ranges(ranges):
            if not 0 <= start < end <= tensor.channels:
                raise ValueError("consumer request outside tensor")
            for left, right in missing_ranges(start, end, self.available.get(tensor_id, {}).get(destination, ())):
                if tensor.source == "EXTERNAL":
                    self.receive(tensor_id, destination, left, right)
                    continue
                covered = 0
                for owner in self.primary[tensor_id]:
                    lower, upper = max(left, owner.start), min(right, owner.end)
                    if lower < upper:
                        if owner.chiplet != destination:
                            transfers.append((owner.chiplet, destination, lower, upper,
                                              (upper-lower)*tensor.elements_per_channel*tensor.bytes_per_element))
                        self.receive(tensor_id, destination, lower, upper)
                        covered += upper-lower
                if covered != right-left:
                    raise ValueError("missing primary data owner")
        return tuple(transfers)

    def retire(self, tensor_id):
        self.primary.pop(tensor_id, None)
        self.available.pop(tensor_id, None)

    def resident_bytes(self):
        sizes = [0]*self.total_chiplets
        for tensor_id, nodes in self.available.items():
            tensor = self.tensors[tensor_id]
            for node, ranges in nodes.items():
                sizes[node] += sum(end-start for start,end in ranges)*tensor.elements_per_channel*tensor.bytes_per_element
        return tuple(sizes)

    def snapshot(self):
        return tuple((name, node, start, end) for name, nodes in sorted(self.available.items())
                     for node, ranges in sorted(nodes.items()) for start,end in ranges)
