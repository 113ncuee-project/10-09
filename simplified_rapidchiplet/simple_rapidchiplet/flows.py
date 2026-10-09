"""Canonical network flows; tensor ownership generation is a later phase.

Legacy metadata uses rounded MiB, so bytes can currently be fractional.
Future operator-derived flows can use integer byte counts without changing
this representation. Local and EXTERNAL transfers are not mesh flows.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class Flow:
    tensor_id: str
    src: int
    dst: int
    bytes: float
    kind: str
    source_group_index: int
    destination_group_index: int
    producer_block: str = ""
    consumer_block: str = ""
    legacy_kind: str = ""
    producer_operator: str = ""
    consumer_operator: str = ""
    channel_start: int = 0
    channel_end: int = 0
    event_id: str = ""

    def __post_init__(self) -> None:
        if not self.tensor_id:
            raise ValueError("flow requires a tensor_id")
        if self.src < 0 or self.dst < 0 or self.src == self.dst:
            raise ValueError("mesh flows require distinct non-negative endpoints")
        if not math.isfinite(self.bytes) or self.bytes <= 0:
            raise ValueError("flow bytes must be finite and positive")

    @property
    def event_key(self) -> tuple[str, int, int, str]:
        return self.kind, self.source_group_index, self.destination_group_index, self.event_id

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def traffic_bits_by_pair(flows: Iterable[Flow]) -> dict[tuple[int, int], float]:
    """Lossless byte-to-bit conversion, retaining separate tensor flows."""
    traffic: dict[tuple[int, int], float] = {}
    for flow in flows:
        pair = flow.src, flow.dst
        traffic[pair] = traffic.get(pair, 0.0) + flow.bytes * 8
    return traffic
