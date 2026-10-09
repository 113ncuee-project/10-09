from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChipletConfig:
    pe_rows: int
    pe_cols: int
    frequency_hz: float
    op_per_mac: float
    ops_per_pe_per_cycle: float
    utilization: float
    width_mm: float
    spacing_mm: float
    internal_latency_cycles: float
    phy_latency_cycles: float
    # Local SRAM budget used by the architecture-level feasibility checker.
    # Existing configs that do not provide it use this conservative default.
    sram_mb: float = 16.0

    @property
    def peak_ops_per_second(self) -> float:
        return self.pe_rows * self.pe_cols * self.frequency_hz * self.ops_per_pe_per_cycle

    @property
    def peak_tops(self) -> float:
        return self.peak_ops_per_second / 1e12


@dataclass(frozen=True)
class NetworkConfig:
    # Official E2E uses backend chiplet/packaging bandwidths. These values
    # apply only to the explicitly labelled local proxy and scheduler.
    link_bandwidth_bits_per_cycle: float
    link_latency_base_cycles: float
    link_latency_cycles_per_mm: float


@dataclass(frozen=True)
class PowerConfig:
    chiplet_static_w: float
    chiplet_peak_dynamic_w: float
    phy_w: float
    link_static_w_per_mm: float
    link_dynamic_pj_per_bit: float
    # Existing legacy proxy calibration, independent of report FPS.
    proxy_reference_activity_hz: float


@dataclass(frozen=True)
class ReferenceModelConfig:
    name: str
    variant: str
    input_shape: tuple[int, ...]
    num_classes: int
    activation_precision: str
    weight_precision: str
    activation_bytes_per_element: int
    weight_bytes_per_element: int
    initial_input_source: str
    weights_source: str


@dataclass(frozen=True)
class RapidChipletConfig:
    root: str
    technology_file: str
    chiplet_file: str
    chiplet_name: str
    packaging_file: str
    technology: str
    chiplet_power_w: float
    fraction_power_bumps: float
    phy_fraction_bump_area: float


@dataclass(frozen=True)
class EvalConfig:
    max_chiplets: int
    chiplet: ChipletConfig
    network: NetworkConfig
    power: PowerConfig
    rapidchiplet: RapidChipletConfig
    reference_model: ReferenceModelConfig
    search: "SearchConfig" = None


@dataclass(frozen=True)
class SearchConfig:
    assignment_top_k: int = 4
    learning_rate: float = 0.35
    discount: float = 1.0
    exploration: float = 0.25
    exhaustive_max_evaluations: int = 200000

    def __post_init__(self):
        if self.assignment_top_k < 1 or self.exhaustive_max_evaluations < 1:
            raise ValueError("search bounds must be positive")
        if not 0 < self.learning_rate <= 1 or not 0 <= self.discount <= 1 or not 0 <= self.exploration <= 1:
            raise ValueError("invalid Q-learning parameters")


def load_config(path: str | Path) -> EvalConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    chiplet = ChipletConfig(**raw["chiplet"])
    network = NetworkConfig(**raw["network"])
    power = PowerConfig(**raw["power"])
    native_config = dict(raw["rapidchiplet"])
    native_root = Path(native_config["root"])
    if not native_root.is_absolute():
        native_config["root"] = str((Path(path).resolve().parent / native_root).resolve())
    rapidchiplet = RapidChipletConfig(**native_config)
    if set(raw["evaluation"]) != {"max_chiplets"}:
        raise ValueError("evaluation accepts only max_chiplets; migrate obsolete constraint settings")
    reference = dict(raw["reference_model"])
    reference["input_shape"] = tuple(reference["input_shape"])
    return EvalConfig(
        max_chiplets=int(raw["evaluation"]["max_chiplets"]),
        chiplet=chiplet,
        network=network,
        power=power,
        rapidchiplet=rapidchiplet,
        reference_model=ReferenceModelConfig(**reference),
        search=SearchConfig(**raw.get("search", {})),
    )
