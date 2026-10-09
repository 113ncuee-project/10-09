from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Stage:
    name: str
    macs_g: float
    output_mb: float
    depends_on: tuple[str, ...] = ()
    block_type: str = "legacy_stage"
    input_mb: float = 0.0
    weight_mb: float = 0.0
    operators: tuple[str, ...] = ()
    branch_closed: bool = True
    source_stage: str = ""


# ``Stage`` is kept as the public compatibility name used by the original
# simplified evaluator.  A stage is now treated as a schedulable block.
Block = Stage


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dataset: str
    input_shape: str
    macs_g: float
    params_m: float
    stages: tuple[Stage, ...]
    block_source: str = "legacy"

    @property
    def blocks(self) -> tuple[Block, ...]:
        """Return the fine-grained blocks used by the DSE search.

        The old code called these records ``stages``.  Keeping the property
        avoids breaking existing callers while making the new block-level
        terminology explicit.
        """

        return self.stages

    def block_graph(self) -> dict[str, Any]:
        """Return a serialisable block dependency graph.

        Models without explicit dependencies are interpreted as a pipeline;
        ``load_models`` fills those predecessor links during parsing.
        """

        nodes = [
            {
                "id": index,
                "name": block.name,
                "type": block.block_type,
                "macs_g": block.macs_g,
                "output_mb": block.output_mb,
                "memory": {
                    "input_mb": block.input_mb,
                    "output_mb": block.output_mb,
                    "weight_mb": block.weight_mb,
                },
                "tensor_io": {
                    "input_mb": block.input_mb,
                    "output_mb": block.output_mb,
                },
                "operators": list(block.operators),
                "branch_closed": block.branch_closed,
                "source_stage": block.source_stage,
            }
            for index, block in enumerate(self.blocks)
        ]
        edges = [
            {
                "source": dependency,
                "target": block.name,
                "traffic_bytes": next(
                    candidate.output_mb * 1024 * 1024
                    for candidate in self.blocks
                    if candidate.name == dependency
                ),
            }
            for block in self.blocks
            for dependency in block.depends_on
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "kind": "semantic_block_dag" if self.block_source == "semantic" else "block_dag",
            "block_source": self.block_source,
        }

    def ops_per_inference(self, op_per_mac: float) -> float:
        return self.macs_g * 1e9 * op_per_mac


def load_models(path: str | Path) -> dict[str, ModelSpec]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    models: dict[str, ModelSpec] = {}
    for item in raw["models"]:
        has_explicit_blocks = "blocks" in item
        raw_blocks = item.get("blocks", item.get("stages", ()))
        if not raw_blocks:
            raise ValueError(f"Model {item.get('name', '<unknown>')} has no blocks")

        names = [str(block["name"]) for block in raw_blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"Model {item.get('name', '<unknown>')} has duplicate block names")

        stages_list: list[Stage] = []
        for index, block in enumerate(raw_blocks):
            raw_dependencies = block.get("depends_on")
            if raw_dependencies is None:
                dependencies = (names[index - 1],) if index else ()
            else:
                dependencies = tuple(str(value) for value in raw_dependencies)
            unknown = [value for value in dependencies if value not in names]
            if unknown:
                raise ValueError(
                    f"Model {item.get('name', '<unknown>')} block {names[index]} "
                    f"depends on unknown blocks: {', '.join(unknown)}"
                )
            stages_list.append(
                Stage(
                    name=names[index],
                    macs_g=float(block["macs_g"]),
                    output_mb=float(block["output_mb"]),
                    depends_on=dependencies,
                    block_type=str(
                        block.get("type", block.get("semantic_type", "semantic_block" if has_explicit_blocks else "legacy_stage"))
                    ),
                    input_mb=float(
                        block.get(
                            "input_mb",
                            stages_list[-1].output_mb if stages_list else 0.0,
                        )
                    ),
                    weight_mb=float(block.get("weight_mb", 0.0)),
                    operators=tuple(str(value) for value in block.get("operators", (names[index],))),
                    branch_closed=bool(block.get("branch_closed", True)),
                    source_stage=str(block.get("source_stage", "")),
                )
            )
        stages = tuple(stages_list)
        model = ModelSpec(
            name=str(item["name"]),
            dataset=str(item.get("dataset", raw.get("dataset", "unknown"))),
            input_shape=str(item.get("input_shape", raw.get("input_shape", ""))),
            macs_g=float(item["macs_g"]),
            params_m=float(item.get("params_m", 0.0)),
            stages=stages,
            block_source="semantic" if has_explicit_blocks else "legacy",
        )
        models[model.name] = model
    return models


def select_models(models: dict[str, ModelSpec], names: Iterable[str] | None) -> list[ModelSpec]:
    if not names:
        return list(models.values())

    selected: list[ModelSpec] = []
    for name in names:
        if name not in models:
            known = ", ".join(sorted(models))
            raise KeyError(f"Unknown model '{name}'. Known models: {known}")
        selected.append(models[name])
    return selected


def refine_model_blocks(model: ModelSpec, split_factor: int = 1) -> ModelSpec:
    """Compatibility refinement for legacy stage-only model descriptions.

    Explicit semantic block graphs are never split here.  They already encode
    the model-native units (for example ResNet-50 Bottlenecks); splitting them
    would destroy the branch-closure and operator-fusion boundary.  The
    equal-MAC fallback remains only for old configs that still contain
    ``stages`` and is intentionally marked as inferred.
    """

    if split_factor < 1:
        raise ValueError("split_factor must be >= 1")
    if split_factor == 1 or model.block_source == "semantic":
        return model

    refined: list[Stage] = []
    previous_name: str | None = None
    for block in model.blocks:
        per_block_macs = block.macs_g / split_factor
        for part in range(split_factor):
            name = f"{block.name}.b{part + 1}"
            dependencies = (previous_name,) if previous_name is not None else ()
            refined.append(
                Stage(
                    name=name,
                    macs_g=per_block_macs,
                    output_mb=block.output_mb,
                    depends_on=dependencies,
                    block_type="inferred_micro_block",
                    input_mb=block.input_mb,
                    weight_mb=block.weight_mb / split_factor,
                    operators=(f"inferred part of {block.name}",),
                    branch_closed=block.branch_closed,
                    source_stage=block.name,
                )
            )
            previous_name = name

    return ModelSpec(
        name=model.name,
        dataset=model.dataset,
        input_shape=model.input_shape,
        macs_g=model.macs_g,
        params_m=model.params_m,
        stages=tuple(refined),
        block_source="legacy-inferred",
    )
