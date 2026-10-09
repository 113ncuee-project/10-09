"""Semantic block extraction for the simplified RapidChiplet IR.

The extraction boundary follows the model's native semantics instead of an
arbitrary MAC-count split:

* a block is a useful tensor-to-tensor unit;
* fused operators such as Conv+BN+Activation stay together;
* residual/concat branches are closed before the block boundary;
* the DSE agent only sees the resulting block graph.

The current project receives model metadata through ``models.json`` rather
than importing PyTorch or ONNX at runtime.  ``load_models`` is therefore the
small parser/front-end, while this module owns the semantic-block validation
and the compatibility fallback for old stage-only configs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import ModelSpec, refine_model_blocks


@dataclass(frozen=True)
class BlockExtractionReport:
    """Small audit record stored by callers alongside an extracted model."""

    model: str
    source: str
    block_count: int
    semantic_units: tuple[str, ...]
    fused_operator_blocks: int
    branch_closed_blocks: int

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "source": self.source,
            "block_count": self.block_count,
            "semantic_units": list(self.semantic_units),
            "fused_operator_blocks": self.fused_operator_blocks,
            "branch_closed_blocks": self.branch_closed_blocks,
        }


def validate_block_graph(model: ModelSpec) -> None:
    """Validate IDs, dependencies and semantic boundary invariants."""

    blocks = model.blocks
    names = {block.name for block in blocks}
    if len(names) != len(blocks):
        raise ValueError(f"Model {model.name} has duplicate block names")

    for block in blocks:
        if block.macs_g < 0.0 or block.output_mb < 0.0:
            raise ValueError(f"Block {block.name} has a negative compute/output size")
        if not block.operators:
            raise ValueError(f"Block {block.name} must contain at least one fused operator")
        if model.block_source == "semantic" and not block.branch_closed:
            raise ValueError(
                f"Semantic block {block.name} must close residual/concat branches "
                "before the DSE boundary"
            )
        if block.name in block.depends_on:
            raise ValueError(f"Block {block.name} cannot depend on itself")
        unknown = [dependency for dependency in block.depends_on if dependency not in names]
        if unknown:
            raise ValueError(
                f"Block {block.name} depends on unknown blocks: {', '.join(unknown)}"
            )

    # A DFS catches cycles even when a future config uses a non-sequential DAG.
    dependencies = {block.name: block.depends_on for block in blocks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError(f"Model {model.name} block graph contains a cycle at {name}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in dependencies[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for block in blocks:
        visit(block.name)


def extract_semantic_blocks(
    model: ModelSpec,
    *,
    legacy_split_factor: int = 1,
) -> tuple[ModelSpec, BlockExtractionReport]:
    """Return a model whose blocks are the units exposed to DSE.

    An explicit ``blocks`` section is authoritative.  This is how ResNet-50
    is represented: Stem, 16 Bottlenecks, and Head.  The optional split factor
    is used only for old ``stages`` configs and is labelled as inferred; it is
    never applied to an explicit semantic graph.
    """

    if legacy_split_factor < 1:
        raise ValueError("legacy_split_factor must be >= 1")

    extracted = (
        model
        if model.block_source == "semantic"
        else refine_model_blocks(model, legacy_split_factor)
    )
    validate_block_graph(extracted)
    report = BlockExtractionReport(
        model=extracted.name,
        source=extracted.block_source,
        block_count=len(extracted.blocks),
        semantic_units=tuple(block.block_type for block in extracted.blocks),
        fused_operator_blocks=sum(len(block.operators) > 1 for block in extracted.blocks),
        branch_closed_blocks=sum(block.branch_closed for block in extracted.blocks),
    )
    return extracted, report
