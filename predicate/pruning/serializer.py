"""
Serializer for pruned snapshot contexts.
"""

from __future__ import annotations

from .types import PrunedSnapshotContext


def serialize_pruned_snapshot(ctx: PrunedSnapshotContext) -> str:
    """Serialize a pruned snapshot context into a compact prompt block."""

    lines = [
        f"Category: {ctx.category.value}",
        f"URL: {ctx.url}",
        "Nodes:",
    ]
    for node in ctx.nodes:
        line = f'[{node.id}] {node.role} text="{node.text or ""}"'
        if node.semantic_tags:
            line += f" tags={','.join(node.semantic_tags)}"
        if node.region:
            line += f" region={node.region}"
        if node.href:
            line += f" href={node.href}"
        lines.append(line)
    return "\n".join(lines)
