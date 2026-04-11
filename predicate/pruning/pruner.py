"""
Deterministic snapshot pruning entry points.

Supports over-pruning recovery via relaxation levels:
- If pruning leaves too few elements, increase relaxation level and re-prune
- Relaxation progressively loosens allow predicates and increases node budgets
"""

from __future__ import annotations

from typing import Any

from .policies import get_pruning_policy
from .serializer import serialize_pruned_snapshot
from .types import PrunedSnapshotContext, PruningTaskCategory, SkeletonDomNode

# Minimum elements threshold - if pruning leaves fewer, consider relaxation
MIN_PRUNED_ELEMENTS = 5


def _node_score(el: Any, goal: str) -> float:
    """Score an element for ranking within the pruned set."""
    text = str(getattr(el, "text", "") or "").lower()
    goal_lower = (goal or "").lower()

    score = float(getattr(el, "importance", 0) or 0)
    if bool(getattr(el, "in_viewport", True)):
        score += 25.0
    if bool(getattr(el, "in_dominant_group", False)):
        score += 20.0
    visual_cues = getattr(el, "visual_cues", None)
    if visual_cues is not None and bool(getattr(visual_cues, "is_clickable", False)):
        score += 15.0
    if text and goal_lower and any(token in text for token in goal_lower.split()):
        score += 10.0
    if "$" in text:
        score += 8.0
    return score


def _semantic_tags(el: Any) -> tuple[str, ...]:
    """Derive semantic tags from element properties."""
    text = str(getattr(el, "text", "") or "").lower()
    role = str(getattr(el, "role", "") or "").lower()
    tags: list[str] = []
    if "$" in text:
        tags.append("price")
    if "add to cart" in text or "add to bag" in text:
        tags.append("add_to_cart")
    if "checkout" in text or "cart" in text:
        tags.append("checkout")
    if role in {"searchbox", "textbox", "combobox"} and "search" in text:
        tags.append("search_input")
    if role == "link" and len(text.strip()) >= 3:
        tags.append("product_title")
    return tuple(tags)


def prune_snapshot_for_task(
    snapshot: Any,
    *,
    goal: str,
    category: PruningTaskCategory,
    relaxation_level: int = 0,
) -> PrunedSnapshotContext:
    """
    Prune a snapshot deterministically for the given category.

    Args:
        snapshot: The snapshot to prune
        goal: The task goal for context-aware scoring
        category: The detected task category
        relaxation_level: 0=strict, 1=relaxed, 2=loose, 3+=fallback

    Returns:
        PrunedSnapshotContext with the pruned nodes and metadata
    """
    all_elements = getattr(snapshot, "elements", []) or []
    raw_count = len(all_elements)

    policy = get_pruning_policy(category, relaxation_level)
    kept: list[Any] = []

    for el in all_elements:
        if policy.block(el, goal):
            continue
        if policy.allow(el, goal):
            kept.append(el)

    kept.sort(key=lambda el: _node_score(el, goal), reverse=True)
    selected = kept[: policy.max_nodes]

    nodes = tuple(
        SkeletonDomNode(
            id=int(getattr(el, "id")),
            role=str(getattr(el, "role", "") or ""),
            text=getattr(el, "text", None),
            href=getattr(el, "href", None),
            region=getattr(getattr(el, "layout", None), "region", None) or "unknown",
            semantic_tags=_semantic_tags(el),
        )
        for el in selected
    )

    ctx = PrunedSnapshotContext(
        category=category,
        url=str(getattr(snapshot, "url", "") or ""),
        nodes=nodes,
        prompt_block="",
        relaxation_level=relaxation_level,
        raw_element_count=raw_count,
        pruned_element_count=len(nodes),
    )

    return PrunedSnapshotContext(
        category=ctx.category,
        url=ctx.url,
        nodes=ctx.nodes,
        prompt_block=serialize_pruned_snapshot(ctx),
        relaxation_level=ctx.relaxation_level,
        raw_element_count=ctx.raw_element_count,
        pruned_element_count=ctx.pruned_element_count,
    )


def prune_with_recovery(
    snapshot: Any,
    *,
    goal: str,
    category: PruningTaskCategory,
    max_relaxation: int = 3,
    verbose: bool = False,
) -> PrunedSnapshotContext:
    """
    Prune with automatic recovery via relaxation if over-pruning is detected.

    This function progressively relaxes the pruning policy if the initial
    pruning leaves too few elements.

    Args:
        snapshot: The snapshot to prune
        goal: The task goal
        category: The detected task category
        max_relaxation: Maximum relaxation level to try (default 3)
        verbose: Print relaxation info

    Returns:
        PrunedSnapshotContext with the best pruning result
    """
    for level in range(max_relaxation + 1):
        ctx = prune_snapshot_for_task(
            snapshot,
            goal=goal,
            category=category,
            relaxation_level=level,
        )

        if verbose and level > 0:
            print(
                f"  [PRUNING] Relaxation level {level}: "
                f"{ctx.raw_element_count} -> {ctx.pruned_element_count} elements",
                flush=True,
            )

        # If we have enough elements, stop relaxing
        if not ctx.is_sparse:
            return ctx

        # If we're at max relaxation, return whatever we have
        if level == max_relaxation:
            if verbose:
                print(
                    f"  [PRUNING] Max relaxation reached, "
                    f"returning {ctx.pruned_element_count} elements",
                    flush=True,
                )
            return ctx

    # Shouldn't reach here, but return last result
    return ctx
