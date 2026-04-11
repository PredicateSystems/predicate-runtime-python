"""
Category-specific snapshot pruning helpers.

Supports category-aware pruning with automatic over-pruning recovery:
- Rule-based category classification (no LLM needed)
- Deterministic allow/block policies per category
- Relaxation levels for recovery when pruning is too aggressive
"""

from .classifier import classify_task_category
from .policies import get_pruning_policy, PruningPolicy
from .pruner import prune_snapshot_for_task, prune_with_recovery
from .serializer import serialize_pruned_snapshot
from .types import (
    CategoryDetectionResult,
    PrunedSnapshotContext,
    PruningTaskCategory,
    SkeletonDomNode,
)

__all__ = [
    "CategoryDetectionResult",
    "PrunedSnapshotContext",
    "PruningPolicy",
    "PruningTaskCategory",
    "SkeletonDomNode",
    "classify_task_category",
    "get_pruning_policy",
    "prune_snapshot_for_task",
    "prune_with_recovery",
    "serialize_pruned_snapshot",
]
