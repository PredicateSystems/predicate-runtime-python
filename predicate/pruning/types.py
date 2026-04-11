"""
Core types for category-specific snapshot pruning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class PruningTaskCategory(str, Enum):
    """Task categories used by the pruning pipeline."""

    SHOPPING = "shopping"
    FORM_FILLING = "form_filling"
    SEARCH = "search"
    EXTRACTION = "extraction"
    NAVIGATION = "navigation"
    AUTH = "auth"
    CHECKOUT = "checkout"
    VERIFICATION = "verification"
    GENERIC = "generic"


@dataclass(frozen=True)
class CategoryDetectionResult:
    """Result of pruning category detection."""

    category: PruningTaskCategory
    confidence: float
    source: Literal["rule", "llm"] = "rule"


@dataclass(frozen=True)
class SkeletonDomNode:
    """Compact node retained after deterministic pruning."""

    id: int
    role: str
    text: str | None = None
    href: str | None = None
    region: str = "unknown"
    semantic_tags: tuple[str, ...] = ()
    ordinal: int | None = None


@dataclass(frozen=True)
class PrunedSnapshotContext:
    """Pruned snapshot plus compact prompt block."""

    category: PruningTaskCategory
    url: str
    nodes: tuple[SkeletonDomNode, ...]
    prompt_block: str
    relaxation_level: int = 0
    raw_element_count: int = 0
    pruned_element_count: int = 0

    @property
    def is_sparse(self) -> bool:
        """Check if pruning left too few elements (potential over-pruning)."""
        return len(self.nodes) < 5
