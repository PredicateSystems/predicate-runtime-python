"""
Unit tests for PlannerExecutorAgent pruning integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from predicate.agents.automation_task import AutomationTask, TaskCategory
from predicate.agents.planner_executor_agent import (
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    SnapshotEscalationConfig,
)
from predicate.models import BBox, Element, Snapshot, VisualCues
from predicate.pruning import (
    PruningTaskCategory,
    prune_snapshot_for_task,
    prune_with_recovery,
)


def make_element(
    *,
    id: int,
    role: str,
    text: str,
    importance: int,
    href: str | None = None,
    in_dominant_group: bool | None = None,
) -> Element:
    return Element(
        id=id,
        role=role,
        text=text,
        importance=importance,
        href=href,
        in_dominant_group=in_dominant_group,
        bbox=BBox(x=0, y=0, width=100, height=20),
        visual_cues=VisualCues(is_primary=False, is_clickable=role in {"button", "link"}),
    )


def make_snapshot(elements: list[Element]) -> Snapshot:
    return Snapshot(status="success", url="https://shop.example.com", elements=elements)


class TestPlannerExecutorPruningIntegration:
    """Tests for pruning-aware context formatting and escalation."""

    def test_format_context_uses_pruned_snapshot_when_task_category_is_known(self) -> None:
        """Pruned context should include category and prioritize relevant elements."""
        agent = PlannerExecutorAgent(
            planner=MagicMock(),
            executor=MagicMock(),
        )
        agent._current_task = AutomationTask(
            task_id="shopping-1",
            starting_url="https://shop.example.com",
            task="add the product to cart",
            category=TaskCategory.TRANSACTION,
            domain_hints=("ecommerce",),
        )
        snap = make_snapshot(
            [
                make_element(id=1, role="button", text="Add to Cart", importance=950, in_dominant_group=True),
                make_element(id=2, role="link", text="Privacy Policy", importance=50, href="/privacy"),
            ]
        )

        result = agent._format_context(snap, "add the product to cart")

        # Should include category and the Add to Cart button
        assert "Category: shopping" in result
        assert "[1] button" in result
        assert "Add to Cart" in result

    def test_format_context_excludes_blocked_elements_at_strict_level(self) -> None:
        """At relaxation level 0, common footer links should be blocked."""
        snap = make_snapshot(
            [
                make_element(id=1, role="button", text="Add to Cart", importance=950, in_dominant_group=True),
                make_element(id=2, role="link", text="Privacy Policy", importance=50, href="/privacy"),
                make_element(id=3, role="link", text="Terms of Service", importance=50, href="/terms"),
            ]
        )

        # Test strict pruning (level 0)
        ctx = prune_snapshot_for_task(
            snap,
            goal="add the product to cart",
            category=PruningTaskCategory.SHOPPING,
            relaxation_level=0,
        )

        # Privacy Policy and Terms should be blocked at level 0
        node_texts = [n.text for n in ctx.nodes]
        assert "Add to Cart" in node_texts
        assert "Privacy Policy" not in node_texts
        assert "Terms of Service" not in node_texts


class TestPruningRecovery:
    """Tests for over-pruning recovery via relaxation levels."""

    def test_relaxation_increases_node_count(self) -> None:
        """Higher relaxation levels should allow more elements."""
        snap = make_snapshot(
            [
                make_element(id=1, role="button", text="Add to Cart", importance=950, in_dominant_group=True),
                make_element(id=2, role="link", text="Privacy Policy", importance=50, href="/privacy"),
                make_element(id=3, role="link", text="Terms of Service", importance=50, href="/terms"),
                make_element(id=4, role="button", text="Close", importance=100),
            ]
        )

        ctx_strict = prune_snapshot_for_task(
            snap,
            goal="add the product to cart",
            category=PruningTaskCategory.SHOPPING,
            relaxation_level=0,
        )

        ctx_relaxed = prune_snapshot_for_task(
            snap,
            goal="add the product to cart",
            category=PruningTaskCategory.SHOPPING,
            relaxation_level=2,
        )

        # Relaxed should have more elements
        assert len(ctx_relaxed.nodes) >= len(ctx_strict.nodes)

    def test_prune_with_recovery_auto_relaxes(self) -> None:
        """prune_with_recovery should auto-relax if initial pruning is sparse."""
        # Create a snapshot with very few matching elements at level 0
        snap = make_snapshot(
            [
                make_element(id=1, role="heading", text="Welcome", importance=100),
                make_element(id=2, role="paragraph", text="Some text", importance=50),
                make_element(id=3, role="button", text="OK", importance=200),
            ]
        )

        ctx = prune_with_recovery(
            snap,
            goal="find something",
            category=PruningTaskCategory.SHOPPING,
            max_relaxation=3,
            verbose=False,
        )

        # Should have relaxed to include the button
        assert ctx.relaxation_level > 0 or len(ctx.nodes) >= 1

    def test_pruned_context_includes_metadata(self) -> None:
        """PrunedSnapshotContext should include element count metadata."""
        snap = make_snapshot(
            [
                make_element(id=1, role="button", text="Add to Cart", importance=950, in_dominant_group=True),
                make_element(id=2, role="link", text="Product A", importance=800, href="/a", in_dominant_group=True),
                make_element(id=3, role="link", text="Product B", importance=700, href="/b", in_dominant_group=True),
            ]
        )

        ctx = prune_snapshot_for_task(
            snap,
            goal="add to cart",
            category=PruningTaskCategory.SHOPPING,
        )

        assert ctx.raw_element_count == 3
        assert ctx.pruned_element_count == len(ctx.nodes)
        assert ctx.relaxation_level == 0


class TestCategorySpecificExecutorHints:
    """Tests for category-specific hints in executor prompts."""

    def test_shopping_category_hints(self) -> None:
        """Shopping category should provide relevant hints."""
        from predicate.agents.planner_executor_agent import _get_category_executor_hints

        hints = _get_category_executor_hints("shopping")
        assert "Add to Cart" in hints
        assert "Buy Now" in hints

    def test_form_filling_category_hints(self) -> None:
        """Form filling category should provide relevant hints."""
        from predicate.agents.planner_executor_agent import _get_category_executor_hints

        hints = _get_category_executor_hints("form_filling")
        assert "input" in hints.lower()
        assert "submit" in hints.lower()

    def test_search_category_hints(self) -> None:
        """Search category should provide relevant hints."""
        from predicate.agents.planner_executor_agent import _get_category_executor_hints

        hints = _get_category_executor_hints("search")
        assert "search" in hints.lower()
        assert "result" in hints.lower()

    def test_unknown_category_returns_empty(self) -> None:
        """Unknown categories should return empty hints."""
        from predicate.agents.planner_executor_agent import _get_category_executor_hints

        hints = _get_category_executor_hints("unknown_category")
        assert hints == ""

    def test_none_category_returns_empty(self) -> None:
        """None category should return empty hints."""
        from predicate.agents.planner_executor_agent import _get_category_executor_hints

        hints = _get_category_executor_hints(None)
        assert hints == ""
