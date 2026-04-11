"""
Unit tests for pruning category classification.
"""

from predicate.pruning import (
    CategoryDetectionResult,
    PruningTaskCategory,
    classify_task_category,
)


class TestCategoryClassifier:
    """Tests for rule-based pruning category detection."""

    def test_classify_task_category_returns_shopping_for_add_to_cart(self) -> None:
        """Shopping tasks should be classified without an LLM call."""
        result = classify_task_category(
            task_text="Search for a hat and add it to cart",
            current_url="https://www.amazon.com",
            domain_hints=("ecommerce",),
        )

        assert isinstance(result, CategoryDetectionResult)
        assert result.category == PruningTaskCategory.SHOPPING
        assert result.source == "rule"
        assert result.confidence >= 0.80

    def test_classify_task_category_returns_form_filling_for_submit_form(self) -> None:
        """Form tasks should be identified from task text."""
        result = classify_task_category(
            task_text="Fill out the contact form and submit it",
            current_url="https://example.com/contact",
            domain_hints=("forms",),
        )

        assert result.category == PruningTaskCategory.FORM_FILLING
        assert result.source == "rule"
        assert result.confidence >= 0.80

    def test_classify_task_category_returns_search_for_search_task(self) -> None:
        """Search tasks should map to the search pruning category."""
        result = classify_task_category(
            task_text="Search for flights from Seattle to New York",
            current_url="https://www.google.com/travel/flights",
            domain_hints=("travel", "flights"),
        )

        assert result.category == PruningTaskCategory.SEARCH
        assert result.source == "rule"
        assert result.confidence >= 0.80

    def test_classify_task_category_falls_back_to_generic_when_no_rule_matches(self) -> None:
        """Unknown tasks should fall back to GENERIC without raising."""
        result = classify_task_category(
            task_text="Observe the page and think about what to do next",
            current_url="https://example.com",
            domain_hints=(),
        )

        assert result.category == PruningTaskCategory.GENERIC
        assert result.source == "rule"
        assert result.confidence == 0.0
