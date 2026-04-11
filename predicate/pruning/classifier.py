"""
Rule-based task classifier for pruning categories.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .types import CategoryDetectionResult, PruningTaskCategory


def _normalize_hints(domain_hints: Iterable[str] | None) -> set[str]:
    return {hint.strip().lower() for hint in domain_hints or () if str(hint).strip()}


def classify_task_category(
    *,
    task_text: str,
    current_url: str | None = None,
    domain_hints: Iterable[str] | None = None,
    task_category: Any | None = None,
) -> CategoryDetectionResult:
    """
    Classify a browser task into a pruning-oriented category using rules only.
    """

    text = (task_text or "").lower()
    url = (current_url or "").lower()
    hints = _normalize_hints(domain_hints)
    category_value = str(getattr(task_category, "value", task_category) or "").lower()

    if category_value == "form_fill":
        return CategoryDetectionResult(PruningTaskCategory.FORM_FILLING, 0.90)
    if category_value == "search":
        return CategoryDetectionResult(PruningTaskCategory.SEARCH, 0.90)
    if category_value == "extraction":
        return CategoryDetectionResult(PruningTaskCategory.EXTRACTION, 0.90)
    if category_value == "navigation":
        return CategoryDetectionResult(PruningTaskCategory.NAVIGATION, 0.90)
    if category_value == "verification":
        return CategoryDetectionResult(PruningTaskCategory.VERIFICATION, 0.90)

    if any(keyword in text for keyword in ("add to cart", "add it to cart", "add to bag", "buy now", "purchase")):
        return CategoryDetectionResult(PruningTaskCategory.SHOPPING, 0.95)
    if "checkout" in text:
        return CategoryDetectionResult(PruningTaskCategory.CHECKOUT, 0.95)
    if any(keyword in text for keyword in ("fill out", "submit form", "contact form", "enter email", "type into")):
        return CategoryDetectionResult(PruningTaskCategory.FORM_FILLING, 0.90)
    if "sign in" in text or "login" in text or "password" in text:
        return CategoryDetectionResult(PruningTaskCategory.AUTH, 0.90)
    if any(keyword in text for keyword in ("extract", "list the", "count the", "scrape")):
        return CategoryDetectionResult(PruningTaskCategory.EXTRACTION, 0.80)
    if any(keyword in text for keyword in ("search for", "look up")) or "find" in text:
        return CategoryDetectionResult(PruningTaskCategory.SEARCH, 0.85)

    if "ecommerce" in hints:
        if category_value == "transaction":
            return CategoryDetectionResult(PruningTaskCategory.SHOPPING, 0.85)
        return CategoryDetectionResult(PruningTaskCategory.SHOPPING, 0.65)

    if hints & {"forms", "contact", "signup"}:
        return CategoryDetectionResult(PruningTaskCategory.FORM_FILLING, 0.65)

    if hints & {"reference", "news", "recipes", "weather", "travel", "flights", "hotels", "social"}:
        return CategoryDetectionResult(PruningTaskCategory.SEARCH, 0.65)

    if "checkout" in url or "cart" in url or "bag" in url:
        return CategoryDetectionResult(PruningTaskCategory.CHECKOUT, 0.60)

    return CategoryDetectionResult(PruningTaskCategory.GENERIC, 0.0)
