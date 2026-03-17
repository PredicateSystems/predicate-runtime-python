"""
HeuristicSpec: Planner-generated hints for element selection.

This module provides models for dynamic heuristics composition. The planner
can generate HeuristicHint objects alongside execution plans, allowing
element selection without requiring an LLM call.

Key concepts:
- HeuristicHint: A single hint with intent pattern, text patterns, and role filters
- Hints are generated per-step by the planner
- ComposableHeuristics (separate module) uses these hints at runtime
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HeuristicHint(BaseModel):
    """
    Planner-generated hint for element selection.

    The planner can propose these hints alongside plan steps to guide
    element selection without requiring an LLM executor call.

    Attributes:
        intent_pattern: Pattern to match against step intent (e.g., "add_to_cart")
        text_patterns: Text patterns to search in element text (case-insensitive)
        role_filter: Allowed element roles (e.g., ["button", "link"])
        priority: Priority order (higher = try first)
        attribute_patterns: Optional attribute patterns to match (e.g., {"data-action": "add-to-cart"})

    Example:
        HeuristicHint(
            intent_pattern="add_to_cart",
            text_patterns=["add to cart", "add to bag", "buy now"],
            role_filter=["button"],
            priority=10,
        )

    Example in plan JSON:
        {
            "id": 3,
            "goal": "Add item to cart",
            "action": "CLICK",
            "intent": "add_to_cart",
            "heuristic_hints": [
                {
                    "intent_pattern": "add_to_cart",
                    "text_patterns": ["add to cart", "add to bag"],
                    "role_filter": ["button"],
                    "priority": 10
                }
            ],
            "verify": [{"predicate": "url_contains", "args": ["/cart"]}]
        }
    """

    intent_pattern: str = Field(
        ...,
        description="Intent pattern to match (e.g., 'add_to_cart', 'checkout', 'login')",
    )
    text_patterns: list[str] = Field(
        default_factory=list,
        description="Text patterns to search in elements (case-insensitive)",
    )
    role_filter: list[str] = Field(
        default_factory=list,
        description="Allowed element roles (e.g., ['button', 'link'])",
    )
    priority: int = Field(
        default=0,
        description="Priority order (higher = try first)",
    )
    attribute_patterns: dict[str, str] = Field(
        default_factory=dict,
        description="Attribute patterns to match (e.g., {'data-action': 'add-to-cart'})",
    )

    class Config:
        extra = "allow"

    def matches_intent(self, intent: str) -> bool:
        """
        Check if this hint matches the given intent.

        Args:
            intent: The intent string from the plan step

        Returns:
            True if the hint's intent_pattern is found in the intent
        """
        if not intent:
            return False
        return self.intent_pattern.lower() in intent.lower()

    def matches_element(self, element: object) -> bool:
        """
        Check if an element matches this hint's criteria.

        Args:
            element: Snapshot element with text, role, and attributes

        Returns:
            True if the element matches all criteria
        """
        # Check role filter
        role = getattr(element, "role", "") or ""
        if self.role_filter and role.lower() not in [r.lower() for r in self.role_filter]:
            return False

        # Check text patterns
        text = (getattr(element, "text", "") or "").lower()
        if self.text_patterns:
            if not any(pattern.lower() in text for pattern in self.text_patterns):
                return False

        # Check attribute patterns
        if self.attribute_patterns:
            attributes = getattr(element, "attributes", {}) or {}
            for attr_name, attr_pattern in self.attribute_patterns.items():
                attr_value = attributes.get(attr_name, "")
                if attr_pattern.lower() not in (attr_value or "").lower():
                    return False

        return True


# Common heuristic hints for well-known patterns
COMMON_HINTS = {
    "add_to_cart": HeuristicHint(
        intent_pattern="add_to_cart",
        text_patterns=["add to cart", "add to bag", "add to basket", "buy now"],
        role_filter=["button"],
        priority=10,
    ),
    "checkout": HeuristicHint(
        intent_pattern="checkout",
        text_patterns=["checkout", "proceed to checkout", "go to checkout"],
        role_filter=["button", "link"],
        priority=10,
    ),
    "login": HeuristicHint(
        intent_pattern="login",
        text_patterns=["log in", "login", "sign in", "signin"],
        role_filter=["button", "link"],
        priority=10,
    ),
    "submit": HeuristicHint(
        intent_pattern="submit",
        text_patterns=["submit", "send", "continue", "next", "confirm"],
        role_filter=["button"],
        priority=5,
    ),
    "search": HeuristicHint(
        intent_pattern="search",
        text_patterns=["search", "find", "go"],
        role_filter=["button", "textbox"],
        priority=5,
    ),
    "close": HeuristicHint(
        intent_pattern="close",
        text_patterns=["close", "dismiss", "x", "cancel"],
        role_filter=["button"],
        priority=3,
    ),
    "accept_cookies": HeuristicHint(
        intent_pattern="accept_cookies",
        text_patterns=["accept", "accept all", "allow", "agree", "ok", "got it"],
        role_filter=["button"],
        priority=8,
    ),
}


def get_common_hint(intent: str) -> HeuristicHint | None:
    """
    Get a common heuristic hint for well-known intents.

    Args:
        intent: Intent string (e.g., "add_to_cart", "checkout")

    Returns:
        HeuristicHint if a common hint exists, None otherwise
    """
    return COMMON_HINTS.get(intent.lower().replace(" ", "_").replace("-", "_"))
