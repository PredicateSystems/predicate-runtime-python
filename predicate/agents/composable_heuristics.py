"""
ComposableHeuristics: Dynamic heuristics composition for element selection.

This module provides a heuristics implementation that composes from multiple sources:
1. Planner-provided HeuristicHints (per step, highest priority)
2. Static IntentHeuristics (user-injected at agent construction)
3. TaskCategory defaults (lowest priority)

This allows the planner to dynamically guide element selection without
requiring changes to user-provided heuristics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .automation_task import TaskCategory
from .heuristic_spec import COMMON_HINTS, HeuristicHint

if TYPE_CHECKING:
    pass


@runtime_checkable
class IntentHeuristics(Protocol):
    """
    Protocol for pluggable domain-specific element selection heuristics.

    This protocol is duplicated here to avoid circular imports with
    planner_executor_agent.py. The actual protocol is defined there.
    """

    def find_element_for_intent(
        self,
        intent: str,
        elements: list[Any],
        url: str,
        goal: str,
    ) -> int | None:
        """Find element ID for a given intent."""
        ...

    def priority_order(self) -> list[str]:
        """Return list of intent patterns in priority order."""
        ...


class ComposableHeuristics:
    """
    Heuristics implementation that composes from multiple sources.

    Priority order (highest to lowest):
    1. Planner-provided HeuristicHints for current step
    2. Common hints for well-known patterns (add_to_cart, checkout, etc.)
    3. Static IntentHeuristics (user-injected)
    4. TaskCategory defaults

    Example:
        heuristics = ComposableHeuristics(
            static_heuristics=my_ecommerce_heuristics,
            task_category=TaskCategory.TRANSACTION,
        )

        # Before executing each step, set the step's hints
        heuristics.set_step_hints(step.heuristic_hints)

        # Find element for intent
        element_id = heuristics.find_element_for_intent(
            intent="add_to_cart",
            elements=snapshot.elements,
            url=snapshot.url,
            goal="Add laptop to cart",
        )
    """

    def __init__(
        self,
        *,
        static_heuristics: IntentHeuristics | None = None,
        task_category: TaskCategory | None = None,
        use_common_hints: bool = True,
    ) -> None:
        """
        Initialize ComposableHeuristics.

        Args:
            static_heuristics: User-provided IntentHeuristics (optional)
            task_category: Task category for default heuristics (optional)
            use_common_hints: Whether to use COMMON_HINTS as fallback
        """
        self._static = static_heuristics
        self._category = task_category
        self._use_common_hints = use_common_hints
        self._current_hints: list[HeuristicHint] = []

    def set_step_hints(self, hints: list[HeuristicHint] | list[dict] | None) -> None:
        """
        Set hints for the current step.

        Called before each step execution with hints from the plan.

        Args:
            hints: List of HeuristicHint objects or dicts
        """
        if not hints:
            self._current_hints = []
            return

        # Convert dicts to HeuristicHint if needed
        parsed_hints: list[HeuristicHint] = []
        for h in hints:
            if isinstance(h, HeuristicHint):
                parsed_hints.append(h)
            elif isinstance(h, dict):
                try:
                    parsed_hints.append(HeuristicHint(**h))
                except Exception:
                    # Skip invalid hints
                    pass

        # Sort by priority (highest first)
        self._current_hints = sorted(parsed_hints, key=lambda h: -h.priority)

    def clear_step_hints(self) -> None:
        """Clear hints for the current step."""
        self._current_hints = []

    def find_element_for_intent(
        self,
        intent: str,
        elements: list[Any],
        url: str,
        goal: str,
    ) -> int | None:
        """
        Find element ID for a given intent using composed heuristics.

        Tries sources in priority order:
        1. Planner-provided hints for current step
        2. Common hints for well-known patterns
        3. Static heuristics (user-provided)
        4. TaskCategory defaults

        Args:
            intent: The intent hint from the plan step
            elements: List of snapshot elements
            url: Current page URL
            goal: Human-readable goal for context

        Returns:
            Element ID if found, None to fall back to LLM executor
        """
        if not intent or not elements:
            return None

        # 1. Try planner-provided hints first
        for hint in self._current_hints:
            if hint.matches_intent(intent):
                element_id = self._match_hint(hint, elements)
                if element_id is not None:
                    return element_id

        # 2. Try common hints for well-known patterns
        if self._use_common_hints:
            common_hint = self._get_common_hint_for_intent(intent)
            if common_hint:
                element_id = self._match_hint(common_hint, elements)
                if element_id is not None:
                    return element_id

        # 3. Try static heuristics
        if self._static is not None:
            try:
                element_id = self._static.find_element_for_intent(
                    intent, elements, url, goal
                )
                if element_id is not None:
                    return element_id
            except Exception:
                # Don't let static heuristics crash the flow
                pass

        # 4. Try category-based defaults
        return self._category_default_match(intent, elements)

    def _match_hint(self, hint: HeuristicHint, elements: list[Any]) -> int | None:
        """
        Match elements against a hint's criteria.

        Args:
            hint: HeuristicHint to match against
            elements: List of snapshot elements

        Returns:
            Element ID if match found, None otherwise
        """
        for el in elements:
            if hint.matches_element(el):
                element_id = getattr(el, "id", None)
                if element_id is not None:
                    return element_id
        return None

    def _get_common_hint_for_intent(self, intent: str) -> HeuristicHint | None:
        """Get common hint for well-known intents."""
        intent_normalized = intent.lower().replace(" ", "_").replace("-", "_")

        # Direct match
        if intent_normalized in COMMON_HINTS:
            return COMMON_HINTS[intent_normalized]

        # Partial match
        for key, hint in COMMON_HINTS.items():
            if key in intent_normalized or intent_normalized in key:
                return hint

        return None

    def _category_default_match(
        self, intent: str, elements: list[Any]
    ) -> int | None:
        """
        Apply category-based default matching.

        Uses TaskCategory to apply sensible defaults for common patterns.

        Args:
            intent: The intent string
            elements: List of snapshot elements

        Returns:
            Element ID if match found, None otherwise
        """
        if not self._category:
            return None

        intent_lower = intent.lower()

        if self._category == TaskCategory.TRANSACTION:
            # Transaction patterns: add to cart, checkout, buy, submit
            transaction_keywords = [
                "add to cart",
                "add to bag",
                "buy now",
                "checkout",
                "proceed",
                "submit",
                "confirm",
                "place order",
            ]
            for el in elements:
                text = (getattr(el, "text", "") or "").lower()
                role = getattr(el, "role", "") or ""
                if role in ("button", "link"):
                    for keyword in transaction_keywords:
                        if keyword in text:
                            return getattr(el, "id", None)

        elif self._category == TaskCategory.FORM_FILL:
            # Form patterns: submit, next, continue
            form_keywords = ["submit", "next", "continue", "save", "update"]
            for el in elements:
                text = (getattr(el, "text", "") or "").lower()
                role = getattr(el, "role", "") or ""
                if role == "button":
                    for keyword in form_keywords:
                        if keyword in text:
                            return getattr(el, "id", None)

        elif self._category == TaskCategory.SEARCH:
            # Search patterns: search button, go
            if "search" in intent_lower:
                for el in elements:
                    text = (getattr(el, "text", "") or "").lower()
                    role = getattr(el, "role", "") or ""
                    if role in ("button", "textbox") and "search" in text:
                        return getattr(el, "id", None)

        elif self._category == TaskCategory.NAVIGATION:
            # Navigation: links matching intent
            for el in elements:
                text = (getattr(el, "text", "") or "").lower()
                role = getattr(el, "role", "") or ""
                if role == "link" and intent_lower in text:
                    return getattr(el, "id", None)

        return None

    def priority_order(self) -> list[str]:
        """
        Return list of intent patterns in priority order.

        Combines patterns from all sources.

        Returns:
            List of intent pattern strings
        """
        patterns: list[str] = []

        # Add current step hints
        patterns.extend(h.intent_pattern for h in self._current_hints)

        # Add common hints
        if self._use_common_hints:
            patterns.extend(COMMON_HINTS.keys())

        # Add static heuristics patterns
        if self._static is not None:
            try:
                patterns.extend(self._static.priority_order())
            except Exception:
                pass

        # Deduplicate while preserving order
        seen = set()
        result = []
        for p in patterns:
            if p not in seen:
                seen.add(p)
                result.append(p)

        return result
