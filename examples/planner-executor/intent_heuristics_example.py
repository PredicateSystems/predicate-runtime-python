#!/usr/bin/env python3
"""
PlannerExecutorAgent example with IntentHeuristics and ExecutorOverride.

This example demonstrates the pluggable heuristics system:
- IntentHeuristics: Domain-specific element selection without LLM
- ExecutorOverride: Validate or override executor element choices
- Pre-step verification: Skip steps if predicates already pass

These features allow the SDK to remain generic while supporting specialized
behavior for different sites (e-commerce, forms, etc.).

Usage:
    export OPENAI_API_KEY="sk-..."
    python intent_heuristics_example.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from predicate import AsyncPredicateBrowser
from predicate.agent_runtime import AgentRuntime
from predicate.agents import (
    ExecutorOverride,
    IntentHeuristics,
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    RecoveryNavigationConfig,
)
from predicate.backends.playwright_backend import PlaywrightBackend
from predicate.llm_provider import OpenAIProvider


# ---------------------------------------------------------------------------
# Example IntentHeuristics Implementation
# ---------------------------------------------------------------------------


class EcommerceHeuristics:
    """
    Example IntentHeuristics for e-commerce sites.

    This heuristics class provides domain-specific element selection for
    common e-commerce actions like adding to cart, checkout, etc.

    When the agent receives a step with an intent like "add_to_cart",
    this class tries to find the matching element without calling the LLM.
    If no match is found, the agent falls back to the LLM executor.
    """

    def find_element_for_intent(
        self,
        intent: str,
        elements: list[Any],
        url: str,
        goal: str,
    ) -> int | None:
        """
        Find element ID for a given intent using domain-specific patterns.

        Args:
            intent: The intent hint from the plan step
            elements: List of snapshot elements
            url: Current page URL
            goal: Human-readable goal for context

        Returns:
            Element ID if match found, None to fall back to LLM
        """
        intent_lower = intent.lower()

        # Add to cart patterns
        if "add_to_cart" in intent_lower or "add to cart" in intent_lower:
            for el in elements:
                text = (getattr(el, "text", "") or "").lower()
                role = getattr(el, "role", "")
                if role == "button" and "add to cart" in text:
                    return getattr(el, "id", None)

        # Search box patterns
        if "search" in intent_lower and "box" in intent_lower:
            for el in elements:
                role = getattr(el, "role", "")
                if role in ("searchbox", "combobox", "textbox"):
                    text = (getattr(el, "text", "") or "").lower()
                    if "search" in text:
                        return getattr(el, "id", None)

        # Checkout patterns
        if "checkout" in intent_lower or "proceed" in intent_lower:
            for el in elements:
                text = (getattr(el, "text", "") or "").lower()
                role = getattr(el, "role", "")
                if role == "button" and ("checkout" in text or "proceed" in text):
                    return getattr(el, "id", None)

        # First product link (common in search results)
        if "first_product" in intent_lower:
            # Find links containing product indicators
            for el in elements:
                role = getattr(el, "role", "")
                if role == "link":
                    # Check if it looks like a product link (has /dp/ or product in href)
                    href = getattr(el, "href", "") or ""
                    if "/dp/" in href or "/product/" in href:
                        return getattr(el, "id", None)

        return None  # Fall back to LLM executor

    def priority_order(self) -> list[str]:
        """
        Return intent patterns in priority order.

        This helps the agent prioritize certain actions when multiple
        matching elements are found.
        """
        return [
            "checkout",
            "proceed_to_checkout",
            "add_to_cart",
            "search_box",
            "first_product",
            "quantity",
        ]


# ---------------------------------------------------------------------------
# Example ExecutorOverride Implementation
# ---------------------------------------------------------------------------


class SafetyOverride:
    """
    Example ExecutorOverride for safety validation.

    This override validates executor element choices before actions are
    executed, providing safety checks like:
    - Block clicks on delete/remove buttons
    - Block form submissions that might cause data loss
    - Block navigation to external sites
    """

    def __init__(self, blocked_patterns: list[str] | None = None):
        """
        Initialize SafetyOverride.

        Args:
            blocked_patterns: Text patterns to block (case-insensitive)
        """
        self.blocked_patterns = blocked_patterns or [
            "delete",
            "remove account",
            "cancel order",
            "unsubscribe",
        ]

    def validate_choice(
        self,
        element_id: int,
        action: str,
        elements: list[Any],
        goal: str,
    ) -> tuple[bool, int | None, str | None]:
        """
        Validate or override the executor's element choice.

        Args:
            element_id: The element ID chosen by the executor
            action: The action type (CLICK, TYPE, etc.)
            elements: List of snapshot elements
            goal: Human-readable goal

        Returns:
            Tuple of (is_valid, override_element_id, rejection_reason)
        """
        # Find the selected element
        selected_element = None
        for el in elements:
            if getattr(el, "id", None) == element_id:
                selected_element = el
                break

        if selected_element is None:
            # Element not found, allow (might be a valid ID)
            return True, None, None

        text = (getattr(selected_element, "text", "") or "").lower()

        # Check against blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.lower() in text:
                return False, None, f"blocked_pattern:{pattern}"

        # All checks passed
        return True, None, None


# ---------------------------------------------------------------------------
# Main Example
# ---------------------------------------------------------------------------


async def example_with_heuristics() -> None:
    """Run agent with IntentHeuristics."""
    print("\n--- Example 1: IntentHeuristics ---")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("  Skipping (no OPENAI_API_KEY)")
        return

    # Create heuristics instance
    heuristics = EcommerceHeuristics()

    # Create agent with heuristics
    agent = PlannerExecutorAgent(
        planner=OpenAIProvider(model="gpt-4o"),
        executor=OpenAIProvider(model="gpt-4o-mini"),
        config=PlannerExecutorConfig(),
        intent_heuristics=heuristics,  # Plug in domain-specific heuristics
    )

    print("  Agent created with EcommerceHeuristics")
    print("  Heuristics priority order:", heuristics.priority_order())


async def example_with_safety_override() -> None:
    """Run agent with ExecutorOverride for safety."""
    print("\n--- Example 2: ExecutorOverride (Safety) ---")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("  Skipping (no OPENAI_API_KEY)")
        return

    # Create safety override
    safety = SafetyOverride(
        blocked_patterns=[
            "delete",
            "remove",
            "cancel subscription",
        ],
    )

    # Create agent with safety override
    agent = PlannerExecutorAgent(
        planner=OpenAIProvider(model="gpt-4o"),
        executor=OpenAIProvider(model="gpt-4o-mini"),
        config=PlannerExecutorConfig(),
        executor_override=safety,  # Add safety validation
    )

    print("  Agent created with SafetyOverride")
    print("  Blocked patterns:", safety.blocked_patterns)


async def example_pre_step_verification() -> None:
    """Demonstrate pre-step verification skipping."""
    print("\n--- Example 3: Pre-step Verification ---")

    # Pre-step verification is enabled by default
    config = PlannerExecutorConfig(
        pre_step_verification=True,  # Default
    )
    print(f"  pre_step_verification: {config.pre_step_verification}")
    print("  When enabled, steps are skipped if their verification predicates already pass")
    print("  This saves time when the desired state is already achieved")

    # Example: If a step's goal is 'go to checkout' with verify=[url_contains('checkout')]
    # and the browser is already on a checkout page, the step will be skipped


async def example_recovery_navigation() -> None:
    """Demonstrate recovery navigation config."""
    print("\n--- Example 4: Recovery Navigation ---")

    config = PlannerExecutorConfig(
        recovery=RecoveryNavigationConfig(
            enabled=True,
            max_recovery_attempts=3,
            track_successful_urls=True,
        ),
    )

    print(f"  recovery.enabled: {config.recovery.enabled}")
    print(f"  recovery.max_recovery_attempts: {config.recovery.max_recovery_attempts}")
    print(f"  recovery.track_successful_urls: {config.recovery.track_successful_urls}")
    print("  When enabled, the agent tracks last_known_good_url for recovery")


async def example_combined() -> None:
    """Combined example with all new features."""
    print("\n--- Example 5: Combined Features ---")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("  Skipping (no OPENAI_API_KEY)")
        return

    # Create instances
    heuristics = EcommerceHeuristics()
    safety = SafetyOverride()

    # Create agent with all features
    agent = PlannerExecutorAgent(
        planner=OpenAIProvider(model="gpt-4o"),
        executor=OpenAIProvider(model="gpt-4o-mini"),
        config=PlannerExecutorConfig(
            pre_step_verification=True,
            recovery=RecoveryNavigationConfig(
                enabled=True,
                max_recovery_attempts=2,
            ),
        ),
        intent_heuristics=heuristics,
        executor_override=safety,
    )

    print("  Agent created with:")
    print("    - EcommerceHeuristics (domain-specific element selection)")
    print("    - SafetyOverride (action validation)")
    print("    - Pre-step verification (skip if already satisfied)")
    print("    - Recovery navigation (track good URLs)")


async def example_run_with_features() -> None:
    """Actually run the agent with new features."""
    print("\n--- Example 6: Run with Features ---")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("  Skipping (no OPENAI_API_KEY)")
        return

    predicate_api_key = os.getenv("PREDICATE_API_KEY")

    # Create agent with heuristics
    heuristics = EcommerceHeuristics()

    agent = PlannerExecutorAgent(
        planner=OpenAIProvider(model="gpt-4o"),
        executor=OpenAIProvider(model="gpt-4o-mini"),
        config=PlannerExecutorConfig(
            pre_step_verification=True,
        ),
        intent_heuristics=heuristics,
    )

    # Simple task on example.com
    task = "Navigate to example.com and verify the page loaded"

    async with AsyncPredicateBrowser(
        api_key=predicate_api_key,
        headless=True,
    ) as browser:
        page = await browser.new_page()
        await page.goto("https://example.com")

        backend = PlaywrightBackend(page)
        runtime = AgentRuntime(backend=backend)

        result = await agent.run(
            runtime=runtime,
            task=task,
        )

        print(f"  Success: {result.success}")
        print(f"  Steps: {result.steps_completed}/{result.steps_total}")

        # Check if any steps were skipped due to pre-step verification
        for outcome in result.step_outcomes:
            if outcome.action_taken and "SKIPPED" in outcome.action_taken:
                print(f"  Skipped step {outcome.step_id}: {outcome.goal}")


async def main() -> None:
    print("PlannerExecutorAgent - New Features Examples")
    print("=" * 50)

    await example_with_heuristics()
    await example_with_safety_override()
    await example_pre_step_verification()
    await example_recovery_navigation()
    await example_combined()
    await example_run_with_features()

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
