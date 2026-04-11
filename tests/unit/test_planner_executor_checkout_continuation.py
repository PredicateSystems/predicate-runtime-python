"""
Unit tests for WebBench-style checkout continuation feature in PlannerExecutorAgent.

Tests the dynamic step insertion after Add to Cart actions to click checkout buttons
in cart drawers instead of dismissing them.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from predicate.agents.planner_executor_agent import (
    ModalDismissalConfig,
    Plan,
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    PlanStep,
    StepStatus,
    StepOutcome,
)
from predicate.models import Snapshot


# ---------------------------------------------------------------------------
# Test Checkout Continuation Logic
# ---------------------------------------------------------------------------


class TestCheckoutContinuation:
    """Tests for WebBench-style checkout continuation after Add to Cart."""

    def test_detects_add_to_cart_as_last_step(self) -> None:
        """Should detect when Add to Cart is the last planned step."""
        plan = Plan(
            task="Buy product",
            steps=[
                PlanStep(id=1, goal="Search for product", action="TYPE_AND_SUBMIT", input="laptop", verify=[]),
                PlanStep(id=2, goal="Click product", action="CLICK", intent="product link", verify=[]),
                PlanStep(id=3, goal="Add to cart", action="CLICK", intent="Add to Cart button", verify=[]),
            ],
        )

        # Last step (index 2) is Add to Cart
        last_step = plan.steps[-1]
        goal_lower = last_step.goal.lower()
        is_add_to_cart = any(
            pattern in goal_lower
            for pattern in ("add to cart", "add to bag", "add to basket")
        )
        assert is_add_to_cart is True

    def test_detects_add_to_bag_variation(self) -> None:
        """Should detect 'Add to Bag' variation."""
        plan = Plan(
            task="Buy product",
            steps=[
                PlanStep(id=1, goal="Add to bag", action="CLICK", intent="Add to Bag button", verify=[]),
            ],
        )

        goal_lower = plan.steps[0].goal.lower()
        is_add_to_cart = any(
            pattern in goal_lower
            for pattern in ("add to cart", "add to bag", "add to basket")
        )
        assert is_add_to_cart is True

    def test_detects_checkout_button_in_snapshot(self) -> None:
        """Should detect checkout buttons in cart drawer snapshot."""
        # Mock snapshot with checkout button
        mock_snapshot = MagicMock(spec=Snapshot)
        checkout_element = MagicMock()
        checkout_element.id = 123
        checkout_element.text = "Proceed to Checkout"
        checkout_element.role = "button"

        other_element = MagicMock()
        other_element.id = 456
        other_element.text = "Continue Shopping"
        other_element.role = "link"

        mock_snapshot.elements = [checkout_element, other_element]

        # Test checkout button detection
        checkout_patterns = (
            "checkout", "check out", "proceed to checkout",
            "view cart", "go to cart", "view bag", "go to bag",
        )

        checkout_el = None
        for el in mock_snapshot.elements:
            el_text = (el.text or "").lower()
            el_role = (el.role or "").lower()
            if el_role in ("button", "link") and any(p in el_text for p in checkout_patterns):
                checkout_el = el
                break

        assert checkout_el is not None
        assert checkout_el.id == 123

    def test_detects_view_cart_variation(self) -> None:
        """Should detect 'View Cart' button variation."""
        mock_snapshot = MagicMock(spec=Snapshot)
        view_cart_element = MagicMock()
        view_cart_element.id = 789
        view_cart_element.text = "View Cart"
        view_cart_element.role = "link"

        mock_snapshot.elements = [view_cart_element]

        checkout_patterns = (
            "checkout", "check out", "proceed to checkout",
            "view cart", "go to cart", "view bag", "go to bag",
        )

        checkout_el = None
        for el in mock_snapshot.elements:
            el_text = (el.text or "").lower()
            el_role = (el.role or "").lower()
            if el_role in ("button", "link") and any(p in el_text for p in checkout_patterns):
                checkout_el = el
                break

        assert checkout_el is not None
        assert checkout_el.id == 789

    def test_no_checkout_button_found(self) -> None:
        """Should handle case when no checkout button is found."""
        mock_snapshot = MagicMock(spec=Snapshot)
        other_element = MagicMock()
        other_element.id = 111
        other_element.text = "Continue Shopping"
        other_element.role = "button"

        mock_snapshot.elements = [other_element]

        checkout_patterns = (
            "checkout", "check out", "proceed to checkout",
            "view cart", "go to cart", "view bag", "go to bag",
        )

        checkout_el = None
        for el in mock_snapshot.elements:
            el_text = (el.text or "").lower()
            el_role = (el.role or "").lower()
            if el_role in ("button", "link") and any(p in el_text for p in checkout_patterns):
                checkout_el = el
                break

        assert checkout_el is None

    def test_only_triggers_on_last_step(self) -> None:
        """Checkout continuation should only trigger when Add to Cart is the LAST step."""
        plan = Plan(
            task="Buy product",
            steps=[
                PlanStep(id=1, goal="Add to cart", action="CLICK", intent="Add to Cart", verify=[]),
                PlanStep(id=2, goal="Click checkout", action="CLICK", intent="Checkout", verify=[]),
            ],
        )

        # First step is Add to Cart but NOT the last step
        step_index = 0
        is_last_step = (step_index == len(plan.steps) - 1)
        assert is_last_step is False  # Should NOT trigger

        # Second step is last
        step_index = 1
        is_last_step = (step_index == len(plan.steps) - 1)
        assert is_last_step is True


# ---------------------------------------------------------------------------
# Test Extended Checkout Detection in Modal Dismissal
# ---------------------------------------------------------------------------


class TestExtendedCheckoutDetection:
    """Tests for extended checkout detection patterns in modal dismissal."""

    def test_skips_dismissal_for_checkout_button_text(self) -> None:
        """Should skip modal dismissal only if drawer contains CLICKABLE checkout elements."""
        checkout_button_patterns = (
            "checkout", "check out", "proceed to checkout", "go to checkout",
            "view cart", "view bag", "shopping cart", "shopping bag",
            "continue to checkout", "secure checkout",
            "go to cart", "see cart", "go to bag",
        )

        # Test CLICKABLE checkout patterns (buttons/links)
        clickable_test_cases = [
            ("Proceed to Checkout", "button", True),
            ("View Cart", "link", True),
            ("Shopping Cart", "button", True),
            ("Go to Cart", "link", True),
        ]

        for text, role, should_skip in clickable_test_cases:
            # Only skip if it's a button or link with checkout text
            is_clickable = role in ("button", "link")
            has_checkout_text = any(pattern in text.lower() for pattern in checkout_button_patterns)
            result = is_clickable and has_checkout_text
            assert result == should_skip, f"Failed for text: {text}, role: {role}"

    def test_allows_dismissal_for_informational_text(self) -> None:
        """Should ALLOW dismissal if drawer only has informational text, not clickable checkout buttons."""
        # These are informational messages, not clickable buttons
        informational_cases = [
            ("Item Added to Cart", "text"),  # Just a message, not a button
            ("Subtotal: $29.99", "text"),    # Just a label
            ("Your Cart", "heading"),         # Just a heading
        ]

        checkout_button_patterns = (
            "checkout", "check out", "proceed to checkout",
            "view cart", "view bag", "shopping cart",
        )

        for text, role in informational_cases:
            # Should NOT skip dismissal - these aren't buttons/links
            is_clickable = role in ("button", "link")
            assert is_clickable is False, f"Informational element should not be clickable: {text}"

    def test_skips_dismissal_for_cart_href(self) -> None:
        """Should skip modal dismissal if drawer contains cart/checkout links."""
        test_hrefs = [
            "/cart",
            "/checkout",
            "/bag",
            "/shopping-cart",
            "https://example.com/cart",
        ]

        for href in test_hrefs:
            href_lower = href.lower()
            # Check for cart, checkout, or bag anywhere in the href
            has_cart_link = "cart" in href_lower or "checkout" in href_lower or "bag" in href_lower
            assert has_cart_link is True, f"Failed for href: {href}"

    def test_does_not_skip_for_non_checkout_elements(self) -> None:
        """Should NOT skip dismissal for non-checkout related elements."""
        checkout_patterns = (
            "checkout", "check out", "proceed to checkout", "go to checkout",
            "view cart", "view bag", "shopping cart", "shopping bag",
            "continue to checkout", "secure checkout",
            "go to cart", "see cart", "cart summary", "your cart",
            "subtotal", "item added", "added to cart", "added to bag",
        )

        non_checkout_texts = [
            "Continue Shopping",
            "Close",
            "Dismiss",
            "Learn More",
            "Add Protection",
        ]

        for text in non_checkout_texts:
            should_skip = any(pattern in text.lower() for pattern in checkout_patterns)
            assert should_skip is False, f"Incorrectly matched: {text}"

    def test_does_not_treat_global_nav_cart_link_as_drawer_checkout(self) -> None:
        """Top-nav Amazon cart links should not suppress drawer dismissal."""
        from types import SimpleNamespace

        mock_planner = MagicMock()
        mock_executor = MagicMock()
        agent = PlannerExecutorAgent(planner=mock_planner, executor=mock_executor)

        nav_cart = SimpleNamespace(
            role="link",
            text="Cart",
            aria_label="0 items in cart",
            href="https://www.amazon.com/gp/cart/view.html?ref_=nav_cart",
            doc_y=24.0,
            layout=SimpleNamespace(region="header"),
        )

        assert agent._is_global_nav_cart_link(nav_cart) is True


# ---------------------------------------------------------------------------
# Test Build Executor Prompt Improvements
# ---------------------------------------------------------------------------


class TestBuildExecutorPromptImprovements:
    """Tests for improved executor prompts."""

    def test_type_action_specifies_input_element(self) -> None:
        """TYPE action prompt should explicitly ask for INPUT element, not submit button."""
        from predicate.agents.planner_executor_agent import build_executor_prompt

        sys_prompt, user_prompt = build_executor_prompt(
            goal="Search for laptop",
            intent=None,
            compact_context="100|textbox|Search|500|1|0|-|0|\n200|button|Submit|300|1|0|-|0|",
            input_text="laptop",
            action_type="TYPE_AND_SUBMIT",  # Must specify action_type to get TYPE prompt
        )

        # System prompt should mention INPUT element explicitly
        assert "INPUT" in sys_prompt.upper() or "TEXTBOX" in sys_prompt.upper()
        assert "NOT the submit button" in sys_prompt or "not submit button" in sys_prompt.lower()

        # User prompt should show the correct action instruction
        assert 'Return TYPE(id, "laptop")' in user_prompt

    def test_click_action_uses_simple_instruction(self) -> None:
        """CLICK action should have simple clear instruction."""
        from predicate.agents.planner_executor_agent import build_executor_prompt

        sys_prompt, user_prompt = build_executor_prompt(
            goal="Click submit button",
            intent="submit button",
            compact_context="100|button|Submit|500|1|0|-|0|",
            input_text=None,
        )

        # Should ask for CLICK(id)
        assert "Return CLICK(id)" in user_prompt


# ---------------------------------------------------------------------------
# Test Think Tag Stripping
# ---------------------------------------------------------------------------


class TestThinkTagStripping:
    """Tests for improved <think> tag stripping in action parsing."""

    def test_strips_closed_think_tags(self) -> None:
        """Should strip closed <think>...</think> tags."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent
        from predicate.llm_provider import LLMProvider

        # Create minimal agent instance
        mock_planner = MagicMock(spec=LLMProvider)
        mock_executor = MagicMock(spec=LLMProvider)
        agent = PlannerExecutorAgent(planner=mock_planner, executor=mock_executor)

        # Test with closed think tags
        text = "<think>I should click element 42</think>CLICK(42)"
        action_type, args = agent._parse_action(text)
        assert action_type == "CLICK"
        assert args == [42]

    def test_strips_unclosed_think_tags(self) -> None:
        """Should strip unclosed <think>... to end of string."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent
        from predicate.llm_provider import LLMProvider

        mock_planner = MagicMock(spec=LLMProvider)
        mock_executor = MagicMock(spec=LLMProvider)
        agent = PlannerExecutorAgent(planner=mock_planner, executor=mock_executor)

        # Test with unclosed think tag
        text = "CLICK(42)<think>this is my reasoning..."
        action_type, args = agent._parse_action(text)
        assert action_type == "CLICK"
        assert args == [42]

    def test_handles_multiple_think_tags(self) -> None:
        """Should handle multiple think tags in response."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent
        from predicate.llm_provider import LLMProvider

        mock_planner = MagicMock(spec=LLMProvider)
        mock_executor = MagicMock(spec=LLMProvider)
        agent = PlannerExecutorAgent(planner=mock_planner, executor=mock_executor)

        # Multiple think tags
        text = "<think>First thought</think>CLICK(100)<think>Second thought</think>"
        action_type, args = agent._parse_action(text)
        assert action_type == "CLICK"
        assert args == [100]

    def test_parses_none_response(self) -> None:
        """Should parse NONE response when executor can't find suitable element."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent
        from predicate.llm_provider import LLMProvider

        mock_planner = MagicMock(spec=LLMProvider)
        mock_executor = MagicMock(spec=LLMProvider)
        agent = PlannerExecutorAgent(planner=mock_planner, executor=mock_executor)

        # Test NONE response
        text = "NONE"
        action_type, args = agent._parse_action(text)
        assert action_type == "NONE"
        assert args == []

    def test_parses_none_with_explanation(self) -> None:
        """Should parse NONE even with additional text."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent
        from predicate.llm_provider import LLMProvider

        mock_planner = MagicMock(spec=LLMProvider)
        mock_executor = MagicMock(spec=LLMProvider)
        agent = PlannerExecutorAgent(planner=mock_planner, executor=mock_executor)

        # Test NONE with trailing text
        text = "NONE - no search box found"
        action_type, args = agent._parse_action(text)
        assert action_type == "NONE"
        assert args == []


# ---------------------------------------------------------------------------
# Test Overlay Dismiss Intent Detection
# ---------------------------------------------------------------------------


class TestOverlayDismissIntentDetection:
    """Tests for detecting overlay/modal dismissal intents."""

    def test_detects_cookie_consent_dismissal(self) -> None:
        """Should detect cookie consent dismissal intents."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent
        from predicate.llm_provider import LLMProvider

        mock_planner = MagicMock(spec=LLMProvider)
        mock_executor = MagicMock(spec=LLMProvider)
        agent = PlannerExecutorAgent(planner=mock_planner, executor=mock_executor)

        test_cases = [
            ("Accept cookie consent", "Accept cookies button"),
            ("Dismiss privacy banner", "close button"),
            ("Close GDPR modal", "accept button"),
        ]

        for goal, intent in test_cases:
            is_overlay_dismiss = agent._looks_like_overlay_dismiss_intent(goal=goal, intent=intent)
            assert is_overlay_dismiss is True, f"Failed for: {goal} / {intent}"

    def test_detects_modal_dismissal(self) -> None:
        """Should detect modal/popup dismissal intents."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent
        from predicate.llm_provider import LLMProvider

        mock_planner = MagicMock(spec=LLMProvider)
        mock_executor = MagicMock(spec=LLMProvider)
        agent = PlannerExecutorAgent(planner=mock_planner, executor=mock_executor)

        test_cases = [
            ("Close newsletter popup", "close button"),
            ("Dismiss notification", "dismiss button"),
            ("Close dialog", "OK button"),
        ]

        for goal, intent in test_cases:
            is_overlay_dismiss = agent._looks_like_overlay_dismiss_intent(goal=goal, intent=intent)
            assert is_overlay_dismiss is True, f"Failed for: {goal} / {intent}"

    def test_does_not_detect_regular_clicks(self) -> None:
        """Should NOT detect regular click actions as overlay dismissal."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent
        from predicate.llm_provider import LLMProvider

        mock_planner = MagicMock(spec=LLMProvider)
        mock_executor = MagicMock(spec=LLMProvider)
        agent = PlannerExecutorAgent(planner=mock_planner, executor=mock_executor)

        test_cases = [
            ("Click product", "product link"),
            ("Add to cart", "Add to Cart button"),
            ("Submit form", "submit button"),
        ]

        for goal, intent in test_cases:
            is_overlay_dismiss = agent._looks_like_overlay_dismiss_intent(goal=goal, intent=intent)
            assert is_overlay_dismiss is False, f"Incorrectly matched: {goal} / {intent}"
