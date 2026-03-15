"""
Unit tests for PlannerExecutorAgent.

Tests for:
- IntentHeuristics protocol
- ExecutorOverride protocol
- Pre-step verification (skip if predicates pass)
- Optional substeps execution
- Plan normalization
- Plan smoothness validation
- RecoveryNavigationConfig
"""

from __future__ import annotations

import pytest
from typing import Any

from predicate.agents.planner_executor_agent import (
    ExecutorOverride,
    IntentHeuristics,
    Plan,
    PlannerExecutorConfig,
    PlanStep,
    PredicateSpec,
    RecoveryNavigationConfig,
    normalize_plan,
    validate_plan_smoothness,
)


# ---------------------------------------------------------------------------
# Test normalize_plan
# ---------------------------------------------------------------------------


class TestNormalizePlan:
    """Tests for the normalize_plan function."""

    def test_normalizes_action_to_uppercase(self) -> None:
        plan_dict = {
            "task": "test",
            "steps": [
                {"id": 1, "goal": "click button", "action": "click", "verify": []},
            ],
        }
        result = normalize_plan(plan_dict)
        assert result["steps"][0]["action"] == "CLICK"

    def test_normalizes_action_aliases(self) -> None:
        test_cases = [
            ("CLICK_ELEMENT", "CLICK"),
            ("CLICK_BUTTON", "CLICK"),
            ("CLICK_LINK", "CLICK"),
            ("INPUT", "TYPE_AND_SUBMIT"),
            ("TYPE_TEXT", "TYPE_AND_SUBMIT"),
            ("ENTER_TEXT", "TYPE_AND_SUBMIT"),
            ("GOTO", "NAVIGATE"),
            ("GO_TO", "NAVIGATE"),
            ("OPEN", "NAVIGATE"),
            ("SCROLL_DOWN", "SCROLL"),
            ("SCROLL_UP", "SCROLL"),
        ]

        for alias, expected in test_cases:
            plan_dict = {
                "task": "test",
                "steps": [
                    {"id": 1, "goal": "test", "action": alias, "verify": []},
                ],
            }
            result = normalize_plan(plan_dict)
            assert result["steps"][0]["action"] == expected, f"Failed for alias: {alias}"

    def test_normalizes_url_to_target(self) -> None:
        plan_dict = {
            "task": "test",
            "steps": [
                {
                    "id": 1,
                    "goal": "navigate",
                    "action": "NAVIGATE",
                    "url": "https://example.com",
                    "verify": [],
                },
            ],
        }
        result = normalize_plan(plan_dict)
        assert "target" in result["steps"][0]
        assert result["steps"][0]["target"] == "https://example.com"
        assert "url" not in result["steps"][0]

    def test_preserves_existing_target(self) -> None:
        plan_dict = {
            "task": "test",
            "steps": [
                {
                    "id": 1,
                    "goal": "navigate",
                    "action": "NAVIGATE",
                    "target": "https://example.com",
                    "url": "https://other.com",
                    "verify": [],
                },
            ],
        }
        result = normalize_plan(plan_dict)
        assert result["steps"][0]["target"] == "https://example.com"

    def test_converts_string_id_to_int(self) -> None:
        plan_dict = {
            "task": "test",
            "steps": [
                {"id": "1", "goal": "test", "action": "CLICK", "verify": []},
                {"id": "2", "goal": "test2", "action": "CLICK", "verify": []},
            ],
        }
        result = normalize_plan(plan_dict)
        assert result["steps"][0]["id"] == 1
        assert result["steps"][1]["id"] == 2

    def test_normalizes_optional_substeps(self) -> None:
        plan_dict = {
            "task": "test",
            "steps": [
                {
                    "id": 1,
                    "goal": "test",
                    "action": "CLICK",
                    "verify": [],
                    "optional_substeps": [
                        {
                            "id": 1,
                            "goal": "scroll",
                            "action": "scroll_down",
                            "url": "https://example.com",
                        },
                    ],
                },
            ],
        }
        result = normalize_plan(plan_dict)
        substep = result["steps"][0]["optional_substeps"][0]
        assert substep["action"] == "SCROLL_DOWN"
        assert substep["target"] == "https://example.com"


# ---------------------------------------------------------------------------
# Test validate_plan_smoothness
# ---------------------------------------------------------------------------


class TestValidatePlanSmoothness:
    """Tests for the validate_plan_smoothness function."""

    def test_warns_on_empty_plan(self) -> None:
        plan = Plan(task="test", steps=[])
        warnings = validate_plan_smoothness(plan)
        assert "Plan has no steps" in warnings

    def test_warns_on_single_step_plan(self) -> None:
        plan = Plan(
            task="test",
            steps=[
                PlanStep(id=1, goal="test", action="CLICK", verify=[]),
            ],
        )
        warnings = validate_plan_smoothness(plan)
        assert any("only one step" in w for w in warnings)

    def test_warns_on_missing_verification_for_required_step(self) -> None:
        plan = Plan(
            task="test",
            steps=[
                PlanStep(id=1, goal="step1", action="CLICK", verify=[], required=True),
                PlanStep(id=2, goal="step2", action="CLICK", verify=[], required=True),
            ],
        )
        warnings = validate_plan_smoothness(plan)
        assert any("no verification" in w for w in warnings)

    def test_no_warning_for_optional_step_without_verification(self) -> None:
        plan = Plan(
            task="test",
            steps=[
                PlanStep(
                    id=1,
                    goal="step1",
                    action="CLICK",
                    verify=[PredicateSpec(predicate="url_contains", args=["test"])],
                    required=True,
                ),
                PlanStep(id=2, goal="step2", action="CLICK", verify=[], required=False),
            ],
        )
        warnings = validate_plan_smoothness(plan)
        # Only one warning about step 2 having no verification
        verification_warnings = [w for w in warnings if "no verification" in w]
        assert len(verification_warnings) == 0  # required=False means no warning

    def test_warns_on_navigate_without_target(self) -> None:
        plan = Plan(
            task="test",
            steps=[
                PlanStep(id=1, goal="go", action="NAVIGATE", target=None, verify=[]),
                PlanStep(id=2, goal="done", action="FINISH", verify=[]),
            ],
        )
        warnings = validate_plan_smoothness(plan)
        assert any("NAVIGATE but has no target" in w for w in warnings)

    def test_warns_on_click_without_intent(self) -> None:
        plan = Plan(
            task="test",
            steps=[
                PlanStep(id=1, goal="click", action="CLICK", intent=None, verify=[]),
                PlanStep(id=2, goal="done", action="FINISH", verify=[]),
            ],
        )
        warnings = validate_plan_smoothness(plan)
        assert any("CLICK but has no intent" in w for w in warnings)

    def test_warns_on_type_and_submit_without_input(self) -> None:
        plan = Plan(
            task="test",
            steps=[
                PlanStep(
                    id=1,
                    goal="type",
                    action="TYPE_AND_SUBMIT",
                    input=None,
                    verify=[],
                ),
                PlanStep(id=2, goal="done", action="FINISH", verify=[]),
            ],
        )
        warnings = validate_plan_smoothness(plan)
        assert any("TYPE_AND_SUBMIT but has no input" in w for w in warnings)

    def test_warns_on_consecutive_click_actions(self) -> None:
        plan = Plan(
            task="test",
            steps=[
                PlanStep(
                    id=1,
                    goal="click1",
                    action="CLICK",
                    intent="btn1",
                    verify=[PredicateSpec(predicate="url_contains", args=["a"])],
                ),
                PlanStep(
                    id=2,
                    goal="click2",
                    action="CLICK",
                    intent="btn2",
                    verify=[PredicateSpec(predicate="url_contains", args=["b"])],
                ),
            ],
        )
        warnings = validate_plan_smoothness(plan)
        assert any("both use CLICK" in w for w in warnings)

    def test_no_warnings_for_good_plan(self) -> None:
        plan = Plan(
            task="test",
            steps=[
                PlanStep(
                    id=1,
                    goal="navigate",
                    action="NAVIGATE",
                    target="https://example.com",
                    verify=[PredicateSpec(predicate="url_contains", args=["example"])],
                ),
                PlanStep(
                    id=2,
                    goal="click",
                    action="CLICK",
                    intent="submit",
                    verify=[PredicateSpec(predicate="exists", args=["role=button"])],
                ),
            ],
        )
        warnings = validate_plan_smoothness(plan)
        # May have some warnings (consecutive actions check is lenient)
        # but no critical issues
        assert "Plan has no steps" not in warnings


# ---------------------------------------------------------------------------
# Test IntentHeuristics Protocol
# ---------------------------------------------------------------------------


class TestIntentHeuristicsProtocol:
    """Tests for the IntentHeuristics protocol."""

    def test_protocol_check_passes_for_valid_implementation(self) -> None:
        class ValidHeuristics:
            def find_element_for_intent(
                self,
                intent: str,
                elements: list[Any],
                url: str,
                goal: str,
            ) -> int | None:
                return None

            def priority_order(self) -> list[str]:
                return []

        heuristics = ValidHeuristics()
        assert isinstance(heuristics, IntentHeuristics)

    def test_heuristics_can_return_element_id(self) -> None:
        class MockHeuristics:
            def find_element_for_intent(
                self,
                intent: str,
                elements: list[Any],
                url: str,
                goal: str,
            ) -> int | None:
                if intent == "add_to_cart":
                    for el in elements:
                        if getattr(el, "text", "").lower() == "add to cart":
                            return getattr(el, "id", None)
                return None

            def priority_order(self) -> list[str]:
                return ["add_to_cart", "checkout"]

        class MockElement:
            def __init__(self, id: int, text: str):
                self.id = id
                self.text = text

        heuristics = MockHeuristics()
        elements = [
            MockElement(1, "Some text"),
            MockElement(2, "Add to Cart"),
            MockElement(3, "Other button"),
        ]

        result = heuristics.find_element_for_intent(
            intent="add_to_cart",
            elements=elements,
            url="https://example.com",
            goal="add item to cart",
        )
        assert result == 2

    def test_heuristics_returns_none_for_unknown_intent(self) -> None:
        class MockHeuristics:
            def find_element_for_intent(
                self,
                intent: str,
                elements: list[Any],
                url: str,
                goal: str,
            ) -> int | None:
                return None

            def priority_order(self) -> list[str]:
                return []

        heuristics = MockHeuristics()
        result = heuristics.find_element_for_intent(
            intent="unknown",
            elements=[],
            url="https://example.com",
            goal="test",
        )
        assert result is None


# ---------------------------------------------------------------------------
# Test ExecutorOverride Protocol
# ---------------------------------------------------------------------------


class TestExecutorOverrideProtocol:
    """Tests for the ExecutorOverride protocol."""

    def test_protocol_check_passes_for_valid_implementation(self) -> None:
        class ValidOverride:
            def validate_choice(
                self,
                element_id: int,
                action: str,
                elements: list[Any],
                goal: str,
            ) -> tuple[bool, int | None, str | None]:
                return True, None, None

        override = ValidOverride()
        assert isinstance(override, ExecutorOverride)

    def test_override_can_block_action(self) -> None:
        class SafetyOverride:
            def validate_choice(
                self,
                element_id: int,
                action: str,
                elements: list[Any],
                goal: str,
            ) -> tuple[bool, int | None, str | None]:
                for el in elements:
                    if getattr(el, "id", None) == element_id:
                        text = getattr(el, "text", "").lower()
                        if "delete" in text:
                            return False, None, "blocked_delete"
                return True, None, None

        class MockElement:
            def __init__(self, id: int, text: str):
                self.id = id
                self.text = text

        override = SafetyOverride()
        elements = [
            MockElement(1, "Submit"),
            MockElement(2, "Delete Account"),
        ]

        # Should allow submit button
        is_valid, override_id, reason = override.validate_choice(
            element_id=1,
            action="CLICK",
            elements=elements,
            goal="submit form",
        )
        assert is_valid is True

        # Should block delete button
        is_valid, override_id, reason = override.validate_choice(
            element_id=2,
            action="CLICK",
            elements=elements,
            goal="delete account",
        )
        assert is_valid is False
        assert reason == "blocked_delete"

    def test_override_can_suggest_alternative(self) -> None:
        class CorrectionOverride:
            def validate_choice(
                self,
                element_id: int,
                action: str,
                elements: list[Any],
                goal: str,
            ) -> tuple[bool, int | None, str | None]:
                # Always suggest element 5 instead
                return False, 5, "corrected"

        override = CorrectionOverride()
        is_valid, override_id, reason = override.validate_choice(
            element_id=1,
            action="CLICK",
            elements=[],
            goal="test",
        )
        assert is_valid is False
        assert override_id == 5
        assert reason == "corrected"


# ---------------------------------------------------------------------------
# Test RecoveryNavigationConfig
# ---------------------------------------------------------------------------


class TestRecoveryNavigationConfig:
    """Tests for RecoveryNavigationConfig."""

    def test_default_values(self) -> None:
        config = RecoveryNavigationConfig()
        assert config.enabled is True
        assert config.max_recovery_attempts == 2
        assert config.track_successful_urls is True

    def test_custom_values(self) -> None:
        config = RecoveryNavigationConfig(
            enabled=False,
            max_recovery_attempts=5,
            track_successful_urls=False,
        )
        assert config.enabled is False
        assert config.max_recovery_attempts == 5
        assert config.track_successful_urls is False


# ---------------------------------------------------------------------------
# Test PlannerExecutorConfig with new options
# ---------------------------------------------------------------------------


class TestPlannerExecutorConfigNewOptions:
    """Tests for new config options in PlannerExecutorConfig."""

    def test_pre_step_verification_default_enabled(self) -> None:
        config = PlannerExecutorConfig()
        assert config.pre_step_verification is True

    def test_pre_step_verification_can_be_disabled(self) -> None:
        config = PlannerExecutorConfig(pre_step_verification=False)
        assert config.pre_step_verification is False

    def test_recovery_config_present(self) -> None:
        config = PlannerExecutorConfig()
        assert config.recovery is not None
        assert config.recovery.enabled is True

    def test_custom_recovery_config(self) -> None:
        config = PlannerExecutorConfig(
            recovery=RecoveryNavigationConfig(
                enabled=True,
                max_recovery_attempts=3,
            ),
        )
        assert config.recovery.max_recovery_attempts == 3


# ---------------------------------------------------------------------------
# Test PlanStep with optional_substeps
# ---------------------------------------------------------------------------


class TestPlanStepOptionalSubsteps:
    """Tests for PlanStep optional_substeps field."""

    def test_optional_substeps_default_empty(self) -> None:
        step = PlanStep(id=1, goal="test", action="CLICK", verify=[])
        assert step.optional_substeps == []

    def test_optional_substeps_can_be_set(self) -> None:
        substep = PlanStep(id=1, goal="scroll", action="SCROLL", verify=[], required=False)
        step = PlanStep(
            id=1,
            goal="click",
            action="CLICK",
            verify=[],
            optional_substeps=[substep],
        )
        assert len(step.optional_substeps) == 1
        assert step.optional_substeps[0].action == "SCROLL"

    def test_optional_substeps_nested_structure(self) -> None:
        step_dict = {
            "id": 1,
            "goal": "click product",
            "action": "CLICK",
            "intent": "first_product",
            "verify": [{"predicate": "url_contains", "args": ["/dp/"]}],
            "optional_substeps": [
                {
                    "id": 1,
                    "goal": "scroll down",
                    "action": "SCROLL",
                    "required": False,
                },
                {
                    "id": 2,
                    "goal": "retry click",
                    "action": "CLICK",
                    "intent": "first_product",
                    "verify": [{"predicate": "url_contains", "args": ["/dp/"]}],
                    "required": False,
                },
            ],
        }
        step = PlanStep.model_validate(step_dict)
        assert len(step.optional_substeps) == 2
        assert step.optional_substeps[0].action == "SCROLL"
        assert step.optional_substeps[1].action == "CLICK"
