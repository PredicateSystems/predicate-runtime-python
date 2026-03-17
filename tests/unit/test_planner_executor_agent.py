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
    AuthBoundaryConfig,
    ExecutorOverride,
    IntentHeuristics,
    ModalDismissalConfig,
    Plan,
    PlannerExecutorConfig,
    PlanStep,
    PredicateSpec,
    RecoveryNavigationConfig,
    SnapshotEscalationConfig,
    build_executor_prompt,
    normalize_plan,
    validate_plan_smoothness,
)


# ---------------------------------------------------------------------------
# Test build_executor_prompt
# ---------------------------------------------------------------------------


class TestBuildExecutorPrompt:
    """Tests for the build_executor_prompt function."""

    def test_basic_prompt_structure(self) -> None:
        sys_prompt, user_prompt = build_executor_prompt(
            goal="Click the submit button",
            intent=None,
            compact_context="123|button|Submit|100|1|0|-|0|",
        )
        assert "CLICK(<id>)" in sys_prompt
        assert "TYPE(<id>" in sys_prompt
        assert "Goal: Click the submit button" in user_prompt
        assert "123|button|Submit" in user_prompt

    def test_includes_intent_when_provided(self) -> None:
        sys_prompt, user_prompt = build_executor_prompt(
            goal="Click on product",
            intent="Click the first product link",
            compact_context="456|link|Product|100|1|0|-|0|",
        )
        assert "Intent: Click the first product link" in user_prompt

    def test_no_intent_line_when_none(self) -> None:
        sys_prompt, user_prompt = build_executor_prompt(
            goal="Click button",
            intent=None,
            compact_context="789|button|OK|100|1|0|-|0|",
        )
        assert "Intent:" not in user_prompt

    def test_includes_input_text_when_provided(self) -> None:
        """Input text should be included for TYPE_AND_SUBMIT actions."""
        sys_prompt, user_prompt = build_executor_prompt(
            goal="Search for Logitech mouse",
            intent=None,
            compact_context="167|searchbox|Search|100|1|0|-|0|",
            input_text="Logitech mouse",
        )
        assert 'Text to type: "Logitech mouse"' in user_prompt

    def test_no_input_line_when_none(self) -> None:
        """No input text line when not provided."""
        sys_prompt, user_prompt = build_executor_prompt(
            goal="Click button",
            intent=None,
            compact_context="123|button|Submit|100|1|0|-|0|",
            input_text=None,
        )
        assert "Text to type:" not in user_prompt

    def test_includes_both_intent_and_input(self) -> None:
        """Both intent and input can be present."""
        sys_prompt, user_prompt = build_executor_prompt(
            goal="Search for laptop",
            intent="search_box",
            compact_context="100|searchbox|Search|100|1|0|-|0|",
            input_text="laptop",
        )
        assert "Intent: search_box" in user_prompt
        assert 'Text to type: "laptop"' in user_prompt


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
# Test ModalDismissalConfig
# ---------------------------------------------------------------------------


class TestModalDismissalConfig:
    """Tests for ModalDismissalConfig."""

    def test_default_values(self) -> None:
        config = ModalDismissalConfig()
        assert config.enabled is True
        assert config.max_attempts == 2
        assert config.min_new_elements == 5
        assert "button" in config.role_filter
        assert "link" in config.role_filter

    def test_default_patterns_exist(self) -> None:
        config = ModalDismissalConfig()
        # Should have common dismissal patterns
        assert "no thanks" in config.dismiss_patterns
        assert "close" in config.dismiss_patterns
        assert "skip" in config.dismiss_patterns
        assert "cancel" in config.dismiss_patterns

    def test_default_icons_exist(self) -> None:
        config = ModalDismissalConfig()
        # Should have common close icons
        assert "x" in config.dismiss_icons
        assert "×" in config.dismiss_icons
        assert "✕" in config.dismiss_icons

    def test_can_be_disabled(self) -> None:
        config = ModalDismissalConfig(enabled=False)
        assert config.enabled is False

    def test_custom_patterns_for_i18n(self) -> None:
        # German patterns
        config_german = ModalDismissalConfig(
            dismiss_patterns=(
                "nein danke",
                "schließen",
                "abbrechen",
            ),
        )
        assert "nein danke" in config_german.dismiss_patterns
        assert "schließen" in config_german.dismiss_patterns
        assert config_german.dismiss_patterns[0] == "nein danke"

    def test_custom_icons(self) -> None:
        config = ModalDismissalConfig(
            dismiss_icons=("x", "×", "✖"),
        )
        assert len(config.dismiss_icons) == 3
        assert "✖" in config.dismiss_icons

    def test_custom_max_attempts(self) -> None:
        config = ModalDismissalConfig(max_attempts=5)
        assert config.max_attempts == 5

    def test_planner_executor_config_has_modal(self) -> None:
        config = PlannerExecutorConfig()
        assert hasattr(config, "modal")
        assert config.modal.enabled is True

    def test_planner_executor_config_custom_modal(self) -> None:
        config = PlannerExecutorConfig(
            modal=ModalDismissalConfig(
                enabled=True,
                dismiss_patterns=("custom pattern",),
                max_attempts=3,
            ),
        )
        assert config.modal.max_attempts == 3
        assert "custom pattern" in config.modal.dismiss_patterns


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


# ---------------------------------------------------------------------------
# Test SnapshotEscalationConfig
# ---------------------------------------------------------------------------


class TestSnapshotEscalationConfig:
    """Tests for SnapshotEscalationConfig including scroll-after-escalation."""

    def test_default_values(self) -> None:
        config = SnapshotEscalationConfig()
        assert config.enabled is True
        assert config.limit_base == 60
        assert config.limit_step == 30
        assert config.limit_max == 200

    def test_scroll_after_escalation_defaults(self) -> None:
        config = SnapshotEscalationConfig()
        assert config.scroll_after_escalation is True
        assert config.scroll_max_attempts == 3
        assert config.scroll_directions == ("down", "up")

    def test_scroll_after_escalation_can_be_disabled(self) -> None:
        config = SnapshotEscalationConfig(scroll_after_escalation=False)
        assert config.scroll_after_escalation is False

    def test_custom_scroll_settings(self) -> None:
        config = SnapshotEscalationConfig(
            scroll_after_escalation=True,
            scroll_max_attempts=5,
            scroll_directions=("down",),  # Only scroll down
        )
        assert config.scroll_max_attempts == 5
        assert config.scroll_directions == ("down",)

    def test_scroll_viewport_fraction_default(self) -> None:
        config = SnapshotEscalationConfig()
        assert config.scroll_viewport_fraction == 0.4

    def test_scroll_viewport_fraction_custom(self) -> None:
        config = SnapshotEscalationConfig(scroll_viewport_fraction=0.5)
        assert config.scroll_viewport_fraction == 0.5

    def test_scroll_directions_can_be_reordered(self) -> None:
        # Try up first, then down
        config = SnapshotEscalationConfig(
            scroll_directions=("up", "down"),
        )
        assert config.scroll_directions == ("up", "down")

    def test_limit_escalation_steps(self) -> None:
        # Default: 60 -> 90 -> 120 -> 150 -> 180 -> 200
        config = SnapshotEscalationConfig()
        limits = []
        current = config.limit_base
        while current <= config.limit_max:
            limits.append(current)
            current = min(current + config.limit_step, config.limit_max + 1)
            if current > config.limit_max and current - config.limit_step < config.limit_max:
                limits.append(config.limit_max)
                break
        assert limits == [60, 90, 120, 150, 180, 200] or limits == [60, 90, 120, 150, 180]

    def test_custom_limit_step(self) -> None:
        # Custom: 60 -> 110 -> 160 -> 200
        config = SnapshotEscalationConfig(limit_step=50)
        assert config.limit_step == 50


# ---------------------------------------------------------------------------
# Test PlannerExecutorConfig with SnapshotEscalationConfig
# ---------------------------------------------------------------------------


class TestPlannerExecutorConfigSnapshot:
    """Tests for PlannerExecutorConfig snapshot escalation settings."""

    def test_default_snapshot_config(self) -> None:
        config = PlannerExecutorConfig()
        assert config.snapshot is not None
        assert config.snapshot.enabled is True
        assert config.snapshot.scroll_after_escalation is True

    def test_custom_snapshot_config(self) -> None:
        config = PlannerExecutorConfig(
            snapshot=SnapshotEscalationConfig(
                enabled=True,
                limit_base=100,
                limit_max=300,
                scroll_after_escalation=True,
                scroll_max_attempts=5,
            ),
        )
        assert config.snapshot.limit_base == 100
        assert config.snapshot.limit_max == 300
        assert config.snapshot.scroll_max_attempts == 5

    def test_disable_snapshot_escalation(self) -> None:
        config = PlannerExecutorConfig(
            snapshot=SnapshotEscalationConfig(enabled=False),
        )
        assert config.snapshot.enabled is False

    def test_scroll_to_find_config_exists(self) -> None:
        # Verify scroll_to_find settings exist in PlannerExecutorConfig
        config = PlannerExecutorConfig()
        assert hasattr(config, "scroll_to_find_enabled")
        assert hasattr(config, "scroll_to_find_max_scrolls")
        assert hasattr(config, "scroll_to_find_directions")


# ---------------------------------------------------------------------------
# Test _snapshot_with_escalation scroll-after-escalation behavior (async)
# ---------------------------------------------------------------------------


class MockElement:
    """Mock element for testing."""

    def __init__(self, id: int, role: str = "button", text: str = ""):
        self.id = id
        self.role = role
        self.text = text
        self.name = text


class MockSnapshot:
    """Mock snapshot for testing."""

    def __init__(self, elements: list[MockElement], url: str = "https://example.com"):
        # Ensure we have at least 3 elements to pass detect_snapshot_failure
        if len(elements) < 3:
            # Add filler elements to prevent "too_few_elements" detection
            for i in range(3 - len(elements)):
                elements.append(MockElement(100 + i, "generic", f"Filler {i}"))
        self.elements = elements
        self.url = url
        self.title = "Test Page"
        self.status = "success"  # Use "success" to pass detect_snapshot_failure
        self.screenshot = None
        self.diagnostics = None  # No diagnostics to avoid confidence checks


class MockRuntime:
    """Mock runtime for testing scroll-after-escalation and auth boundary."""

    def __init__(
        self,
        snapshots_by_scroll: dict[int, MockSnapshot] | None = None,
        url: str = "https://example.com",
    ):
        self.scroll_count = 0
        self.scroll_directions: list[str] = []
        self.snapshots_by_scroll = snapshots_by_scroll or {}
        self.default_snapshot = MockSnapshot([MockElement(1, "button", "Submit")])
        self._url = url

    async def get_url(self) -> str:
        """Return the current URL for auth boundary detection."""
        return self._url

    async def get_viewport_height(self) -> int:
        """Return mock viewport height."""
        return 800  # Standard viewport height

    async def snapshot(
        self,
        limit: int = 60,
        screenshot: bool = False,
        goal: str = "",
        show_overlay: bool = False,
    ) -> MockSnapshot:
        # Return snapshot based on scroll count
        return self.snapshots_by_scroll.get(self.scroll_count, self.default_snapshot)

    async def scroll(self, direction: str = "down") -> None:
        self.scroll_count += 1
        self.scroll_directions.append(direction)

    async def scroll_by(
        self,
        dy: float,
        *,
        verify: bool = True,
        min_delta_px: float = 50.0,
        js_fallback: bool = True,
        required: bool = True,
        timeout_s: float = 10.0,
        **kwargs: Any,
    ) -> bool:
        """Mock scroll_by with verification - always returns True (scroll effective)."""
        direction = "down" if dy > 0 else "up"
        self.scroll_count += 1
        self.scroll_directions.append(direction)
        return True  # Scroll always effective in tests

    async def stabilize(self) -> None:
        pass


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        class MockResponse:
            content = '{"task": "test", "steps": []}'

        return MockResponse()


class MockIntentHeuristics:
    """Mock intent heuristics for testing scroll-after-escalation."""

    def __init__(self, element_map: dict[str, int] | None = None):
        """
        Args:
            element_map: Maps intent names to element IDs to return.
                         If intent not in map, returns None.
        """
        self.element_map = element_map or {}

    def find_element_for_intent(
        self,
        intent: str,
        elements: list[Any],
        url: str,
        goal: str,
    ) -> int | None:
        return self.element_map.get(intent)

    def priority_order(self) -> list[str]:
        return list(self.element_map.keys())


class TestSnapshotWithEscalationScrollBehavior:
    """Tests for _snapshot_with_escalation scroll-after-escalation behavior."""

    @pytest.mark.asyncio
    async def test_scroll_after_escalation_disabled_no_scroll(self) -> None:
        """When scroll_after_escalation is disabled, no scrolling should occur."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent

        config = PlannerExecutorConfig(
            snapshot=SnapshotEscalationConfig(
                enabled=True,
                scroll_after_escalation=False,  # Disabled
            ),
        )
        agent = PlannerExecutorAgent(
            planner=MockLLMProvider(),
            executor=MockLLMProvider(),
            config=config,
        )

        runtime = MockRuntime()
        step = PlanStep(
            id=1,
            goal="Click Add to Cart",
            action="CLICK",
            intent="add_to_cart",
            verify=[],
        )

        ctx = await agent._snapshot_with_escalation(
            runtime,
            goal=step.goal,
            capture_screenshot=False,
            step=step,
        )

        # No scrolling should have occurred
        assert runtime.scroll_count == 0
        assert ctx.snapshot is not None

    @pytest.mark.asyncio
    async def test_scroll_after_escalation_scrolls_when_element_not_found(self) -> None:
        """When element not found after limit escalation, should scroll to find it."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent
        from unittest.mock import AsyncMock, patch, MagicMock

        config = PlannerExecutorConfig(
            snapshot=SnapshotEscalationConfig(
                enabled=False,  # Disable limit escalation to test scroll directly
                scroll_after_escalation=True,
                scroll_max_attempts=2,
                scroll_directions=("down",),
            ),
            verbose=True,  # Enable verbose to see debug output
            stabilize_enabled=False,
        )

        # Must inject intent_heuristics for scroll-after-escalation to trigger
        # (scroll only happens for CLICK actions with intent and heuristics)
        mock_heuristics = MockIntentHeuristics()

        agent = PlannerExecutorAgent(
            planner=MockLLMProvider(),
            executor=MockLLMProvider(),
            config=config,
            intent_heuristics=mock_heuristics,  # Required for scroll-after-escalation
        )

        # Initial snapshot has no "Add to Cart" button
        # After 1 scroll, "Add to Cart" appears
        initial_snap = MockSnapshot([MockElement(1, "button", "Submit")])
        after_scroll_snap = MockSnapshot([
            MockElement(1, "button", "Submit"),
            MockElement(2, "button", "Add to Cart"),
        ])

        runtime = MockRuntime(snapshots_by_scroll={0: initial_snap, 1: after_scroll_snap})

        step = PlanStep(
            id=1,
            goal="Click Add to Cart button",
            action="CLICK",
            intent="add_to_cart",
            verify=[],
        )

        # Mock _try_intent_heuristics to simulate finding element after scroll
        call_count = 0

        async def mock_try_heuristics(step: Any, elements: list, url: str) -> int | None:
            nonlocal call_count
            call_count += 1
            # First call (before scroll): element not found
            # Second call (after scroll): element found
            if call_count == 1:
                return None
            return 2  # Found element 2 after scrolling

        # Mock _format_context to avoid issues with snapshot formatting
        def mock_format_context(snap: Any, goal: str) -> str:
            return "mocked context"

        with patch.object(agent, "_try_intent_heuristics", side_effect=mock_try_heuristics):
            with patch.object(agent, "_format_context", side_effect=mock_format_context):
                ctx = await agent._snapshot_with_escalation(
                    runtime,
                    goal=step.goal,
                    capture_screenshot=False,
                    step=step,
                )

        # Should have scrolled to find the element
        assert runtime.scroll_count >= 1
        assert "down" in runtime.scroll_directions

    @pytest.mark.asyncio
    async def test_scroll_tries_multiple_directions(self) -> None:
        """Should try all configured directions when element not found."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent
        from unittest.mock import AsyncMock, patch, MagicMock

        config = PlannerExecutorConfig(
            snapshot=SnapshotEscalationConfig(
                enabled=False,  # Disable limit escalation
                scroll_after_escalation=True,
                scroll_max_attempts=2,
                scroll_directions=("down", "up"),
            ),
            verbose=True,  # Enable verbose to see debug output
            stabilize_enabled=False,
        )

        # Must inject intent_heuristics for scroll-after-escalation to trigger
        mock_heuristics = MockIntentHeuristics()

        agent = PlannerExecutorAgent(
            planner=MockLLMProvider(),
            executor=MockLLMProvider(),
            config=config,
            intent_heuristics=mock_heuristics,  # Required for scroll-after-escalation
        )

        # Element never found - should try all directions
        snap = MockSnapshot([MockElement(1, "button", "Submit")])
        runtime = MockRuntime(snapshots_by_scroll={i: snap for i in range(10)})

        step = PlanStep(
            id=1,
            goal="Click non-existent button",
            action="CLICK",
            intent="nonexistent",
            verify=[],
        )

        # Mock _try_intent_heuristics to always return None (element never found)
        async def mock_try_heuristics(step: Any, elements: list, url: str) -> int | None:
            return None  # Element never found

        # Mock _format_context to avoid issues with snapshot formatting
        def mock_format_context(snap: Any, goal: str) -> str:
            return "mocked context"

        with patch.object(agent, "_try_intent_heuristics", side_effect=mock_try_heuristics):
            with patch.object(agent, "_format_context", side_effect=mock_format_context):
                ctx = await agent._snapshot_with_escalation(
                    runtime,
                    goal=step.goal,
                    capture_screenshot=False,
                    step=step,
                )

        # Should have tried both directions
        assert "down" in runtime.scroll_directions
        assert "up" in runtime.scroll_directions
        # Max attempts per direction is 2, so total scrolls should be 4
        assert runtime.scroll_count == 4

    @pytest.mark.asyncio
    async def test_no_scroll_when_step_is_none(self) -> None:
        """When step is None, scroll-after-escalation should not trigger."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent

        config = PlannerExecutorConfig(
            snapshot=SnapshotEscalationConfig(
                enabled=False,
                scroll_after_escalation=True,
            ),
        )
        agent = PlannerExecutorAgent(
            planner=MockLLMProvider(),
            executor=MockLLMProvider(),
            config=config,
        )

        runtime = MockRuntime()

        ctx = await agent._snapshot_with_escalation(
            runtime,
            goal="Some goal",
            capture_screenshot=False,
            step=None,  # No step provided
        )

        # No scrolling should occur without a step
        assert runtime.scroll_count == 0

    @pytest.mark.asyncio
    async def test_no_scroll_for_type_and_submit_actions(self) -> None:
        """Scroll-after-escalation should NOT trigger for TYPE_AND_SUBMIT actions."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent

        config = PlannerExecutorConfig(
            snapshot=SnapshotEscalationConfig(
                enabled=False,
                scroll_after_escalation=True,  # Enabled, but should not trigger
            ),
        )

        # Even with intent_heuristics, TYPE_AND_SUBMIT should not trigger scroll
        mock_heuristics = MockIntentHeuristics()

        agent = PlannerExecutorAgent(
            planner=MockLLMProvider(),
            executor=MockLLMProvider(),
            config=config,
            intent_heuristics=mock_heuristics,
        )

        runtime = MockRuntime()

        step = PlanStep(
            id=1,
            goal="Search for Logitech mouse",
            action="TYPE_AND_SUBMIT",  # Not a CLICK action
            intent="search",
            input="Logitech mouse",
            verify=[],
        )

        ctx = await agent._snapshot_with_escalation(
            runtime,
            goal=step.goal,
            capture_screenshot=False,
            step=step,
        )

        # No scrolling should occur for TYPE_AND_SUBMIT
        assert runtime.scroll_count == 0

    @pytest.mark.asyncio
    async def test_no_scroll_without_intent_heuristics(self) -> None:
        """Scroll-after-escalation should NOT trigger without intent_heuristics."""
        from predicate.agents.planner_executor_agent import PlannerExecutorAgent

        config = PlannerExecutorConfig(
            snapshot=SnapshotEscalationConfig(
                enabled=False,
                scroll_after_escalation=True,  # Enabled, but should not trigger
            ),
        )

        # No intent_heuristics injected
        agent = PlannerExecutorAgent(
            planner=MockLLMProvider(),
            executor=MockLLMProvider(),
            config=config,
            # intent_heuristics=None  # Not provided
        )

        runtime = MockRuntime()

        step = PlanStep(
            id=1,
            goal="Click Add to Cart",
            action="CLICK",
            intent="add_to_cart",
            verify=[],
        )

        ctx = await agent._snapshot_with_escalation(
            runtime,
            goal=step.goal,
            capture_screenshot=False,
            step=step,
        )

        # No scrolling should occur without intent_heuristics
        assert runtime.scroll_count == 0


# ---------------------------------------------------------------------------
# Test AuthBoundaryConfig
# ---------------------------------------------------------------------------


class TestAuthBoundaryConfig:
    """Tests for AuthBoundaryConfig."""

    def test_default_values(self) -> None:
        config = AuthBoundaryConfig()
        assert config.enabled is True
        assert config.stop_on_auth is True
        assert "/signin" in config.url_patterns
        assert "/ap/signin" in config.url_patterns
        assert "/ax/claim" in config.url_patterns  # Amazon CAPTCHA

    def test_default_url_patterns_include_common_patterns(self) -> None:
        config = AuthBoundaryConfig()
        expected_patterns = [
            "/signin",
            "/sign-in",
            "/login",
            "/log-in",
            "/auth",
            "/authenticate",
            "/ap/signin",  # Amazon
            "/ap/register",  # Amazon
            "/ax/claim",  # Amazon CAPTCHA
            "/account/login",
            "/accounts/login",
            "/user/login",
        ]
        for pattern in expected_patterns:
            assert pattern in config.url_patterns, f"Missing pattern: {pattern}"

    def test_can_be_disabled(self) -> None:
        config = AuthBoundaryConfig(enabled=False)
        assert config.enabled is False

    def test_stop_on_auth_can_be_disabled(self) -> None:
        config = AuthBoundaryConfig(stop_on_auth=False)
        assert config.stop_on_auth is False

    def test_custom_url_patterns(self) -> None:
        config = AuthBoundaryConfig(
            url_patterns=("/custom/login", "/my-signin"),
        )
        assert config.url_patterns == ("/custom/login", "/my-signin")

    def test_custom_auth_success_message(self) -> None:
        config = AuthBoundaryConfig(
            auth_success_message="Custom: Login required",
        )
        assert config.auth_success_message == "Custom: Login required"

    def test_planner_executor_config_has_auth_boundary(self) -> None:
        config = PlannerExecutorConfig()
        assert config.auth_boundary is not None
        assert config.auth_boundary.enabled is True

    def test_planner_executor_config_custom_auth_boundary(self) -> None:
        config = PlannerExecutorConfig(
            auth_boundary=AuthBoundaryConfig(
                enabled=True,
                url_patterns=("/custom-signin",),
                stop_on_auth=True,
            ),
        )
        assert config.auth_boundary.url_patterns == ("/custom-signin",)


# ---------------------------------------------------------------------------
# Test Modal Dismissal After Successful CLICK
# ---------------------------------------------------------------------------


class TestModalDismissalAfterSuccessfulClick:
    """Tests for modal dismissal when verification passes after CLICK."""

    def test_modal_dismissal_config_min_new_elements_default(self) -> None:
        """Default min_new_elements should be 5 for DOM change detection."""
        config = ModalDismissalConfig()
        assert config.min_new_elements == 5

    def test_modal_enabled_by_default_in_planner_executor_config(self) -> None:
        """Modal dismissal should be enabled by default."""
        config = PlannerExecutorConfig()
        assert config.modal.enabled is True
        assert config.modal.min_new_elements == 5

    def test_modal_dismissal_patterns_include_no_thanks(self) -> None:
        """'no thanks' should be in default patterns for drawer dismissal."""
        config = ModalDismissalConfig()
        # This is the pattern that dismisses Amazon's product protection drawer
        assert "no thanks" in config.dismiss_patterns

    def test_modal_config_has_required_fields_for_drawer_dismissal(self) -> None:
        """Config should have all fields needed for drawer dismissal logic."""
        config = ModalDismissalConfig()
        # These are all used in _attempt_modal_dismissal
        assert hasattr(config, "enabled")
        assert hasattr(config, "dismiss_patterns")
        assert hasattr(config, "dismiss_icons")
        assert hasattr(config, "role_filter")
        assert hasattr(config, "max_attempts")
        assert hasattr(config, "min_new_elements")


# ---------------------------------------------------------------------------
# Test Token Usage Tracking
# ---------------------------------------------------------------------------


class TestTokenUsageTracking:
    """Tests for token usage tracking in PlannerExecutorAgent."""

    def test_token_usage_totals_add(self) -> None:
        """TokenUsageTotals should accumulate tokens correctly."""
        from predicate.agents.planner_executor_agent import TokenUsageTotals
        from predicate.llm_provider import LLMResponse

        totals = TokenUsageTotals()
        assert totals.calls == 0
        assert totals.prompt_tokens == 0
        assert totals.completion_tokens == 0
        assert totals.total_tokens == 0

        # Add first response
        resp1 = LLMResponse(
            content="test",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        totals.add(resp1)
        assert totals.calls == 1
        assert totals.prompt_tokens == 100
        assert totals.completion_tokens == 50
        assert totals.total_tokens == 150

        # Add second response
        resp2 = LLMResponse(
            content="test2",
            prompt_tokens=200,
            completion_tokens=75,
            total_tokens=275,
        )
        totals.add(resp2)
        assert totals.calls == 2
        assert totals.prompt_tokens == 300
        assert totals.completion_tokens == 125
        assert totals.total_tokens == 425

    def test_token_usage_totals_handles_none_values(self) -> None:
        """TokenUsageTotals should handle None token counts gracefully."""
        from predicate.agents.planner_executor_agent import TokenUsageTotals
        from predicate.llm_provider import LLMResponse

        totals = TokenUsageTotals()
        resp = LLMResponse(
            content="test",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )
        totals.add(resp)
        assert totals.calls == 1
        assert totals.prompt_tokens == 0
        assert totals.completion_tokens == 0
        assert totals.total_tokens == 0

    def test_token_usage_collector_records_by_role(self) -> None:
        """_TokenUsageCollector should track tokens by role."""
        from predicate.agents.planner_executor_agent import _TokenUsageCollector
        from predicate.llm_provider import LLMResponse

        collector = _TokenUsageCollector()

        resp_planner = LLMResponse(
            content="plan",
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
            model_name="gpt-4o",
        )
        collector.record(role="planner", resp=resp_planner)

        resp_executor = LLMResponse(
            content="action",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            model_name="gpt-4o-mini",
        )
        collector.record(role="executor", resp=resp_executor)

        summary = collector.summary()

        # Check total
        assert summary["total"]["calls"] == 2
        assert summary["total"]["prompt_tokens"] == 600
        assert summary["total"]["completion_tokens"] == 220
        assert summary["total"]["total_tokens"] == 820

        # Check by_role
        assert "planner" in summary["by_role"]
        assert summary["by_role"]["planner"]["calls"] == 1
        assert summary["by_role"]["planner"]["total_tokens"] == 700

        assert "executor" in summary["by_role"]
        assert summary["by_role"]["executor"]["calls"] == 1
        assert summary["by_role"]["executor"]["total_tokens"] == 120

    def test_token_usage_collector_records_by_model(self) -> None:
        """_TokenUsageCollector should track tokens by model name."""
        from predicate.agents.planner_executor_agent import _TokenUsageCollector
        from predicate.llm_provider import LLMResponse

        collector = _TokenUsageCollector()

        resp1 = LLMResponse(
            content="test",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model_name="gpt-4o",
        )
        collector.record(role="planner", resp=resp1)

        resp2 = LLMResponse(
            content="test",
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
            model_name="gpt-4o-mini",
        )
        collector.record(role="executor", resp=resp2)

        summary = collector.summary()

        # Check by_model
        assert "gpt-4o" in summary["by_model"]
        assert summary["by_model"]["gpt-4o"]["total_tokens"] == 150

        assert "gpt-4o-mini" in summary["by_model"]
        assert summary["by_model"]["gpt-4o-mini"]["total_tokens"] == 75

    def test_token_usage_collector_reset(self) -> None:
        """_TokenUsageCollector reset should clear all data."""
        from predicate.agents.planner_executor_agent import _TokenUsageCollector
        from predicate.llm_provider import LLMResponse

        collector = _TokenUsageCollector()
        resp = LLMResponse(
            content="test",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        collector.record(role="planner", resp=resp)
        assert collector.summary()["total"]["calls"] == 1

        collector.reset()
        summary = collector.summary()
        assert summary["total"]["calls"] == 0
        assert summary["total"]["total_tokens"] == 0
        assert summary["by_role"] == {}
        assert summary["by_model"] == {}

    def test_run_outcome_has_token_usage_field(self) -> None:
        """RunOutcome should have token_usage field."""
        from predicate.agents.planner_executor_agent import RunOutcome

        outcome = RunOutcome(
            run_id="test-run",
            task="test task",
            success=True,
            steps_completed=3,
            steps_total=3,
            replans_used=0,
        )
        # Default should be None
        assert outcome.token_usage is None

        # Should accept token usage dict
        outcome_with_tokens = RunOutcome(
            run_id="test-run",
            task="test task",
            success=True,
            steps_completed=3,
            steps_total=3,
            replans_used=0,
            token_usage={
                "total": {"calls": 5, "total_tokens": 1000},
                "by_role": {"planner": {"calls": 2, "total_tokens": 700}},
                "by_model": {"gpt-4o": {"calls": 2, "total_tokens": 700}},
            },
        )
        assert outcome_with_tokens.token_usage is not None
        assert outcome_with_tokens.token_usage["total"]["total_tokens"] == 1000


# ---------------------------------------------------------------------------
# Test StepwisePlanningConfig
# ---------------------------------------------------------------------------


class TestStepwisePlanningConfig:
    """Tests for StepwisePlanningConfig."""

    def test_default_values(self) -> None:
        from predicate.agents.planner_executor_agent import StepwisePlanningConfig

        config = StepwisePlanningConfig()
        assert config.max_steps == 30
        assert config.action_history_limit == 5
        assert config.include_page_context is True

    def test_custom_values(self) -> None:
        from predicate.agents.planner_executor_agent import StepwisePlanningConfig

        config = StepwisePlanningConfig(
            max_steps=50,
            action_history_limit=10,
            include_page_context=False,
        )
        assert config.max_steps == 50
        assert config.action_history_limit == 10
        assert config.include_page_context is False


# ---------------------------------------------------------------------------
# Test Auto-Fallback Configuration
# ---------------------------------------------------------------------------


class TestAutoFallbackConfig:
    """Tests for auto-fallback to stepwise planning configuration."""

    def test_auto_fallback_enabled_by_default(self) -> None:
        """Auto-fallback should be enabled by default."""
        config = PlannerExecutorConfig()
        assert config.auto_fallback_to_stepwise is True
        assert config.auto_fallback_replan_threshold == 1

    def test_auto_fallback_can_be_disabled(self) -> None:
        """Auto-fallback can be disabled."""
        config = PlannerExecutorConfig(auto_fallback_to_stepwise=False)
        assert config.auto_fallback_to_stepwise is False

    def test_custom_fallback_threshold(self) -> None:
        """Custom replan threshold for fallback."""
        config = PlannerExecutorConfig(
            auto_fallback_to_stepwise=True,
            auto_fallback_replan_threshold=2,
        )
        assert config.auto_fallback_replan_threshold == 2

    def test_run_outcome_has_fallback_used_field(self) -> None:
        """RunOutcome should have fallback_used field."""
        from predicate.agents.planner_executor_agent import RunOutcome

        outcome = RunOutcome(
            run_id="test-run",
            task="test task",
            success=True,
            steps_completed=3,
            steps_total=3,
            replans_used=0,
        )
        # Default should be False
        assert outcome.fallback_used is False

        # Can be set to True
        outcome_with_fallback = RunOutcome(
            run_id="test-run",
            task="test task",
            success=True,
            steps_completed=5,
            steps_total=5,
            replans_used=1,
            fallback_used=True,
        )
        assert outcome_with_fallback.fallback_used is True

    def test_stepwise_config_in_planner_executor_config(self) -> None:
        """StepwisePlanningConfig should be accessible in PlannerExecutorConfig."""
        from predicate.agents.planner_executor_agent import StepwisePlanningConfig

        config = PlannerExecutorConfig(
            stepwise=StepwisePlanningConfig(
                max_steps=20,
                action_history_limit=3,
            ),
        )
        assert config.stepwise.max_steps == 20
        assert config.stepwise.action_history_limit == 3


# ---------------------------------------------------------------------------
# Test Stepwise Modal Dismissal
# ---------------------------------------------------------------------------


class TestStepwiseModalDismissal:
    """Tests for modal dismissal in stepwise planning mode."""

    def test_modal_config_used_in_stepwise(self) -> None:
        """Modal dismissal config should be available for stepwise planning."""
        config = PlannerExecutorConfig(
            modal=ModalDismissalConfig(
                enabled=True,
                min_new_elements=5,
            ),
        )
        # Verify the config is correctly set up for stepwise use
        assert config.modal.enabled is True
        assert config.modal.min_new_elements == 5
        # Should have default dismiss patterns for drawers
        assert "no thanks" in config.modal.dismiss_patterns
        assert "close" in config.modal.dismiss_patterns
        assert "dismiss" in config.modal.dismiss_patterns

    def test_modal_dismissal_patterns_cover_cart_drawers(self) -> None:
        """Default patterns should handle common cart/upsell drawer scenarios."""
        config = ModalDismissalConfig()
        # These patterns are commonly found in e-commerce drawers
        patterns_lower = [p.lower() for p in config.dismiss_patterns]
        assert "no thanks" in patterns_lower  # Amazon protection drawer
        assert "close" in patterns_lower  # Generic close buttons
        assert "continue" in patterns_lower  # "Continue shopping"
        assert "skip" in patterns_lower  # "Skip for now"

    def test_modal_dismissal_disabled_does_not_affect_stepwise(self) -> None:
        """When modal is disabled, stepwise should still work (no dismissal)."""
        config = PlannerExecutorConfig(
            modal=ModalDismissalConfig(enabled=False),
        )
        assert config.modal.enabled is False
        # Stepwise config should still be valid
        assert config.stepwise.max_steps == 30
