"""
Unit tests for AutomationTask and related models.
"""

import pytest
from dataclasses import dataclass

from predicate.agents import (
    AutomationTask,
    ExtractionSpec,
    SuccessCriteria,
    TaskCategory,
    HeuristicHint,
    RecoveryCheckpoint,
    RecoveryState,
    ComposableHeuristics,
    COMMON_HINTS,
    get_common_hint,
)


class TestAutomationTask:
    """Tests for AutomationTask dataclass."""

    def test_basic_creation(self):
        """Test creating a basic AutomationTask."""
        task = AutomationTask(
            task_id="test-001",
            starting_url="https://example.com",
            task="Click the login button",
        )
        assert task.task_id == "test-001"
        assert task.starting_url == "https://example.com"
        assert task.task == "Click the login button"
        assert task.category is None
        assert task.max_steps == 50
        assert task.enable_recovery is True

    def test_with_category(self):
        """Test creating AutomationTask with category."""
        task = AutomationTask(
            task_id="test-002",
            starting_url="https://shop.com",
            task="Add item to cart",
            category=TaskCategory.TRANSACTION,
        )
        assert task.category == TaskCategory.TRANSACTION

    def test_with_success_criteria(self):
        """Test creating AutomationTask with success criteria."""
        task = AutomationTask(
            task_id="test-003",
            starting_url="https://shop.com",
            task="Complete checkout",
            success_criteria=SuccessCriteria(
                predicates=[
                    {"predicate": "url_contains", "args": ["/confirmation"]},
                    {"predicate": "exists", "args": [".order-number"]},
                ],
                require_all=True,
            ),
        )
        assert task.success_criteria is not None
        assert len(task.success_criteria.predicates) == 2
        assert task.success_criteria.require_all is True

    def test_with_extraction_spec(self):
        """Test creating AutomationTask with extraction spec."""
        task = AutomationTask(
            task_id="test-004",
            starting_url="https://shop.com/product",
            task="Extract product details",
            category=TaskCategory.EXTRACTION,
            extraction_spec=ExtractionSpec(
                output_schema={"name": "str", "price": "float"},
                format="json",
            ),
        )
        assert task.extraction_spec is not None
        assert task.extraction_spec.format == "json"
        assert task.extraction_spec.output_schema == {"name": "str", "price": "float"}

    def test_from_string(self):
        """Test creating AutomationTask from string."""
        task = AutomationTask.from_string(
            "Search for laptops",
            "https://amazon.com",
            category=TaskCategory.SEARCH,
        )
        assert task.task == "Search for laptops"
        assert task.starting_url == "https://amazon.com"
        assert task.category == TaskCategory.SEARCH
        assert task.task_id is not None

    def test_with_success_criteria_method(self):
        """Test with_success_criteria method."""
        task = AutomationTask(
            task_id="test-005",
            starting_url="https://shop.com",
            task="Checkout",
        )
        task_with_criteria = task.with_success_criteria(
            {"predicate": "url_contains", "args": ["/success"]},
            {"predicate": "exists", "args": [".confirmation"]},
        )
        assert task_with_criteria.success_criteria is not None
        assert len(task_with_criteria.success_criteria.predicates) == 2
        # Original task unchanged
        assert task.success_criteria is None

    def test_with_extraction_method(self):
        """Test with_extraction method."""
        task = AutomationTask(
            task_id="test-006",
            starting_url="https://shop.com",
            task="Get product info",
        )
        task_with_extraction = task.with_extraction(
            output_schema={"name": "str"},
            format="json",
        )
        assert task_with_extraction.extraction_spec is not None
        assert task_with_extraction.category == TaskCategory.EXTRACTION

    def test_from_webbench_task(self):
        """Test converting from WebBenchTask-like object."""
        @dataclass(frozen=True)
        class MockWebBenchTask:
            id: str
            starting_url: str
            task: str
            category: str | None = None

        wb_task = MockWebBenchTask(
            id="wb-001",
            starting_url="https://example.com",
            task="Read the page title",
            category="READ",
        )
        automation_task = AutomationTask.from_webbench_task(wb_task)

        assert automation_task.task_id == "wb-001"
        assert automation_task.starting_url == "https://example.com"
        assert automation_task.task == "Read the page title"
        assert automation_task.category == TaskCategory.EXTRACTION
        assert automation_task.extraction_spec is not None


class TestHeuristicHint:
    """Tests for HeuristicHint model."""

    def test_basic_creation(self):
        """Test creating a basic HeuristicHint."""
        hint = HeuristicHint(
            intent_pattern="add_to_cart",
            text_patterns=["add to cart", "add to bag"],
            role_filter=["button"],
            priority=10,
        )
        assert hint.intent_pattern == "add_to_cart"
        assert len(hint.text_patterns) == 2
        assert hint.role_filter == ["button"]
        assert hint.priority == 10

    def test_matches_intent(self):
        """Test intent matching."""
        hint = HeuristicHint(
            intent_pattern="checkout",
            text_patterns=["checkout"],
            role_filter=["button"],
        )
        assert hint.matches_intent("checkout") is True
        assert hint.matches_intent("proceed_to_checkout") is True
        assert hint.matches_intent("CHECKOUT") is True
        assert hint.matches_intent("login") is False

    def test_matches_element(self):
        """Test element matching."""
        hint = HeuristicHint(
            intent_pattern="add_to_cart",
            text_patterns=["add to cart"],
            role_filter=["button"],
        )

        @dataclass
        class MockElement:
            id: int
            role: str
            text: str

        matching = MockElement(id=1, role="button", text="Add to Cart")
        non_matching_role = MockElement(id=2, role="link", text="Add to Cart")
        non_matching_text = MockElement(id=3, role="button", text="Remove")

        assert hint.matches_element(matching) is True
        assert hint.matches_element(non_matching_role) is False
        assert hint.matches_element(non_matching_text) is False

    def test_common_hints(self):
        """Test COMMON_HINTS dictionary."""
        assert "add_to_cart" in COMMON_HINTS
        assert "checkout" in COMMON_HINTS
        assert "login" in COMMON_HINTS
        assert "submit" in COMMON_HINTS

    def test_get_common_hint(self):
        """Test get_common_hint function."""
        hint = get_common_hint("add_to_cart")
        assert hint is not None
        assert hint.intent_pattern == "add_to_cart"

        # Test with dashes/spaces
        hint2 = get_common_hint("add-to-cart")
        assert hint2 is not None

        # Test non-existent
        hint3 = get_common_hint("nonexistent_pattern")
        assert hint3 is None


class TestRecoveryState:
    """Tests for RecoveryState and RecoveryCheckpoint."""

    def test_basic_creation(self):
        """Test creating a basic RecoveryState."""
        state = RecoveryState(max_recovery_attempts=3)
        assert state.max_recovery_attempts == 3
        assert state.recovery_attempts_used == 0
        assert len(state.checkpoints) == 0

    def test_record_checkpoint(self):
        """Test recording checkpoints."""
        state = RecoveryState()
        checkpoint = state.record_checkpoint(
            url="https://shop.com/cart",
            step_index=2,
            snapshot_digest="abc123",
            predicates_passed=["url_contains('/cart')"],
        )

        assert checkpoint.url == "https://shop.com/cart"
        assert checkpoint.step_index == 2
        assert len(state) == 1

    def test_get_recovery_target(self):
        """Test getting recovery target."""
        state = RecoveryState()

        # No checkpoints
        assert state.get_recovery_target() is None

        # Add checkpoints
        state.record_checkpoint("https://a.com", 0, "a", [])
        state.record_checkpoint("https://b.com", 1, "b", [])

        target = state.get_recovery_target()
        assert target is not None
        assert target.url == "https://b.com"

    def test_can_recover(self):
        """Test can_recover logic."""
        state = RecoveryState(max_recovery_attempts=2)

        # No checkpoints
        assert state.can_recover() is False

        # Add checkpoint
        state.record_checkpoint("https://a.com", 0, "a", [])
        assert state.can_recover() is True

        # Consume attempts
        state.consume_recovery_attempt()
        assert state.can_recover() is True

        state.consume_recovery_attempt()
        assert state.can_recover() is False

    def test_consume_recovery_attempt(self):
        """Test consuming recovery attempts."""
        state = RecoveryState(max_recovery_attempts=2)
        state.record_checkpoint("https://a.com", 0, "a", [])

        checkpoint = state.consume_recovery_attempt()
        assert checkpoint is not None
        assert state.recovery_attempts_used == 1
        assert state.current_recovery_target == checkpoint

    def test_max_checkpoints_limit(self):
        """Test that checkpoints are bounded."""
        state = RecoveryState(max_checkpoints=3)

        for i in range(5):
            state.record_checkpoint(f"https://a.com/{i}", i, str(i), [])

        # Should only keep last 3
        assert len(state) == 3
        assert state.checkpoints[0].url == "https://a.com/2"

    def test_reset(self):
        """Test resetting state."""
        state = RecoveryState()
        state.record_checkpoint("https://a.com", 0, "a", [])
        state.consume_recovery_attempt()

        state.reset()

        assert len(state) == 0
        assert state.recovery_attempts_used == 0
        assert state.current_recovery_target is None

    def test_last_successful_url(self):
        """Test last_successful_url property."""
        state = RecoveryState()
        assert state.last_successful_url is None

        state.record_checkpoint("https://a.com", 0, "a", [])
        assert state.last_successful_url == "https://a.com"


class TestComposableHeuristics:
    """Tests for ComposableHeuristics."""

    def test_basic_creation(self):
        """Test creating ComposableHeuristics."""
        heuristics = ComposableHeuristics()
        assert heuristics.priority_order() is not None

    def test_with_task_category(self):
        """Test with task category."""
        heuristics = ComposableHeuristics(
            task_category=TaskCategory.TRANSACTION,
        )

        @dataclass
        class MockElement:
            id: int
            role: str
            text: str

        elements = [
            MockElement(id=1, role="button", text="Add to Cart"),
            MockElement(id=2, role="link", text="Home"),
        ]

        # Category defaults should find transaction-related buttons
        element_id = heuristics.find_element_for_intent(
            "add_item",
            elements,
            "https://shop.com",
            "Add item to cart",
        )
        assert element_id == 1

    def test_set_step_hints(self):
        """Test setting step hints."""
        heuristics = ComposableHeuristics()

        heuristics.set_step_hints([
            {"intent_pattern": "custom", "text_patterns": ["custom text"], "role_filter": ["button"]},
        ])

        # Verify hints are set
        order = heuristics.priority_order()
        assert "custom" in order

    def test_clear_step_hints(self):
        """Test clearing step hints."""
        heuristics = ComposableHeuristics()
        heuristics.set_step_hints([
            {"intent_pattern": "test", "text_patterns": ["test"]},
        ])

        heuristics.clear_step_hints()

        # Should not have "test" in priority anymore (unless it's a common hint)
        order = heuristics.priority_order()
        # Common hints should still be there
        assert "add_to_cart" in order

    def test_hint_priority_over_static(self):
        """Test that planner hints take priority."""
        class StaticHeuristics:
            def find_element_for_intent(self, intent, elements, url, goal):
                return 999  # Always return 999

            def priority_order(self):
                return ["static"]

        heuristics = ComposableHeuristics(
            static_heuristics=StaticHeuristics(),
        )

        @dataclass
        class MockElement:
            id: int
            role: str
            text: str

        elements = [
            MockElement(id=1, role="button", text="Custom Button"),
            MockElement(id=2, role="button", text="Other"),
        ]

        # Set hints that match element 1
        heuristics.set_step_hints([
            HeuristicHint(
                intent_pattern="custom",
                text_patterns=["custom button"],
                role_filter=["button"],
                priority=100,
            ),
        ])

        # Hints should be tried first
        element_id = heuristics.find_element_for_intent(
            "custom_action",
            elements,
            "https://example.com",
            "Do custom action",
        )
        assert element_id == 1  # Not 999 from static heuristics


class TestTaskCategory:
    """Tests for TaskCategory enum."""

    def test_all_categories_exist(self):
        """Test all expected categories exist."""
        assert TaskCategory.NAVIGATION == "navigation"
        assert TaskCategory.SEARCH == "search"
        assert TaskCategory.FORM_FILL == "form_fill"
        assert TaskCategory.EXTRACTION == "extraction"
        assert TaskCategory.TRANSACTION == "transaction"
        assert TaskCategory.VERIFICATION == "verification"

    def test_category_values(self):
        """Test category string values."""
        assert TaskCategory.TRANSACTION.value == "transaction"
        assert str(TaskCategory.EXTRACTION) == "TaskCategory.EXTRACTION"


class TestExtractionSpec:
    """Tests for ExtractionSpec model."""

    def test_defaults(self):
        """Test default values."""
        spec = ExtractionSpec()
        assert spec.output_schema is None
        assert spec.target_selectors == []
        assert spec.format == "json"
        assert spec.require_evidence is True

    def test_with_schema(self):
        """Test with schema."""
        spec = ExtractionSpec(
            output_schema={"name": "str", "price": "float"},
            format="json",
        )
        assert spec.output_schema == {"name": "str", "price": "float"}


class TestSuccessCriteria:
    """Tests for SuccessCriteria model."""

    def test_defaults(self):
        """Test default values."""
        criteria = SuccessCriteria()
        assert criteria.predicates == []
        assert criteria.require_all is True

    def test_with_predicates(self):
        """Test with predicates."""
        criteria = SuccessCriteria(
            predicates=[
                {"predicate": "url_contains", "args": ["/success"]},
            ],
            require_all=False,
        )
        assert len(criteria.predicates) == 1
        assert criteria.require_all is False
