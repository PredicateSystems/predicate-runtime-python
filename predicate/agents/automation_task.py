"""
AutomationTask: Generic task model for browser automation.

This module provides a task abstraction that generalizes WebBenchTask to support
broad web automation use cases like "buy a laptop on xyz.com".

Key features:
- Natural language task description with optional structured goal
- Task category hints for heuristics selection
- Budget constraints (timeout, max_steps, max_replans)
- Extraction specification for data extraction tasks
- Human-defined success criteria (optional override)
- Recovery configuration for rollback on failure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskCategory(str, Enum):
    """
    Task category for heuristics and verification selection.

    Categories help the planner and executor make better decisions about:
    - Element selection strategies
    - Verification stringency
    - Recovery behavior
    """

    NAVIGATION = "navigation"  # Navigate to a destination
    SEARCH = "search"  # Search and find information
    FORM_FILL = "form_fill"  # Fill out forms
    EXTRACTION = "extraction"  # Extract data from pages
    TRANSACTION = "transaction"  # Purchase, submit, create actions
    VERIFICATION = "verification"  # Verify state/information exists


class ExtractionSpec(BaseModel):
    """
    Specification for data extraction tasks.

    Used when the task involves extracting structured data from pages,
    such as product information, search results, or table data.
    """

    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema for expected extraction output",
    )
    target_selectors: list[str] = Field(
        default_factory=list,
        description="Suggested selectors for extraction targets",
    )
    format: Literal["json", "text", "markdown", "table"] = Field(
        default="json",
        description="Output format for extracted data",
    )
    require_evidence: bool = Field(
        default=True,
        description="Whether to require grounding evidence for extractions",
    )

    class Config:
        extra = "allow"


class SuccessCriteria(BaseModel):
    """
    Human-defined success criteria for task completion.

    If provided, these criteria override planner-proposed verification.
    This allows users to define exactly what "success" means for their task.

    Example:
        SuccessCriteria(
            predicates=[
                {"predicate": "url_contains", "args": ["/confirmation"]},
                {"predicate": "exists", "args": ["[data-testid='order-number']"]},
            ],
            require_all=True,
        )
    """

    predicates: list[dict[str, Any]] = Field(
        default_factory=list,
        description="PredicateSpec definitions for success verification",
    )
    require_all: bool = Field(
        default=True,
        description="If True, all predicates must pass. If False, any passing is success.",
    )

    class Config:
        extra = "allow"


@dataclass(frozen=True)
class AutomationTask:
    """
    Generic automation task for the PlannerExecutorAgent.

    This replaces WebBenchTask with a more flexible structure that supports:
    - Broad tasks like "buy a laptop on xyz.com"
    - Optional structured goals and extraction specs
    - Timeout and budget constraints
    - Category hints for heuristics selection
    - Recovery configuration

    Example:
        task = AutomationTask(
            task_id="purchase-laptop-001",
            starting_url="https://amazon.com",
            task="Find a laptop under $1000 with good reviews and add to cart",
            category=TaskCategory.TRANSACTION,
            timeout_s=300.0,
            max_steps=50,
        )

    Example with extraction:
        task = AutomationTask(
            task_id="extract-product-info",
            starting_url="https://amazon.com/dp/B0...",
            task="Extract the product name, price, and rating",
            category=TaskCategory.EXTRACTION,
            extraction_spec=ExtractionSpec(
                schema={"name": "str", "price": "float", "rating": "float"},
                format="json",
            ),
        )

    Example with human-defined success criteria:
        task = AutomationTask(
            task_id="checkout-flow",
            starting_url="https://shop.com/cart",
            task="Complete the checkout process",
            category=TaskCategory.TRANSACTION,
            success_criteria=SuccessCriteria(
                predicates=[
                    {"predicate": "url_contains", "args": ["/confirmation"]},
                    {"predicate": "exists", "args": [".order-number"]},
                ],
                require_all=True,
            ),
        )
    """

    # Required fields
    task_id: str
    starting_url: str
    task: str  # Natural language task description

    # Optional: Structured goal for more precise planning
    goal: dict[str, Any] | None = None

    # Optional: Category hint for heuristics/verification selection
    category: TaskCategory | None = None

    # Budget constraints
    timeout_s: float | None = None
    max_steps: int = 50
    max_replans: int = 2
    max_vision_calls: int = 3

    # Extraction specification (for data extraction tasks)
    extraction_spec: ExtractionSpec | None = None

    # Human-defined success criteria (optional override)
    success_criteria: SuccessCriteria | None = None

    # Recovery configuration
    enable_recovery: bool = True
    max_recovery_attempts: int = 2

    # Domain hints for heuristics (e.g., ["ecommerce", "amazon"])
    domain_hints: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_webbench_task(cls, task: Any) -> "AutomationTask":
        """
        Factory method to convert WebBenchTask to AutomationTask.

        Preserves backward compatibility with webbench.

        Args:
            task: WebBenchTask instance with id, starting_url, task, category

        Returns:
            AutomationTask instance

        Example:
            from webbench.models import WebBenchTask

            wb_task = WebBenchTask(
                id="task-001",
                starting_url="https://example.com",
                task="Click the login button",
                category="CREATE",
            )
            automation_task = AutomationTask.from_webbench_task(wb_task)
        """
        # Map WebBench categories to TaskCategory
        category_map = {
            "READ": TaskCategory.EXTRACTION,
            "CREATE": TaskCategory.TRANSACTION,
            "UPDATE": TaskCategory.FORM_FILL,
            "DELETE": TaskCategory.TRANSACTION,
            "FILE_MANIPULATION": TaskCategory.TRANSACTION,
        }
        wb_category = getattr(task, "category", None)
        category = category_map.get(wb_category) if wb_category else None

        # For READ tasks, create a basic extraction spec
        extraction_spec = None
        if wb_category == "READ":
            extraction_spec = ExtractionSpec(
                format="json",
                require_evidence=True,
            )

        return cls(
            task_id=task.id,
            starting_url=task.starting_url,
            task=task.task,
            category=category,
            extraction_spec=extraction_spec,
        )

    @classmethod
    def from_string(
        cls,
        task: str,
        starting_url: str,
        *,
        task_id: str | None = None,
        category: TaskCategory | None = None,
    ) -> "AutomationTask":
        """
        Create an AutomationTask from a simple string description.

        Args:
            task: Natural language task description
            starting_url: URL to start automation from
            task_id: Optional task ID (auto-generated if not provided)
            category: Optional task category hint

        Returns:
            AutomationTask instance

        Example:
            task = AutomationTask.from_string(
                "Search for 'laptop' and add the first result to cart",
                "https://amazon.com",
            )
        """
        import uuid

        return cls(
            task_id=task_id or str(uuid.uuid4()),
            starting_url=starting_url,
            task=task,
            category=category,
        )

    def with_success_criteria(self, *predicates: dict[str, Any], require_all: bool = True) -> "AutomationTask":
        """
        Return a new AutomationTask with the specified success criteria.

        Args:
            *predicates: PredicateSpec dictionaries
            require_all: If True, all predicates must pass

        Returns:
            New AutomationTask with success_criteria set

        Example:
            task = task.with_success_criteria(
                {"predicate": "url_contains", "args": ["/success"]},
                {"predicate": "exists", "args": [".confirmation"]},
            )
        """
        return AutomationTask(
            task_id=self.task_id,
            starting_url=self.starting_url,
            task=self.task,
            goal=self.goal,
            category=self.category,
            timeout_s=self.timeout_s,
            max_steps=self.max_steps,
            max_replans=self.max_replans,
            max_vision_calls=self.max_vision_calls,
            extraction_spec=self.extraction_spec,
            success_criteria=SuccessCriteria(
                predicates=list(predicates),
                require_all=require_all,
            ),
            enable_recovery=self.enable_recovery,
            max_recovery_attempts=self.max_recovery_attempts,
            domain_hints=self.domain_hints,
        )

    def with_extraction(
        self,
        output_schema: dict[str, Any] | None = None,
        format: Literal["json", "text", "markdown", "table"] = "json",
    ) -> "AutomationTask":
        """
        Return a new AutomationTask with extraction specification.

        Args:
            output_schema: JSON schema for expected output
            format: Output format

        Returns:
            New AutomationTask with extraction_spec set
        """
        return AutomationTask(
            task_id=self.task_id,
            starting_url=self.starting_url,
            task=self.task,
            goal=self.goal,
            category=self.category or TaskCategory.EXTRACTION,
            timeout_s=self.timeout_s,
            max_steps=self.max_steps,
            max_replans=self.max_replans,
            max_vision_calls=self.max_vision_calls,
            extraction_spec=ExtractionSpec(output_schema=output_schema, format=format),
            success_criteria=self.success_criteria,
            enable_recovery=self.enable_recovery,
            max_recovery_attempts=self.max_recovery_attempts,
            domain_hints=self.domain_hints,
        )
