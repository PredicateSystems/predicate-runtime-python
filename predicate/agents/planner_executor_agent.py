"""
PlannerExecutorAgent: Two-tier agent architecture with Planner (7B+) and Executor (3B-7B) models.

This module implements the Planner + Executor design pattern from docs/PLANNER_EXECUTOR_DESIGN.md:
- Planner: Generates JSON execution plans with predicates
- Executor: Executes each step with snapshot-first verification

Key features:
- Incremental limit escalation for snapshot capture
- Vision fallback for canvas pages or low-confidence snapshots
- SnapshotContext sharing between Planner and Executor
- Full tracing integration for Predicate Studio visualization
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..agent_runtime import AgentRuntime
from ..llm_provider import LLMProvider, LLMResponse
from ..models import Snapshot, SnapshotOptions, StepHookContext
from ..trace_event_builder import TraceEventBuilder
from ..tracing import Tracer
from ..verification import (
    Predicate,
    all_of,
    any_of,
    element_count,
    exists,
    not_exists,
    url_contains,
    url_matches,
)

from .automation_task import AutomationTask, ExtractionSpec, SuccessCriteria, TaskCategory
from .browser_agent import CaptchaConfig, VisionFallbackConfig
from .composable_heuristics import ComposableHeuristics
from .heuristic_spec import HeuristicHint
from .recovery import RecoveryCheckpoint, RecoveryState


# ---------------------------------------------------------------------------
# IntentHeuristics Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IntentHeuristics(Protocol):
    """
    Protocol for pluggable domain-specific element selection heuristics.

    Developers can implement this protocol to provide domain-specific logic
    for selecting elements based on the step intent. This allows the SDK to
    remain generic while supporting specialized behavior for different sites.

    Example implementation for an e-commerce site:

        class EcommerceHeuristics:
            def find_element_for_intent(
                self,
                intent: str,
                elements: list[Any],
                url: str,
                goal: str,
            ) -> int | None:
                if "add to cart" in intent.lower():
                    for el in elements:
                        text = getattr(el, "text", "") or ""
                        if "add to cart" in text.lower():
                            return getattr(el, "id", None)
                return None  # Fall back to LLM

            def priority_order(self) -> list[str]:
                return ["add_to_cart", "checkout", "search"]

        # Usage:
        agent = PlannerExecutorAgent(
            planner=planner,
            executor=executor,
            intent_heuristics=EcommerceHeuristics(),
        )
    """

    def find_element_for_intent(
        self,
        intent: str,
        elements: list[Any],
        url: str,
        goal: str,
    ) -> int | None:
        """
        Find element ID for a given intent using domain-specific heuristics.

        Args:
            intent: The intent hint from the plan step (e.g., "add_to_cart", "checkout")
            elements: List of snapshot elements with id, role, text, etc.
            url: Current page URL
            goal: Human-readable goal for context

        Returns:
            Element ID if a match is found, None to fall back to LLM executor
        """
        ...

    def priority_order(self) -> list[str]:
        """
        Return list of intent patterns in priority order.

        The agent will try heuristics for each intent pattern in order.
        This helps prioritize certain actions (e.g., checkout over add-to-cart).

        Returns:
            List of intent pattern strings
        """
        ...


@runtime_checkable
class ExecutorOverride(Protocol):
    """
    Protocol for validating or overriding executor element choices.

    This allows developers to add validation logic or override the executor's
    choice before an action is executed. Useful for safety checks or
    domain-specific corrections.

    Example:
        class SafetyOverride:
            def validate_choice(
                self,
                element_id: int,
                action: str,
                elements: list[Any],
                goal: str,
            ) -> tuple[bool, int | None, str | None]:
                # Block clicks on delete buttons
                for el in elements:
                    if getattr(el, "id", None) == element_id:
                        text = getattr(el, "text", "") or ""
                        if "delete" in text.lower():
                            return False, None, "blocked_delete_button"
                return True, None, None
    """

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
            - is_valid: True if choice is acceptable
            - override_element_id: Alternative element ID, or None
            - rejection_reason: Reason for rejection, or None
        """
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotEscalationConfig:
    """
    Configuration for incremental snapshot limit escalation.

    When verification fails or elements are missing, the agent progressively
    increases the snapshot limit to capture more of the page.

    Attributes:
        enabled: If True (default), escalate limit on low element count or verification failure.
                 If False, always use limit_base without escalation.
        limit_base: Initial snapshot limit (default: 60)
        limit_step: Increment size for each escalation (default: 30)
                    Example with limit_step=30: 60 -> 90 -> 120 -> 150 -> 180 -> 200
                    Example with limit_step=50: 60 -> 110 -> 160 -> 200
        limit_max: Maximum snapshot limit (default: 200)

    Example:
        # Default: escalation enabled with step=30
        config = SnapshotEscalationConfig()  # 60 -> 90 -> 120 -> ... -> 200

        # Disable escalation (always use limit_base=60)
        config = SnapshotEscalationConfig(enabled=False)

        # Custom step size
        config = SnapshotEscalationConfig(limit_step=50)  # 60 -> 110 -> 160 -> 200

        # Larger initial limit, smaller steps
        config = SnapshotEscalationConfig(limit_base=100, limit_step=20, limit_max=180)
    """

    enabled: bool = True
    limit_base: int = 60
    limit_step: int = 30
    limit_max: int = 200


@dataclass(frozen=True)
class RetryConfig:
    """
    Retry configuration for verification and action execution.
    """

    verify_timeout_s: float = 10.0
    verify_poll_s: float = 0.5
    verify_max_attempts: int = 5
    executor_repair_attempts: int = 2
    max_replans: int = 1


@dataclass(frozen=True)
class RecoveryNavigationConfig:
    """
    Configuration for recovery navigation when agent gets off-track.

    The agent tracks the last known good URL (where verification passed)
    and can navigate back if subsequent steps fail.

    Attributes:
        enabled: If True, track last_known_good_url and attempt recovery navigation.
        max_recovery_attempts: Maximum navigation recovery attempts per step.
        recovery_predicates: Optional predicates to verify recovery succeeded.
    """

    enabled: bool = True
    max_recovery_attempts: int = 2
    track_successful_urls: bool = True


@dataclass(frozen=True)
class PlannerExecutorConfig:
    """
    High-level configuration for PlannerExecutorAgent.

    This config focuses on:
    - Snapshot escalation settings
    - Retry/verification settings
    - Vision fallback settings
    - Recovery navigation settings
    - Planner/Executor LLM settings
    - Tracing settings
    """

    # Snapshot escalation
    snapshot: SnapshotEscalationConfig = SnapshotEscalationConfig()

    # Retry configuration
    retry: RetryConfig = RetryConfig()

    # Vision fallback
    vision: VisionFallbackConfig = VisionFallbackConfig(
        enabled=True,
        max_vision_calls=3,
        trigger_requires_vision=True,
        trigger_canvas_or_low_actionables=True,
    )

    # CAPTCHA handling
    captcha: CaptchaConfig = CaptchaConfig()

    # Recovery navigation
    recovery: RecoveryNavigationConfig = RecoveryNavigationConfig()

    # Planner LLM settings
    planner_max_tokens: int = 2048
    planner_temperature: float = 0.0

    # Executor LLM settings
    executor_max_tokens: int = 96
    executor_temperature: float = 0.0

    # Stabilization (wait for DOM to settle after actions)
    stabilize_enabled: bool = True
    stabilize_poll_s: float = 0.35
    stabilize_max_attempts: int = 6

    # Pre-step verification (skip step if predicates already pass)
    pre_step_verification: bool = True

    # Tracing
    trace_screenshots: bool = True
    trace_screenshot_format: str = "jpeg"
    trace_screenshot_quality: int = 80


# ---------------------------------------------------------------------------
# Plan Models (Pydantic for validation)
# ---------------------------------------------------------------------------


class PredicateSpec(BaseModel):
    """Specification for a verification predicate."""

    predicate: str = Field(..., description="Predicate type: url_contains, exists, not_exists, any_of, all_of, element_count")
    args: list[Any] = Field(default_factory=list, description="Predicate arguments")

    class Config:
        extra = "allow"


class PlanStep(BaseModel):
    """A single step in the execution plan."""

    id: int = Field(..., description="Step ID (1-indexed, contiguous)")
    goal: str = Field(..., description="Human-readable goal for this step")
    action: str = Field(..., description="Action type: NAVIGATE, CLICK, TYPE_AND_SUBMIT, SCROLL")
    target: str | None = Field(None, description="URL for NAVIGATE action")
    intent: str | None = Field(None, description="Intent hint for CLICK action")
    input: str | None = Field(None, description="Text for TYPE_AND_SUBMIT action")
    verify: list[PredicateSpec] = Field(default_factory=list, description="Verification predicates")
    required: bool = Field(True, description="If True, step failure triggers replan")
    stop_if_true: bool = Field(False, description="If True, stop execution when verification passes")
    optional_substeps: list["PlanStep"] = Field(default_factory=list, description="Optional fallback steps")
    heuristic_hints: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Planner-generated hints for element selection",
    )

    class Config:
        extra = "allow"


class Plan(BaseModel):
    """Execution plan generated by the Planner."""

    task: str = Field(..., description="Original task description")
    notes: list[str] = Field(default_factory=list, description="Planner notes/assumptions")
    steps: list[PlanStep] = Field(..., description="Ordered execution steps")

    class Config:
        extra = "allow"


class ReplanPatch(BaseModel):
    """Patch for modifying an existing plan after step failure."""

    mode: Literal["patch"] = "patch"
    replace_steps: list[dict[str, Any]] = Field(..., description="Steps to replace by ID")


# ---------------------------------------------------------------------------
# SnapshotContext: Shared page state between Planner and Executor
# ---------------------------------------------------------------------------


@dataclass
class SnapshotContext:
    """
    Shared page state between Planner and Executor.

    This class enables snapshot sharing to avoid redundant captures and
    tracks metadata for vision fallback decisions.
    """

    snapshot: Snapshot
    compact_representation: str
    screenshot_base64: str | None
    captured_at: datetime
    limit_used: int
    snapshot_success: bool = True
    requires_vision: bool = False
    vision_reason: str | None = None

    def is_stale(self, max_age_seconds: float = 5.0) -> bool:
        """Check if snapshot is too old to reuse."""
        return (datetime.now() - self.captured_at).total_seconds() > max_age_seconds

    def should_use_vision(self) -> bool:
        """Check if executor should use vision fallback."""
        return not self.snapshot_success or self.requires_vision

    def digest(self) -> str:
        """Compute a digest for loop/change detection."""
        parts = [
            self.snapshot.url[:200] if self.snapshot.url else "",
            self.snapshot.title[:200] if self.snapshot.title else "",
            f"count:{len(self.snapshot.elements or [])}",
        ]
        for el in (self.snapshot.elements or [])[:100]:
            eid = getattr(el, "id", None)
            role = getattr(el, "role", None)
            text = (getattr(el, "text", None) or getattr(el, "name", None) or "")[:40]
            parts.append(f"{eid}|{role}|{text}")
        h = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
        return f"sha256:{h}"


def detect_snapshot_failure(snap: Snapshot) -> tuple[bool, str | None]:
    """
    Detect if snapshot is unusable and should trigger vision fallback.

    Returns:
        (should_use_vision, reason)
    """
    # Check explicit status field (tri-state: success, error, require_vision)
    status = getattr(snap, "status", "success")
    if status == "require_vision":
        return True, "require_vision"

    if status == "error":
        error = getattr(snap, "error", None)
        return True, f"snapshot_error:{error}"

    # Check diagnostics if available
    diag = getattr(snap, "diagnostics", None)
    if diag:
        confidence = getattr(diag, "confidence", 1.0)
        if confidence is not None and float(confidence) < 0.3:
            return True, "low_confidence"

        has_canvas = getattr(diag, "has_canvas", False)
        elements = getattr(snap, "elements", []) or []
        if has_canvas and len(elements) < 5:
            return True, "canvas_page"

    # Very few elements usually indicates a problem
    elements = getattr(snap, "elements", []) or []
    if len(elements) < 3:
        return True, "too_few_elements"

    return False, None


# ---------------------------------------------------------------------------
# Step Outcome
# ---------------------------------------------------------------------------


class StepStatus(str, Enum):
    """Status of a step execution."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    VISION_FALLBACK = "vision_fallback"


@dataclass
class StepOutcome:
    """Result of executing a single plan step."""

    step_id: int
    goal: str
    status: StepStatus
    action_taken: str | None = None
    verification_passed: bool = False
    used_vision: bool = False
    error: str | None = None
    duration_ms: int = 0
    url_before: str | None = None
    url_after: str | None = None


@dataclass
class RunOutcome:
    """Result of a complete agent run."""

    run_id: str
    task: str
    success: bool
    steps_completed: int
    steps_total: int
    replans_used: int
    step_outcomes: list[StepOutcome] = field(default_factory=list)
    total_duration_ms: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Predicate Builder
# ---------------------------------------------------------------------------


def build_predicate(spec: PredicateSpec | dict[str, Any]) -> Predicate:
    """
    Build a Predicate from a specification.

    Supports: url_contains, url_matches, exists, not_exists, element_count, any_of, all_of
    """
    if isinstance(spec, dict):
        name = spec.get("predicate", "")
        args = spec.get("args", [])
    else:
        name = spec.predicate
        args = spec.args

    if name == "url_contains":
        return url_contains(args[0])
    if name == "url_matches":
        pattern = args[0]
        if isinstance(pattern, str) and "/dp/" in pattern and not pattern.startswith("http"):
            return url_contains("/dp/")
        return url_matches(pattern)
    if name == "exists":
        return exists(args[0])
    if name == "not_exists":
        return not_exists(args[0])
    if name == "element_count":
        selector = args[0]
        min_count = args[1] if len(args) > 1 else 0
        max_count = args[2] if len(args) > 2 else None
        return element_count(selector, min_count=min_count, max_count=max_count)
    if name == "any_of":
        return any_of(*(build_predicate(p) for p in args))
    if name == "all_of":
        return all_of(*(build_predicate(p) for p in args))

    raise ValueError(f"Unsupported predicate: {name}")


# ---------------------------------------------------------------------------
# Plan Normalization and Validation
# ---------------------------------------------------------------------------


def normalize_plan(plan_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize plan dictionary to handle LLM output variations.

    This function handles common variations in LLM output:
    - url vs target field names
    - action aliases (click vs CLICK)
    - step id variations (string vs int)

    Args:
        plan_dict: Raw plan dictionary from LLM

    Returns:
        Normalized plan dictionary
    """
    # Normalize steps
    if "steps" in plan_dict:
        for step in plan_dict["steps"]:
            # Normalize action names to uppercase
            if "action" in step:
                action = step["action"].upper()
                # Handle common aliases
                action_aliases = {
                    "CLICK_ELEMENT": "CLICK",
                    "CLICK_BUTTON": "CLICK",
                    "CLICK_LINK": "CLICK",
                    "INPUT": "TYPE_AND_SUBMIT",
                    "TYPE_TEXT": "TYPE_AND_SUBMIT",
                    "ENTER_TEXT": "TYPE_AND_SUBMIT",
                    "GOTO": "NAVIGATE",
                    "GO_TO": "NAVIGATE",
                    "OPEN": "NAVIGATE",
                    "SCROLL_DOWN": "SCROLL",
                    "SCROLL_UP": "SCROLL",
                }
                step["action"] = action_aliases.get(action, action)

            # Normalize url -> target for NAVIGATE actions
            if "url" in step and "target" not in step:
                step["target"] = step.pop("url")

            # Ensure step id is int
            if "id" in step and isinstance(step["id"], str):
                try:
                    step["id"] = int(step["id"])
                except ValueError:
                    pass

            # Normalize optional_substeps recursively
            if "optional_substeps" in step:
                for substep in step["optional_substeps"]:
                    if "action" in substep:
                        substep["action"] = substep["action"].upper()
                    if "url" in substep and "target" not in substep:
                        substep["target"] = substep.pop("url")

    return plan_dict


def validate_plan_smoothness(plan: "Plan") -> list[str]:
    """
    Validate plan quality and smoothness.

    Checks for common issues that indicate a low-quality plan:
    - Missing verification predicates
    - Consecutive same actions
    - Empty or too short plans
    - Missing required fields

    Args:
        plan: Parsed Plan object

    Returns:
        List of warning strings (empty if plan is smooth)
    """
    warnings: list[str] = []

    # Check for empty plan
    if not plan.steps:
        warnings.append("Plan has no steps")
        return warnings

    # Check for very short plans (might be incomplete)
    if len(plan.steps) < 2:
        warnings.append("Plan has only one step - might be incomplete")

    # Check each step
    prev_action = None
    for i, step in enumerate(plan.steps):
        # Check for missing verification
        if not step.verify and step.required:
            warnings.append(f"Step {step.id} has no verification predicates")

        # Check for consecutive same actions (might indicate loop)
        if step.action == prev_action and step.action == "CLICK":
            warnings.append(f"Steps {step.id - 1} and {step.id} both use {step.action}")

        # Check for NAVIGATE without target
        if step.action == "NAVIGATE" and not step.target:
            warnings.append(f"Step {step.id} is NAVIGATE but has no target URL")

        # Check for CLICK without intent
        if step.action == "CLICK" and not step.intent:
            warnings.append(f"Step {step.id} is CLICK but has no intent hint")

        # Check for TYPE_AND_SUBMIT without input
        if step.action == "TYPE_AND_SUBMIT" and not step.input:
            warnings.append(f"Step {step.id} is TYPE_AND_SUBMIT but has no input")

        prev_action = step.action

    return warnings


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------


def build_planner_prompt(
    task: str,
    *,
    start_url: str | None = None,
    site_type: str = "general",
    auth_state: str = "unknown",
    strict: bool = False,
    schema_errors: str | None = None,
) -> tuple[str, str]:
    """
    Build system and user prompts for the Planner LLM.

    Returns:
        (system_prompt, user_prompt)
    """
    strict_note = "\nReturn ONLY a JSON object. No explanations, no code fences.\n" if strict else ""
    schema_note = f"\nSchema errors from last attempt:\n{schema_errors}\n" if schema_errors else ""

    system = f"""You are the PLANNER. Output a JSON execution plan for the web automation task.
{strict_note}
Your output must be a valid JSON object with:
- task: string (the task description)
- notes: list of strings (assumptions, constraints)
- steps: list of step objects

Each step must have:
- id: int (1-indexed, contiguous)
- goal: string (human-readable goal)
- action: string (NAVIGATE, CLICK, TYPE_AND_SUBMIT, or SCROLL)
- verify: list of predicate specs

Available predicates:
- url_contains(substring): URL contains the given string
- url_matches(pattern): URL matches regex pattern
- exists(selector): Element matching selector exists
- not_exists(selector): Element matching selector does not exist
- element_count(selector, min, max): Element count within range
- any_of(predicates...): Any predicate is true
- all_of(predicates...): All predicates are true

Selectors: role=button, role=link, text~'text', role=textbox, etc.

Return ONLY valid JSON. No prose, no code fences."""

    user = f"""Task: {task}
{schema_note}
Starting URL: {start_url or "browser's current page"}
Site type: {site_type}
Auth state: {auth_state}

Output a JSON plan to accomplish this task."""

    return system, user


def build_executor_prompt(
    goal: str,
    intent: str | None,
    compact_context: str,
) -> tuple[str, str]:
    """
    Build system and user prompts for the Executor LLM.

    Returns:
        (system_prompt, user_prompt)
    """
    intent_line = f"Intent: {intent}\n" if intent else ""

    system = """You are a careful web automation executor.
You must respond with exactly ONE action in this format:
- CLICK(<id>)
- TYPE(<id>, "text")
- PRESS('key')
- SCROLL(direction)
- FINISH()

Output only the action. No explanations."""

    user = f"""You are controlling a browser via element IDs.

Goal: {goal}
{intent_line}
Elements (ID|role|text|importance|clickable|...):
{compact_context}

Return ONLY the action to take."""

    return system, user


# ---------------------------------------------------------------------------
# PlannerExecutorAgent
# ---------------------------------------------------------------------------


class PlannerExecutorAgent:
    """
    Two-tier agent architecture with Planner and Executor models.

    The Planner (typically 7B+ parameters) generates JSON execution plans
    with predicates. The Executor (3B-7B parameters) executes each step
    using a snapshot-first approach.

    Features:
    - Incremental limit escalation for reliable element capture
    - Vision fallback for canvas pages or low-confidence snapshots
    - SnapshotContext sharing to avoid redundant captures
    - Full tracing integration for Predicate Studio visualization
    - Replanning on step failure
    - Pre-step verification: skip execution if predicates already pass
    - Optional substeps: fallback steps for edge cases (scroll, close drawer)
    - Plan normalization: handles LLM output variations (url vs target, etc.)
    - Plan smoothness validation: quality checks on generated plans
    - Pluggable IntentHeuristics: domain-specific element selection without LLM
    - ExecutorOverride: validate/override executor element choices
    - Recovery navigation: track last known good URL for off-track recovery

    Example:
        >>> from predicate.agents import PlannerExecutorAgent, PlannerExecutorConfig
        >>> from predicate.llm_provider import OpenAIProvider
        >>>
        >>> planner = OpenAIProvider(model="gpt-4o")
        >>> executor = OpenAIProvider(model="gpt-4o-mini")
        >>>
        >>> agent = PlannerExecutorAgent(
        ...     planner=planner,
        ...     executor=executor,
        ...     config=PlannerExecutorConfig(),
        ... )
        >>>
        >>> async with AsyncPredicateBrowser() as browser:
        ...     runtime = AgentRuntime.from_browser(browser)
        ...     result = await agent.run(
        ...         runtime=runtime,
        ...         task="Search for 'laptop' on Amazon and add first result to cart",
        ...         start_url="https://amazon.com",
        ...     )
        ...     print(f"Success: {result.success}")

    Example with IntentHeuristics:
        >>> class EcommerceHeuristics:
        ...     def find_element_for_intent(self, intent, elements, url, goal):
        ...         if "add to cart" in intent.lower():
        ...             for el in elements:
        ...                 if "add to cart" in (getattr(el, "text", "") or "").lower():
        ...                     return getattr(el, "id", None)
        ...         return None  # Fall back to LLM
        ...
        ...     def priority_order(self):
        ...         return ["add_to_cart", "checkout"]
        >>>
        >>> agent = PlannerExecutorAgent(
        ...     planner=planner,
        ...     executor=executor,
        ...     intent_heuristics=EcommerceHeuristics(),
        ... )
    """

    def __init__(
        self,
        *,
        planner: LLMProvider,
        executor: LLMProvider,
        vision_executor: LLMProvider | None = None,
        vision_verifier: LLMProvider | None = None,
        config: PlannerExecutorConfig | None = None,
        tracer: Tracer | None = None,
        context_formatter: Callable[[Snapshot, str], str] | None = None,
        intent_heuristics: IntentHeuristics | None = None,
        executor_override: ExecutorOverride | None = None,
    ) -> None:
        """
        Initialize PlannerExecutorAgent.

        Args:
            planner: LLM for generating plans (recommend 7B+ model)
            executor: LLM for executing steps (3B-7B model)
            vision_executor: Optional LLM for vision-based action selection
            vision_verifier: Optional LLM for vision-based verification
            config: Agent configuration
            tracer: Tracer for Predicate Studio visualization
            context_formatter: Custom function to format snapshot for LLM
            intent_heuristics: Optional pluggable heuristics for domain-specific
                element selection. When provided, the agent tries heuristics
                before falling back to the LLM executor.
            executor_override: Optional hook to validate or override executor
                element choices before action execution.
        """
        self.planner = planner
        self.executor = executor
        self.vision_executor = vision_executor
        self.vision_verifier = vision_verifier
        self.config = config or PlannerExecutorConfig()
        self.tracer = tracer
        self._context_formatter = context_formatter
        self._intent_heuristics = intent_heuristics
        self._executor_override = executor_override

        # State tracking
        self._current_plan: Plan | None = None
        self._step_index: int = 0
        self._replans_used: int = 0
        self._vision_calls: int = 0
        self._snapshot_context: SnapshotContext | None = None
        self._run_id: str | None = None
        self._last_known_good_url: str | None = None

        # Recovery state (initialized per-run)
        self._recovery_state: RecoveryState | None = None

        # Composable heuristics (wraps static heuristics with dynamic hints)
        self._composable_heuristics: ComposableHeuristics | None = None

        # Current automation task (for run-level context)
        self._current_task: AutomationTask | None = None

    def _format_context(self, snap: Snapshot, goal: str) -> str:
        """Format snapshot for LLM context."""
        if self._context_formatter is not None:
            return self._context_formatter(snap, goal)

        # Default compact format
        lines = []
        for el in snap.elements[:120]:
            eid = getattr(el, "id", "?")
            role = getattr(el, "role", "")
            text = (getattr(el, "text", "") or "")[:50]
            importance = getattr(el, "importance", 0)
            clickable = 1 if getattr(el, "clickable", False) else 0
            lines.append(f"{eid}|{role}|{text}|{importance}|{clickable}")
        return "\n".join(lines)

    async def _attempt_recovery(
        self,
        runtime: AgentRuntime,
    ) -> bool:
        """
        Attempt to recover to the last known good state.

        Navigates back to the checkpoint URL and verifies we're in a recoverable state.

        Args:
            runtime: AgentRuntime instance

        Returns:
            True if recovery succeeded, False otherwise
        """
        if self._recovery_state is None:
            return False

        checkpoint = self._recovery_state.consume_recovery_attempt()
        if checkpoint is None:
            return False

        # Emit recovery event
        if self.tracer:
            self.tracer.emit(
                "recovery_attempt",
                {
                    "target_url": checkpoint.url,
                    "target_step_index": checkpoint.step_index,
                    "attempt": self._recovery_state.recovery_attempts_used,
                },
            )

        try:
            # Navigate to checkpoint URL
            await runtime.goto(checkpoint.url)

            # Wait for page to settle
            await runtime.stabilize()

            # Verify we're at the expected URL (basic check)
            snapshot = await runtime.snapshot()
            current_url = snapshot.url or ""

            # Check if URL matches (allowing for minor variations)
            url_matches = (
                checkpoint.url in current_url
                or current_url in checkpoint.url
                or checkpoint.url.rstrip("/") == current_url.rstrip("/")
            )

            if url_matches:
                if self.tracer:
                    self.tracer.emit(
                        "recovery_success",
                        {
                            "recovered_to_url": checkpoint.url,
                            "actual_url": current_url,
                        },
                    )
                return True
            else:
                if self.tracer:
                    self.tracer.emit(
                        "recovery_url_mismatch",
                        {
                            "expected_url": checkpoint.url,
                            "actual_url": current_url,
                        },
                    )
                return False

        except Exception as e:
            if self.tracer:
                self.tracer.emit(
                    "recovery_failed",
                    {
                        "error": str(e),
                        "checkpoint_url": checkpoint.url,
                    },
                )
            return False

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from LLM response, handling code fences and think tags."""
        # Remove <think>...</think> tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Remove code fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text)

        # Find JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())

        raise ValueError("No JSON object found in response")

    def _parse_action(self, text: str) -> tuple[str, list[Any]]:
        """Parse action from executor response."""
        text = text.strip()

        # CLICK(<id>)
        match = re.match(r"CLICK\((\d+)\)", text)
        if match:
            return "CLICK", [int(match.group(1))]

        # TYPE(<id>, "text")
        match = re.match(r'TYPE\((\d+),\s*["\'](.+?)["\']\)', text)
        if match:
            return "TYPE", [int(match.group(1)), match.group(2)]

        # PRESS('key')
        match = re.match(r"PRESS\(['\"](.+?)['\"]\)", text)
        if match:
            return "PRESS", [match.group(1)]

        # SCROLL(direction)
        match = re.match(r"SCROLL\((\w+)\)", text)
        if match:
            return "SCROLL", [match.group(1)]

        # FINISH()
        if text.startswith("FINISH"):
            return "FINISH", []

        return "UNKNOWN", [text]

    # -------------------------------------------------------------------------
    # Tracing Helpers
    # -------------------------------------------------------------------------

    def _emit_run_start(self, task: str, start_url: str | None) -> None:
        """Emit run_start trace event."""
        if self.tracer is None:
            return
        try:
            self.tracer.emit_run_start(
                agent="PlannerExecutorAgent",
                llm_model=f"{self.planner.model_name} / {self.executor.model_name}",
                config={
                    "task": task,
                    "start_url": start_url,
                    "snapshot_limit_base": self.config.snapshot.limit_base,
                    "snapshot_limit_max": self.config.snapshot.limit_max,
                    "max_replans": self.config.retry.max_replans,
                    "vision_enabled": self.config.vision.enabled,
                },
            )
        except Exception:
            pass

    def _emit_step_start(
        self,
        step: PlanStep,
        step_index: int,
        pre_url: str | None,
    ) -> str:
        """Emit step_start trace event and return step_id."""
        step_id = f"step-{step_index}"
        if self.tracer is None:
            return step_id
        try:
            self.tracer.emit_step_start(
                step_id=step_id,
                step_index=step_index,
                goal=step.goal,
                attempt=0,
                pre_url=pre_url,
            )
        except Exception:
            pass
        return step_id

    def _emit_snapshot(
        self,
        ctx: SnapshotContext,
        step_id: str,
        step_index: int,
    ) -> None:
        """Emit snapshot trace event with screenshot for Studio visualization."""
        if self.tracer is None:
            return
        try:
            data = TraceEventBuilder.build_snapshot_event(
                ctx.snapshot,
                include_all_elements=True,
                step_index=step_index,
            )

            # Add screenshot if available
            if ctx.screenshot_base64:
                # Handle data URL format
                screenshot_b64 = ctx.screenshot_base64
                if screenshot_b64.startswith("data:image"):
                    screenshot_b64 = screenshot_b64.split(",", 1)[1] if "," in screenshot_b64 else screenshot_b64
                data["screenshot_base64"] = screenshot_b64
                data["screenshot_format"] = self.config.trace_screenshot_format

            self.tracer.emit("snapshot", data=data, step_id=step_id)
        except Exception:
            pass

    def _emit_step_end(
        self,
        step_id: str,
        step_index: int,
        step: PlanStep,
        outcome: StepOutcome,
        pre_url: str | None,
        post_url: str | None,
        llm_response: str | None,
        snapshot_digest: str | None,
    ) -> None:
        """Emit step_end trace event."""
        if self.tracer is None:
            return
        try:
            step_end_data = {
                "v": 1,
                "step_id": step_id,
                "step_index": step_index,
                "goal": step.goal,
                "attempt": 0,
                "pre": {
                    "url": pre_url or "",
                    "snapshot_digest": snapshot_digest,
                },
                "llm": {
                    "response_text": llm_response,
                    "model": self.executor.model_name,
                },
                "exec": {
                    "action": outcome.action_taken or "",
                    "success": outcome.status == StepStatus.SUCCESS,
                    "error": outcome.error,
                    "used_vision": outcome.used_vision,
                },
                "post": {
                    "url": post_url or "",
                },
                "verify": {
                    "passed": outcome.verification_passed,
                    "predicates": [v.model_dump() for v in step.verify] if step.verify else [],
                },
            }
            self.tracer.emit("step_end", step_end_data, step_id=step_id)
        except Exception:
            pass

    def _emit_plan_event(self, plan: Plan, raw_output: str) -> None:
        """Emit plan_generated trace event."""
        if self.tracer is None:
            return
        try:
            self.tracer.emit(
                "plan_generated",
                {
                    "task": plan.task,
                    "step_count": len(plan.steps),
                    "steps": [{"id": s.id, "goal": s.goal, "action": s.action} for s in plan.steps],
                    "raw_output": raw_output[:2000],  # Truncate for storage
                    "planner_model": self.planner.model_name,
                },
            )
        except Exception:
            pass

    def _emit_replan_event(
        self,
        failed_step: PlanStep,
        failure_reason: str,
        new_plan: Plan,
        raw_output: str,
    ) -> None:
        """Emit replan trace event."""
        if self.tracer is None:
            return
        try:
            self.tracer.emit(
                "replan",
                {
                    "failed_step_id": failed_step.id,
                    "failed_step_goal": failed_step.goal,
                    "failure_reason": failure_reason,
                    "new_step_count": len(new_plan.steps),
                    "raw_output": raw_output[:2000],
                    "replans_used": self._replans_used,
                },
            )
        except Exception:
            pass

    def _emit_run_end(self, outcome: RunOutcome) -> None:
        """Emit run_end trace event."""
        if self.tracer is None:
            return
        try:
            self.tracer.set_final_status("success" if outcome.success else "failure")
            self.tracer.emit_run_end(
                steps=outcome.steps_total,
                status="success" if outcome.success else "failure",
            )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Snapshot with Escalation
    # -------------------------------------------------------------------------

    async def _snapshot_with_escalation(
        self,
        runtime: AgentRuntime,
        goal: str,
        capture_screenshot: bool = True,
    ) -> SnapshotContext:
        """
        Capture snapshot with incremental limit escalation.

        Progressively increases snapshot limit if elements are missing or
        confidence is low. Escalation can be disabled via config.

        When escalation is disabled (config.snapshot.enabled=False), only a
        single snapshot at limit_base is captured.
        """
        cfg = self.config.snapshot
        current_limit = cfg.limit_base
        max_limit = cfg.limit_max if cfg.enabled else cfg.limit_base  # Disable escalation if not enabled
        last_snap: Snapshot | None = None
        last_compact: str = ""
        screenshot_b64: str | None = None
        requires_vision = False
        vision_reason: str | None = None

        while current_limit <= max_limit:
            try:
                snap = await runtime.snapshot(
                    limit=current_limit,
                    screenshot=capture_screenshot,
                    goal=goal,
                )
                if snap is None:
                    if not cfg.enabled:
                        break  # No escalation, exit immediately
                    current_limit = min(current_limit + cfg.limit_step, max_limit + 1)
                    continue

                last_snap = snap

                # Extract screenshot
                if capture_screenshot:
                    raw_screenshot = getattr(snap, "screenshot", None)
                    if raw_screenshot:
                        screenshot_b64 = raw_screenshot

                # Check for vision fallback
                needs_vision, reason = detect_snapshot_failure(snap)
                if needs_vision:
                    requires_vision = True
                    vision_reason = reason
                    break

                # Format context
                compact = self._format_context(snap, goal)
                last_compact = compact

                # If escalation disabled, we're done after first successful snapshot
                if not cfg.enabled:
                    break

                # Check element count - if sufficient, no need to escalate
                elements = getattr(snap, "elements", []) or []
                if len(elements) >= 10:
                    break

                # Escalate limit
                if current_limit < max_limit:
                    current_limit = min(current_limit + cfg.limit_step, max_limit)
                else:
                    break

            except Exception:
                if not cfg.enabled:
                    break  # No escalation on error
                current_limit = min(current_limit + cfg.limit_step, max_limit + 1)

        # Fallback for failed capture
        if last_snap is None:
            last_snap = Snapshot(
                status="error",
                elements=[],
                url="",
                title="",
                error="snapshot_capture_failed",
            )
            requires_vision = True
            vision_reason = "snapshot_capture_failed"

        return SnapshotContext(
            snapshot=last_snap,
            compact_representation=last_compact,
            screenshot_base64=screenshot_b64,
            captured_at=datetime.now(),
            limit_used=current_limit,
            snapshot_success=not requires_vision,
            requires_vision=requires_vision,
            vision_reason=vision_reason,
        )

    # -------------------------------------------------------------------------
    # Plan Generation
    # -------------------------------------------------------------------------

    async def plan(
        self,
        task: str,
        *,
        start_url: str | None = None,
        max_attempts: int = 2,
    ) -> Plan:
        """
        Generate execution plan for the given task.

        Args:
            task: Task description
            start_url: Starting URL
            max_attempts: Maximum planning attempts

        Returns:
            Plan object with steps
        """
        last_output = ""
        last_errors = ""

        for attempt in range(1, max_attempts + 1):
            max_tokens = self.config.planner_max_tokens if attempt == 1 else self.config.planner_max_tokens + 512

            sys_prompt, user_prompt = build_planner_prompt(
                task,
                start_url=start_url,
                strict=(attempt > 1),
                schema_errors=last_errors or None,
            )

            resp = self.planner.generate(
                sys_prompt,
                user_prompt,
                temperature=self.config.planner_temperature,
                max_new_tokens=max_tokens,
            )
            last_output = resp.content

            try:
                plan_dict = self._extract_json(resp.content)

                # Normalize plan to handle LLM output variations
                plan_dict = normalize_plan(plan_dict)

                plan = Plan.model_validate(plan_dict)

                # Validate plan smoothness (warnings only, don't fail)
                warnings = validate_plan_smoothness(plan)
                if warnings and self.tracer:
                    try:
                        self.tracer.emit("plan_warnings", {"warnings": warnings})
                    except Exception:
                        pass

                self._current_plan = plan
                self._step_index = 0

                # Emit trace event
                self._emit_plan_event(plan, last_output)

                return plan

            except Exception as e:
                last_errors = str(e)
                continue

        raise RuntimeError(f"Planner failed after {max_attempts} attempts. Last output:\n{last_output[:500]}")

    async def replan(
        self,
        task: str,
        failed_step: PlanStep,
        failure_reason: str,
        *,
        max_attempts: int = 2,
    ) -> Plan:
        """
        Generate patched plan after step failure.

        Args:
            task: Original task description
            failed_step: The step that failed
            failure_reason: Reason for failure
            max_attempts: Maximum replan attempts

        Returns:
            Updated Plan
        """
        if self._current_plan is None:
            raise RuntimeError("Cannot replan without an existing plan")

        self._replans_used += 1
        last_output = ""

        system = """You are the PLANNER. Output a JSON patch to edit an existing plan.
Edit ONLY the failed step and optionally the next step.
Return ONLY a JSON object with mode="patch" and replace_steps array."""

        user = f"""Task: {task}

Failure:
- Step ID: {failed_step.id}
- Step goal: {failed_step.goal}
- Reason: {failure_reason}

Return JSON patch:
{{
  "mode": "patch",
  "replace_steps": [
    {{
      "id": {failed_step.id},
      "step": {{ "id": {failed_step.id}, "goal": "...", "action": "...", "verify": [...] }}
    }}
  ]
}}"""

        for attempt in range(1, max_attempts + 1):
            resp = self.planner.generate(
                system,
                user,
                temperature=self.config.planner_temperature,
                max_new_tokens=1024,
            )
            last_output = resp.content

            try:
                patch_dict = self._extract_json(resp.content)

                # Apply patch
                steps = list(self._current_plan.steps)
                for item in patch_dict.get("replace_steps", []):
                    step_id = item.get("id")
                    step_data = item.get("step")
                    if step_id and step_data:
                        for i, s in enumerate(steps):
                            if s.id == step_id:
                                steps[i] = PlanStep.model_validate(step_data)
                                break

                new_plan = Plan(
                    task=self._current_plan.task,
                    notes=self._current_plan.notes,
                    steps=steps,
                )
                self._current_plan = new_plan

                # Emit trace event
                self._emit_replan_event(failed_step, failure_reason, new_plan, last_output)

                return new_plan

            except Exception:
                continue

        raise RuntimeError(f"Replan failed after {max_attempts} attempts")

    # -------------------------------------------------------------------------
    # Step Execution
    # -------------------------------------------------------------------------

    async def _check_pre_step_verification(
        self,
        runtime: AgentRuntime,
        step: PlanStep,
    ) -> bool:
        """
        Check if step verification predicates already pass before executing.

        This optimization skips step execution if the desired state is already
        achieved (e.g., already on checkout page when step goal is "go to checkout").

        Returns:
            True if all predicates pass (step can be skipped), False otherwise
        """
        if not step.verify:
            return False

        for verify_spec in step.verify:
            try:
                pred = build_predicate(verify_spec)
                # Quick check without retries
                snap = await runtime.snapshot(limit=30, screenshot=False, goal=step.goal)
                if snap is None:
                    return False
                if not pred.evaluate(snap):
                    return False
            except Exception:
                return False

        return True

    async def _try_intent_heuristics(
        self,
        step: PlanStep,
        elements: list[Any],
        url: str,
    ) -> int | None:
        """
        Try pluggable intent heuristics to find element without LLM.

        Returns:
            Element ID if heuristics found a match, None otherwise
        """
        if self._intent_heuristics is None:
            return None

        if not step.intent:
            return None

        try:
            element_id = self._intent_heuristics.find_element_for_intent(
                intent=step.intent,
                elements=elements,
                url=url,
                goal=step.goal,
            )
            return element_id
        except Exception:
            return None

    async def _execute_optional_substeps(
        self,
        substeps: list[PlanStep],
        runtime: AgentRuntime,
        parent_step_index: int,
    ) -> list[StepOutcome]:
        """
        Execute optional substeps (fallback steps for edge cases).

        Optional substeps are executed when the main step's verification fails.
        They handle scenarios like scroll-to-reveal, closing drawers, etc.

        Returns:
            List of substep outcomes
        """
        outcomes: list[StepOutcome] = []

        for i, substep in enumerate(substeps):
            substep_index = parent_step_index * 100 + i + 1  # e.g., 101, 102 for step 1's substeps

            # Execute substep with simplified logic
            try:
                ctx = await self._snapshot_with_escalation(
                    runtime,
                    goal=substep.goal,
                    capture_screenshot=False,
                )

                # Determine element and action
                action_type = substep.action
                element_id: int | None = None

                if action_type in ("CLICK", "TYPE_AND_SUBMIT"):
                    # Try heuristics first
                    elements = getattr(ctx.snapshot, "elements", []) or []
                    url = getattr(ctx.snapshot, "url", "") or ""
                    element_id = await self._try_intent_heuristics(substep, elements, url)

                    if element_id is None:
                        # Fall back to executor
                        sys_prompt, user_prompt = build_executor_prompt(
                            substep.goal,
                            substep.intent,
                            ctx.compact_representation,
                        )
                        resp = self.executor.generate(
                            sys_prompt,
                            user_prompt,
                            temperature=self.config.executor_temperature,
                            max_new_tokens=self.config.executor_max_tokens,
                        )
                        parsed_action, parsed_args = self._parse_action(resp.content)
                        if parsed_action == "CLICK" and parsed_args:
                            element_id = parsed_args[0]

                # Execute the action
                if action_type == "CLICK" and element_id is not None:
                    await runtime.click(element_id)
                elif action_type == "SCROLL":
                    direction = "down"  # Default
                    await runtime.scroll(direction)
                elif action_type == "NAVIGATE" and substep.target:
                    await runtime.goto(substep.target)

                outcomes.append(StepOutcome(
                    step_id=substep.id,
                    goal=substep.goal,
                    status=StepStatus.SUCCESS,
                    action_taken=f"{action_type}({element_id})" if element_id else action_type,
                    verification_passed=True,
                ))

            except Exception as e:
                outcomes.append(StepOutcome(
                    step_id=substep.id,
                    goal=substep.goal,
                    status=StepStatus.FAILED,
                    error=str(e),
                ))

        return outcomes

    async def _execute_step(
        self,
        step: PlanStep,
        runtime: AgentRuntime,
        step_index: int,
    ) -> StepOutcome:
        """Execute a single plan step with pre-verification, heuristics, and optional substeps."""
        start_time = time.time()
        pre_url = await runtime.get_url() if hasattr(runtime, "get_url") else None
        step_id = self._emit_step_start(step, step_index, pre_url)

        llm_response: str | None = None
        action_taken: str | None = None
        used_vision = False
        used_heuristics = False
        error: str | None = None
        verification_passed = False

        try:
            # Pre-step verification check: skip if predicates already pass
            if self.config.pre_step_verification and step.verify:
                if await self._check_pre_step_verification(runtime, step):
                    # Step already satisfied, skip execution
                    verification_passed = True
                    action_taken = "SKIPPED(pre_verification_passed)"

                    # Track successful URL for recovery
                    if self.config.recovery.track_successful_urls:
                        self._last_known_good_url = pre_url

                    outcome = StepOutcome(
                        step_id=step.id,
                        goal=step.goal,
                        status=StepStatus.SKIPPED,
                        action_taken=action_taken,
                        verification_passed=True,
                        duration_ms=int((time.time() - start_time) * 1000),
                        url_before=pre_url,
                        url_after=pre_url,
                    )

                    self._emit_step_end(
                        step_id=step_id,
                        step_index=step_index,
                        step=step,
                        outcome=outcome,
                        pre_url=pre_url,
                        post_url=pre_url,
                        llm_response=None,
                        snapshot_digest=None,
                    )

                    return outcome

            # Capture snapshot with escalation
            ctx = await self._snapshot_with_escalation(
                runtime,
                goal=step.goal,
                capture_screenshot=self.config.trace_screenshots,
            )
            self._snapshot_context = ctx

            # Emit snapshot trace
            self._emit_snapshot(ctx, step_id, step_index)

            # Check for vision fallback
            if ctx.should_use_vision() and self.config.vision.enabled:
                if self._vision_calls < self.config.vision.max_vision_calls:
                    self._vision_calls += 1
                    used_vision = True
                    # Vision execution would go here
                    # For now, fall through to standard executor

            # Determine element and action
            action_type = step.action
            element_id: int | None = None

            if action_type in ("CLICK", "TYPE_AND_SUBMIT"):
                # Try intent heuristics first (if available)
                elements = getattr(ctx.snapshot, "elements", []) or []
                url = getattr(ctx.snapshot, "url", "") or ""
                element_id = await self._try_intent_heuristics(step, elements, url)

                if element_id is not None:
                    used_heuristics = True
                    action_taken = f"{action_type}({element_id}) [heuristic]"
                else:
                    # Fall back to LLM executor
                    sys_prompt, user_prompt = build_executor_prompt(
                        step.goal,
                        step.intent,
                        ctx.compact_representation,
                    )

                    resp = self.executor.generate(
                        sys_prompt,
                        user_prompt,
                        temperature=self.config.executor_temperature,
                        max_new_tokens=self.config.executor_max_tokens,
                    )
                    llm_response = resp.content

                    # Parse action
                    parsed_action, parsed_args = self._parse_action(resp.content)
                    action_type = parsed_action
                    if parsed_action == "CLICK" and parsed_args:
                        element_id = parsed_args[0]
                    elif parsed_action == "TYPE" and len(parsed_args) >= 2:
                        element_id = parsed_args[0]

                    action_taken = f"{action_type}({', '.join(str(a) for a in parsed_args)})"

                    # Apply executor override if configured
                    if self._executor_override and element_id is not None:
                        try:
                            is_valid, override_id, reason = self._executor_override.validate_choice(
                                element_id=element_id,
                                action=action_type,
                                elements=elements,
                                goal=step.goal,
                            )
                            if not is_valid:
                                if override_id is not None:
                                    element_id = override_id
                                    action_taken = f"{action_type}({element_id}) [override]"
                                else:
                                    error = f"Executor override rejected: {reason}"
                        except Exception:
                            pass  # Ignore override errors

            elif action_type == "NAVIGATE":
                action_taken = f"NAVIGATE({step.target})"
            elif action_type == "SCROLL":
                action_taken = "SCROLL(down)"
            else:
                action_taken = action_type

            # Execute action via runtime
            if error is None:
                if action_type == "CLICK" and element_id is not None:
                    await runtime.click(element_id)
                elif action_type == "TYPE" and element_id is not None:
                    text = step.input or ""
                    await runtime.type(element_id, text)
                elif action_type == "TYPE_AND_SUBMIT" and element_id is not None:
                    text = step.input or ""
                    await runtime.type(element_id, text)
                    await runtime.press("Enter")
                elif action_type == "PRESS":
                    key = "Enter"  # Default
                    await runtime.press(key)
                elif action_type == "SCROLL":
                    await runtime.scroll("down")
                elif action_type == "NAVIGATE" and step.target:
                    await runtime.goto(step.target)
                elif action_type == "FINISH":
                    pass  # No action needed
                elif action_type not in ("CLICK", "TYPE", "TYPE_AND_SUBMIT") or element_id is None:
                    if action_type in ("CLICK", "TYPE", "TYPE_AND_SUBMIT"):
                        error = f"No element ID for {action_type}"
                    else:
                        error = f"Unknown action: {action_type}"

            # Record action for tracing
            if action_taken:
                await runtime.record_action(action_taken)

            # Run verifications
            if step.verify and error is None:
                verification_passed = await self._verify_step(runtime, step)

                # If verification failed and we have optional substeps, try them
                if not verification_passed and step.optional_substeps:
                    substep_outcomes = await self._execute_optional_substeps(
                        step.optional_substeps,
                        runtime,
                        step_index,
                    )
                    # Re-run verification after substeps
                    if any(o.status == StepStatus.SUCCESS for o in substep_outcomes):
                        verification_passed = await self._verify_step(runtime, step)
            else:
                verification_passed = error is None

            # Track successful URL for recovery
            if verification_passed and self.config.recovery.track_successful_urls:
                post_url = await runtime.get_url() if hasattr(runtime, "get_url") else None
                if post_url:
                    self._last_known_good_url = post_url

        except Exception as e:
            error = str(e)
            verification_passed = False

        # Build outcome
        post_url = await runtime.get_url() if hasattr(runtime, "get_url") else None
        duration_ms = int((time.time() - start_time) * 1000)

        status = StepStatus.SUCCESS if verification_passed else StepStatus.FAILED
        if used_vision:
            status = StepStatus.VISION_FALLBACK

        outcome = StepOutcome(
            step_id=step.id,
            goal=step.goal,
            status=status,
            action_taken=action_taken,
            verification_passed=verification_passed,
            used_vision=used_vision,
            error=error,
            duration_ms=duration_ms,
            url_before=pre_url,
            url_after=post_url,
        )

        # Emit step_end trace
        self._emit_step_end(
            step_id=step_id,
            step_index=step_index,
            step=step,
            outcome=outcome,
            pre_url=pre_url,
            post_url=post_url,
            llm_response=llm_response,
            snapshot_digest=self._snapshot_context.digest() if self._snapshot_context else None,
        )

        return outcome

    async def _verify_step(
        self,
        runtime: AgentRuntime,
        step: PlanStep,
    ) -> bool:
        """Run verification predicates with limit escalation."""
        cfg = self.config.retry

        for verify_spec in step.verify:
            pred = build_predicate(verify_spec)
            label = f"verify_{step.id}"

            # Use runtime's check with eventually for required verifications
            if step.required:
                ok = await runtime.check(pred, label=label, required=True).eventually(
                    timeout_s=cfg.verify_timeout_s,
                    poll_s=cfg.verify_poll_s,
                    max_snapshot_attempts=cfg.verify_max_attempts,
                )
            else:
                ok = runtime.assert_(pred, label=label, required=False)

            if not ok:
                return False

        return True

    # -------------------------------------------------------------------------
    # Main Run Loop
    # -------------------------------------------------------------------------

    async def run(
        self,
        runtime: AgentRuntime,
        task: AutomationTask | str,
        *,
        start_url: str | None = None,
        run_id: str | None = None,
    ) -> RunOutcome:
        """
        Execute complete task with planning, execution, and replanning.

        Args:
            runtime: AgentRuntime instance
            task: AutomationTask instance or task description string
            start_url: Starting URL (only needed if task is a string)
            run_id: Run ID for tracing (optional)

        Returns:
            RunOutcome with execution results

        Example with AutomationTask:
            task = AutomationTask(
                task_id="purchase-laptop",
                starting_url="https://amazon.com",
                task="Find a laptop under $1000 and add to cart",
                category=TaskCategory.TRANSACTION,
            )
            result = await agent.run(runtime, task)

        Example with string:
            result = await agent.run(
                runtime,
                "Search for laptops",
                start_url="https://amazon.com",
            )
        """
        # Normalize task to AutomationTask
        if isinstance(task, str):
            if start_url is None:
                raise ValueError("start_url is required when task is a string")
            automation_task = AutomationTask(
                task_id=run_id or str(uuid.uuid4()),
                starting_url=start_url,
                task=task,
            )
        else:
            automation_task = task
            if start_url is None:
                start_url = automation_task.starting_url

        self._current_task = automation_task
        self._run_id = run_id or automation_task.task_id
        self._replans_used = 0
        self._vision_calls = 0
        start_time = time.time()

        # Initialize recovery state if enabled
        if automation_task.enable_recovery:
            self._recovery_state = RecoveryState(
                max_recovery_attempts=automation_task.max_recovery_attempts,
            )
        else:
            self._recovery_state = None

        # Initialize composable heuristics
        self._composable_heuristics = ComposableHeuristics(
            static_heuristics=self._intent_heuristics,
            task_category=automation_task.category,
        )

        # Use task description for planning
        task_description = automation_task.task

        # Emit run start
        self._emit_run_start(task_description, start_url)

        step_outcomes: list[StepOutcome] = []
        error: str | None = None

        try:
            # Generate plan
            plan = await self.plan(task_description, start_url=start_url)

            # Execute steps
            step_index = 0
            while step_index < len(plan.steps):
                step = plan.steps[step_index]

                # Set step-specific heuristic hints
                if self._composable_heuristics and step.heuristic_hints:
                    self._composable_heuristics.set_step_hints(step.heuristic_hints)

                outcome = await self._execute_step(step, runtime, step_index)
                step_outcomes.append(outcome)

                # Record checkpoint on success (for recovery)
                if outcome.status in (StepStatus.SUCCESS, StepStatus.VISION_FALLBACK):
                    if self._recovery_state is not None and outcome.url_after:
                        self._recovery_state.record_checkpoint(
                            url=outcome.url_after,
                            step_index=step_index,
                            snapshot_digest=hashlib.sha256(
                                (outcome.url_after or "").encode()
                            ).hexdigest()[:16],
                            predicates_passed=[],
                        )

                # Handle failure
                if outcome.status == StepStatus.FAILED and step.required:
                    # Try recovery first if available
                    if self._recovery_state and self._recovery_state.can_recover():
                        recovered = await self._attempt_recovery(runtime)
                        if recovered:
                            # Resume from checkpoint step
                            checkpoint = self._recovery_state.current_recovery_target
                            if checkpoint:
                                step_index = checkpoint.step_index + 1
                                self._recovery_state.clear_recovery_target()
                                continue

                    # Try replanning
                    if self._replans_used < self.config.retry.max_replans:
                        try:
                            plan = await self.replan(
                                task_description,
                                step,
                                outcome.error or "verification_failed",
                            )
                            # Continue from failed step
                            continue
                        except Exception as e:
                            error = f"Replan failed: {e}"
                            break
                    else:
                        error = f"Step {step.id} failed: {outcome.error}"
                        break

                # Check stop condition
                if step.stop_if_true and outcome.verification_passed:
                    break

                step_index += 1

                # Clear step hints for next iteration
                if self._composable_heuristics:
                    self._composable_heuristics.clear_step_hints()

        except Exception as e:
            error = str(e)

        # Build final outcome
        success = error is None and all(
            o.status in (StepStatus.SUCCESS, StepStatus.VISION_FALLBACK)
            for o in step_outcomes
            if self._current_plan and any(s.required and s.id == o.step_id for s in self._current_plan.steps)
        )

        run_outcome = RunOutcome(
            run_id=self._run_id,
            task=task_description,
            success=success,
            steps_completed=len(step_outcomes),
            steps_total=len(self._current_plan.steps) if self._current_plan else 0,
            replans_used=self._replans_used,
            step_outcomes=step_outcomes,
            total_duration_ms=int((time.time() - start_time) * 1000),
            error=error,
        )

        # Emit run end
        self._emit_run_end(run_outcome)

        return run_outcome

    async def step(
        self,
        runtime: AgentRuntime,
        step: PlanStep,
        step_index: int = 0,
    ) -> StepOutcome:
        """
        Execute a single step (for manual step-by-step execution).

        Args:
            runtime: AgentRuntime instance
            step: PlanStep to execute
            step_index: Step index for tracing

        Returns:
            StepOutcome with step results
        """
        return await self._execute_step(step, runtime, step_index)
