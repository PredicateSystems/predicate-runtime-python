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

import asyncio
import base64
import hashlib
import inspect
import json
import re
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import SimpleNamespace
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..actions import clear_async, type_text_async
from ..agent_runtime import AgentRuntime
from ..llm_provider import LLMProvider, LLMResponse
from ..models import Snapshot, SnapshotOptions, StepHookContext
from ..pruning import (
    PrunedSnapshotContext,
    PruningTaskCategory,
    classify_task_category,
    prune_with_recovery,
)
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
# Token Usage Tracking
# ---------------------------------------------------------------------------


@dataclass
class TokenUsageTotals:
    """Accumulated token counts for a single role or model."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, resp: LLMResponse) -> None:
        """Add token counts from an LLM response."""
        self.calls += 1
        pt = resp.prompt_tokens if isinstance(resp.prompt_tokens, int) else 0
        ct = resp.completion_tokens if isinstance(resp.completion_tokens, int) else 0
        tt = resp.total_tokens if isinstance(resp.total_tokens, int) else (pt + ct)
        self.prompt_tokens += max(0, int(pt))
        self.completion_tokens += max(0, int(ct))
        self.total_tokens += max(0, int(tt))


class _TokenUsageCollector:
    """Collects token usage statistics by role (planner/executor) and model."""

    def __init__(self) -> None:
        self._by_role: dict[str, TokenUsageTotals] = {}
        self._by_model: dict[str, TokenUsageTotals] = {}

    def record(self, *, role: str, resp: LLMResponse) -> None:
        """Record token usage from an LLM response."""
        self._by_role.setdefault(role, TokenUsageTotals()).add(resp)
        m = str(resp.model_name or "").strip() or "unknown"
        self._by_model.setdefault(m, TokenUsageTotals()).add(resp)

    def reset(self) -> None:
        """Clear all recorded statistics."""
        self._by_role.clear()
        self._by_model.clear()

    def summary(self) -> dict[str, Any]:
        """
        Get a summary of all token usage.

        Returns:
            Dictionary with:
            - total: aggregate counts across all calls
            - by_role: breakdown by role (planner, executor, replan)
            - by_model: breakdown by model name
        """
        def _sum(items: dict[str, TokenUsageTotals]) -> TokenUsageTotals:
            out = TokenUsageTotals()
            for t in items.values():
                out.calls += t.calls
                out.prompt_tokens += t.prompt_tokens
                out.completion_tokens += t.completion_tokens
                out.total_tokens += t.total_tokens
            return out

        total = _sum(self._by_role)
        return {
            "total": {
                "calls": total.calls,
                "prompt_tokens": total.prompt_tokens,
                "completion_tokens": total.completion_tokens,
                "total_tokens": total.total_tokens,
            },
            "by_role": {
                k: {
                    "calls": v.calls,
                    "prompt_tokens": v.prompt_tokens,
                    "completion_tokens": v.completion_tokens,
                    "total_tokens": v.total_tokens,
                }
                for k, v in self._by_role.items()
            },
            "by_model": {
                k: {
                    "calls": v.calls,
                    "prompt_tokens": v.prompt_tokens,
                    "completion_tokens": v.completion_tokens,
                    "total_tokens": v.total_tokens,
                }
                for k, v in self._by_model.items()
            },
        }


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

        # Enable scroll-after-escalation to find elements below/above viewport
        config = SnapshotEscalationConfig(scroll_after_escalation=True, scroll_directions=("down", "up"))

        # Custom scroll amount as fraction of viewport height (default: 0.4 = 40%)
        config = SnapshotEscalationConfig(scroll_viewport_fraction=0.5)  # 50% of viewport
    """

    enabled: bool = True
    limit_base: int = 60
    limit_step: int = 30
    limit_max: int = 200
    # Scroll after exhausting limit escalation to find elements in different viewports
    scroll_after_escalation: bool = True
    scroll_max_attempts: int = 3  # Max scrolls per direction
    scroll_directions: tuple[str, ...] = ("down", "up")  # Directions to try
    scroll_viewport_fraction: float = 0.4  # Scroll by 40% of viewport height (adaptive to screen size)


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
class ModalDismissalConfig:
    """
    Configuration for automatic modal/drawer dismissal after DOM changes.

    When a CLICK action triggers a DOM change (e.g., modal/drawer appears),
    this feature attempts to dismiss blocking overlays using common patterns.

    This handles common blocking scenarios:
    - Product protection/warranty upsells (Amazon, etc.)
    - Cookie consent banners
    - Newsletter signup popups
    - Promotional overlays
    - Cart upsell drawers

    The dismissal logic looks for buttons with common dismissal text patterns
    and clicks them to clear the overlay.

    Attributes:
        enabled: If True, attempt to dismiss modals after DOM change detection.
        dismiss_patterns: Text patterns to match dismissal buttons (case-insensitive).
        role_filter: Element roles to consider for dismissal buttons.
        max_attempts: Maximum dismissal attempts per modal.
        min_new_elements: Minimum new DOM elements to trigger modal detection.

    Example:
        # Default: enabled with common dismissal patterns
        config = ModalDismissalConfig()

        # Disable modal dismissal
        config = ModalDismissalConfig(enabled=False)

        # Custom patterns (e.g., for non-English sites)
        config = ModalDismissalConfig(
            dismiss_patterns=("nein danke", "schließen", "abbrechen"),
        )
    """

    enabled: bool = True
    # Patterns that require word boundary matching (longer patterns)
    # Ordered by preference: decline > close > accept (we prefer not to accept upsells)
    dismiss_patterns: tuple[str, ...] = (
        # Decline/Skip patterns (highest priority - user is declining an offer)
        "no thanks",
        "no, thanks",
        "no thank you",
        "not now",
        "not interested",
        "maybe later",
        "skip",
        "decline",
        "reject",
        "deny",
        "continue without",
        # Close patterns
        "close",
        "close dialog",
        "close modal",
        "close popup",
        "dismiss",
        "dismiss banner",
        "dismiss dialog",
        "cancel",
        # Continue patterns (when modal offers upgrade vs continue)
        "continue",
        "continue to",
        "proceed",
    )
    # Icon characters that require exact match (entire label is just this character)
    dismiss_icons: tuple[str, ...] = (
        "x",
        "×",  # Unicode multiplication sign
        "✕",  # Unicode X mark
        "✖",  # Heavy multiplication X
        "✗",  # Ballot X
        "╳",  # Box drawings
    )
    role_filter: tuple[str, ...] = ("button", "link")
    max_attempts: int = 2
    min_new_elements: int = 5  # Same threshold as DOM change fallback


@dataclass(frozen=True)
class CheckoutDetectionConfig:
    """
    Configuration for checkout page detection.

    After modal dismissal or action completion, the agent checks if the
    current page is a checkout-relevant page. If detected, this triggers
    a replan to continue the checkout flow.

    This handles scenarios where:
    - Agent clicks "Add to Cart" and modal is dismissed, but agent stops
    - Agent lands on cart page but plan doesn't include checkout steps
    - Agent reaches login page during checkout flow

    Attributes:
        enabled: If True, detect checkout pages and trigger continuation.
        url_patterns: URL patterns that indicate checkout-relevant pages.
        element_patterns: Element text patterns that indicate checkout pages.
        trigger_replan: If True, trigger replanning when checkout page detected.

    Example:
        # Default: enabled with common checkout patterns
        config = CheckoutDetectionConfig()

        # Disable checkout detection
        config = CheckoutDetectionConfig(enabled=False)

        # Custom patterns
        config = CheckoutDetectionConfig(
            url_patterns=("/warenkorb", "/kasse"),  # German
        )
    """

    enabled: bool = True
    # URL patterns that indicate checkout-relevant pages
    url_patterns: tuple[str, ...] = (
        # Cart pages
        "/cart",
        "/basket",
        "/bag",
        "/shopping-cart",
        "/gp/cart",  # Amazon
        # Checkout pages
        "/checkout",
        "/buy",
        "/order",
        "/payment",
        "/pay",
        "/purchase",
        "/gp/buy",  # Amazon
        "/gp/checkout",  # Amazon
        # Sign-in during checkout
        "/signin",
        "/sign-in",
        "/login",
        "/ap/signin",  # Amazon
        "/auth",
        "/authenticate",
    )
    # Element text patterns that indicate checkout pages (case-insensitive)
    element_patterns: tuple[str, ...] = (
        "proceed to checkout",
        "proceed to buy",
        "go to checkout",
        "view cart",
        "shopping cart",
        "your cart",
        "sign in to checkout",
        "continue to payment",
        "place your order",
        "buy now",
        "checkout",
    )
    # If True, trigger replanning when checkout page is detected
    trigger_replan: bool = True


@dataclass(frozen=True)
class AuthBoundaryConfig:
    """
    Configuration for authentication boundary detection.

    When the agent reaches a login/sign-in page and doesn't have credentials,
    it should stop gracefully instead of failing or spinning endlessly.

    This is a "terminal state" - the agent has successfully navigated as far
    as possible without authentication.

    Attributes:
        enabled: If True, detect auth boundaries and stop gracefully.
        url_patterns: URL patterns indicating authentication pages.
        stop_on_auth: If True, mark run as successful when auth boundary reached.
        auth_success_message: Message to include in outcome when stopping at auth.

    Example:
        # Default: enabled, stop gracefully at login pages
        config = AuthBoundaryConfig()

        # Disable (try to continue past auth pages)
        config = AuthBoundaryConfig(enabled=False)
    """

    enabled: bool = True
    # URL patterns that indicate authentication/login pages
    url_patterns: tuple[str, ...] = (
        "/signin",
        "/sign-in",
        "/login",
        "/log-in",
        "/auth",
        "/authenticate",
        "/ap/signin",  # Amazon sign-in
        "/ap/register",  # Amazon registration
        "/ax/claim",  # Amazon CAPTCHA/verification
        "/account/login",
        "/accounts/login",
        "/user/login",
    )
    # If True, mark the run as successful when auth boundary is reached
    # (since the agent did everything it could without credentials)
    stop_on_auth: bool = True
    # Message to include when stopping at auth boundary
    auth_success_message: str = "Reached authentication boundary (login required)"


@dataclass(frozen=True)
class StepwisePlanningConfig:
    """
    Configuration for stepwise (ReAct-style) planning.

    Instead of generating a full plan upfront, the agent plans one step at a time
    based on the current page state. This allows the agent to adapt to unexpected
    site layouts and flows.

    Attributes:
        max_steps: Maximum number of steps before giving up (safety limit)
        action_history_limit: How many past actions to include in planning context
        include_page_context: Whether to include compact element representation in prompts

    Example:
        config = StepwisePlanningConfig(
            max_steps=20,
            action_history_limit=5,
        )
    """

    max_steps: int = 30
    action_history_limit: int = 5
    include_page_context: bool = True


@dataclass
class ActionRecord:
    """
    Record of an executed action for history tracking in stepwise planning.

    Attributes:
        step_num: Step number (1-indexed)
        action: Action type (CLICK, TYPE_AND_SUBMIT, SCROLL, etc.)
        target: Element description or URL
        result: Outcome (success, failed)
        url_after: URL after action completed
    """

    step_num: int
    action: str
    target: str | None
    result: str
    url_after: str | None


@dataclass(frozen=True)
class PlannerExecutorConfig:
    """
    High-level configuration for PlannerExecutorAgent.

    This config focuses on:
    - Snapshot escalation settings
    - Retry/verification settings
    - Vision fallback settings
    - Recovery navigation settings
    - Modal dismissal settings
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

    # Modal dismissal (for blocking overlays after DOM changes)
    modal: ModalDismissalConfig = ModalDismissalConfig()

    # Checkout page detection (continue workflow when reaching checkout pages)
    checkout: CheckoutDetectionConfig = CheckoutDetectionConfig()

    # Authentication boundary detection (stop gracefully at login pages)
    auth_boundary: AuthBoundaryConfig = AuthBoundaryConfig()

    # Stepwise planning (ReAct-style, plan one step at a time)
    stepwise: StepwisePlanningConfig = StepwisePlanningConfig()

    # Auto-fallback: switch from upfront to stepwise planning on repeated failures
    # When enabled, if upfront planning fails (max_replans exhausted), automatically
    # retry with stepwise planning for better adaptability on unfamiliar sites.
    auto_fallback_to_stepwise: bool = True
    auto_fallback_replan_threshold: int = 1  # Fallback after this many replans fail

    # Planner LLM settings
    planner_max_tokens: int = 2048
    planner_temperature: float = 0.0

    # Page context for planning: when enabled, extracts page content as markdown
    # during initial planning to help the planner understand page type and structure.
    # This adds token cost but improves plan quality for complex pages.
    use_page_context: bool = False
    page_context_max_chars: int = 8000  # Max chars of markdown to include

    # Executor LLM settings
    executor_max_tokens: int = 96
    executor_temperature: float = 0.0
    type_delay_ms: float | None = 17.0

    # Stabilization (wait for DOM to settle after actions)
    stabilize_enabled: bool = True
    stabilize_poll_s: float = 0.35
    stabilize_max_attempts: int = 6

    # Pre-step verification (skip step if predicates already pass)
    pre_step_verification: bool = True

    # Scroll-to-find: automatically scroll to find elements when not in viewport
    scroll_to_find_enabled: bool = True
    scroll_to_find_max_scrolls: int = 3  # Max scroll attempts per direction
    scroll_to_find_directions: tuple[str, ...] = ("down", "up")  # Try down first, then up

    # Tracing
    trace_screenshots: bool = True
    trace_screenshot_format: str = "jpeg"
    trace_screenshot_quality: int = 80

    # Verbose mode (print plan, prompts, and LLM responses to stdout)
    verbose: bool = False


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
    pruning_category: str | None = None
    pruned_node_count: int = 0

    def is_stale(self, max_age_seconds: float = 5.0) -> bool:
        """Check if snapshot is too old to reuse."""
        return (datetime.now() - self.captured_at).total_seconds() > max_age_seconds

    def should_use_vision(self) -> bool:
        """Check if executor should use vision fallback."""
        return not self.snapshot_success or self.requires_vision

    def digest(self) -> str:
        """Compute a digest for loop/change detection."""
        title = getattr(self.snapshot, "title", None) or ""
        parts = [
            self.snapshot.url[:200] if self.snapshot.url else "",
            title[:200] if title else "",
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

    Note: If we have sufficient elements (10+), we should NOT trigger vision
    fallback even if diagnostics suggest it. This handles cases where the
    API incorrectly flags normal HTML pages as requiring vision.
    """
    elements = getattr(snap, "elements", []) or []
    element_count = len(elements)

    # If we have sufficient elements, the snapshot is usable
    # regardless of what diagnostics say
    if element_count >= 10:
        return False, None

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
        if has_canvas and element_count < 5:
            return True, "canvas_page"

        # Check diagnostics.requires_vision only if few elements
        requires_vision = getattr(diag, "requires_vision", False)
        if requires_vision and element_count < 5:
            return True, "diagnostics_requires_vision"

    # Very few elements usually indicates a problem
    if element_count < 3:
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
    extracted_data: Any | None = None


@dataclass
class SearchSubmitTelemetry:
    """Tracks search-submit behavior for debugging and diagnostics."""

    first_submit_method: Literal["click", "enter"] | None = None
    retry_submit_method: Literal["click", "enter"] | None = None
    observed_search_results_dom: bool = False


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
    token_usage: dict[str, Any] | None = None  # Token usage summary from get_token_stats()
    fallback_used: bool = False  # True if auto-fallback to stepwise was triggered


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
# Extraction Keywords for Markdown-based Text Extraction
# ---------------------------------------------------------------------------

# Keywords that indicate a simple text extraction task suitable for read_markdown()
# These tasks don't need LLM-based extraction - just return the page content as markdown
TEXT_EXTRACTION_KEYWORDS = frozenset([
    # Direct extraction verbs
    "extract",
    "read",
    "parse",
    "scrape",
    "get",
    "fetch",
    "retrieve",
    "capture",
    "grab",
    "copy",
    "pull",
    # Question words that indicate reading content
    "what is",
    "what are",
    "what's",
    "show me",
    "tell me",
    "find",
    "list",
    "display",
    # Content-specific patterns
    "title",
    "headline",
    "heading",
    "text",
    "content",
    "body",
    "paragraph",
    "article",
    "post",
    "message",
    "description",
    "summary",
    "excerpt",
    # Data extraction patterns
    "price",
    "cost",
    "amount",
    "name",
    "label",
    "value",
    "number",
    "date",
    "time",
    "address",
    "email",
    "phone",
    "rating",
    "review",
    "comment",
    "author",
    "username",
    # Table/list extraction
    "table",
    "row",
    "column",
    "item",
    "entry",
    "record",
])


def _is_text_extraction_task(task: str) -> bool:
    """
    Determine if a task is a simple text extraction that can use read_markdown().

    Returns True if the task contains keywords indicating text extraction,
    where returning the page markdown is sufficient without LLM-based extraction.

    Args:
        task: The task description to analyze

    Returns:
        True if this is a text extraction task suitable for read_markdown()
    """
    if not task:
        return False

    task_lower = task.lower()

    # Check for extraction keyword patterns using word boundary matching
    # to avoid false positives (e.g., "time" in "sentiment")
    for keyword in TEXT_EXTRACTION_KEYWORDS:
        # Multi-word keywords (like "what is") use substring matching
        if " " in keyword:
            if keyword in task_lower:
                return True
        else:
            # Single-word keywords use word boundary matching via regex
            # Match keyword at word boundaries, allowing for plurals (optional 's' or 'es')
            # e.g., "title" matches "title", "titles", "title's"
            pattern = rf"\b{re.escape(keyword)}(s|es)?\b"
            if re.search(pattern, task_lower):
                return True

    return False


# ---------------------------------------------------------------------------
# Plan Normalization and Validation
# ---------------------------------------------------------------------------


def _parse_string_predicate(pred_str: str) -> dict[str, Any] | None:
    """
    Parse a string predicate into a normalized dict.

    LLMs sometimes output predicates as strings like:
    - "url_contains('amazon.com')" -> {"predicate": "url_contains", "args": ["amazon.com"]}
    - "url_matches(^https://www\\.amazon\\.com/.*)" -> {"predicate": "url_matches", "args": ["^https://www\\.amazon\\.com/.*"]}
    - "exists(role=button)" -> {"predicate": "exists", "args": ["role=button"]}

    Args:
        pred_str: String representation of a predicate

    Returns:
        Normalized predicate dict or None if parsing fails
    """
    import re

    pred_str = pred_str.strip()

    # Try to match function-call style: predicate_name(args)
    match = re.match(r'^(\w+)\s*\(\s*(.+?)\s*\)$', pred_str, re.DOTALL)
    if match:
        pred_name = match.group(1)
        args_str = match.group(2)

        # Strip quotes from args if present
        args_str = args_str.strip()
        if (args_str.startswith("'") and args_str.endswith("'")) or \
           (args_str.startswith('"') and args_str.endswith('"')):
            args_str = args_str[1:-1]

        return {
            "predicate": pred_name,
            "args": [args_str],
        }

    # Try simple predicate name without args
    if re.match(r'^[\w_]+$', pred_str):
        return {
            "predicate": pred_str,
            "args": [],
        }

    return None


def _normalize_verify_predicate(pred: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a verify predicate to the expected format.

    LLMs may output predicates in various formats:
    - {"url_contains": "amazon.com"} -> {"predicate": "url_contains", "args": ["amazon.com"]}
    - {"exists": "text~'Logitech'"} -> {"predicate": "exists", "args": ["text~'Logitech'"]}
    - {"predicate": "url_contains", "input": "x"} -> {"predicate": "url_contains", "args": ["x"]}
    - {"type": "url_contains", "input": "x"} -> {"predicate": "url_contains", "args": ["x"]}

    Args:
        pred: Raw predicate dictionary

    Returns:
        Normalized predicate with "predicate" and "args" fields
    """
    # Handle "type" field as alternative to "predicate" (common LLM variation)
    if "type" in pred and "predicate" not in pred:
        pred["predicate"] = pred.pop("type")

    # Already has predicate field - normalize args
    if "predicate" in pred:
        # Handle "input" field as alternative to "args"
        if "args" not in pred or not pred["args"]:
            if "input" in pred:
                pred["args"] = [pred.pop("input")]
            elif "value" in pred:
                pred["args"] = [pred.pop("value")]
            elif "pattern" in pred:  # For url_matches
                pred["args"] = [pred.pop("pattern")]
            elif "substring" in pred:  # For url_contains
                pred["args"] = [pred.pop("substring")]
            elif "selector" in pred:  # For exists/not_exists
                pred["args"] = [pred.pop("selector")]
        return pred

    # Predicate type is a key in the dict (e.g., {"url_contains": "amazon.com"})
    known_predicates = [
        "url_contains", "url_equals", "url_matches",
        "exists", "not_exists",
        "element_count", "element_visible",
        "any_of", "all_of",
        "text_contains", "text_equals",
    ]

    for pred_type in known_predicates:
        if pred_type in pred:
            return {
                "predicate": pred_type,
                "args": [pred[pred_type]] if pred[pred_type] else [],
            }

    # Unknown format - return as-is and let validation fail with clear error
    return pred


def normalize_plan(plan_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize plan dictionary to handle LLM output variations.

    This function handles common variations in LLM output:
    - url vs target field names
    - action aliases (click vs CLICK)
    - step id variations (string vs int)
    - verify predicate format variations

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
                    "EXTRACT_TEXT": "EXTRACT",
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

            # Normalize verify predicates
            if "verify" in step and isinstance(step["verify"], list):
                normalized_verify = []
                for pred in step["verify"]:
                    if isinstance(pred, dict):
                        normalized_verify.append(_normalize_verify_predicate(pred))
                    elif isinstance(pred, str):
                        # Try to parse string predicates like "url_contains('text')"
                        parsed = _parse_string_predicate(pred)
                        if parsed:
                            normalized_verify.append(parsed)
                        else:
                            # Keep as-is, let validation fail with clear error
                            normalized_verify.append({"predicate": "unknown", "args": [pred]})
                    else:
                        normalized_verify.append(pred)
                step["verify"] = normalized_verify

            # Normalize optional_substeps recursively
            if "optional_substeps" in step:
                for substep in step["optional_substeps"]:
                    if "action" in substep:
                        substep["action"] = substep["action"].upper()
                    if "url" in substep and "target" not in substep:
                        substep["target"] = substep.pop("url")
                    # Normalize verify in substeps too
                    if "verify" in substep and isinstance(substep["verify"], list):
                        normalized_verify = []
                        for pred in substep["verify"]:
                            if isinstance(pred, dict):
                                normalized_verify.append(_normalize_verify_predicate(pred))
                            elif isinstance(pred, str):
                                parsed = _parse_string_predicate(pred)
                                if parsed:
                                    normalized_verify.append(parsed)
                                else:
                                    normalized_verify.append({"predicate": "unknown", "args": [pred]})
                            else:
                                normalized_verify.append(pred)
                        substep["verify"] = normalized_verify

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
    page_context: str | None = None,
) -> tuple[str, str]:
    """
    Build system and user prompts for the Planner LLM.

    Args:
        task: Task description
        start_url: Starting URL
        site_type: Type of site (general, e-commerce, etc.)
        auth_state: Authentication state
        strict: If True, emphasize JSON-only output
        schema_errors: Errors from previous parsing attempt
        page_context: Optional markdown content of the current page for context

    Returns:
        (system_prompt, user_prompt)
    """
    strict_note = "\nReturn ONLY a JSON object. No explanations, no code fences.\n" if strict else ""
    schema_note = f"\nSchema errors from last attempt:\n{schema_errors}\n" if schema_errors else ""

    # Detect task type for domain-specific guidance
    task_lower = task.lower()
    is_ecommerce_task = any(keyword in task_lower for keyword in [
        "buy", "purchase", "add to cart", "checkout", "order", "shop",
        "amazon", "ebay", "walmart", "target", "bestbuy",
        "cart", "mouse", "keyboard", "laptop", "product",
    ])
    is_search_task = any(keyword in task_lower for keyword in [
        "search", "find", "look for", "look up", "google",
    ])

    # Build domain-specific planning guidance
    domain_guidance = ""
    if is_ecommerce_task:
        domain_guidance = """

IMPORTANT: E-Commerce Task Planning Rules
=========================================
For shopping/purchase tasks, include ALL necessary steps in order:
1. NAVIGATE to the site (if not already there)
2. Find the product - choose ONE of these approaches IN PRIORITY ORDER:
   a) DIRECT MATCH (BEST): Scan page for text closely matching the goal. CLICK any product/category with matching text.
   b) CATEGORY BROWSE: If no exact match, click a category link that relates to the goal (e.g., "Tablecloths" for "vinyl tablecloth")
   c) SEARCH: ONLY if you see an input EXPLICITLY labeled "Search" with placeholder="Search..." or aria-label="Search"
   d) SEARCH ICON: Only if you see a magnifying glass icon linked to search
3. CLICK on specific product from results (not filters or categories)
4. CLICK "Add to Cart" button on product page
5. CLICK "Proceed to Checkout" or cart icon
6. Handle login/signup if required (may need CLICK + TYPE_AND_SUBMIT)
7. CLICK through checkout process

CRITICAL - CATEGORY NAVIGATION (MOST RELIABLE FOR HOMEPAGES):
- On homepage/landing pages, browse via CATEGORY LINKS - this is the MOST RELIABLE method
- Look for category links like "Rally Home Goods", "Tablecloths", "Kitchen", "Catalog", etc.
- Category links are usually in the main navigation, sidebar, or footer and are always clickable
- Example: Goal "vinyl tablecloth" → Click "Rally Home Goods" or "Catalog" category first
- After clicking a category, THEN look for the specific product

SECONDARY - Direct Product Click (ONLY on collection/category pages):
- If a product appears on a CATEGORY/COLLECTION page (not homepage), click it directly
- WARNING: Products in "Hot Products", carousels, or grid sections on HOME PAGES are often NOT clickable
- The snapshot may not capture product titles in carousels - use category navigation instead

AVOID - Searching on sites without visible search box:
- Many e-commerce sites hide search or don't have search at all
- If you don't see a clear "Search" textbox in the page markdown, DO NOT try to search
- Prefer category navigation over searching - it's more reliable

CRITICAL - Search Box Identification (ONLY WHEN NO MATCHING TEXT):
- Only use TYPE_AND_SUBMIT if you see an input EXPLICITLY labeled for SEARCH
- Valid search indicators: placeholder="Search...", aria-label="Search", text "Search products"
- DO NOT type into fields with these labels (they are NOT search boxes):
  * "Your email address", "Email", "Newsletter", "Subscribe"
  * "Zip code", "Location", "Enter your email"
  * Any field asking for personal information
- If unsure whether a field is a search box, DO NOT use it - click products/categories instead

Common mistakes to AVOID:
- Do NOT skip "Add to Cart" step - clicking a product link is NOT adding to cart
- Do NOT combine multiple distinct actions into one step
- Do NOT confuse filter/category clicks with product selection
- Do NOT assume a search box exists - if none is clearly visible, click products/categories directly
- Do NOT hallucinate search boxes - if page content doesn't show an obvious search input, use direct navigation
- Do NOT type into email/newsletter/subscription fields - they are NOT search boxes
- Do NOT use search when matching text is visible - click directly instead
- Each distinct user action should be its own step

Intent hints are critical - ALWAYS include the specific product/element name:
- intent: "Click product Vinyl Tablecloth" (GOOD - includes product name)
- intent: "Click on product title or image" (BAD - too generic, will click wrong product)
- intent: "Click category link Tablecloths" (GOOD - includes category name)
- intent: "Click Add to Cart button"
- intent: "Click Proceed to Checkout"
- intent: "Click sign in button"
"""
    elif is_search_task:
        domain_guidance = """

IMPORTANT: Search Task Planning Rules
=====================================
For search tasks, include steps to:
1. NAVIGATE to the search site (if not already there)
2. TYPE_AND_SUBMIT the search query
3. Wait for/verify search results
4. If selecting a result: CLICK on specific result item
"""

    # Check for extraction tasks
    is_extraction_task = any(keyword in task_lower for keyword in [
        "extract", "get the", "what is", "read the", "find the text", "scrape",
        "title of", "price of", "name of", "content of",
    ])

    if is_extraction_task:
        domain_guidance = """

IMPORTANT: Extraction Task Planning Rules
=========================================
For extraction tasks where data is already visible on the page:

1. If the data you need is VISIBLE in the page context markdown above:
   - Use EXTRACT directly as the ONLY step - no clicking needed
   - The EXTRACT action will read the visible text from the page

2. If you need to navigate to see the data:
   - First CLICK or NAVIGATE to the right page
   - Then use EXTRACT

CRITICAL: Do NOT click on links to external sites when extracting.
- Hacker News post titles link to EXTERNAL sites, not to HN pages
- To extract a title that's visible, use EXTRACT directly on the current page
- Only click if you need to navigate to an HN item page (e.g., for comments)

Example for "Extract the title of the first post":
{
  "steps": [
    {
      "id": 1,
      "goal": "Extract the first post title from the page",
      "action": "EXTRACT",
      "target": "first post title",
      "verify": []
    }
  ]
}
"""

    system = f"""You are the PLANNER. Output a JSON execution plan for the web automation task.
{strict_note}
Your output must be a valid JSON object with this EXACT structure:

{{
  "task": "description of task",
  "notes": ["assumption 1", "assumption 2"],
  "steps": [
    {{
      "id": 1,
      "goal": "Navigate to website",
      "action": "NAVIGATE",
      "target": "https://example.com",
      "verify": [{{"predicate": "url_contains", "args": ["example.com"]}}]
    }},
    {{
      "id": 2,
      "goal": "Search for product",
      "action": "TYPE_AND_SUBMIT",
      "input": "search query",
      "verify": [{{"predicate": "url_contains", "args": ["search"]}}]
    }},
    {{
      "id": 3,
      "goal": "Click on product from results",
      "action": "CLICK",
      "intent": "Click on product title",
      "verify": []
    }}
  ]
}}

CRITICAL: Each verify predicate MUST be an object with "predicate" and "args" keys:
- {{"predicate": "url_contains", "args": ["substring"]}}
- {{"predicate": "exists", "args": ["role=button"]}}
- {{"predicate": "not_exists", "args": ["text~'error'"]}}

DO NOT use string format like "url_contains('text')" - use object format only.

CRITICAL - url_contains RULES:
1. Use ONLY generic keywords, NEVER site-specific paths like "/product/", "/products/", "/collections/"
2. Different sites use different URL patterns - don't guess the path structure
3. For product pages: use "verify": [] (empty) or use the product keyword like ["snow-blower"]
4. For search: "search" or "query=" work across most sites
5. For checkout: "checkout" or "cart" work across most sites
6. NEVER use paths like "/product/", "/products/", "/p/", "/dp/" - these are site-specific

Examples:
- GOOD: {{"predicate": "url_contains", "args": ["snow-blower"]}} - uses product keyword
- GOOD: {{"predicate": "url_contains", "args": ["search"]}} - generic search indicator
- BAD: {{"predicate": "url_contains", "args": ["/product/"]}} - site-specific path
- BAD: {{"predicate": "url_contains", "args": ["/products/vinyl-tablecloth"]}} - guessing path structure
{domain_guidance}
Return ONLY valid JSON. No prose, no code fences, no markdown."""

    # Build page context section if provided
    page_context_section = ""
    if page_context:
        page_context_section = f"""

Current Page Content:
The following is a markdown representation of the current page content. Use this to understand
the page structure, available elements (buttons, links, forms), and content to inform your plan.
Note: This may be truncated if the page is large.

---
{page_context}
---
"""

    user = f"""Task: {task}
{schema_note}
Starting URL: {start_url or "browser's current page"}
Site type: {site_type}
Auth state: {auth_state}
{page_context_section}
Output a JSON plan to accomplish this task. Each step should represent ONE distinct action."""

    return system, user


def build_stepwise_planner_prompt(
    goal: str,
    current_url: str,
    page_context: str,
    action_history: list["ActionRecord"],
) -> tuple[str, str]:
    """
    Build system and user prompts for stepwise (ReAct-style) planning.

    Instead of generating a full plan upfront, this prompt asks the LLM to
    decide the next single action based on current page state and history.

    Args:
        goal: The overall task goal
        current_url: Current page URL
        page_context: Compact representation of page elements
        action_history: List of previously executed actions

    Returns:
        (system_prompt, user_prompt)
    """
    # Build action history text
    history_text = ""
    if action_history:
        history_text = "Actions taken so far:\n"
        for rec in action_history:
            target_str = f"({rec.target})" if rec.target else ""
            history_text += f"  {rec.step_num}. {rec.action}{target_str} → {rec.result}"
            if rec.url_after:
                history_text += f" [URL: {rec.url_after[:60]}...]"
            history_text += "\n"
        history_text += "\n"

    # Tight prompt optimized for small local models (7B)
    system = """You are a browser automation planner. Decide the NEXT action.

Actions:
- CLICK: Click an element. Set "intent" to element type/role. Set "input" to EXACT text from elements list.
- TYPE_AND_SUBMIT: Type and submit. ONLY use if you see a "searchbox" or "textbox" with "search" in the text.
- SCROLL: Scroll page. Set "direction" to "up" or "down".
- DONE: Goal achieved. Return this when the goal is complete.

CRITICAL RULE FOR CLICK:
- The "input" field MUST contain text that ACTUALLY APPEARS in the elements list below
- Do NOT guess or invent text - copy EXACT text from an element
- If product title "vinyl tablecloth" is NOT in the elements list, click a category link instead (e.g., "Catalog", "Home Goods")
- Only click a specific product if you see its EXACT name in the elements

Output ONLY valid JSON (no markdown, no ```):
{"action":"CLICK","intent":"category link","input":"Catalog","reasoning":"browse products via category"}
{"action":"CLICK","intent":"product link","input":"Vinyl Round Tablecloth","reasoning":"found exact product name"}
{"action":"DONE","intent":"completed","reasoning":"goal achieved"}

RULES:
1. ONLY use text that appears EXACTLY in the elements list - do NOT invent names
2. For shopping: start with category links (Catalog, Shop Now, Home Goods) to find products
3. ONLY use TYPE_AND_SUBMIT if you see a textbox labeled "search"
4. Do NOT type into "email" or "newsletter" fields
5. Do NOT repeat the same action twice
6. Output ONLY JSON - no <think> tags, no markdown, no prose"""

    user = f"""Goal: {goal}

Current URL: {current_url}

{history_text}Current page elements (ID|role|text|importance|clickable|...):
{page_context}

Based on the goal and current page state, what is the NEXT action to take?"""

    return system, user


def _get_category_executor_hints(category: str | None) -> str:
    """
    Get category-specific hints for the executor.

    These hints guide the executor to prioritize certain element types
    based on the detected task category, improving accuracy without
    adding tokens to the planner.
    """
    if not category:
        return ""

    category_lower = category.lower() if isinstance(category, str) else str(category).lower()

    hints = {
        "shopping": (
            "Priority: 'Add to Cart', 'Buy Now', 'Add to Bag', product links, price elements."
        ),
        "checkout": (
            "Priority: 'Checkout', 'Proceed to Checkout', 'Place Order', payment fields."
        ),
        "form_filling": (
            "Priority: input fields, textboxes, submit/send buttons, form labels."
        ),
        "search": (
            "Priority: search box, search button, result links, filter controls."
        ),
        "auth": (
            "Priority: username/email field, password field, sign in/login button."
        ),
        "extraction": (
            "Priority: data elements, table cells, list items, content containers."
        ),
        "navigation": (
            "Priority: navigation links, menu items, breadcrumbs."
        ),
    }

    return hints.get(category_lower, "")


def build_executor_prompt(
    goal: str,
    intent: str | None,
    compact_context: str,
    input_text: str | None = None,
    category: str | None = None,
    action_type: str | None = None,
) -> tuple[str, str]:
    """
    Build system and user prompts for the Executor LLM.

    Args:
        goal: Human-readable goal for this step
        intent: Intent hint for element selection (optional)
        compact_context: Compact representation of page elements
        input_text: For TYPE_AND_SUBMIT: text to type. For CLICK: target text to match (optional)
        category: Task category for category-specific hints (optional)
        action_type: Action type (CLICK, TYPE_AND_SUBMIT, etc.) to determine prompt variant

    Returns:
        (system_prompt, user_prompt)
    """
    intent_line = f"Intent: {intent}\n" if intent else ""

    # For CLICK actions, input_text is target to match (not text to type)
    is_type_action = action_type in ("TYPE_AND_SUBMIT", "TYPE")
    if is_type_action and input_text:
        input_line = f"Text to type: \"{input_text}\"\n"
    elif input_text:
        input_line = f"Target to find: \"{input_text}\"\n"
    else:
        input_line = ""

    # Get category-specific hints
    category_hints = _get_category_executor_hints(category)
    category_line = f"{category_hints}\n" if category_hints else ""

    # Tight prompt optimized for small local models (4B-7B)
    # Key: explicit format, no reasoning, clear failure consequence
    if is_type_action and input_text:
        # TYPE action needed - find the INPUT element (textbox/combobox), not the submit button
        system = (
            "You are an executor for browser automation.\n"
            "Task: Find the INPUT element (textbox, combobox, searchbox) to type into.\n"
            "Return ONLY ONE line: TYPE(<id>, \"text\")\n"
            "IMPORTANT: Return the ID of the INPUT/TEXTBOX element, NOT the submit button.\n"
            "CRITICAL - AVOID these fields (they are NOT search boxes):\n"
            "- Fields with 'email', 'newsletter', 'subscribe', 'signup' in the text\n"
            "- Fields labeled 'Your email address', 'Email', 'Enter your email'\n"
            "- Fields in footer/newsletter sections\n"
            "ONLY use fields explicitly labeled for SEARCH (placeholder='Search', aria='Search').\n"
            "If NO search field exists, return NONE instead of guessing.\n"
            "If you output anything else, the action fails.\n"
            "Do NOT output <think> or any reasoning.\n"
            "No prose, no markdown, no extra whitespace.\n"
            "Example: TYPE(42, \"hello world\")"
        )
    else:
        # CLICK action (most common)
        # Check if this is a search-related action (from intent or goal)
        search_keywords = ["search", "magnify", "magnifier", "find"]
        is_search_action = (
            (intent and any(kw in intent.lower() for kw in search_keywords))
            or any(kw in goal.lower() for kw in search_keywords)
        )
        # Check if this is a product click action (from intent or goal)
        product_keywords = ["product", "item", "result", "listing"]
        is_product_action = (
            (intent and any(kw in intent.lower() for kw in product_keywords))
            or any(kw in goal.lower() for kw in product_keywords)
        )
        # Check if this is an Add to Cart action
        add_to_cart_keywords = ["add to cart", "add to bag", "add to basket", "buy now"]
        is_add_to_cart_action = (
            (intent and any(kw in intent.lower() for kw in add_to_cart_keywords))
            or any(kw in goal.lower() for kw in add_to_cart_keywords)
        )
        # Check if intent asks to match text (e.g., "Click element with text matching [keyword]")
        is_text_matching_action = intent and "matching" in intent.lower()
        # Check if input_text specifies a target to match (for CLICK actions, input_text is target text)
        has_target_text = bool(input_text)

        if is_search_action:
            system = (
                "You are an executor for browser automation.\n"
                "Return ONLY a single-line CLICK(id) action.\n"
                "If you output anything else, the action fails.\n"
                "Do NOT output <think> or any reasoning.\n"
                "SEARCH ICON HINTS: Look for links/buttons with 'search' in text/href, "
                "or icon-only elements (text='0' or empty) with 'search' in href.\n"
                "Output MUST match exactly: CLICK(<digits>) with no spaces.\n"
                "Example: CLICK(12)"
            )
        elif is_text_matching_action or has_target_text:
            # When planner specifies target text (input field), executor must match it
            target_text = input_text or ""
            system = (
                "You are an executor for browser automation.\n"
                "Return ONLY a single-line CLICK(id) action.\n"
                "If you output anything else, the action fails.\n"
                "Do NOT output <think> or any reasoning.\n"
                f"CRITICAL: Find an element with text matching '{target_text}'.\n"
                "- Look for: product titles, category names, link text, button labels\n"
                "- Text must contain the target words (case-insensitive partial match is OK)\n"
                "- If NO element contains the target text, return NONE instead of clicking something random\n"
                "Output: CLICK(<digits>) or NONE\n"
                "Example: CLICK(42) or NONE"
            )
        elif is_product_action:
            # Product click action without specific target - guide executor to find product cards/links
            system = (
                "You are an executor for browser automation.\n"
                "Return ONLY a single-line CLICK(id) action.\n"
                "If you output anything else, the action fails.\n"
                "Do NOT output <think> or any reasoning.\n"
                "PRODUCT CLICK HINTS:\n"
                "- Look for LINK elements (role=link) with product IDs in href (e.g., /7027762, /dp/B...)\n"
                "- Prefer links with delivery info text like 'Delivery', 'Ships to Store', 'Get it...'\n"
                "- These are inside product cards and will navigate to product detail pages\n"
                "- AVOID buttons like 'Search', 'Shop', category buttons, or filter buttons\n"
                "- AVOID image slider options (slider image 1, 2, etc.)\n"
                "Output MUST match exactly: CLICK(<digits>) with no spaces.\n"
                "Example: CLICK(1268)"
            )
        elif is_add_to_cart_action:
            # Add to Cart action - may need to click product first if on search results page
            system = (
                "You are an executor for browser automation.\n"
                "Return ONLY a single-line CLICK(id) action.\n"
                "If you output anything else, the action fails.\n"
                "Do NOT output <think> or any reasoning.\n"
                "ADD TO CART HINTS:\n"
                "- FIRST: Look for buttons with text: 'Add to Cart', 'Add to Bag', 'Add to Basket', 'Buy Now'\n"
                "- If found, click that button directly\n"
                "- FALLBACK: If NO 'Add to Cart' button is visible, you are likely on a SEARCH RESULTS page\n"
                "  - In this case, click a PRODUCT LINK to go to the product details page first\n"
                "  - Look for LINK elements with product IDs in href (e.g., /7027762, /dp/B...)\n"
                "  - Prefer links with product names, prices, or delivery info\n"
                "- AVOID: 'Search' buttons, category buttons, filter buttons, pagination\n"
                "Output MUST match exactly: CLICK(<digits>) with no spaces.\n"
                "Example: CLICK(42)"
            )
        else:
            system = (
                "You are an executor for browser automation.\n"
                "Return ONLY a single-line CLICK(id) action.\n"
                "If you output anything else, the action fails.\n"
                "Do NOT output <think> or any reasoning.\n"
                "No prose, no markdown, no extra whitespace.\n"
                "Output MUST match exactly: CLICK(<digits>) with no spaces.\n"
                "Example: CLICK(12)"
            )

    # Choose the appropriate closing instruction based on action type
    if is_type_action and input_text:
        # For TYPE actions, explicitly ask for TYPE with the text
        action_instruction = f'Return TYPE(id, "{input_text}"):'
    elif input_text:
        # For CLICK with target text, remind to match target or return NONE
        action_instruction = f'Return CLICK(id) for element matching "{input_text}", or NONE if not found:'
    else:
        action_instruction = "Return CLICK(id):"

    user = f"""Goal: {goal}
{intent_line}{category_line}{input_line}
Elements:
{compact_context}

{action_instruction}"""

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

        # Cached pruning category (run-scoped, avoids re-classification per step)
        self._cached_pruning_category: PruningTaskCategory | None = None

        # Token usage tracking
        self._token_collector = _TokenUsageCollector()

    def get_token_stats(self) -> dict[str, Any]:
        """
        Get token usage statistics for the agent session.

        Returns:
            Dictionary with:
            - total: aggregate counts (calls, prompt_tokens, completion_tokens, total_tokens)
            - by_role: breakdown by role (planner, executor, replan, vision)
            - by_model: breakdown by model name

        Example:
            >>> stats = agent.get_token_stats()
            >>> print(f"Total tokens: {stats['total']['total_tokens']}")
            >>> print(f"Planner tokens: {stats['by_role'].get('planner', {}).get('total_tokens', 0)}")
        """
        return self._token_collector.summary()

    def reset_token_stats(self) -> None:
        """Reset token usage statistics to zero."""
        self._token_collector.reset()

    def _record_token_usage(self, role: str, resp: LLMResponse) -> None:
        """Record token usage from an LLM response."""
        try:
            self._token_collector.record(role=role, resp=resp)
        except Exception:
            pass  # Don't fail on token tracking errors

    def _detect_pruning_category(
        self,
        snap: Snapshot,
        goal: str,
    ) -> PruningTaskCategory | None:
        """Resolve the pruning category from task context, then goal-based rules.

        The category is cached for the duration of a run to ensure consistency
        and avoid re-classification on every step.
        """
        # Return cached category if available
        if self._cached_pruning_category is not None:
            return self._cached_pruning_category

        if self._current_task is not None:
            try:
                category = self._current_task.pruning_category_hint()
                if category != PruningTaskCategory.GENERIC:
                    self._cached_pruning_category = category
                    if self.config.verbose:
                        print(f"  [CATEGORY] Detected category from task hint: {category.value}", flush=True)
                    return category
            except Exception:
                pass

            result = classify_task_category(
                task_text=self._current_task.task,
                current_url=self._current_task.starting_url or getattr(snap, "url", "") or "",
                domain_hints=self._current_task.domain_hints,
                task_category=self._current_task.category,
            )
        else:
            result = classify_task_category(
                task_text=goal,
                current_url=getattr(snap, "url", "") or "",
            )

        if result.category == PruningTaskCategory.GENERIC:
            return None

        # Cache the category for this run
        self._cached_pruning_category = result.category
        if self.config.verbose:
            print(f"  [CATEGORY] Detected category: {result.category.value} (confidence={result.confidence:.2f})", flush=True)
        return result.category

    def _get_cached_category_str(self) -> str | None:
        """Get the cached category as a string for executor hints."""
        if self._cached_pruning_category is not None:
            return self._cached_pruning_category.value
        return None

    def _build_pruned_context(
        self,
        snap: Snapshot,
        goal: str,
    ) -> PrunedSnapshotContext | None:
        """Build a category-specific pruned context when task intent is known.

        Uses automatic over-pruning recovery via relaxation levels if the
        initial pruning leaves too few elements.
        """
        if self._context_formatter is not None:
            return None

        category = self._detect_pruning_category(snap, goal)
        if category is None:
            return None

        try:
            ctx = prune_with_recovery(
                snap,
                goal=goal,
                category=category,
                max_relaxation=3,
                verbose=self.config.verbose,
            )
            if self.config.verbose and ctx.relaxation_level == 0:
                print(
                    f"  [PRUNING] {ctx.raw_element_count} -> {ctx.pruned_element_count} elements "
                    f"(category={category.value})",
                    flush=True,
                )
            return ctx
        except Exception:
            return None

    def _format_context(self, snap: Snapshot, goal: str) -> str:
        """
        Format snapshot for LLM context.

        Uses compact format: id|role|text|importance|is_primary|bg|clickable|nearby_text|ord|DG|href
        Same format as documented in reddit_post_planner_executor_local_llm.md
        """
        if self._context_formatter is not None:
            return self._context_formatter(snap, goal)

        pruned_context = self._build_pruned_context(snap, goal)
        if pruned_context is not None and pruned_context.nodes:
            return pruned_context.prompt_block

        import re

        # Filter to interactive elements
        interactive_roles = {
            "button", "link", "textbox", "searchbox", "combobox",
            "checkbox", "radio", "slider", "tab", "menuitem",
            "option", "switch", "cell", "a", "input", "select", "textarea",
        }

        elements = []
        for el in snap.elements:
            role = (getattr(el, "role", "") or "").lower()
            if role in interactive_roles or getattr(el, "clickable", False):
                elements.append(el)

        # Sort by importance
        elements.sort(key=lambda el: getattr(el, "importance", 0) or 0, reverse=True)

        # Build dominant group rank map
        rank_in_group_map: dict[int, int] = {}
        dg_key = getattr(snap, "dominant_group_key", None)
        dg_elements = [el for el in elements if getattr(el, "in_dominant_group", False)]
        if not dg_elements and dg_key:
            dg_elements = [el for el in elements if getattr(el, "group_key", None) == dg_key]

        # Sort dominant group by position
        def rank_sort_key(el):
            doc_y = getattr(el, "doc_y", None) or float("inf")
            bbox = getattr(el, "bbox", None)
            bbox_y = bbox.y if bbox else float("inf")
            bbox_x = bbox.x if bbox else float("inf")
            neg_imp = -(getattr(el, "importance", 0) or 0)
            return (doc_y, bbox_y, bbox_x, neg_imp)

        dg_elements.sort(key=rank_sort_key)
        for rank, el in enumerate(dg_elements):
            rank_in_group_map[el.id] = rank

        # Format lines - take top elements by importance + from dominant group + by position
        selected_ids: set[int] = set()
        selected: list = []

        # Top 40 by importance
        for el in elements[:40]:
            if el.id not in selected_ids:
                selected_ids.add(el.id)
                selected.append(el)

        # Top 20 from dominant group
        for el in dg_elements[:20]:
            if el.id not in selected_ids:
                selected_ids.add(el.id)
                selected.append(el)

        # Top 20 by position
        elements_by_pos = sorted(elements, key=lambda el: (
            getattr(el, "doc_y", None) or float("inf"),
            -(getattr(el, "importance", 0) or 0)
        ))
        for el in elements_by_pos[:20]:
            if el.id not in selected_ids:
                selected_ids.add(el.id)
                selected.append(el)

        def compress_href(href: str | None) -> str:
            if not href:
                return ""
            # Extract domain or last path segment
            href = href.strip()
            if href.startswith("/"):
                parts = href.split("/")
                return parts[-1][:20] if parts[-1] else ""
            try:
                from urllib.parse import urlparse
                parsed = urlparse(href)
                if parsed.path and parsed.path != "/":
                    parts = parsed.path.rstrip("/").split("/")
                    return parts[-1][:20] if parts[-1] else parsed.netloc[:15]
                return parsed.netloc[:15]
            except Exception:
                return href[:20]

        def get_bg_color(el) -> str:
            """Extract background color name from visual_cues."""
            visual_cues = getattr(el, "visual_cues", None)
            if not visual_cues:
                return ""
            # Try bg_color_name first, then bg_fallback
            bg = getattr(visual_cues, "bg_color_name", None) or ""
            if not bg:
                bg = getattr(visual_cues, "bg_fallback", None) or ""
            return bg[:10] if bg else ""

        def get_nearby_text(el) -> str:
            """Extract nearby text for context."""
            nearby = getattr(el, "nearby_text", None) or ""
            if not nearby:
                # Try aria_label or placeholder as fallback
                nearby = getattr(el, "aria_label", None) or ""
            if not nearby:
                nearby = getattr(el, "placeholder", None) or ""
            # Truncate and normalize
            nearby = re.sub(r"\s+", " ", nearby.strip())
            return nearby[:20] if nearby else ""

        lines = []
        for el in selected:
            eid = el.id
            role = getattr(el, "role", "") or ""
            if getattr(el, "href", None):
                role = "link"

            # Truncate and normalize text
            text = getattr(el, "text", "") or ""
            text = re.sub(r"\s+", " ", text.strip())
            if len(text) > 30:
                text = text[:27] + "..."

            importance = getattr(el, "importance", 0) or 0

            # is_primary from visual_cues
            is_primary = "0"
            visual_cues = getattr(el, "visual_cues", None)
            if visual_cues and getattr(visual_cues, "is_primary", False):
                is_primary = "1"

            # bg: background color
            bg = get_bg_color(el)

            # clickable flag
            clickable = "1" if getattr(el, "clickable", False) else "0"

            # nearby_text
            nearby_text = get_nearby_text(el)

            # in dominant group
            in_dg = getattr(el, "in_dominant_group", False)
            if not in_dg and dg_key:
                in_dg = getattr(el, "group_key", None) == dg_key

            # ord: rank in dominant group
            ord_val = rank_in_group_map.get(eid, "") if in_dg else ""
            dg_flag = "1" if in_dg else "0"

            # href
            href = compress_href(getattr(el, "href", None))

            # Format: id|role|text|importance|is_primary|bg|clickable|nearby_text|ord|DG|href
            line = f"{eid}|{role}|{text}|{importance}|{is_primary}|{bg}|{clickable}|{nearby_text}|{ord_val}|{dg_flag}|{href}"
            lines.append(line)

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
        """Parse action from executor response.

        Handles various LLM output formats:
        - CLICK(42)
        - - CLICK(42)  (with leading dash/bullet)
        - TYPE(42, "text")
        - - TYPE(42, "Logitech mouse")
        """
        text = text.strip()

        # Strip <think>...</think> tags (Qwen/DeepSeek reasoning output)
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        # If <think> never closed, strip from first <think> to end
        text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.IGNORECASE).strip()

        # Strip common prefixes (bullets, dashes, asterisks)
        text = re.sub(r"^[-*•]\s*", "", text)

        # CLICK(<id>)
        match = re.search(r"CLICK\((\d+)\)", text)
        if match:
            return "CLICK", [int(match.group(1))]

        # TYPE(<id>, "text") - also handle without quotes
        match = re.search(r'TYPE\((\d+),\s*["\']?([^"\']+?)["\']?\)', text)
        if match:
            return "TYPE", [int(match.group(1)), match.group(2).strip()]

        # PRESS('key')
        match = re.search(r"PRESS\(['\"]?(.+?)['\"]?\)", text)
        if match:
            return "PRESS", [match.group(1)]

        # SCROLL(direction)
        match = re.search(r"SCROLL\((\w+)\)", text)
        if match:
            return "SCROLL", [match.group(1)]

        # FINISH()
        if "FINISH" in text:
            return "FINISH", []

        # NONE - executor couldn't find a suitable element (e.g., no search box found)
        if text.upper() == "NONE" or "NONE" in text.upper():
            return "NONE", []

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
        step: PlanStep | None = None,
    ) -> SnapshotContext:
        """
        Capture snapshot with incremental limit escalation and optional scroll-to-find.

        Progressively increases snapshot limit if elements are missing or
        confidence is low. Escalation can be disabled via config.

        After exhausting limit escalation, if scroll_after_escalation is enabled,
        scrolls down/up to find elements that may be outside the current viewport.

        When escalation is disabled (config.snapshot.enabled=False), only a
        single snapshot at limit_base is captured.

        Args:
            runtime: AgentRuntime for capturing snapshots and scrolling
            goal: Goal string for context formatting
            capture_screenshot: Whether to capture screenshot
            step: Optional PlanStep for goal-aware element detection during scroll
        """
        cfg = self.config.snapshot
        current_limit = cfg.limit_base
        max_limit = cfg.limit_max if cfg.enabled else cfg.limit_base  # Disable escalation if not enabled
        last_snap: Snapshot | None = None
        last_compact: str = ""
        last_pruned_context: PrunedSnapshotContext | None = None
        screenshot_b64: str | None = None
        requires_vision = False
        vision_reason: str | None = None

        while current_limit <= max_limit:
            try:
                snap = await runtime.snapshot(
                    limit=current_limit,
                    screenshot=capture_screenshot,
                    goal=goal,
                    show_overlay=True,  # Show visual overlay for debugging
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

                # Format context FIRST - we always want the compact representation
                # even if vision fallback is required, so the planner can see available elements
                pruned_context = self._build_pruned_context(snap, goal)
                compact = (
                    pruned_context.prompt_block
                    if pruned_context is not None and pruned_context.nodes
                    else self._format_context(snap, goal)
                )
                last_compact = compact
                last_pruned_context = pruned_context

                # Check for vision fallback
                needs_vision, reason = detect_snapshot_failure(snap)
                if needs_vision:
                    requires_vision = True
                    vision_reason = reason
                    break

                # If escalation disabled, we're done after first successful snapshot
                if not cfg.enabled:
                    break

                # Check element count - if sufficient, no need to escalate
                # NOTE: Limit escalation is based on element COUNT only, not on whether
                # a specific target element was found. Intent heuristics are only used
                # for scroll-after-escalation AFTER limit escalation is exhausted.
                elements = getattr(snap, "elements", []) or []
                pruned_node_count = len(pruned_context.nodes) if pruned_context is not None else 0
                if len(elements) >= 10 and (pruned_context is None or pruned_node_count > 0):
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

        # Scroll-after-escalation: if we have a step and scroll is enabled,
        # try scrolling to find elements that may be outside the viewport
        #
        # IMPORTANT: Only trigger scroll-after-escalation for CLICK actions with
        # specific intents that suggest an element may be below the viewport
        # (e.g., "add_to_cart", "checkout"). Do NOT scroll for TYPE_AND_SUBMIT
        # or generic actions where elements are typically visible.
        should_try_scroll = (
            cfg.scroll_after_escalation
            and step is not None
            and last_snap is not None
            and not requires_vision
            and step.action == "CLICK"  # Only for CLICK actions
            and step.intent  # Must have a specific intent
            and self._intent_heuristics is not None  # Must have heuristics to detect
        )

        if should_try_scroll:
            # Check if we can find the target element using intent heuristics
            elements = getattr(last_snap, "elements", []) or []
            url = getattr(last_snap, "url", "") or ""
            found_element = await self._try_intent_heuristics(step, elements, url)

            if found_element is None:
                # Element not found in current viewport - try scrolling
                if self.config.verbose:
                    print(f"  [SNAPSHOT-ESCALATION] Target element not found, trying scroll-after-escalation...", flush=True)

                # Get viewport height and calculate scroll delta
                viewport_height = await runtime.get_viewport_height()
                scroll_delta = viewport_height * cfg.scroll_viewport_fraction

                for direction in cfg.scroll_directions:
                    # Map direction to dy (pixels): down=positive, up=negative
                    scroll_dy = scroll_delta if direction == "down" else -scroll_delta

                    for scroll_num in range(cfg.scroll_max_attempts):
                        if self.config.verbose:
                            print(f"  [SNAPSHOT-ESCALATION] Scrolling {direction} ({scroll_num + 1}/{cfg.scroll_max_attempts})...", flush=True)

                        # Scroll with deterministic verification
                        # scroll_by() returns False if scroll had no effect (reached page boundary)
                        scroll_effective = await runtime.scroll_by(
                            dy=scroll_dy,
                            verify=True,
                            min_delta_px=50.0,
                            js_fallback=True,
                            required=False,  # Don't fail the task if scroll doesn't work
                            timeout_s=5.0,
                        )

                        if not scroll_effective:
                            if self.config.verbose:
                                print(f"  [SNAPSHOT-ESCALATION] Scroll {direction} had no effect (reached boundary), skipping remaining attempts", flush=True)
                            break  # No point trying more scrolls in this direction

                        # Wait for stabilization after successful scroll
                        if self.config.stabilize_enabled:
                            await asyncio.sleep(self.config.stabilize_poll_s)

                        # Take new snapshot at max limit (we already escalated)
                        try:
                            snap = await runtime.snapshot(
                                limit=cfg.limit_max,
                                screenshot=capture_screenshot,
                                goal=goal,
                                show_overlay=True,
                            )
                            if snap is None:
                                continue

                            last_snap = snap
                            last_pruned_context = self._build_pruned_context(snap, goal)
                            last_compact = (
                                last_pruned_context.prompt_block
                                if last_pruned_context is not None and last_pruned_context.nodes
                                else self._format_context(snap, goal)
                            )

                            # Extract screenshot
                            if capture_screenshot:
                                raw_screenshot = getattr(snap, "screenshot", None)
                                if raw_screenshot:
                                    screenshot_b64 = raw_screenshot

                            # Check if target element is now visible
                            elements = getattr(snap, "elements", []) or []
                            url = getattr(snap, "url", "") or ""
                            found_element = await self._try_intent_heuristics(step, elements, url)

                            if found_element is not None:
                                if self.config.verbose:
                                    print(f"  [SNAPSHOT-ESCALATION] Found target element {found_element} after scrolling {direction}", flush=True)
                                # Break out of both loops
                                break
                        except Exception:
                            continue

                    # If found, break out of direction loop
                    if found_element is not None:
                        break

                if found_element is None and self.config.verbose:
                    print(f"  [SNAPSHOT-ESCALATION] Target element not found after scrolling", flush=True)

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
            pruning_category=(
                last_pruned_context.category.value
                if last_pruned_context is not None
                else None
            ),
            pruned_node_count=(
                len(last_pruned_context.nodes)
                if last_pruned_context is not None
                else 0
            ),
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
        page_context: str | None = None,
    ) -> Plan:
        """
        Generate execution plan for the given task.

        Args:
            task: Task description
            start_url: Starting URL
            max_attempts: Maximum planning attempts
            page_context: Optional markdown content of current page for better planning

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
                page_context=page_context if attempt == 1 else None,  # Only include on first attempt
            )

            if self.config.verbose:
                print("\n" + "=" * 60, flush=True)
                print(f"[PLANNER] Attempt {attempt}/{max_attempts}", flush=True)
                print("=" * 60, flush=True)
                print(f"Task: {task}", flush=True)
                print(f"Start URL: {start_url}", flush=True)
                print("-" * 40, flush=True)

            resp = self.planner.generate(
                sys_prompt,
                user_prompt,
                temperature=self.config.planner_temperature,
                max_new_tokens=max_tokens,
            )
            self._record_token_usage("planner", resp)
            last_output = resp.content

            if self.config.verbose:
                print("\n--- Planner Response ---", flush=True)
                print(resp.content, flush=True)
                print("--- End Response ---\n", flush=True)

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

                if self.config.verbose:
                    import json
                    print("\n=== PLAN GENERATED ===", flush=True)
                    print(json.dumps(plan.model_dump(), indent=2, default=str), flush=True)
                    print("=== END PLAN ===\n", flush=True)

                return plan

            except Exception as e:
                last_errors = str(e)
                if self.config.verbose:
                    print(f"[PLANNER] Parse error: {e}", flush=True)
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
Return ONLY a JSON object with mode="patch" and replace_steps array.

IMPORTANT - Alternative approaches when CLICK fails:
- If a product/category navigation failed, USE SITE SEARCH instead:
  * Replace the failed CLICK with a TYPE_AND_SUBMIT to search for the product
  * This is the MOST RELIABLE fallback - site search works on all websites
- If clicking a specific element failed:
  * Try a different selector or button (e.g., "Quick Shop", "View Details")
- Don't just retry the same approach with minor changes"""

        # Extract product/item name from step for search suggestion
        product_hint = ""
        step_labels = " ".join([
            failed_step.goal or "",
            failed_step.target or "",
            failed_step.intent or "",
        ]).lower()
        # Common patterns to extract product name
        for pattern in [r"snow\s*blower", r"product\s+(\w+)", r"click\s+(?:on\s+)?(.+?)(?:\s+product|\s+category)?$"]:
            match = re.search(pattern, step_labels, re.IGNORECASE)
            if match:
                product_hint = match.group(0) if match.lastindex is None else match.group(1)
                break
        if not product_hint and failed_step.target:
            product_hint = failed_step.target

        user = f"""Task: {task}

Failure:
- Step ID: {failed_step.id}
- Step goal: {failed_step.goal}
- Reason: {failure_reason}

IMPORTANT: The element could not be found or clicked. The current page likely doesn't have the target.
The BEST approach is to USE SITE SEARCH to find the product directly.

RECOMMENDED: Replace the failed step with a site search:
{{
  "mode": "patch",
  "replace_steps": [
    {{
      "id": {failed_step.id},
      "step": {{ "id": {failed_step.id}, "goal": "Search for {product_hint or 'the product'}", "action": "TYPE_AND_SUBMIT", "input": "{product_hint or 'product name'}", "intent": "Type in search box and submit", "verify": [{{"predicate": "url_contains", "args": ["search"]}}] }}
    }}
  ]
}}

Alternative approaches (if search doesn't apply):
1. Click "Catalog" or "Shop All" to browse products
2. Click "Quick Shop" or "View Details" buttons

Return JSON patch:"""

        for attempt in range(1, max_attempts + 1):
            resp = self.planner.generate(
                system,
                user,
                temperature=self.config.planner_temperature,
                max_new_tokens=1024,
            )
            self._record_token_usage("replan", resp)
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

    async def _scroll_to_find_element(
        self,
        runtime: AgentRuntime,
        step: PlanStep,
        goal: str,
    ) -> tuple[int | None, "SnapshotContext | None"]:
        """
        Scroll the page to find an element that matches the step's goal/intent.

        This is used when the initial snapshot doesn't contain the target element
        (e.g., "Add to Cart" button below the viewport).

        Returns:
            (element_id, new_snapshot_context) if found, (None, None) otherwise
        """
        if not self.config.scroll_to_find_enabled:
            return None, None

        for direction in self.config.scroll_to_find_directions:
            for scroll_num in range(self.config.scroll_to_find_max_scrolls):
                if self.config.verbose:
                    print(f"  [SCROLL-TO-FIND] Scrolling {direction} ({scroll_num + 1}/{self.config.scroll_to_find_max_scrolls})...", flush=True)

                # Scroll
                await runtime.scroll(direction)

                # Wait for stabilization
                if self.config.stabilize_enabled:
                    await asyncio.sleep(self.config.stabilize_poll_s)

                # Take new snapshot (don't pass step to avoid recursive scroll-to-find)
                ctx = await self._snapshot_with_escalation(
                    runtime,
                    goal=goal,
                    capture_screenshot=self.config.trace_screenshots,
                    step=None,  # Avoid recursive scroll-after-escalation
                )

                # Try heuristics on new snapshot
                elements = getattr(ctx.snapshot, "elements", []) or []
                url = getattr(ctx.snapshot, "url", "") or ""
                element_id = await self._try_intent_heuristics(step, elements, url)

                if element_id is not None:
                    if self.config.verbose:
                        print(f"  [SCROLL-TO-FIND] Found element {element_id} after scrolling {direction}", flush=True)
                    return element_id, ctx

                # Try LLM executor on new snapshot
                sys_prompt, user_prompt = build_executor_prompt(
                    step.goal,
                    step.intent,
                    ctx.compact_representation,
                    input_text=step.input,
                    category=self._get_cached_category_str(),
                    action_type=step.action,
                )
                resp = self.executor.generate(
                    sys_prompt,
                    user_prompt,
                    temperature=self.config.executor_temperature,
                    max_new_tokens=self.config.executor_max_tokens,
                )
                self._record_token_usage("executor", resp)
                parsed_action, parsed_args = self._parse_action(resp.content)

                if parsed_action == "CLICK" and parsed_args:
                    element_id = parsed_args[0]
                    # Verify element exists in snapshot
                    if any(getattr(el, "id", None) == element_id for el in elements):
                        if self.config.verbose:
                            print(f"  [SCROLL-TO-FIND] LLM found element {element_id} after scrolling {direction}", flush=True)
                        return element_id, ctx

                # If LLM suggests SCROLL, it means element still not visible
                if parsed_action == "SCROLL":
                    continue

        if self.config.verbose:
            print(f"  [SCROLL-TO-FIND] Element not found after scrolling", flush=True)
        return None, None

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
                    step=substep,  # Enable scroll-after-escalation for substeps
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
                            input_text=substep.input,
                            category=self._get_cached_category_str(),
                            action_type=substep.action,
                        )
                        resp = self.executor.generate(
                            sys_prompt,
                            user_prompt,
                            temperature=self.config.executor_temperature,
                            max_new_tokens=self.config.executor_max_tokens,
                        )
                        self._record_token_usage("executor", resp)
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

    def _word_boundary_match(self, pattern: str, text: str) -> bool:
        """
        Match pattern as a word boundary to avoid false positives.

        For example, "x" should match "x" but not "mexico" or "boxer".
        "close" should match "close" or "Close Dialog" but not "enclosed".
        """
        import re

        if not pattern or not text:
            return False

        pattern_lower = pattern.lower()
        text_lower = text.lower()

        # For very short patterns (1-2 chars like "x", "×"), require exact match
        if len(pattern) <= 2:
            return text_lower == pattern_lower or text_lower.strip() == pattern_lower

        # For longer patterns, use word boundary matching
        try:
            return bool(re.search(r'\b' + re.escape(pattern_lower) + r'\b', text_lower))
        except re.error:
            # Fall back to contains if regex fails
            return pattern_lower in text_lower

    async def _attempt_modal_dismissal(
        self,
        runtime: AgentRuntime,
        post_snap: Any,
    ) -> bool:
        """
        Attempt to dismiss a modal/drawer overlay after DOM change detection.

        This handles common blocking scenarios like:
        - Product protection/warranty upsells
        - Cookie consent banners
        - Newsletter signup popups
        - Promotional overlays
        - Cart upsell drawers

        The method looks for buttons with common dismissal text patterns
        and clicks them to clear the overlay.

        Uses word boundary matching to avoid false positives like:
        - "mexico" matching "x"
        - "enclosed" matching "close"
        - "boxer" matching "x"

        Args:
            runtime: The AgentRuntime for browser control
            post_snap: The snapshot after DOM change was detected

        Returns:
            True if a dismissal was attempted, False otherwise
        """
        if not self.config.modal.enabled:
            return False

        cfg = self.config.modal
        elements = getattr(post_snap, "elements", []) or []

        # CRITICAL: Check if this drawer/modal contains CLICKABLE checkout-related elements.
        # Only skip dismissal if there's an actual button/link the user should interact with.
        # Informational text like "Added to cart" or "Subtotal" doesn't count.
        checkout_button_patterns = (
            "checkout", "check out", "proceed to checkout", "go to checkout",
            "view cart", "view bag", "shopping cart", "shopping bag",
            "continue to checkout", "secure checkout",
            "go to cart", "see cart", "go to bag",
        )

        # Check for clickable checkout buttons/links
        for el in elements:
            role = (getattr(el, "role", "") or "").lower()
            if role not in ("button", "link"):
                continue

            if self._is_global_nav_cart_link(el):
                continue

            text = (getattr(el, "text", "") or "").lower()
            aria_label = (getattr(el, "aria_label", "") or getattr(el, "ariaLabel", "") or "").lower()
            href = (getattr(el, "href", "") or "").lower()

            # Check text/aria-label for checkout patterns
            for pattern in checkout_button_patterns:
                if pattern in text or pattern in aria_label:
                    if self.config.verbose:
                        print(f"  [MODAL] Skipping dismissal - found clickable checkout element: '{pattern}' in {role}", flush=True)
                    return False

            # Check href for cart/checkout links
            if "cart" in href or "checkout" in href or "bag" in href:
                if self.config.verbose:
                    print(f"  [MODAL] Skipping dismissal - found cart/checkout link: '{href}'", flush=True)
                return False

        # Find candidates that match dismissal patterns
        candidates: list[tuple[int, int, str]] = []  # (element_id, score, matched_pattern)

        for el in elements:
            el_id = getattr(el, "id", None)
            if el_id is None:
                continue

            role = (getattr(el, "role", "") or "").lower()
            if role not in cfg.role_filter:
                continue

            # Get text and aria_label for matching
            text = (getattr(el, "text", "") or "").lower().strip()
            aria_label = (getattr(el, "aria_label", "") or getattr(el, "ariaLabel", "") or "").lower().strip()
            labels = [lbl for lbl in [text, aria_label] if lbl]

            if not labels:
                continue

            matched = False

            # Check icon patterns first (require exact match)
            for i, icon in enumerate(cfg.dismiss_icons):
                icon_lower = icon.lower()
                score = 200 + len(cfg.dismiss_icons) - i  # Icons get high priority

                for lbl in labels:
                    if lbl == icon_lower:
                        candidates.append((el_id, score, icon))
                        matched = True
                        break
                if matched:
                    break

            if matched:
                continue

            # Check word boundary patterns
            for i, pattern in enumerate(cfg.dismiss_patterns):
                score = len(cfg.dismiss_patterns) - i  # Higher score for earlier patterns

                for lbl in labels:
                    if self._word_boundary_match(pattern, lbl):
                        candidates.append((el_id, score, pattern))
                        matched = True
                        break
                if matched:
                    break

        if not candidates:
            if self.config.verbose:
                print("  [MODAL] No dismissal candidates found", flush=True)
            return False

        # Sort by score (highest first) and try the best candidates
        candidates.sort(key=lambda x: x[1], reverse=True)

        for attempt in range(min(cfg.max_attempts, len(candidates))):
            el_id, score, pattern = candidates[attempt]
            if self.config.verbose:
                print(f"  [MODAL] Attempting dismissal: clicking element {el_id} (pattern: '{pattern}')", flush=True)

            try:
                await runtime.click(el_id)
                # Wait briefly for modal to close
                await asyncio.sleep(0.3)

                # Check if modal was dismissed (DOM changed again)
                try:
                    new_snap = await runtime.snapshot(emit_trace=False)
                    new_elements = set(getattr(el, "id", 0) for el in (new_snap.elements or []))
                    post_elements = set(getattr(el, "id", 0) for el in (post_snap.elements or []))
                    removed_elements = post_elements - new_elements

                    if len(removed_elements) > 3:
                        if self.config.verbose:
                            print(f"  [MODAL] Dismissal successful ({len(removed_elements)} elements removed)", flush=True)
                        return True
                except Exception:
                    pass  # Continue with next attempt

            except Exception as e:
                if self.config.verbose:
                    print(f"  [MODAL] Dismissal click failed: {e}", flush=True)
                continue

        if self.config.verbose:
            print("  [MODAL] All dismissal attempts exhausted", flush=True)
        return False

    def _is_global_nav_cart_link(self, el: Any) -> bool:
        """
        Detect persistent header/nav cart links that should not be treated as
        drawer-local checkout controls.
        """
        href = (getattr(el, "href", "") or "").lower()
        text = (getattr(el, "text", "") or "").lower().strip()
        aria_label = (getattr(el, "aria_label", "") or getattr(el, "ariaLabel", "") or "").lower().strip()
        label = text or aria_label

        layout = getattr(el, "layout", None)
        region = (getattr(layout, "region", "") or "").lower()
        doc_y = getattr(el, "doc_y", None)

        if "nav_cart" in href or "ref_=nav_cart" in href:
            return True

        if region in {"header", "nav"} and (
            "cart" in href or label in {"cart", "0 items in cart"} or "items in cart" in label
        ):
            return True

        try:
            if doc_y is not None and float(doc_y) <= 120 and (
                "cart" in href or label in {"cart", "0 items in cart"} or "items in cart" in label
            ):
                return True
        except (TypeError, ValueError):
            pass

        return False

    def _looks_like_search_submission(self, step: PlanStep, element: Any | None) -> bool:
        """Detect TYPE_AND_SUBMIT actions that are likely site search submissions."""
        role = (getattr(element, "role", "") or "").lower() if element is not None else ""
        if role in {"searchbox", "combobox"}:
            return True

        labels = " ".join(
            str(part or "")
            for part in (
                step.goal,
                step.intent,
                step.input,
                getattr(element, "text", None),
                getattr(element, "name", None),
                getattr(element, "aria_label", None),
                getattr(element, "ariaLabel", None),
            )
        ).lower()
        return "search" in labels

    def _is_add_to_cart_step(self, step: PlanStep) -> bool:
        """Detect if a step is an Add to Cart action."""
        add_to_cart_keywords = ["add to cart", "add to bag", "add to basket", "buy now"]
        labels = " ".join(
            str(part or "").lower()
            for part in (step.goal, step.intent, step.input)
        )
        return any(kw in labels for kw in add_to_cart_keywords)

    def _is_search_results_url(self, url: str) -> bool:
        """Check if URL looks like a search results page."""
        url_lower = url.lower()
        # Common patterns for search results pages
        search_patterns = [
            "search",
            "query=",
            "q=",
            "s=",
            "/s?",
            "keyword=",
            "keywords=",
            "results",
        ]
        return any(pattern in url_lower for pattern in search_patterns)

    def _is_category_navigation_step(self, step: PlanStep) -> bool:
        """Check if this step is navigating to a category/section."""
        nav_keywords = [
            "navigate to", "go to", "click category", "category link",
            "click on", "browse", "section", "department"
        ]
        labels = " ".join(
            str(part or "").lower()
            for part in (step.goal, step.intent)
        )
        return any(kw in labels for kw in nav_keywords)

    def _url_change_matches_intent(self, step: PlanStep, pre_url: str, post_url: str) -> bool:
        """
        Check if URL change actually matches the step's intent.

        For category navigation, the new URL should contain keywords from the target.
        This prevents accepting unrelated URL changes as successful navigation.
        """
        # Extract target keywords from step
        target = step.target or ""
        intent = step.intent or ""
        goal = step.goal or ""

        post_url_lower = post_url.lower()

        # Special case: checkout/cart related steps
        # These steps may go to /cart first before /checkout, which is valid
        checkout_keywords = ["checkout", "proceed to checkout", "cart", "view cart"]
        step_labels = f"{goal} {intent}".lower()
        is_checkout_step = any(kw in step_labels for kw in checkout_keywords)
        if is_checkout_step:
            # Accept cart or checkout URLs as valid for checkout steps
            checkout_url_patterns = ["cart", "checkout", "basket", "bag"]
            if any(pattern in post_url_lower for pattern in checkout_url_patterns):
                return True

        # Get keywords from target (e.g., "Outdoor Power Equipment" -> ["outdoor", "power", "equipment"])
        target_words = set(
            word.lower() for word in re.split(r'[\s\-_]+', target)
            if len(word) >= 3  # Skip short words like "to", "and"
        )

        # Also check predicates for expected URL patterns
        expected_patterns = []
        for pred in (step.verify or []):
            if pred.predicate == "url_contains" and pred.args:
                expected_patterns.append(pred.args[0].lower())

        # If predicates specify URL patterns, check those
        if expected_patterns:
            if any(pattern in post_url_lower for pattern in expected_patterns):
                return True
            # For non-checkout steps, reject URL changes that don't match predicates
            # But only if we have a target to validate against
            if target_words:
                return False
            # No target and no predicate match - be permissive
            return True

        # Otherwise check if target keywords appear in URL
        if target_words:
            # At least one target word should appear in URL
            if any(word in post_url_lower for word in target_words):
                return True
            # URL doesn't contain any target keywords - suspicious
            return False

        # No target specified - can't validate, allow fallback
        return True

    def _find_submit_button_for_type_and_submit(
        self,
        *,
        elements: list[Any],
        input_element_id: int | None,
        step: PlanStep,
    ) -> int | None:
        """Find an explicit search/submit control for search-style TYPE_AND_SUBMIT steps."""
        selected_element = None
        for el in elements:
            if getattr(el, "id", None) == input_element_id:
                selected_element = el
                break

        if not self._looks_like_search_submission(step, selected_element):
            return None

        candidates: list[tuple[int, int]] = []
        for el in elements:
            el_id = getattr(el, "id", None)
            if el_id is None or el_id == input_element_id:
                continue

            role = (getattr(el, "role", "") or "").lower()
            if role not in {"button", "link"}:
                continue

            label = " ".join(
                str(part or "")
                for part in (
                    getattr(el, "text", None),
                    getattr(el, "name", None),
                    getattr(el, "aria_label", None),
                    getattr(el, "ariaLabel", None),
                )
            ).lower()
            href = (getattr(el, "href", "") or "").lower()

            score = 0
            if "submit search" in label:
                score += 120
            if "search" in label:
                score += 80
            if "submit" in label:
                score += 60
            if label.strip() in {"go", "search"}:
                score += 50
            if "/search" in href or "search?" in href or "q=" in href:
                score += 40

            if score > 0:
                score += int(getattr(el, "importance", 0) or 0) // 100
                candidates.append((int(el_id), score))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0]

    def _type_and_submit_url_change_looks_valid(
        self,
        *,
        pre_url: str,
        post_url: str,
        step: PlanStep,
        element: Any | None,
        typed_text: str,
    ) -> bool:
        """
        Allow URL-change fallback for TYPE_AND_SUBMIT only when the resulting URL
        still matches the expected semantics of the typed action.
        """
        if not self._looks_like_search_submission(step, element):
            return True

        from urllib.parse import quote_plus, urlparse

        post_lower = post_url.lower()
        if any(marker in post_lower for marker in ("/search", "?q=", "&q=", "query=", "search=", "keyword=")):
            return True

        encoded_query = quote_plus((typed_text or "").strip().lower())
        if encoded_query and encoded_query in post_lower:
            return True

        parsed = urlparse(post_url)
        searchable = f"{parsed.path}?{parsed.query}".lower()
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", (typed_text or "").lower()) if len(tok) >= 3]
        if tokens:
            matched = sum(1 for tok in tokens[:4] if tok in searchable)
            if matched >= min(2, len(tokens[:4])):
                return True

        return False

    def _choose_type_and_submit_submit_method(
        self,
        *,
        elements: list[Any],
        input_element_id: int | None,
        step: PlanStep,
        prefer_alternate_of: Literal["click", "enter"] | None = None,
    ) -> tuple[Literal["click", "enter"], int | None]:
        """Choose the submit method for TYPE_AND_SUBMIT, optionally preferring the alternate path.

        NOTE: For search-like submissions, we prefer Enter key by default (matching WebBench behavior).
        Many search boxes (e.g., lifeisgood.com) don't have a proper submit button, or clicking the
        "submit" button navigates to a category page instead of performing a search. Pressing Enter
        is more reliable for search inputs.

        When prefer_alternate_of is set, we try to return the opposite method for retry purposes:
        - If prefer_alternate_of="enter", try to return "click" (if a submit button exists)
        - If prefer_alternate_of="click", return "enter"
        """
        submit_button_id = self._find_submit_button_for_type_and_submit(
            elements=elements,
            input_element_id=input_element_id,
            step=step,
        )

        # For search-like submissions, prefer Enter key by default (more reliable)
        # Only fall back to button click if Enter doesn't work (via prefer_alternate_of)
        default_method: Literal["click", "enter"] = "enter"

        # Handle retry case: prefer the alternate method
        if prefer_alternate_of == "enter" and submit_button_id is not None:
            # First attempt used Enter, retry with click (if button available)
            return "click", submit_button_id
        if prefer_alternate_of == "click":
            # First attempt used click, retry with Enter
            return "enter", None

        return default_method, submit_button_id

    def _get_runtime_page(self, runtime: AgentRuntime) -> Any | None:
        """Best-effort access to the live browser page for immediate URL observation."""
        backend = getattr(runtime, "backend", None)
        candidates = [
            getattr(backend, "page", None),
            getattr(backend, "_page", None),
            getattr(runtime, "_legacy_page", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if type(candidate).__module__.startswith("unittest.mock"):
                continue
            if inspect.getattr_static(candidate, "url", None) is not None:
                return candidate
        return None

    async def _read_focused_input_value(self, runtime: AgentRuntime) -> str | None:
        """Best-effort read of the currently focused input value."""
        page = self._get_runtime_page(runtime)
        if page is None:
            return None
        try:
            value = await page.evaluate(
                """
                () => {
                    const el = document.activeElement;
                    if (!el) return null;
                    if ("value" in el) return el.value ?? "";
                    return null;
                }
                """
            )
        except Exception:
            return None
        return value if isinstance(value, str) else None

    def _normalize_input_value(self, value: str | None) -> str:
        """Normalize input text for equality checks across controlled inputs."""
        return " ".join((value or "").strip().lower().split())

    async def _clear_and_type_search_input(
        self,
        *,
        runtime: AgentRuntime,
        input_element_id: int,
        text: str,
    ) -> bool:
        """Clear and type into a search input using the live page when available."""
        page = self._get_runtime_page(runtime)
        if page is None:
            return False

        browser_like = getattr(runtime, "_legacy_browser", None) or SimpleNamespace(page=page)

        try:
            await runtime.click(input_element_id)
        except Exception:
            return False

        try:
            clear_result = await clear_async(browser_like, int(input_element_id), take_snapshot=False)
            if getattr(clear_result, "success", False):
                await runtime.record_action(f"CLEAR({input_element_id})")
        except Exception:
            pass

        try:
            select_all_key = "Meta+A" if sys.platform == "darwin" else "Control+A"
            await page.keyboard.press(select_all_key)
            await page.keyboard.press("Backspace")
            await runtime.record_action(f"PRESS({select_all_key})")
            await runtime.record_action('PRESS("Backspace")')
        except Exception:
            pass

        try:
            delay_ms = float(self.config.type_delay_ms or 0)
            type_result = await type_text_async(
                browser_like,
                int(input_element_id),
                str(text),
                take_snapshot=False,
                delay_ms=delay_ms,
            )
            if not getattr(type_result, "success", False):
                return False
            await runtime.record_action(
                f"TYPE({input_element_id}, '{text[:20]}...')" if len(text) > 20 else f"TYPE({input_element_id}, '{text}')"
            )
            return True
        except Exception:
            return False

    async def _submit_if_already_typed(
        self,
        *,
        runtime: AgentRuntime,
        elements: list[Any],
        input_element_id: int,
        step: PlanStep,
        text: str,
        pre_url: str,
        typed_element: Any | None,
        telemetry: SearchSubmitTelemetry,
    ) -> bool:
        """Submit without retyping when the focused input already contains the desired text."""
        page = self._get_runtime_page(runtime)
        if page is None:
            return False

        try:
            await runtime.click(input_element_id)
        except Exception:
            return False

        try:
            await page.wait_for_timeout(80)
        except Exception:
            pass

        current_value = await self._read_focused_input_value(runtime)
        if self._normalize_input_value(current_value) != self._normalize_input_value(text):
            return False

        await self._submit_type_and_submit(
            runtime=runtime,
            elements=elements,
            input_element_id=input_element_id,
            step=step,
            text=text,
            pre_url=pre_url,
            typed_element=typed_element,
            telemetry=telemetry,
        )
        await runtime.record_action(f"SUBMIT_ALREADY_TYPED({input_element_id})")
        return True

    def _snapshot_looks_like_search_results(self, snapshot: Any, typed_text: str) -> bool:
        """Best-effort heuristic for product/search results-like pages."""
        elements = getattr(snapshot, "elements", []) or []
        if not elements:
            return False

        tokens = [tok for tok in re.split(r"[^a-z0-9]+", (typed_text or "").lower()) if len(tok) >= 3]
        product_like_matches = 0
        token_matches = 0

        for el in elements:
            role = (getattr(el, "role", "") or "").lower()
            if role != "link":
                continue
            href = (getattr(el, "href", "") or "").lower()
            label = " ".join(
                str(part or "")
                for part in (
                    getattr(el, "text", None),
                    getattr(el, "name", None),
                    getattr(el, "aria_label", None),
                    getattr(el, "ariaLabel", None),
                )
            ).lower()
            blob = f"{href} {label}"

            if any(p in href for p in ("/product/", "/products/", "/p/", "/dp/")):
                product_like_matches += 1
            if tokens:
                token_matches += sum(1 for tok in tokens[:4] if tok in blob)

        return product_like_matches > 0 or token_matches >= 2

    async def _capture_search_results_snapshot_evidence(
        self,
        *,
        runtime: AgentRuntime,
        typed_text: str,
        telemetry: SearchSubmitTelemetry,
    ) -> None:
        """Capture one post-submit snapshot to track results-like evidence."""
        try:
            snap = await runtime.snapshot(emit_trace=False)
        except Exception:
            return

        telemetry.observed_search_results_dom = self._snapshot_looks_like_search_results(snap, typed_text)

    async def _submit_type_and_submit(
        self,
        *,
        runtime: AgentRuntime,
        elements: list[Any],
        input_element_id: int,
        step: PlanStep,
        text: str,
        pre_url: str,
        typed_element: Any | None,
        telemetry: SearchSubmitTelemetry,
        prefer_alternate_of: Literal["click", "enter"] | None = None,
    ) -> None:
        """Submit a search-like TYPE_AND_SUBMIT using the chosen method and record telemetry."""
        submit_method, submit_target = self._choose_type_and_submit_submit_method(
            elements=elements,
            input_element_id=input_element_id,
            step=step,
            prefer_alternate_of=prefer_alternate_of,
        )

        if submit_method == "click" and submit_target is not None:
            await runtime.click(submit_target)
            if self.config.verbose:
                print(f"  [ACTION] TYPE_AND_SUBMIT submit via CLICK({submit_target})", flush=True)
            # Wait briefly for page load after clicking submit button
            page = self._get_runtime_page(runtime)
            if page is not None:
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=2000)
                except Exception:
                    pass
        else:
            await runtime.press("Enter")
            if self.config.verbose:
                print("  [ACTION] TYPE_AND_SUBMIT submit via PRESS(Enter)", flush=True)
            submit_method = "enter"

            # Wait briefly for URL change after Enter
            page = self._get_runtime_page(runtime)
            if page is not None:
                try:
                    await page.wait_for_url(lambda url: url != pre_url, timeout=3000)
                    if self.config.verbose:
                        print(f"  [SEARCH] URL changed to: {page.url}", flush=True)
                except Exception:
                    if self.config.verbose:
                        print("  [SEARCH] URL unchanged after Enter", flush=True)

        if prefer_alternate_of is None:
            telemetry.first_submit_method = submit_method
        else:
            telemetry.retry_submit_method = submit_method

        await runtime.stabilize()
        await self._capture_search_results_snapshot_evidence(
            runtime=runtime,
            typed_text=text,
            telemetry=telemetry,
        )

    async def _retry_search_widget_submission(
        self,
        *,
        runtime: AgentRuntime,
        elements: list[Any],
        input_element_id: int,
        step: PlanStep,
        text: str,
        pre_url: str,
        typed_element: Any | None,
        telemetry: SearchSubmitTelemetry,
    ) -> bool:
        """Retry a failed search submission once with a clean field and the alternate submit method."""
        if telemetry.first_submit_method not in {"click", "enter"}:
            return False

        # Take a fresh snapshot in case DOM changed
        fresh_elements = elements
        try:
            fresh_snap = await runtime.snapshot(emit_trace=False)
            fresh_elements = getattr(fresh_snap, "elements", []) or elements
            if self.config.verbose:
                print(f"  [SEARCH-RETRY] Fresh snapshot: {len(fresh_elements)} elements", flush=True)
        except Exception:
            pass

        retry_method, _ = self._choose_type_and_submit_submit_method(
            elements=fresh_elements,
            input_element_id=input_element_id,
            step=step,
            prefer_alternate_of=telemetry.first_submit_method,
        )
        if retry_method == telemetry.first_submit_method:
            if self.config.verbose:
                print("  [SEARCH-RETRY] No alternate submit method available", flush=True)
            return False

        select_all_key = "Meta+A" if sys.platform == "darwin" else "Control+A"
        if self.config.verbose:
            print(f"  [SEARCH-RETRY] Retrying search via alternate submit method ({retry_method})", flush=True)
        typed_ok = await self._clear_and_type_search_input(
            runtime=runtime,
            input_element_id=input_element_id,
            text=text,
        )
        if not typed_ok:
            await runtime.click(input_element_id)
            await runtime.press(select_all_key)
            await runtime.press("Backspace")
            await runtime.type(input_element_id, text, delay_ms=self.config.type_delay_ms)
        await self._submit_type_and_submit(
            runtime=runtime,
            elements=fresh_elements,
            input_element_id=input_element_id,
            step=step,
            text=text,
            pre_url=pre_url,
            typed_element=typed_element,
            telemetry=telemetry,
            prefer_alternate_of=telemetry.first_submit_method,
        )
        return await self._verify_step(runtime, step)

    def _looks_like_overlay_dismiss_intent(self, *, goal: str, intent: str) -> bool:
        """
        Detect steps that are dismissing overlays, modals, cookie banners, or popups.
        These clicks should verify that the page state changed (modal was dismissed).
        """
        g = (goal or "").lower()
        i = (intent or "").lower()
        text = f"{g} {i}"
        # Overlay/modal/popup dismissal patterns
        overlay_keywords = (
            "cookie",
            "consent",
            "gdpr",
            "privacy",
            "accept",
            "dismiss",
            "close",
            "overlay",
            "modal",
            "popup",
            "pop-up",
            "banner",
            "dialog",
            "notification",
            "newsletter",
            "subscribe",
        )
        dismiss_verbs = ("accept", "dismiss", "close", "clear", "decline", "reject", "got it", "ok", "okay")
        has_overlay_keyword = any(kw in text for kw in overlay_keywords)
        has_dismiss_verb = any(v in text for v in dismiss_verbs)
        # Match if both overlay context and dismiss intent are present
        if has_overlay_keyword and has_dismiss_verb:
            return True
        # Also match explicit overlay/modal/banner mentions
        if any(kw in text for kw in ("overlay", "modal", "banner", "popup", "pop-up", "dialog")):
            return True
        return False

    async def _detect_checkout_page(
        self,
        runtime: AgentRuntime,
        snapshot: Any | None = None,
    ) -> tuple[bool, str | None]:
        """
        Detect if the current page is a checkout-relevant page.

        This detects pages that indicate the agent should continue
        the checkout flow, such as:
        - Cart pages
        - Checkout pages
        - Sign-in pages during checkout

        Args:
            runtime: The AgentRuntime for browser control
            snapshot: Optional snapshot to check (avoids re-capture)

        Returns:
            (is_checkout_page, page_type) - page_type is 'cart', 'checkout', 'login', etc.
        """
        if not self.config.checkout.enabled:
            return False, None

        cfg = self.config.checkout

        # Get current URL
        try:
            current_url = await runtime.get_url() if hasattr(runtime, "get_url") else None
        except Exception:
            current_url = None

        if not current_url:
            return False, None

        url_lower = current_url.lower()

        # Check URL patterns
        for pattern in cfg.url_patterns:
            if pattern.lower() in url_lower:
                page_type = self._classify_checkout_page(pattern)
                if self.config.verbose:
                    print(f"  [CHECKOUT] Detected {page_type} page (URL: {pattern})", flush=True)
                return True, page_type

        # Check element patterns if we have a snapshot
        if snapshot:
            elements = getattr(snapshot, "elements", []) or []
            for el in elements:
                text = (getattr(el, "text", "") or "").lower()
                for pattern in cfg.element_patterns:
                    if pattern.lower() in text:
                        page_type = self._classify_checkout_page(pattern)
                        if self.config.verbose:
                            print(f"  [CHECKOUT] Detected {page_type} page (element: '{pattern}')", flush=True)
                        return True, page_type

        return False, None

    def _classify_checkout_page(self, pattern: str) -> str:
        """Classify the type of checkout page based on the matched pattern."""
        pattern_lower = pattern.lower()

        if any(kw in pattern_lower for kw in ["cart", "basket", "bag"]):
            return "cart"
        elif any(kw in pattern_lower for kw in ["signin", "sign-in", "login", "auth"]):
            return "login"
        elif any(kw in pattern_lower for kw in ["payment", "pay"]):
            return "payment"
        elif any(kw in pattern_lower for kw in ["order", "place"]):
            return "order"
        else:
            return "checkout"

    def _should_continue_after_checkout_detection(
        self,
        page_type: str | None,
        current_step_index: int,
        total_steps: int,
    ) -> bool:
        """
        Determine if the agent should trigger a replan after detecting a checkout page.

        Returns True if:
        - We're on the last step or near the end
        - The page type indicates we need more steps (e.g., login, payment)
        """
        if not page_type:
            return False

        # If we're on the last few steps and detected checkout, we may need to continue
        steps_remaining = total_steps - current_step_index - 1

        # Always continue if we hit a login page (authentication required)
        if page_type == "login":
            return True

        # Continue if we're on cart/checkout and have no more steps
        if page_type in ("cart", "checkout", "payment", "order") and steps_remaining <= 1:
            return True

        return False

    def _build_checkout_continuation_task(
        self,
        original_task: str,
        page_type: str | None,
    ) -> str:
        """
        Build a task description for continuing from a checkout page.

        Args:
            original_task: The original task description
            page_type: Type of checkout page detected

        Returns:
            A new task description that continues from the current page
        """
        if page_type == "login":
            return f"Continue from sign-in page: Complete sign-in if credentials are available, or proceed as guest if possible. Original task: {original_task}"
        elif page_type == "cart":
            return f"Continue from cart page: Proceed to checkout to complete the purchase. Original task: {original_task}"
        elif page_type == "payment":
            return f"Continue from payment page: Complete payment details and place order. Original task: {original_task}"
        elif page_type == "order":
            return f"Continue from order page: Review and place the order. Original task: {original_task}"
        else:
            return f"Continue checkout process from current page. Original task: {original_task}"

    async def _detect_auth_boundary(
        self,
        runtime: AgentRuntime,
    ) -> bool:
        """
        Detect if the current page is an authentication boundary.

        An auth boundary is a login/sign-in page where the agent cannot
        proceed without credentials. This is a terminal state.

        Args:
            runtime: The AgentRuntime for browser control

        Returns:
            True if on an authentication page, False otherwise
        """
        if not self.config.auth_boundary.enabled:
            return False

        cfg = self.config.auth_boundary

        # Get current URL
        try:
            current_url = await runtime.get_url() if hasattr(runtime, "get_url") else None
        except Exception:
            current_url = None

        if not current_url:
            return False

        url_lower = current_url.lower()

        # Check URL patterns
        for pattern in cfg.url_patterns:
            if pattern.lower() in url_lower:
                if self.config.verbose:
                    print(f"  [AUTH] Detected authentication boundary (URL: {pattern})", flush=True)
                return True

        return False

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

        if self.config.verbose:
            print(f"\n[STEP {step.id}] {step.goal}", flush=True)
            print(f"  Action: {step.action}", flush=True)
            if step.intent:
                print(f"  Intent: {step.intent}", flush=True)
            if step.input:
                print(f"  Input: {step.input}", flush=True)
            if step.target:
                print(f"  Target: {step.target}", flush=True)

        llm_response: str | None = None
        action_taken: str | None = None
        used_vision = False
        used_heuristics = False
        error: str | None = None
        verification_passed = False
        extraction_succeeded = False
        extracted_data: Any | None = None
        search_submit_telemetry = SearchSubmitTelemetry()

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

            # Pre-step auth boundary check: stop early if on signin page without credentials
            # This prevents executing login steps that would fail or use fake credentials
            if self.config.auth_boundary.enabled and self.config.auth_boundary.stop_on_auth:
                is_auth_page = await self._detect_auth_boundary(runtime)
                if is_auth_page:
                    if self.config.verbose:
                        print(f"  [AUTH] Auth boundary detected at step start - stopping gracefully", flush=True)

                    outcome = StepOutcome(
                        step_id=step.id,
                        goal=step.goal,
                        status=StepStatus.SUCCESS,  # Graceful stop = success
                        action_taken="AUTH_BOUNDARY_REACHED",
                        verification_passed=True,
                        error=self.config.auth_boundary.auth_success_message,
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

                    # Return special outcome that signals run completion
                    return outcome

            # Wait for page to stabilize before snapshot
            if self.config.stabilize_enabled:
                await runtime.stabilize()

            # Capture snapshot with escalation (includes scroll-after-escalation if enabled)
            ctx = await self._snapshot_with_escalation(
                runtime,
                goal=step.goal,
                capture_screenshot=self.config.trace_screenshots,
                step=step,  # Enable scroll-after-escalation to find elements outside viewport
            )
            self._snapshot_context = ctx

            if self.config.verbose:
                elements = getattr(ctx.snapshot, "elements", []) or []
                print(f"  [SNAPSHOT] Elements: {len(elements)}, URL: {getattr(ctx.snapshot, 'url', 'N/A')}", flush=True)
                if ctx.requires_vision:
                    print(f"  [SNAPSHOT] Vision required: {ctx.requires_vision}", flush=True)

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
            original_action = step.action  # Keep original plan action for submit logic
            action_type = step.action
            element_id: int | None = None
            executor_text: str | None = None  # Text from executor response (for TYPE actions)

            if action_type == "EXTRACT":
                action_taken = "EXTRACT"
                # Determine extraction query from step goal or task
                extract_query = step.goal or (
                    self._current_task.task if self._current_task is not None else "Extract relevant data from the current page"
                )

                # Check if this is a text extraction task that can use markdown-based extraction
                use_markdown_extraction = _is_text_extraction_task(extract_query)

                if use_markdown_extraction:
                    # Step 1: Get page content as markdown (faster than snapshot-based extraction)
                    markdown_content = await runtime.read_markdown(max_chars=8000)
                    if markdown_content:
                        if self.config.verbose:
                            preview = markdown_content[:160].replace("\n", " ")
                            print(f"  [ACTION] EXTRACT - got markdown: {preview}...", flush=True)

                        # Step 2: Use LLM (executor) to extract specific data from markdown
                        extraction_prompt = f"""You are a text extraction assistant. Given the page content in markdown format, extract the specific information requested.

PAGE CONTENT (MARKDOWN):
{markdown_content}

EXTRACTION REQUEST:
{extract_query}

INSTRUCTIONS:
1. Read the markdown content carefully
2. Find and extract ONLY the specific information requested
3. Return ONLY the extracted text, nothing else
4. If the information is not found, return "NOT_FOUND"

EXTRACTED TEXT:"""

                        resp = self.executor.generate(
                            "You extract specific text from markdown content. Return only the extracted text.",
                            extraction_prompt,
                            temperature=0.0,
                            max_new_tokens=500,
                        )
                        self._record_token_usage("extract", resp)

                        extracted_text = resp.content.strip()
                        if extracted_text and extracted_text != "NOT_FOUND":
                            extraction_succeeded = True
                            extracted_data = {"text": extracted_text, "query": extract_query}
                            if self.config.verbose:
                                print(f"  [ACTION] EXTRACT ok: {extracted_text[:160]}", flush=True)
                        else:
                            error = f"Could not find requested data: {extract_query}"
                    else:
                        error = "Failed to extract markdown from page"
                else:
                    # Use LLM-based extraction for complex extraction tasks
                    page = (
                        getattr(getattr(runtime, "backend", None), "page", None)
                        or getattr(getattr(runtime, "backend", None), "_page", None)
                        or getattr(runtime, "_legacy_page", None)
                    )
                    if page is None:
                        error = "No page available for EXTRACT"
                    else:
                        from types import SimpleNamespace

                        from ..read import extract_async

                        browser_like = SimpleNamespace(page=page)
                        result = await extract_async(
                            browser_like,
                            self.planner,
                            query=extract_query,
                            schema=None,
                        )
                        llm_resp = getattr(result, "llm_response", None)
                        if llm_resp is not None:
                            self._record_token_usage("extract", llm_resp)
                        if result.ok:
                            extraction_succeeded = True
                            extracted_data = result.data
                            if self.config.verbose:
                                preview = str(result.raw or "")[:160]
                                print(f"  [ACTION] EXTRACT ok: {preview}", flush=True)
                        else:
                            error = result.error or "Extraction failed"
            elif action_type in ("CLICK", "TYPE_AND_SUBMIT"):
                # Try intent heuristics first (if available)
                elements = getattr(ctx.snapshot, "elements", []) or []
                url = getattr(ctx.snapshot, "url", "") or ""
                element_id = await self._try_intent_heuristics(step, elements, url)

                if element_id is not None:
                    used_heuristics = True
                    action_taken = f"{action_type}({element_id}) [heuristic]"
                    if self.config.verbose:
                        print(f"  [EXECUTOR] Using heuristic: {action_taken}", flush=True)
                else:
                    # Fall back to LLM executor
                    sys_prompt, user_prompt = build_executor_prompt(
                        step.goal,
                        step.intent,
                        ctx.compact_representation,
                        input_text=step.input,
                        category=self._get_cached_category_str(),
                        action_type=step.action,
                    )

                    if self.config.verbose:
                        print("\n--- Compact Context (Snapshot) ---", flush=True)
                        print(ctx.compact_representation, flush=True)
                        print("--- End Compact Context ---\n", flush=True)

                    resp = self.executor.generate(
                        sys_prompt,
                        user_prompt,
                        temperature=self.config.executor_temperature,
                        max_new_tokens=self.config.executor_max_tokens,
                    )
                    self._record_token_usage("executor", resp)
                    llm_response = resp.content

                    if self.config.verbose:
                        print(f"  [EXECUTOR] LLM Response: {resp.content}", flush=True)

                    # Parse action
                    parsed_action, parsed_args = self._parse_action(resp.content)
                    action_type = parsed_action
                    if parsed_action == "CLICK" and parsed_args:
                        element_id = parsed_args[0]
                    elif parsed_action == "TYPE" and len(parsed_args) >= 2:
                        element_id = parsed_args[0]
                        executor_text = parsed_args[1]  # Store text from executor response

                    action_taken = f"{action_type}({', '.join(str(a) for a in parsed_args)})"

                    if self.config.verbose:
                        print(f"  [EXECUTOR] Parsed action: {action_taken}", flush=True)

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
                                    if self.config.verbose:
                                        print(f"  [EXECUTOR] Override: {action_taken}", flush=True)
                                else:
                                    error = f"Executor override rejected: {reason}"
                        except Exception:
                            pass  # Ignore override errors

            elif action_type == "NAVIGATE":
                action_taken = f"NAVIGATE({step.target or 'current'})"
            elif action_type == "SCROLL":
                action_taken = "SCROLL(down)"
            else:
                action_taken = action_type

            # If executor suggested SCROLL or selected a non-actionable element for CLICK,
            # try scroll-to-find to locate the actual target element
            if error is None and original_action == "CLICK":
                should_scroll_to_find = False
                reason = ""

                if action_type == "SCROLL":
                    # Executor explicitly said to scroll - element not visible
                    should_scroll_to_find = True
                    reason = "executor suggested SCROLL"
                elif element_id is not None:
                    # Check if selected element is a non-actionable type (input fields)
                    selected_element = None
                    for el in elements:
                        if getattr(el, "id", None) == element_id:
                            selected_element = el
                            break
                    if selected_element:
                        role = (getattr(selected_element, "role", "") or "").lower()
                        if role in {"searchbox", "textbox", "combobox", "input", "textarea"}:
                            should_scroll_to_find = True
                            reason = f"selected input element ({role})"

                if should_scroll_to_find and self.config.scroll_to_find_enabled:
                    if self.config.verbose:
                        print(f"  [SCROLL-TO-FIND] Triggering because {reason}", flush=True)
                    found_id, new_ctx = await self._scroll_to_find_element(runtime, step, step.goal)
                    if found_id is not None and new_ctx is not None:
                        element_id = found_id
                        ctx = new_ctx
                        action_type = "CLICK"
                        action_taken = f"CLICK({element_id}) [scroll-to-find]"

            # Execute action via runtime
            if error is None:
                if action_type == "CLICK" and element_id is not None:
                    await runtime.click(element_id)
                    if self.config.verbose:
                        print(f"  [ACTION] CLICK({element_id})", flush=True)
                    # Wait for page to respond to click (JS navigation, etc.)
                    await runtime.stabilize()
                elif action_type == "TYPE" and element_id is not None:
                    # Use text from executor response first, then fall back to step.input
                    text = executor_text or step.input or ""
                    typed_element = None
                    for el in elements:
                        if getattr(el, "id", None) == element_id:
                            typed_element = el
                            break
                    # If original plan action was TYPE_AND_SUBMIT, press Enter to submit
                    if original_action == "TYPE_AND_SUBMIT":
                        submitted_without_retyping = False
                        if self._looks_like_search_submission(step, typed_element):
                            submitted_without_retyping = await self._submit_if_already_typed(
                                runtime=runtime,
                                elements=elements,
                                input_element_id=element_id,
                                step=step,
                                text=text,
                                pre_url=pre_url or (ctx.snapshot.url or ""),
                                typed_element=typed_element,
                                telemetry=search_submit_telemetry,
                            )
                        if not submitted_without_retyping:
                            typed_ok = False
                            if self._looks_like_search_submission(step, typed_element):
                                typed_ok = await self._clear_and_type_search_input(
                                    runtime=runtime,
                                    input_element_id=element_id,
                                    text=text,
                                )
                            if not typed_ok:
                                await runtime.type(element_id, text, delay_ms=self.config.type_delay_ms)
                            await self._submit_type_and_submit(
                                runtime=runtime,
                                elements=elements,
                                input_element_id=element_id,
                                step=step,
                                text=text,
                                pre_url=pre_url or (ctx.snapshot.url or ""),
                                typed_element=typed_element,
                                telemetry=search_submit_telemetry,
                            )
                        if self.config.verbose:
                            print(f"  [ACTION] TYPE_AND_SUBMIT({element_id}, '{text}')", flush=True)
                    elif self.config.verbose:
                        print(f"  [ACTION] TYPE({element_id}, '{text[:30]}...')" if len(text) > 30 else f"  [ACTION] TYPE({element_id}, '{text}')", flush=True)
                elif action_type == "TYPE_AND_SUBMIT" and element_id is not None:
                    # Use text from executor response first, then step.input, then extract from goal
                    text = executor_text or step.input or ""
                    if not text:
                        # Try to extract from goal like "Search for Logitech mouse"
                        import re
                        match = re.search(r"[Ss]earch\s+(?:for\s+)?['\"]?([^'\"]+)['\"]?", step.goal)
                        if match:
                            text = match.group(1).strip()

                    # Type the text
                    typed_element = None
                    for el in elements:
                        if getattr(el, "id", None) == element_id:
                            typed_element = el
                            break
                    submitted_without_retyping = False
                    if self._looks_like_search_submission(step, typed_element):
                        submitted_without_retyping = await self._submit_if_already_typed(
                            runtime=runtime,
                            elements=elements,
                            input_element_id=element_id,
                            step=step,
                            text=text,
                            pre_url=pre_url or (ctx.snapshot.url or ""),
                            typed_element=typed_element,
                            telemetry=search_submit_telemetry,
                        )
                    if not submitted_without_retyping:
                        typed_ok = False
                        if self._looks_like_search_submission(step, typed_element):
                            typed_ok = await self._clear_and_type_search_input(
                                runtime=runtime,
                                input_element_id=element_id,
                                text=text,
                            )
                        if not typed_ok:
                            await runtime.type(element_id, text, delay_ms=self.config.type_delay_ms)
                        await self._submit_type_and_submit(
                            runtime=runtime,
                            elements=elements,
                            input_element_id=element_id,
                            step=step,
                            text=text,
                            pre_url=pre_url or (ctx.snapshot.url or ""),
                            typed_element=typed_element,
                            telemetry=search_submit_telemetry,
                        )

                    if self.config.verbose:
                        print(f"  [ACTION] TYPE_AND_SUBMIT({element_id}, '{text}')", flush=True)
                elif action_type == "PRESS":
                    key = "Enter"  # Default
                    await runtime.press(key)
                    if self.config.verbose:
                        print(f"  [ACTION] PRESS({key})", flush=True)
                elif action_type == "SCROLL":
                    await runtime.scroll("down")
                    if self.config.verbose:
                        print(f"  [ACTION] SCROLL(down)", flush=True)
                elif action_type == "NAVIGATE":
                    if step.target:
                        await runtime.goto(step.target)
                        if self.config.verbose:
                            print(f"  [ACTION] NAVIGATE({step.target})", flush=True)
                        # Wait for page to load after navigation
                        await runtime.stabilize()
                    else:
                        # No target URL - we're already at the page, just verify
                        if self.config.verbose:
                            print(f"  [ACTION] NAVIGATE(skip - already at page)", flush=True)
                elif action_type == "EXTRACT":
                    pass  # Extraction already executed above
                elif action_type == "FINISH":
                    pass  # No action needed
                elif action_type == "NONE":
                    # Executor couldn't find a suitable element (e.g., no search box)
                    # This triggers replanning to try an alternative approach
                    error = f"No suitable element found for step: {step.goal}"
                    if self.config.verbose:
                        print(f"  [EXECUTOR] NONE - no suitable element found, will trigger replan", flush=True)
                elif action_type not in ("CLICK", "TYPE", "TYPE_AND_SUBMIT") or element_id is None:
                    if action_type in ("CLICK", "TYPE", "TYPE_AND_SUBMIT"):
                        error = f"No element ID for {action_type}"
                    else:
                        error = f"Unknown action: {action_type}"

            # Record action for tracing
            if action_taken:
                await runtime.record_action(action_taken)

            # Run verifications
            if action_type == "EXTRACT" and error is None:
                verification_passed = extraction_succeeded
                if self.config.verbose:
                    print(
                        f"  [VERIFY] Using extraction result: {'PASS' if verification_passed else 'FAIL'}",
                        flush=True,
                    )
            elif step.verify and error is None:
                if self.config.verbose:
                    print(f"  [VERIFY] Running {len(step.verify)} verification predicates...", flush=True)
                verification_passed = await self._verify_step(runtime, step)
                if self.config.verbose:
                    print(f"  [VERIFY] Predicate result: {'PASS' if verification_passed else 'FAIL'}", flush=True)

                # For successful CLICK actions, check if a modal/drawer appeared
                # IMPORTANT: For Add to Cart actions, look for checkout button FIRST
                # before attempting modal dismissal (WebBench-style checkout continuation)
                if verification_passed and original_action == "CLICK" and self.config.modal.enabled:
                    try:
                        post_snap = await runtime.snapshot(emit_trace=False)
                        pre_elements = set(getattr(el, "id", 0) for el in (ctx.snapshot.elements or []))
                        post_elements = set(getattr(el, "id", 0) for el in (post_snap.elements or []))
                        new_elements = post_elements - pre_elements
                        if len(new_elements) >= self.config.modal.min_new_elements:
                            # Significant DOM change after CLICK - might be a modal/drawer
                            # Check if this was an Add to Cart action - if so, look for checkout button
                            goal_lower = (step.goal or "").lower()
                            intent_lower = (step.intent or "").lower()
                            is_add_to_cart = any(
                                p in goal_lower or p in intent_lower
                                for p in ("add to cart", "add to bag", "add to basket")
                            )

                            # NOTE: Add to Cart checkout continuation is now handled after step completion
                            # (see WebBench-style logic at step_index += 1), so just dismiss modals here
                            await self._attempt_modal_dismissal(runtime, post_snap)
                    except Exception:
                        pass  # Ignore snapshot errors

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

                # Fallback: For navigation-causing actions, if URL changed significantly,
                # consider the action successful even if predicate verification failed.
                # This handles cases where local LLMs generate imprecise predicates.
                if not verification_passed and original_action in ("TYPE_AND_SUBMIT", "CLICK"):
                    current_url = await runtime.get_url() if hasattr(runtime, "get_url") else None
                    if current_url and pre_url and current_url != pre_url:
                        # Check if this is a meaningful URL change (not just anchor change)
                        # Strip anchors (#...) before comparing
                        pre_url_base = pre_url.split("#")[0]
                        current_url_base = current_url.split("#")[0]
                        is_meaningful_change = pre_url_base != current_url_base

                        fallback_ok = is_meaningful_change
                        if original_action == "TYPE_AND_SUBMIT":
                            typed_element = None
                            for el in (ctx.snapshot.elements or []):
                                if getattr(el, "id", None) == element_id:
                                    typed_element = el
                                    break
                            fallback_ok = self._type_and_submit_url_change_looks_valid(
                                pre_url=pre_url,
                                post_url=current_url,
                                step=step,
                                element=typed_element,
                                typed_text=executor_text or step.input or "",
                            )
                        elif original_action == "CLICK" and is_meaningful_change:
                            # For CLICK actions, validate URL change matches step intent
                            # This prevents accepting wrong category navigations
                            url_matches_intent = self._url_change_matches_intent(
                                step=step,
                                pre_url=pre_url,
                                post_url=current_url,
                            )
                            if not url_matches_intent:
                                fallback_ok = False
                                if self.config.verbose:
                                    print(f"  [VERIFY] URL changed but doesn't match step intent", flush=True)
                                    print(f"  [VERIFY] Step target: {step.target}, URL: {current_url}", flush=True)

                        # Special handling for Add to Cart steps: if we were on search results
                        # and navigated to a product page, retry the step on the new page
                        # instead of accepting URL change as success
                        is_add_to_cart = self._is_add_to_cart_step(step)
                        was_on_search_results = self._is_search_results_url(pre_url)
                        now_on_product_page = not self._is_search_results_url(current_url)

                        if is_add_to_cart and was_on_search_results and now_on_product_page and is_meaningful_change:
                            # We clicked a product link instead of Add to Cart
                            # Retry the step on the product page
                            if self.config.verbose:
                                print(f"  [ADD-TO-CART] Navigated from search results to product page: {pre_url} -> {current_url}", flush=True)
                                print(f"  [ADD-TO-CART] Retrying Add to Cart action on product page...", flush=True)

                            # Get fresh snapshot on the product page
                            await asyncio.sleep(0.5)  # Brief wait for page to load
                            try:
                                retry_ctx = await self._get_execution_context(
                                    runtime, step, step_index
                                )
                                # Build prompt for retry - looking for Add to Cart on product page
                                retry_prompt = self.build_executor_prompt(
                                    goal=step.goal,
                                    elements=retry_ctx.snapshot.elements or [],
                                    intent=step.intent,
                                    task_category=retry_ctx.task_category,
                                    input_text=step.input,
                                )
                                if self.config.verbose:
                                    print(f"  [ADD-TO-CART] Asking executor to find Add to Cart button...", flush=True)

                                retry_resp = self.executor.generate(
                                    retry_prompt["system"],
                                    retry_prompt["user"],
                                    max_tokens=self.config.executor_max_tokens,
                                )
                                self._usage.record(role="executor", resp=retry_resp)
                                retry_action = retry_resp.content.strip()
                                if self.config.verbose:
                                    print(f"  [ADD-TO-CART] Retry executor output: {retry_action}", flush=True)

                                # Parse and execute retry action
                                retry_match = re.match(r"CLICK\((\d+)\)", retry_action)
                                if retry_match:
                                    retry_element_id = int(retry_match.group(1))
                                    await runtime.click(retry_element_id)
                                    if self.config.verbose:
                                        print(f"  [ADD-TO-CART] Clicked element {retry_element_id}", flush=True)

                                    # Wait and verify
                                    await asyncio.sleep(0.5)
                                    verification_passed = await self._verify_step(runtime, step)
                                    if verification_passed:
                                        if self.config.verbose:
                                            print(f"  [ADD-TO-CART] Add to Cart successful after retry!", flush=True)
                                    else:
                                        # Check for DOM change (cart drawer/modal)
                                        post_retry_snap = await runtime.snapshot(SnapshotOptions(limit=50))
                                        if post_retry_snap and hasattr(post_retry_snap, "elements"):
                                            post_els = post_retry_snap.elements or []
                                            cart_indicators = ["cart", "bag", "basket", "checkout", "added", "item"]
                                            has_cart_indicator = any(
                                                any(ind in (getattr(el, "text", "") or "").lower() for ind in cart_indicators)
                                                for el in post_els[:30]
                                            )
                                            if has_cart_indicator:
                                                if self.config.verbose:
                                                    print(f"  [ADD-TO-CART] Cart indicator detected, accepting as success", flush=True)
                                                verification_passed = True
                            except Exception as retry_err:
                                if self.config.verbose:
                                    print(f"  [ADD-TO-CART] Retry failed: {retry_err}", flush=True)

                            # Skip the normal URL fallback since we handled Add to Cart specially
                            fallback_ok = False

                        if fallback_ok:
                            if self.config.verbose:
                                print(f"  [VERIFY] Predicate failed but URL changed: {pre_url} -> {current_url}", flush=True)
                                print(f"  [VERIFY] Accepting {original_action} as successful (URL change fallback)", flush=True)
                            verification_passed = True
                        else:
                            if self.config.verbose:
                                print(f"  [VERIFY] URL changed but does not match {original_action} intent: {pre_url} -> {current_url}", flush=True)
                            if original_action == "TYPE_AND_SUBMIT":
                                typed_element = None
                                for el in (ctx.snapshot.elements or []):
                                    if getattr(el, "id", None) == element_id:
                                        typed_element = el
                                        break
                                if self._looks_like_search_submission(step, typed_element):
                                    # Try retry submission with alternate method
                                    verification_passed = await self._retry_search_widget_submission(
                                        runtime=runtime,
                                        elements=elements,
                                        input_element_id=element_id,
                                        step=step,
                                        pre_url=pre_url,
                                        text=executor_text or step.input or "",
                                        typed_element=typed_element,
                                        telemetry=search_submit_telemetry,
                                    )
                    elif original_action == "CLICK" and error is None and element_id is not None:
                        # For CLICK actions that don't change URL, check if DOM changed
                        # (e.g., modal appeared, cart drawer opened)
                        # But first verify we clicked an actionable element, not an input field
                        clicked_element = None
                        for el in (ctx.snapshot.elements or []):
                            if getattr(el, "id", None) == element_id:
                                clicked_element = el
                                break

                        # Don't accept DOM change fallback for input/search elements
                        # (clicking them causes focus changes but doesn't complete actions)
                        clicked_role = (getattr(clicked_element, "role", "") or "").lower() if clicked_element else ""
                        input_roles = {"searchbox", "textbox", "combobox", "input", "textarea"}
                        if clicked_role in input_roles:
                            if self.config.verbose:
                                print(f"  [VERIFY] Clicked input element ({clicked_role}), not accepting DOM change fallback", flush=True)
                        else:
                            try:
                                post_snap = await runtime.snapshot(emit_trace=False)
                                pre_elements = set(getattr(el, "id", 0) for el in (ctx.snapshot.elements or []))
                                post_elements = set(getattr(el, "id", 0) for el in (post_snap.elements or []))
                                new_elements = post_elements - pre_elements
                                if len(new_elements) >= self.config.modal.min_new_elements:  # Significant DOM change
                                    if self.config.verbose:
                                        print(f"  [VERIFY] Predicate failed but DOM changed ({len(new_elements)} new elements)", flush=True)
                                        print(f"  [VERIFY] Accepting CLICK as successful (DOM change fallback)", flush=True)
                                    verification_passed = True

                                    # Attempt modal dismissal if enabled
                                    # This handles blocking overlays like product protection drawers
                                    if self.config.modal.enabled:
                                        await self._attempt_modal_dismissal(runtime, post_snap)
                            except Exception:
                                pass  # Ignore snapshot errors
            else:
                verification_passed = error is None
                if self.config.verbose:
                    if not step.verify:
                        print(f"  [VERIFY] No predicates defined, using error check: {'PASS' if verification_passed else 'FAIL'}", flush=True)
                    elif error:
                        print(f"  [VERIFY] Skipped due to error: {error}", flush=True)

                # For overlay/modal dismissal steps without verify predicates,
                # check if the modal is actually dismissed
                if verification_passed and not step.verify and original_action == "CLICK":
                    if self._looks_like_overlay_dismiss_intent(
                        goal=str(step.goal or ""),
                        intent=str(step.intent or ""),
                    ):
                        try:
                            post_snap = await runtime.snapshot(emit_trace=False)
                            modal_detected = getattr(post_snap, "modal_detected", None)
                            # If modal is still detected, the dismissal didn't work
                            if modal_detected is True:
                                if self.config.verbose:
                                    print(f"  [VERIFY] Overlay dismissal check: modal still detected", flush=True)
                                verification_passed = False
                            elif self.config.verbose:
                                print(f"  [VERIFY] Overlay dismissal check: modal_detected={modal_detected}", flush=True)
                        except Exception:
                            pass  # Don't fail on snapshot errors

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
            extracted_data=extracted_data,
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

            # Log what predicate we're checking
            if self.config.verbose:
                import json
                spec_str = json.dumps(verify_spec, default=str) if isinstance(verify_spec, dict) else str(verify_spec)
                print(f"  [VERIFY] Checking predicate: {spec_str[:100]}...", flush=True)

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
                if self.config.verbose:
                    print(f"  [VERIFY] Predicate FAILED", flush=True)
                return False
            else:
                if self.config.verbose:
                    print(f"  [VERIFY] Predicate PASSED", flush=True)

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
        self._cached_pruning_category = None  # Reset category cache for new run
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

        # Optionally fetch page context (markdown) for better planning
        page_context: str | None = None
        if self.config.use_page_context:
            try:
                page_context = await runtime.read_markdown(
                    max_chars=self.config.page_context_max_chars
                )
                if self.config.verbose and page_context:
                    print(f"  [PAGE-CONTEXT] Extracted {len(page_context)} chars of markdown for planning", flush=True)
                    print("\n--- Page Context (Markdown) ---", flush=True)
                    print(page_context, flush=True)
                    print("--- End Page Context ---\n", flush=True)
            except Exception:
                pass  # Fail silently - page context is optional

        try:
            # Generate plan
            plan = await self.plan(task_description, start_url=start_url, page_context=page_context)

            # Execute steps
            step_index = 0
            while step_index < len(plan.steps):
                step = plan.steps[step_index]

                # Set step-specific heuristic hints
                if self._composable_heuristics and step.heuristic_hints:
                    self._composable_heuristics.set_step_hints(step.heuristic_hints)

                outcome = await self._execute_step(step, runtime, step_index)
                step_outcomes.append(outcome)

                # Check if auth boundary was reached at step start (graceful termination)
                if outcome.action_taken == "AUTH_BOUNDARY_REACHED":
                    if self.config.verbose:
                        print(f"  [AUTH] Run completed at authentication boundary", flush=True)
                    break  # Graceful termination - no error

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
                    # Check if we've reached an authentication boundary
                    # This is a graceful terminal state - agent did all it could
                    if self.config.auth_boundary.enabled:
                        is_auth_page = await self._detect_auth_boundary(runtime)
                        if is_auth_page and self.config.auth_boundary.stop_on_auth:
                            if self.config.verbose:
                                print(f"  [AUTH] Stopping at authentication boundary", flush=True)
                            # Mark as success since we reached the auth page successfully
                            # The "failure" is just that we can't proceed without credentials
                            error = None  # Clear any error
                            # Update the outcome to indicate auth boundary
                            outcome = StepOutcome(
                                step_id=outcome.step_id,
                                goal=outcome.goal,
                                status=StepStatus.SUCCESS,  # Treat as success
                                action_taken=outcome.action_taken,
                                verification_passed=True,
                                used_vision=outcome.used_vision,
                                error=self.config.auth_boundary.auth_success_message,
                                duration_ms=outcome.duration_ms,
                                url_before=outcome.url_before,
                                url_after=outcome.url_after,
                            )
                            step_outcomes[-1] = outcome  # Replace the failed outcome
                            break  # Stop execution gracefully

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
                        # Replanning exhausted - check if we should fallback to stepwise
                        if (
                            self.config.auto_fallback_to_stepwise
                            and self._replans_used >= self.config.auto_fallback_replan_threshold
                        ):
                            if self.config.verbose:
                                print(
                                    f"\n[FALLBACK] Upfront planning failed after {self._replans_used} replans. "
                                    f"Switching to stepwise planning...",
                                    flush=True,
                                )
                            # Run stepwise planning as fallback
                            # This will handle the rest of the task adaptively
                            stepwise_result = await self.run_stepwise(
                                runtime,
                                automation_task,
                                run_id=self._run_id,
                            )
                            # Combine outcomes: existing steps + stepwise steps
                            combined_outcomes = step_outcomes + stepwise_result.step_outcomes
                            return RunOutcome(
                                run_id=self._run_id,
                                task=task_description,
                                success=stepwise_result.success,
                                steps_completed=len(combined_outcomes),
                                steps_total=len(combined_outcomes),
                                replans_used=self._replans_used,
                                step_outcomes=combined_outcomes,
                                total_duration_ms=int((time.time() - start_time) * 1000),
                                error=stepwise_result.error,
                                token_usage=self.get_token_stats(),
                                fallback_used=True,
                            )
                        else:
                            error = f"Step {step.id} failed: {outcome.error}"
                            break

                # Check stop condition
                if step.stop_if_true and outcome.verification_passed:
                    break

                # Post-Add-to-Cart checkout continuation (WebBench-style):
                # If this was an Add to Cart step and it's the last planned step,
                # check if a cart drawer appeared with checkout buttons and add a step.
                is_last_step = (step_index == len(plan.steps) - 1)
                if (
                    outcome.status in (StepStatus.SUCCESS, StepStatus.VISION_FALLBACK)
                    and is_last_step
                ):
                    goal_lower = (step.goal or "").lower()
                    is_add_to_cart = any(
                        pattern in goal_lower
                        for pattern in ("add to cart", "add to bag", "add to basket")
                    )
                    if is_add_to_cart:
                        try:
                            if self.config.verbose:
                                print(f"  [CHECKOUT-CONTINUATION] Detected Add to Cart as last step, checking for checkout buttons", flush=True)
                            checkout_snap = await runtime.snapshot(emit_trace=False)
                            # Look for checkout buttons in the snapshot
                            checkout_patterns = (
                                "checkout", "check out", "proceed to checkout",
                                "view cart", "go to cart", "view bag", "go to bag",
                            )
                            checkout_el = None
                            for el in getattr(checkout_snap, "elements", []) or []:
                                el_text = (getattr(el, "text", "") or "").lower()
                                el_role = (getattr(el, "role", "") or "").lower()
                                if el_role in ("button", "link") and any(p in el_text for p in checkout_patterns):
                                    checkout_el = el
                                    if self.config.verbose:
                                        print(f"  [CHECKOUT-CONTINUATION] Found checkout button: id={el.id} text={el_text!r}", flush=True)
                                    break
                            if checkout_el is not None:
                                # Dynamically add a checkout step to the plan
                                from .models import PlanStep
                                new_step = PlanStep(
                                    id=step.id + 1,
                                    goal="Click checkout button in cart drawer",
                                    action="CLICK",
                                    target=None,
                                    intent="Checkout or View Cart button",
                                    input=None,
                                    verify=[],
                                    required=True,
                                    stop_if_true=False,
                                    optional_substeps=[],
                                    heuristic_hints=[],
                                )
                                plan.steps.append(new_step)
                                if self.config.verbose:
                                    print(f"  [CHECKOUT-CONTINUATION] Added checkout step id={new_step.id}", flush=True)
                        except Exception as e:
                            if self.config.verbose:
                                print(f"  [CHECKOUT-CONTINUATION] Error: {e}", flush=True)

                step_index += 1

                # Check for checkout page detection at end of plan
                # This handles cases where modal dismissal leaves us on a checkout page
                # but the plan has no more steps
                if (
                    self.config.checkout.enabled
                    and self.config.checkout.trigger_replan
                    and step_index >= len(plan.steps)  # Just finished last step
                    and outcome.status in (StepStatus.SUCCESS, StepStatus.VISION_FALLBACK)
                    and self._replans_used < self.config.retry.max_replans
                ):
                    # Check if we're on a checkout-relevant page
                    is_checkout, page_type = await self._detect_checkout_page(runtime)
                    if is_checkout and self._should_continue_after_checkout_detection(
                        page_type, step_index - 1, len(plan.steps)
                    ):
                        if self.config.verbose:
                            print(f"  [CHECKOUT] Triggering replan for {page_type} page", flush=True)
                        try:
                            # Replan with context about where we are
                            continuation_task = self._build_checkout_continuation_task(
                                task_description, page_type
                            )
                            # Note: page_context (markdown) is only extracted once during initial planning
                            plan = await self.plan(continuation_task, start_url=None)
                            step_index = 0  # Start from beginning of new plan
                            self._replans_used += 1
                            continue
                        except Exception as e:
                            if self.config.verbose:
                                print(f"  [CHECKOUT] Replan failed: {e}", flush=True)
                            # Continue without replanning

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
            token_usage=self.get_token_stats(),
        )

        # Emit run end
        self._emit_run_end(run_outcome)

        return run_outcome

    # -------------------------------------------------------------------------
    # Stepwise Planning (ReAct-style)
    # -------------------------------------------------------------------------

    async def _plan_next_step(
        self,
        goal: str,
        current_url: str,
        page_context: str,
        action_history: list[ActionRecord],
    ) -> dict[str, Any]:
        """
        Use the planner LLM to decide the next action based on current page state.

        Args:
            goal: The overall task goal
            current_url: Current page URL
            page_context: Compact representation of page elements
            action_history: List of previously executed actions

        Returns:
            Dictionary with action details:
            {
                "action": "CLICK" | "TYPE_AND_SUBMIT" | "SCROLL" | "DONE" | "STUCK",
                "intent": "description of target element",
                "input": "text to type",
                "direction": "up" | "down",
                "reasoning": "explanation"
            }
        """
        sys_prompt, user_prompt = build_stepwise_planner_prompt(
            goal=goal,
            current_url=current_url,
            page_context=page_context,
            action_history=action_history,
        )

        if self.config.verbose:
            print("\n" + "=" * 60, flush=True)
            print("[STEPWISE PLANNER] Deciding next action", flush=True)
            print("=" * 60, flush=True)
            print(f"Goal: {goal}", flush=True)
            print(f"URL: {current_url}", flush=True)
            print(f"History: {len(action_history)} actions", flush=True)

        resp = self.planner.generate(
            sys_prompt,
            user_prompt,
            max_tokens=self.config.planner_max_tokens,
            temperature=self.config.planner_temperature,
        )

        # Track token usage
        self._token_collector.record(role="stepwise_planner", resp=resp)

        raw_text = resp.content.strip()

        if self.config.verbose:
            print(f"\n--- Stepwise Planner Response ---", flush=True)
            print(raw_text, flush=True)
            print("--- End Response ---\n", flush=True)

        # Parse JSON response
        try:
            # Handle code fences if present
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                # Find start and end of JSON
                start_idx = 1 if lines[0].startswith("```") else 0
                end_idx = len(lines)
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == "```":
                        end_idx = i
                        break
                raw_text = "\n".join(lines[start_idx:end_idx])

            action_data = json.loads(raw_text)
            return action_data
        except json.JSONDecodeError as e:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            return {
                "action": "STUCK",
                "reasoning": f"Failed to parse planner response: {e}",
            }

    async def run_stepwise(
        self,
        runtime: AgentRuntime,
        task: AutomationTask | str,
        *,
        start_url: str | None = None,
        run_id: str | None = None,
    ) -> RunOutcome:
        """
        Execute task using stepwise (ReAct-style) planning.

        Instead of generating a full plan upfront, the agent plans one step at
        a time based on the current page state. This allows adaptation to
        unexpected site layouts and flows.

        Args:
            runtime: AgentRuntime instance
            task: AutomationTask instance or task description string
            start_url: Starting URL (only needed if task is a string)
            run_id: Run ID for tracing (optional)

        Returns:
            RunOutcome with execution results

        Example:
            result = await agent.run_stepwise(
                runtime,
                "Find a laptop and add to cart",
                start_url="https://example.com",
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
        self._cached_pruning_category = None  # Reset category cache for new run
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

        task_description = automation_task.task
        stepwise_cfg = self.config.stepwise

        # Emit run start
        self._emit_run_start(task_description, start_url)

        action_history: list[ActionRecord] = []
        step_outcomes: list[StepOutcome] = []
        error: str | None = None
        step_num = 0

        if self.config.verbose:
            print("\n" + "=" * 60, flush=True)
            print("[STEPWISE] Starting stepwise execution", flush=True)
            print("=" * 60, flush=True)
            print(f"Task: {task_description}", flush=True)
            print(f"Start URL: {start_url}", flush=True)
            print(f"Max steps: {stepwise_cfg.max_steps}", flush=True)
            print("=" * 60 + "\n", flush=True)

        try:
            while step_num < stepwise_cfg.max_steps:
                step_num += 1

                if self.config.verbose:
                    print(f"\n[STEP {step_num}] Planning next action...", flush=True)

                # 0. Stabilize before taking snapshot (wait for DOM to settle)
                # This is critical for pages with delayed hydration/rendering
                if self.config.stabilize_enabled:
                    await runtime.stabilize()

                # 1. Take snapshot with escalation
                self._snapshot_context = await self._snapshot_with_escalation(
                    runtime,
                    goal=task_description,
                )

                # Debug: log snapshot context details
                if self.config.verbose:
                    snap = self._snapshot_context.snapshot
                    elem_count = len(snap.elements) if snap and snap.elements else 0
                    compact_len = len(self._snapshot_context.compact_representation) if self._snapshot_context.compact_representation else 0
                    requires_vision = self._snapshot_context.requires_vision
                    print(f"  [STEPWISE-SNAPSHOT] Elements: {elem_count}, Compact len: {compact_len}, Requires vision: {requires_vision}", flush=True)
                current_url = await runtime.get_url() if hasattr(runtime, "get_url") else ""

                # 2. Build page context
                if stepwise_cfg.include_page_context:
                    page_context = self._snapshot_context.compact_representation or "(no elements captured)"
                else:
                    page_context = "(page context disabled)"

                # Debug: print page context for stepwise planning
                if self.config.verbose and stepwise_cfg.include_page_context:
                    print("\n--- Stepwise Page Context ---", flush=True)
                    # Truncate to first 20 lines for readability
                    context_lines = page_context.split("\n")
                    if len(context_lines) > 20:
                        print("\n".join(context_lines[:20]), flush=True)
                        print(f"... ({len(context_lines) - 20} more lines)", flush=True)
                    else:
                        print(page_context, flush=True)
                    print("--- End Page Context ---\n", flush=True)

                # 3. Get recent action history
                recent_history = action_history[-stepwise_cfg.action_history_limit:]

                # 4. Ask planner for next action
                next_action = await self._plan_next_step(
                    goal=task_description,
                    current_url=current_url,
                    page_context=page_context,
                    action_history=recent_history,
                )

                action_type = next_action.get("action", "STUCK").upper()
                reasoning = next_action.get("reasoning", "")

                if self.config.verbose:
                    print(f"  [PLANNER] Action: {action_type}", flush=True)
                    print(f"  [PLANNER] Reasoning: {reasoning}", flush=True)

                # 4.5 Loop detection: If the same action+target is repeated 2+ times, force DONE
                current_target = next_action.get("intent") or next_action.get("input") or ""
                if len(action_history) >= 2 and action_type == "CLICK":
                    # Check if the last 2 actions were the same CLICK on the same target
                    last_two = action_history[-2:]
                    if all(r.action == "CLICK" and r.target and current_target.lower() in r.target.lower()
                           for r in last_two):
                        if self.config.verbose:
                            print(f"  [LOOP DETECTED] Same CLICK action repeated 3+ times, forcing DONE", flush=True)
                        action_type = "DONE"
                        reasoning = "Loop detected - same CLICK action repeated, goal likely achieved"

                # 5. Check for terminal states
                if action_type == "DONE":
                    if self.config.verbose:
                        print(f"\n[STEPWISE] Task completed: {reasoning}", flush=True)
                    break

                if action_type == "STUCK":
                    error = f"Agent stuck: {reasoning}"
                    if self.config.verbose:
                        print(f"\n[STEPWISE] Agent stuck: {reasoning}", flush=True)

                    # Check if this is an auth boundary
                    if self.config.auth_boundary.enabled:
                        is_auth = await self._detect_auth_boundary(runtime)
                        if is_auth:
                            if self.config.verbose:
                                print(f"  [AUTH] Detected auth boundary - treating as success", flush=True)
                            error = None  # Clear error - this is a graceful stop
                    break

                # 6. Convert action to PlanStep and execute
                plan_step = self._action_to_plan_step(next_action, step_num)

                if self.config.verbose:
                    print(f"  [EXECUTE] {plan_step.action}", flush=True)
                    if plan_step.intent:
                        print(f"  [EXECUTE] Intent: {plan_step.intent}", flush=True)
                    if plan_step.input:
                        print(f"  [EXECUTE] Input: {plan_step.input}", flush=True)

                # Execute the step
                outcome = await self._execute_step(plan_step, runtime, step_num - 1)
                step_outcomes.append(outcome)

                # 7. Record action in history
                action_record = ActionRecord(
                    step_num=step_num,
                    action=action_type,
                    target=next_action.get("intent") or next_action.get("input") or next_action.get("direction"),
                    result="success" if outcome.verification_passed else "failed",
                    url_after=outcome.url_after,
                )
                action_history.append(action_record)

                if self.config.verbose:
                    status = "OK" if outcome.verification_passed else "FAIL"
                    print(f"  [RESULT] {status}", flush=True)

                # 8. Record checkpoint on success (for recovery)
                if outcome.status in (StepStatus.SUCCESS, StepStatus.VISION_FALLBACK):
                    if self._recovery_state and outcome.url_after:
                        self._recovery_state.record_checkpoint(
                            url=outcome.url_after,
                            step_index=step_num - 1,
                            snapshot_digest=hashlib.sha256(
                                (outcome.url_after or "").encode()
                            ).hexdigest()[:16],
                            predicates_passed=[],
                        )

                    # 8.1 Modal/drawer handling after successful CLICK actions
                    # For Add to Cart actions, look for checkout button first (checkout continuation)
                    # For other actions, use normal modal dismissal
                    if action_type == "CLICK" and self.config.modal.enabled:
                        try:
                            post_snap = await runtime.snapshot(emit_trace=False)
                            pre_elements = set(getattr(el, "id", 0) for el in (snap.elements or []))
                            post_elements = set(getattr(el, "id", 0) for el in (post_snap.elements or []))
                            new_elements = post_elements - pre_elements
                            if len(new_elements) >= self.config.modal.min_new_elements:
                                # Significant DOM change after CLICK - might be a modal/drawer
                                if self.config.verbose:
                                    print(f"  [MODAL] Detected {len(new_elements)} new elements after CLICK, checking for dismissible overlay...", flush=True)

                                # Check if this was an Add to Cart action
                                goal_lower = (plan_step.goal or "").lower()
                                intent_lower = (plan_step.intent or "").lower()
                                is_add_to_cart = any(
                                    p in goal_lower or p in intent_lower
                                    for p in ("add to cart", "add to bag", "add to basket")
                                )

                                if is_add_to_cart:
                                    # Look for checkout button in the drawer
                                    checkout_patterns = (
                                        "checkout", "check out", "proceed to checkout",
                                        "view cart", "go to cart", "view bag", "go to bag",
                                    )
                                    checkout_el = None
                                    for el in getattr(post_snap, "elements", []) or []:
                                        el_text = (getattr(el, "text", "") or "").lower()
                                        el_role = (getattr(el, "role", "") or "").lower()
                                        if el_role in ("button", "link") and any(p in el_text for p in checkout_patterns):
                                            checkout_el = el
                                            break

                                    if checkout_el is not None:
                                        if self.config.verbose:
                                            print(f"  [CHECKOUT-CONTINUATION] Found checkout button: id={checkout_el.id} text={getattr(checkout_el, 'text', '')!r}", flush=True)
                                        await runtime.click(checkout_el.id)
                                        await runtime.stabilize()
                                        if self.config.verbose:
                                            print(f"  [CHECKOUT-CONTINUATION] Clicked checkout button", flush=True)
                                    else:
                                        dismissed = await self._attempt_modal_dismissal(runtime, post_snap)
                                        if dismissed and self.config.verbose:
                                            print(f"  [MODAL] Dismissed overlay", flush=True)
                                else:
                                    dismissed = await self._attempt_modal_dismissal(runtime, post_snap)
                                    if dismissed and self.config.verbose:
                                        print(f"  [MODAL] Dismissed overlay", flush=True)
                        except Exception:
                            pass  # Ignore snapshot errors

                # 9. Handle failure
                if outcome.status == StepStatus.FAILED:
                    # Check for auth boundary
                    if self.config.auth_boundary.enabled:
                        is_auth = await self._detect_auth_boundary(runtime)
                        if is_auth and self.config.auth_boundary.stop_on_auth:
                            if self.config.verbose:
                                print(f"  [AUTH] Stopping at authentication boundary", flush=True)
                            # Update the outcome to indicate auth boundary
                            outcome = StepOutcome(
                                step_id=outcome.step_id,
                                goal=outcome.goal,
                                status=StepStatus.SUCCESS,
                                action_taken=outcome.action_taken,
                                verification_passed=True,
                                used_vision=outcome.used_vision,
                                error=self.config.auth_boundary.auth_success_message,
                                duration_ms=outcome.duration_ms,
                                url_before=outcome.url_before,
                                url_after=outcome.url_after,
                            )
                            step_outcomes[-1] = outcome
                            break

                    # Try recovery if available
                    if self._recovery_state and self._recovery_state.can_recover():
                        if self.config.verbose:
                            print(f"  [RECOVERY] Attempting recovery to last known good state...", flush=True)
                        recovered = await self._attempt_recovery(runtime)
                        if recovered:
                            checkpoint = self._recovery_state.current_recovery_target
                            if checkpoint:
                                # Roll back action history to checkpoint
                                action_history = [
                                    a for a in action_history
                                    if a.step_num <= checkpoint.step_index + 1
                                ]
                                self._recovery_state.clear_recovery_target()
                                if self.config.verbose:
                                    print(f"  [RECOVERY] Recovered to step {checkpoint.step_index + 1}", flush=True)
                                continue

                    # If action failed, the stepwise planner will see it in history
                    # and adapt on the next iteration (no explicit replan needed)
                    if self.config.verbose:
                        print(f"  [STEPWISE] Action failed, will adapt on next iteration", flush=True)

        except Exception as e:
            error = str(e)
            if self.config.verbose:
                print(f"\n[STEPWISE] Exception: {error}", flush=True)

        # Build final outcome
        success = error is None and step_num > 0

        # If we exited due to DONE, it's a success
        # If we exited due to STUCK with auth boundary, it's a success
        # If we hit max_steps, it's a failure
        if step_num >= stepwise_cfg.max_steps and error is None:
            error = f"Max steps ({stepwise_cfg.max_steps}) reached without completing task"
            success = False

        run_outcome = RunOutcome(
            run_id=self._run_id,
            task=task_description,
            success=success,
            steps_completed=len(step_outcomes),
            steps_total=step_num,
            replans_used=0,  # Stepwise doesn't use replanning
            step_outcomes=step_outcomes,
            total_duration_ms=int((time.time() - start_time) * 1000),
            error=error,
            token_usage=self.get_token_stats(),
        )

        # Emit run end
        self._emit_run_end(run_outcome)

        if self.config.verbose:
            print("\n" + "=" * 60, flush=True)
            print("[STEPWISE] Run Complete", flush=True)
            print("=" * 60, flush=True)
            print(f"Success: {success}", flush=True)
            print(f"Steps: {len(step_outcomes)}", flush=True)
            print(f"Duration: {run_outcome.total_duration_ms}ms", flush=True)
            if error:
                print(f"Error: {error}", flush=True)
            print("=" * 60 + "\n", flush=True)

        return run_outcome

    def _action_to_plan_step(self, action_data: dict[str, Any], step_id: int) -> PlanStep:
        """
        Convert a stepwise planner action to a PlanStep for execution.

        Args:
            action_data: Dictionary from stepwise planner
            step_id: Step ID number

        Returns:
            PlanStep object
        """
        action_type = action_data.get("action", "").upper()
        intent = action_data.get("intent")
        input_text = action_data.get("input")
        direction = action_data.get("direction")
        reasoning = action_data.get("reasoning", "")

        # Map action type
        if action_type == "TYPE_AND_SUBMIT":
            action = "TYPE_AND_SUBMIT"
            goal = f"Type '{input_text}' and submit"
        elif action_type == "CLICK":
            action = "CLICK"
            # Include input in goal if provided (e.g., "Click: button (Add Note)")
            if intent and input_text:
                goal = f"Click: {intent} ({input_text})"
            elif intent:
                goal = f"Click: {intent}"
            else:
                goal = "Click element"
        elif action_type == "SCROLL":
            action = "SCROLL"
            goal = f"Scroll {direction}"
        else:
            action = action_type
            goal = reasoning or action_type

        return PlanStep(
            id=step_id,
            goal=goal,
            action=action,
            target=None,
            intent=intent,
            input=input_text,
            verify=[],  # Stepwise mode doesn't use predefined verification
            required=True,
            stop_if_true=False,
            optional_substeps=[],
            heuristic_hints=[],
        )

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
