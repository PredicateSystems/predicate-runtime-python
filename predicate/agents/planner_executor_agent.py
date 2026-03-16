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

        # Enable scroll-after-escalation to find elements below/above viewport
        config = SnapshotEscalationConfig(scroll_after_escalation=True, scroll_directions=("down", "up"))
    """

    enabled: bool = True
    limit_base: int = 60
    limit_step: int = 30
    limit_max: int = 200
    # Scroll after exhausting limit escalation to find elements in different viewports
    scroll_after_escalation: bool = True
    scroll_max_attempts: int = 3  # Max scrolls per direction
    scroll_directions: tuple[str, ...] = ("down", "up")  # Directions to try


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
        "/ap/signin",  # Amazon
        "/ap/register",  # Amazon
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
) -> tuple[str, str]:
    """
    Build system and user prompts for the Planner LLM.

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
2. TYPE_AND_SUBMIT search query in search box
3. CLICK on specific product from results (not filters or categories)
4. CLICK "Add to Cart" button on product page
5. CLICK "Proceed to Checkout" or cart icon
6. Handle login/signup if required (may need CLICK + TYPE_AND_SUBMIT)
7. CLICK through checkout process

Common mistakes to AVOID:
- Do NOT skip "Add to Cart" step - clicking a product link is NOT adding to cart
- Do NOT combine multiple distinct actions into one step
- Do NOT confuse filter/category clicks with product selection
- Each distinct user action should be its own step

Intent hints are critical - use clear hints like:
- intent: "Click Add to Cart button"
- intent: "Click Proceed to Checkout"
- intent: "Click on product title or image"
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
      "goal": "Click on result",
      "action": "CLICK",
      "intent": "Click on product title",
      "verify": [{{"predicate": "url_contains", "args": ["/product/"]}}]
    }}
  ]
}}

CRITICAL: Each verify predicate MUST be an object with "predicate" and "args" keys:
- {{"predicate": "url_contains", "args": ["substring"]}}
- {{"predicate": "exists", "args": ["role=button"]}}
- {{"predicate": "not_exists", "args": ["text~'error'"]}}

DO NOT use string format like "url_contains('text')" - use object format only.
{domain_guidance}
Return ONLY valid JSON. No prose, no code fences, no markdown."""

    user = f"""Task: {task}
{schema_note}
Starting URL: {start_url or "browser's current page"}
Site type: {site_type}
Auth state: {auth_state}

Output a JSON plan to accomplish this task. Each step should represent ONE distinct action."""

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
        """
        Format snapshot for LLM context.

        Uses compact format: id|role|text|importance|is_primary|bg|clickable|nearby_text|ord|DG|href
        Same format as documented in reddit_post_planner_executor_local_llm.md
        """
        if self._context_formatter is not None:
            return self._context_formatter(snap, goal)

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

                for direction in cfg.scroll_directions:
                    for scroll_num in range(cfg.scroll_max_attempts):
                        if self.config.verbose:
                            print(f"  [SNAPSHOT-ESCALATION] Scrolling {direction} ({scroll_num + 1}/{cfg.scroll_max_attempts})...", flush=True)

                        # Scroll
                        await runtime.scroll(direction)

                        # Wait for stabilization
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
                            last_compact = self._format_context(snap, goal)

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

            if action_type in ("CLICK", "TYPE_AND_SUBMIT"):
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
                elif action_type == "TYPE" and element_id is not None:
                    # Use text from executor response first, then fall back to step.input
                    text = executor_text or step.input or ""
                    await runtime.type(element_id, text)
                    # If original plan action was TYPE_AND_SUBMIT, press Enter to submit
                    if original_action == "TYPE_AND_SUBMIT":
                        await runtime.press("Enter")
                        if self.config.verbose:
                            print(f"  [ACTION] TYPE_AND_SUBMIT({element_id}, '{text}')", flush=True)
                        await runtime.stabilize()
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
                    await runtime.type(element_id, text)
                    await runtime.press("Enter")
                    if self.config.verbose:
                        print(f"  [ACTION] TYPE_AND_SUBMIT({element_id}, '{text}')", flush=True)
                    # Wait for page to load after submit
                    await runtime.stabilize()
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
                if self.config.verbose:
                    print(f"  [VERIFY] Running {len(step.verify)} verification predicates...", flush=True)
                verification_passed = await self._verify_step(runtime, step)
                if self.config.verbose:
                    print(f"  [VERIFY] Predicate result: {'PASS' if verification_passed else 'FAIL'}", flush=True)

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
                        # URL changed - the action likely achieved navigation
                        if self.config.verbose:
                            print(f"  [VERIFY] Predicate failed but URL changed: {pre_url} -> {current_url}", flush=True)
                            print(f"  [VERIFY] Accepting {original_action} as successful (URL change fallback)", flush=True)
                        verification_passed = True
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
                        error = f"Step {step.id} failed: {outcome.error}"
                        break

                # Check stop condition
                if step.stop_if_true and outcome.verification_passed:
                    break

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
