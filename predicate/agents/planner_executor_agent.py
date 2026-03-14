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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

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

from .browser_agent import CaptchaConfig, VisionFallbackConfig


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
class PlannerExecutorConfig:
    """
    High-level configuration for PlannerExecutorAgent.

    This config focuses on:
    - Snapshot escalation settings
    - Retry/verification settings
    - Vision fallback settings
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
        """
        self.planner = planner
        self.executor = executor
        self.vision_executor = vision_executor
        self.vision_verifier = vision_verifier
        self.config = config or PlannerExecutorConfig()
        self.tracer = tracer
        self._context_formatter = context_formatter

        # State tracking
        self._current_plan: Plan | None = None
        self._step_index: int = 0
        self._replans_used: int = 0
        self._vision_calls: int = 0
        self._snapshot_context: SnapshotContext | None = None
        self._run_id: str | None = None

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
                plan = Plan.model_validate(plan_dict)
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

    async def _execute_step(
        self,
        step: PlanStep,
        runtime: AgentRuntime,
        step_index: int,
    ) -> StepOutcome:
        """Execute a single plan step."""
        start_time = time.time()
        pre_url = await runtime.get_url() if hasattr(runtime, "get_url") else None
        step_id = self._emit_step_start(step, step_index, pre_url)

        llm_response: str | None = None
        action_taken: str | None = None
        used_vision = False
        error: str | None = None
        verification_passed = False

        try:
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

            # Build executor prompt
            sys_prompt, user_prompt = build_executor_prompt(
                step.goal,
                step.intent,
                ctx.compact_representation,
            )

            # Call executor
            resp = self.executor.generate(
                sys_prompt,
                user_prompt,
                temperature=self.config.executor_temperature,
                max_new_tokens=self.config.executor_max_tokens,
            )
            llm_response = resp.content

            # Parse and execute action
            action_type, action_args = self._parse_action(resp.content)
            action_taken = f"{action_type}({', '.join(str(a) for a in action_args)})"

            # Execute action via runtime
            if action_type == "CLICK" and action_args:
                element_id = action_args[0]
                await runtime.click(element_id)
            elif action_type == "TYPE" and len(action_args) >= 2:
                element_id, text = action_args[0], action_args[1]
                await runtime.type(element_id, text)
            elif action_type == "PRESS" and action_args:
                key = action_args[0]
                await runtime.press(key)
            elif action_type == "SCROLL":
                direction = action_args[0] if action_args else "down"
                await runtime.scroll(direction)
            elif action_type == "FINISH":
                pass  # No action needed
            else:
                error = f"Unknown action: {action_type}"

            # Record action for tracing
            await runtime.record_action(action_taken)

            # Run verifications
            if step.verify:
                verification_passed = await self._verify_step(runtime, step)
            else:
                verification_passed = True

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
        task: str,
        *,
        start_url: str | None = None,
        run_id: str | None = None,
    ) -> RunOutcome:
        """
        Execute complete task with planning, execution, and replanning.

        Args:
            runtime: AgentRuntime instance
            task: Task description
            start_url: Starting URL (optional)
            run_id: Run ID for tracing (optional)

        Returns:
            RunOutcome with execution results
        """
        self._run_id = run_id or str(uuid.uuid4())
        self._replans_used = 0
        self._vision_calls = 0
        start_time = time.time()

        # Emit run start
        self._emit_run_start(task, start_url)

        step_outcomes: list[StepOutcome] = []
        error: str | None = None

        try:
            # Generate plan
            plan = await self.plan(task, start_url=start_url)

            # Execute steps
            step_index = 0
            while step_index < len(plan.steps):
                step = plan.steps[step_index]

                outcome = await self._execute_step(step, runtime, step_index)
                step_outcomes.append(outcome)

                # Handle failure
                if outcome.status == StepStatus.FAILED and step.required:
                    if self._replans_used < self.config.retry.max_replans:
                        try:
                            plan = await self.replan(
                                task,
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
            task=task,
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
