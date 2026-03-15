"""
Agent-level orchestration helpers (snapshot-first, verification-first).

This package provides a "browser-use-like" agent surface built on top of:
- AgentRuntime (snapshots, verification, tracing)
- RuntimeAgent (execution loop and bounded vision fallback)

Agent types:
- PredicateBrowserAgent: Single-executor agent with manual step definitions
- PlannerExecutorAgent: Two-tier agent with LLM-generated plans
"""

from .browser_agent import (
    CaptchaConfig,
    PermissionRecoveryConfig,
    PredicateBrowserAgent,
    PredicateBrowserAgentConfig,
    VisionFallbackConfig,
)
from .planner_executor_agent import (
    ExecutorOverride,
    IntentHeuristics,
    Plan,
    PlanStep,
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    PredicateSpec,
    RecoveryNavigationConfig,
    RetryConfig,
    RunOutcome,
    SnapshotContext,
    SnapshotEscalationConfig,
    StepOutcome,
    StepStatus,
    normalize_plan,
    validate_plan_smoothness,
)

__all__ = [
    # Browser Agent
    "CaptchaConfig",
    "PermissionRecoveryConfig",
    "PredicateBrowserAgent",
    "PredicateBrowserAgentConfig",
    "VisionFallbackConfig",
    # Planner + Executor Agent
    "ExecutorOverride",
    "IntentHeuristics",
    "Plan",
    "PlanStep",
    "PlannerExecutorAgent",
    "PlannerExecutorConfig",
    "PredicateSpec",
    "RecoveryNavigationConfig",
    "RetryConfig",
    "RunOutcome",
    "SnapshotContext",
    "SnapshotEscalationConfig",
    "StepOutcome",
    "StepStatus",
    "normalize_plan",
    "validate_plan_smoothness",
]

