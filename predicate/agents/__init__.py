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
    Plan,
    PlanStep,
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    PredicateSpec,
    RetryConfig,
    RunOutcome,
    SnapshotContext,
    SnapshotEscalationConfig,
    StepOutcome,
    StepStatus,
)

__all__ = [
    # Browser Agent
    "CaptchaConfig",
    "PermissionRecoveryConfig",
    "PredicateBrowserAgent",
    "PredicateBrowserAgentConfig",
    "VisionFallbackConfig",
    # Planner + Executor Agent
    "Plan",
    "PlanStep",
    "PlannerExecutorAgent",
    "PlannerExecutorConfig",
    "PredicateSpec",
    "RetryConfig",
    "RunOutcome",
    "SnapshotContext",
    "SnapshotEscalationConfig",
    "StepOutcome",
    "StepStatus",
]

