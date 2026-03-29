"""
Agent-level orchestration helpers (snapshot-first, verification-first).

This package provides a "browser-use-like" agent surface built on top of:
- AgentRuntime (snapshots, verification, tracing)
- RuntimeAgent (execution loop and bounded vision fallback)

Agent types:
- PredicateBrowserAgent: Single-executor agent with manual step definitions
- PlannerExecutorAgent: Two-tier agent with LLM-generated plans

Task abstractions:
- AutomationTask: Generic task model for browser automation
- TaskCategory: Task category hints for heuristics selection

Heuristics:
- HeuristicHint: Planner-generated hints for element selection
- ComposableHeuristics: Dynamic heuristics composition

Recovery:
- RecoveryState: Checkpoint tracking for rollback recovery
- RecoveryCheckpoint: Individual recovery checkpoint
"""

from .automation_task import (
    AutomationTask,
    ExtractionSpec,
    SuccessCriteria,
    TaskCategory,
)
from .browser_agent import (
    CaptchaConfig,
    PermissionRecoveryConfig,
    PredicateBrowserAgent,
    PredicateBrowserAgentConfig,
    VisionFallbackConfig,
)
from .composable_heuristics import ComposableHeuristics
from .heuristic_spec import COMMON_HINTS, HeuristicHint, get_common_hint
from .planner_executor_agent import (
    ActionRecord,
    AuthBoundaryConfig,
    CheckoutDetectionConfig,
    ExecutorOverride,
    IntentHeuristics,
    ModalDismissalConfig,
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
    StepwisePlanningConfig,
    normalize_plan,
    validate_plan_smoothness,
)
from .recovery import RecoveryCheckpoint, RecoveryState
from .agent_factory import (
    ConfigPreset,
    create_planner_executor_agent,
    get_config_preset,
)

__all__ = [
    # Automation Task
    "AutomationTask",
    "ExtractionSpec",
    "SuccessCriteria",
    "TaskCategory",
    # Browser Agent
    "CaptchaConfig",
    "PermissionRecoveryConfig",
    "PredicateBrowserAgent",
    "PredicateBrowserAgentConfig",
    "VisionFallbackConfig",
    # Heuristics
    "COMMON_HINTS",
    "ComposableHeuristics",
    "HeuristicHint",
    "get_common_hint",
    # Planner + Executor Agent
    "ActionRecord",
    "AuthBoundaryConfig",
    "CheckoutDetectionConfig",
    "ExecutorOverride",
    "IntentHeuristics",
    "ModalDismissalConfig",
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
    "StepwisePlanningConfig",
    "normalize_plan",
    "validate_plan_smoothness",
    # Recovery
    "RecoveryCheckpoint",
    "RecoveryState",
    # Agent Factory
    "ConfigPreset",
    "create_planner_executor_agent",
    "get_config_preset",
]
