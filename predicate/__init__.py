"""
Sentience Python SDK - AI Agent Browser Automation
"""

# Extension helpers (for browser-use integration)
from ._extension_loader import (
    get_extension_dir,
    get_extension_version,
    verify_extension_injected,
    verify_extension_injected_async,
    verify_extension_version,
    verify_extension_version_async,
)
from .actions import (
    back,
    check,
    clear,
    click,
    click_rect,
    press,
    scroll_to,
    search,
    search_async,
    select_option,
    send_keys,
    send_keys_async,
    submit,
    type_text,
    uncheck,
    upload_file,
)
from .agent import SentienceAgent, SentienceAgentAsync
from .agent_config import AgentConfig
from .agent_runtime import AgentRuntime, AssertionHandle

# Snapshot-first browser agent (new high-level surface)
from .agents import (
    CaptchaConfig,
    PermissionRecoveryConfig,
    PredicateBrowserAgent,
    PredicateBrowserAgentConfig,
    VisionFallbackConfig,
)

# Backend-agnostic actions (aliased to avoid conflict with existing actions)
# Browser backends (for browser-use integration)
from .backends import (
    BrowserBackend,
    BrowserUseAdapter,
    BrowserUseCDPTransport,
    CachedSnapshot,
    CDPBackendV0,
    CDPTransport,
    LayoutMetrics,
    PlaywrightBackend,
    ViewportInfo,
)
from .backends import click as backend_click
from .backends import scroll as backend_scroll
from .backends import scroll_to_element as backend_scroll_to_element
from .backends import snapshot as backend_snapshot
from .backends import type_text as backend_type_text
from .backends import wait_for_stable as backend_wait_for_stable

# Agent Layer (Phase 1 & 2)
from .base_agent import BaseAgent
from .browser import AsyncSentienceBrowser, SentienceBrowser
from .captcha import CaptchaContext, CaptchaHandlingError, CaptchaOptions, CaptchaResolution
from .captcha_strategies import ExternalSolver, HumanHandoffSolver, VisionSolver

# Tracing (v0.12.0+)
from .cloud_tracing import CloudTraceSink, SentienceLogger
from .conversational_agent import ConversationalAgent
from .cursor_policy import CursorPolicy
from .debugger import SentienceDebugger
from .expect import expect
from .generator import ScriptGenerator, generate
from .inspector import Inspector, inspect
from .llm_provider import (
    AnthropicProvider,
    DeepInfraProvider,
    LLMProvider,
    LLMResponse,
    LocalLLMProvider,
    LocalVisionLLMProvider,
    MLXVLMProvider,
    OllamaProvider,
    OpenAIProvider,
)
from .models import (  # Agent Layer Models
    ActionHistory,
    ActionResult,
    ActionTokenUsage,
    AgentActionResult,
    BBox,
    Cookie,
    Element,
    LLMStepData,
    LLMUsage,
    LocalStorageItem,
    OriginStorage,
    ScreenshotConfig,
    Snapshot,
    SnapshotFilter,
    SnapshotOptions,
    StepHookContext,
    StorageState,
    TextContext,
    TextMatch,
    TextRect,
    TextRectSearchResult,
    TokenStats,
    Viewport,
    ViewportRect,
    WaitResult,
)

# Ordinal support (Phase 3)
from .ordinal import OrdinalIntent, boost_ordinal_elements, detect_ordinal_intent, select_by_ordinal
from .overlay import clear_overlay, show_overlay
from .overlay_dismissal import OverlayDismissResult, dismiss_overlays, dismiss_overlays_before_agent
from .permissions import PermissionPolicy
from .pruning import (
    CategoryDetectionResult,
    PrunedSnapshotContext,
    PruningTaskCategory,
    SkeletonDomNode,
    classify_task_category,
    prune_snapshot_for_task,
    serialize_pruned_snapshot,
)
from .query import find, query
from .read import extract, extract_async, read, read_best_effort
from .recorder import Recorder, Trace, TraceStep, record
from .runtime_agent import RuntimeAgent, RuntimeStep, StepVerification
from .screenshot import screenshot
from .sentience_methods import AgentAction, SentienceMethod
from .snapshot import snapshot
from .text_search import find_text_rect
from .tools import BackendCapabilities, ToolContext, ToolRegistry, ToolSpec, register_default_tools
from .constants import PREDICATE_API_URL, SENTIENCE_API_URL
from .tracer_factory import create_tracer
from .tracing import JsonlTraceSink, TraceEvent, Tracer, TraceSink

# Utilities (v0.12.0+)
# Import from utils package (re-exports from submodules for backward compatibility)
from .utils import (
    canonical_snapshot_loose,
    canonical_snapshot_strict,
    compute_snapshot_digests,
    save_storage_state,
    sha256_digest,
)

# Formatting (v0.12.0+)
from .utils.formatting import format_snapshot_for_llm

# Verification (agent assertion loop)
from .verification import (
    AssertContext,
    AssertOutcome,
    Predicate,
    all_of,
    any_of,
    custom,
    download_completed,
    element_count,
    exists,
    is_checked,
    is_collapsed,
    is_disabled,
    is_enabled,
    is_expanded,
    is_unchecked,
    not_exists,
    url_contains,
    url_matches,
    value_contains,
    value_equals,
)

# Vision executor primitives (shared parsing/execution helpers)
from .vision_executor import (
    VisionExecutorAction,
    execute_vision_executor_action,
    parse_vision_executor_action,
)
from .visual_agent import SentienceVisualAgent, SentienceVisualAgentAsync
from .wait import wait_for

# Predicate-named class counterparts (canonical moving forward).
# Keep Sentience* names for backward compatibility.
PredicateBrowser = SentienceBrowser
AsyncPredicateBrowser = AsyncSentienceBrowser
PredicateAgent = SentienceAgent
PredicateAgentAsync = SentienceAgentAsync
PredicateVisualAgent = SentienceVisualAgent
PredicateVisualAgentAsync = SentienceVisualAgentAsync
PredicateDebugger = SentienceDebugger

__version__ = "1.3.0"

__all__ = [
    # Extension helpers (for browser-use integration)
    "get_extension_dir",
    "get_extension_version",
    "verify_extension_injected",
    "verify_extension_injected_async",
    "verify_extension_version",
    "verify_extension_version_async",
    # Browser backends (for browser-use integration)
    "BrowserBackend",
    "CDPTransport",
    "CDPBackendV0",
    "PlaywrightBackend",
    "BrowserUseAdapter",
    "BrowserUseCDPTransport",
    "ViewportInfo",
    "LayoutMetrics",
    "backend_snapshot",
    "CachedSnapshot",
    # Backend-agnostic actions (prefixed to avoid conflicts)
    "backend_click",
    "backend_type_text",
    "backend_scroll",
    "backend_scroll_to_element",
    "backend_wait_for_stable",
    # Core SDK
    "PredicateBrowser",
    "AsyncPredicateBrowser",
    "SentienceBrowser",
    "AsyncSentienceBrowser",
    "Snapshot",
    "Element",
    "BBox",
    "Viewport",
    "ActionResult",
    "WaitResult",
    "snapshot",
    "query",
    "find",
    "click",
    "type_text",
    "press",
    "scroll_to",
    "click_rect",
    "CursorPolicy",
    "wait_for",
    "expect",
    "Inspector",
    "inspect",
    "Recorder",
    "Trace",
    "TraceStep",
    "record",
    "ScriptGenerator",
    "generate",
    "read",
    "read_best_effort",
    "screenshot",
    "show_overlay",
    "clear_overlay",
    # Overlay dismissal (proactive popup/banner removal)
    "OverlayDismissResult",
    "dismiss_overlays",
    "dismiss_overlays_before_agent",
    # Text Search
    "find_text_rect",
    "TextRectSearchResult",
    "TextMatch",
    "TextRect",
    "ViewportRect",
    "TextContext",
    # Agent Layer (Phase 1 & 2)
    "BaseAgent",
    "LLMProvider",
    "LLMResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "DeepInfraProvider",
    "LocalLLMProvider",
    "LocalVisionLLMProvider",
    "MLXVLMProvider",
    "OllamaProvider",
    "PredicateAgent",
    "PredicateAgentAsync",
    "SentienceAgent",
    "SentienceAgentAsync",
    "RuntimeAgent",
    "RuntimeStep",
    "StepVerification",
    "PredicateVisualAgent",
    "PredicateVisualAgentAsync",
    "SentienceVisualAgent",
    "SentienceVisualAgentAsync",
    "ConversationalAgent",
    # Agent Layer Models
    "AgentActionResult",
    "TokenStats",
    "ActionHistory",
    "ActionTokenUsage",
    "LLMStepData",
    "LLMUsage",
    "SnapshotOptions",
    "SnapshotFilter",
    "ScreenshotConfig",
    # Storage State Models (Auth Injection)
    "StorageState",
    "Cookie",
    "LocalStorageItem",
    "OriginStorage",
    # Tracing (v0.12.0+)
    "Tracer",
    "TraceSink",
    "JsonlTraceSink",
    "CloudTraceSink",
    "SentienceLogger",
    "TraceEvent",
    "create_tracer",
    "PREDICATE_API_URL",
    "SENTIENCE_API_URL",
    # Utilities (v0.12.0+)
    "canonical_snapshot_strict",
    "canonical_snapshot_loose",
    "compute_snapshot_digests",
    "sha256_digest",
    "save_storage_state",
    # Formatting (v0.12.0+)
    "format_snapshot_for_llm",
    "CategoryDetectionResult",
    "PrunedSnapshotContext",
    "PruningTaskCategory",
    "SkeletonDomNode",
    "classify_task_category",
    "prune_snapshot_for_task",
    "serialize_pruned_snapshot",
    # Agent Config (v0.12.0+)
    "AgentConfig",
    # Enums
    "SentienceMethod",
    "AgentAction",
    # Verification (agent assertion loop)
    "AgentRuntime",
    "PredicateDebugger",
    "SentienceDebugger",
    "AssertContext",
    "AssertOutcome",
    "Predicate",
    "url_matches",
    "url_contains",
    "exists",
    "not_exists",
    "element_count",
    "all_of",
    "any_of",
    "custom",
    # Ordinal support (Phase 3)
    "OrdinalIntent",
    "detect_ordinal_intent",
    "select_by_ordinal",
    "boost_ordinal_elements",
]
