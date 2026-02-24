"""
Async API for Predicate SDK - Convenience re-exports

This module re-exports all async functions for backward compatibility and developer convenience.
You can also import directly from their respective modules:

    # Option 1: From async_api (recommended for convenience)
    from predicate.async_api import (
        AsyncPredicateBrowser,
        snapshot_async,
        click_async,
        wait_for_async,
        screenshot_async,
        find_text_rect_async,
        # ... all async functions in one place
    )

    # Option 2: From respective modules (also works)
    from predicate.browser import AsyncSentienceBrowser
    from predicate.snapshot import snapshot_async
    from predicate.actions import click_async
"""

# ========== Actions (Phase 1) ==========
# Re-export async action functions from actions.py
from predicate.actions import (
    click_async,
    click_rect_async,
    press_async,
    scroll_to_async,
    type_text_async,
)

# ========== Phase 2C: Agent Layer ==========
# Re-export async agent classes from agent.py and base_agent.py
from predicate.agent import SentienceAgentAsync
from predicate.base_agent import BaseAgentAsync

# ========== Browser ==========
# Re-export AsyncSentienceBrowser from browser.py (moved there for better organization)
from predicate.browser import AsyncSentienceBrowser

# Predicate-named class counterparts (canonical moving forward).
# Keep Sentience* names for backward compatibility.
AsyncPredicateBrowser = AsyncSentienceBrowser
PredicateAgentAsync = SentienceAgentAsync

# Re-export async expect functions from expect.py
from predicate.expect import ExpectationAsync, expect_async
from predicate.inspector import InspectorAsync, inspect_async

# Re-export async overlay functions from overlay.py
from predicate.overlay import clear_overlay_async, show_overlay_async

# ========== Query Functions (Pure Functions - No Async Needed) ==========
# Re-export query functions (pure functions, no async needed)
from predicate.query import find, query

# ========== Phase 2B: Supporting Utilities ==========
# Re-export async read functions from read.py
from predicate.read import read_async, read_best_effort_async

# ========== Phase 2D: Developer Tools ==========
# Re-export async recorder and inspector from their modules
from predicate.recorder import RecorderAsync, record_async

# Re-export async screenshot function from screenshot.py
from predicate.screenshot import screenshot_async

# ========== Snapshot (Phase 1) ==========
# Re-export async snapshot functions from snapshot.py
from predicate.snapshot import snapshot_async

# Re-export async text search function from text_search.py
from predicate.text_search import find_text_rect_async

# ========== Phase 2A: Core Utilities ==========
# Re-export async wait function from wait.py
from predicate.wait import wait_for_async

__all__ = [
    # Browser
    "AsyncSentienceBrowser",  # Re-exported from browser.py
    "AsyncPredicateBrowser",  # Predicate-named alias (canonical)
    # Snapshot (Phase 1)
    "snapshot_async",  # Re-exported from snapshot.py
    # Actions (Phase 1)
    "click_async",  # Re-exported from actions.py
    "type_text_async",  # Re-exported from actions.py
    "press_async",  # Re-exported from actions.py
    "scroll_to_async",  # Re-exported from actions.py
    "click_rect_async",  # Re-exported from actions.py
    # Phase 2A: Core Utilities
    "wait_for_async",  # Re-exported from wait.py
    "screenshot_async",  # Re-exported from screenshot.py
    "find_text_rect_async",  # Re-exported from text_search.py
    # Phase 2B: Supporting Utilities
    "read_async",  # Re-exported from read.py
    "read_best_effort_async",  # Re-exported from read.py
    "show_overlay_async",  # Re-exported from overlay.py
    "clear_overlay_async",  # Re-exported from overlay.py
    "expect_async",  # Re-exported from expect.py
    "ExpectationAsync",  # Re-exported from expect.py
    # Phase 2C: Agent Layer
    "SentienceAgentAsync",  # Re-exported from agent.py (legacy name)
    "PredicateAgentAsync",  # Predicate-named alias (canonical)
    "BaseAgentAsync",  # Re-exported from base_agent.py
    # Phase 2D: Developer Tools
    "RecorderAsync",  # Re-exported from recorder.py
    "record_async",  # Re-exported from recorder.py
    "InspectorAsync",  # Re-exported from inspector.py
    "inspect_async",  # Re-exported from inspector.py
    # Query Functions
    "find",  # Re-exported from query.py
    "query",  # Re-exported from query.py
]
