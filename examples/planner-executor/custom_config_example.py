#!/usr/bin/env python3
"""
PlannerExecutorAgent example with custom configuration.

This example demonstrates various configuration options:
- Snapshot escalation (enable/disable, custom step sizes)
- Retry configuration (timeouts, max attempts)
- Vision fallback settings

Usage:
    export OPENAI_API_KEY="sk-..."
    python custom_config_example.py
"""

from __future__ import annotations

import asyncio
import os

from predicate import AsyncPredicateBrowser
from predicate.agent_runtime import AgentRuntime
from predicate.agents import (
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    RetryConfig,
    SnapshotEscalationConfig,
)
from predicate.agents.browser_agent import VisionFallbackConfig
from predicate.backends.playwright_backend import PlaywrightBackend
from predicate.llm_provider import OpenAIProvider


async def example_default_config() -> None:
    """Default configuration: escalation enabled, step=30."""
    print("\n--- Example 1: Default Config ---")
    print("Escalation: 60 -> 90 -> 120 -> 150 -> 180 -> 200")

    config = PlannerExecutorConfig()

    print(f"  snapshot.enabled: {config.snapshot.enabled}")
    print(f"  snapshot.limit_base: {config.snapshot.limit_base}")
    print(f"  snapshot.limit_step: {config.snapshot.limit_step}")
    print(f"  snapshot.limit_max: {config.snapshot.limit_max}")


async def example_disabled_escalation() -> None:
    """Disable escalation: always use limit_base."""
    print("\n--- Example 2: Disabled Escalation ---")
    print("Escalation: disabled (always 60)")

    config = PlannerExecutorConfig(
        snapshot=SnapshotEscalationConfig(enabled=False),
    )

    print(f"  snapshot.enabled: {config.snapshot.enabled}")
    print(f"  snapshot.limit_base: {config.snapshot.limit_base}")


async def example_custom_step_size() -> None:
    """Custom step size for faster escalation."""
    print("\n--- Example 3: Custom Step Size ---")
    print("Escalation: 60 -> 110 -> 160 -> 200 (step=50)")

    config = PlannerExecutorConfig(
        snapshot=SnapshotEscalationConfig(
            limit_step=50,  # Larger steps = fewer iterations
        ),
    )

    print(f"  snapshot.limit_step: {config.snapshot.limit_step}")


async def example_custom_limits() -> None:
    """Custom base and max limits."""
    print("\n--- Example 4: Custom Limits ---")
    print("Escalation: 100 -> 125 -> 150 -> 175 -> 200 -> 225 -> 250")

    config = PlannerExecutorConfig(
        snapshot=SnapshotEscalationConfig(
            limit_base=100,  # Start higher
            limit_step=25,   # Smaller increments
            limit_max=250,   # Higher maximum
        ),
    )

    print(f"  snapshot.limit_base: {config.snapshot.limit_base}")
    print(f"  snapshot.limit_step: {config.snapshot.limit_step}")
    print(f"  snapshot.limit_max: {config.snapshot.limit_max}")


async def example_retry_config() -> None:
    """Custom retry configuration."""
    print("\n--- Example 5: Retry Config ---")

    config = PlannerExecutorConfig(
        retry=RetryConfig(
            verify_timeout_s=15.0,       # Longer timeout for slow pages
            verify_poll_s=0.3,           # Faster polling
            verify_max_attempts=10,      # More verification attempts
            executor_repair_attempts=3,  # More repair attempts
            max_replans=2,               # Allow 2 replans on failure
        ),
    )

    print(f"  retry.verify_timeout_s: {config.retry.verify_timeout_s}")
    print(f"  retry.verify_max_attempts: {config.retry.verify_max_attempts}")
    print(f"  retry.max_replans: {config.retry.max_replans}")


async def example_vision_fallback() -> None:
    """Vision fallback configuration."""
    print("\n--- Example 6: Vision Fallback ---")

    config = PlannerExecutorConfig(
        vision=VisionFallbackConfig(
            enabled=True,
            max_vision_calls=5,                      # Up to 5 vision calls per run
            trigger_requires_vision=True,            # Trigger on require_vision status
            trigger_canvas_or_low_actionables=True,  # Trigger on canvas pages
        ),
    )

    print(f"  vision.enabled: {config.vision.enabled}")
    print(f"  vision.max_vision_calls: {config.vision.max_vision_calls}")


async def example_full_custom() -> None:
    """Full custom configuration with all options."""
    print("\n--- Example 7: Full Custom Config ---")

    config = PlannerExecutorConfig(
        # Snapshot escalation
        snapshot=SnapshotEscalationConfig(
            enabled=True,
            limit_base=80,
            limit_step=40,
            limit_max=240,
        ),
        # Retry settings
        retry=RetryConfig(
            verify_timeout_s=12.0,
            verify_poll_s=0.4,
            verify_max_attempts=6,
            max_replans=2,
        ),
        # Vision fallback
        vision=VisionFallbackConfig(
            enabled=True,
            max_vision_calls=3,
        ),
        # Planner settings
        planner_max_tokens=3000,
        planner_temperature=0.0,
        # Executor settings
        executor_max_tokens=128,
        executor_temperature=0.0,
        # Tracing
        trace_screenshots=True,
        trace_screenshot_format="jpeg",
        trace_screenshot_quality=85,
    )

    print("  Full config created successfully!")
    print(f"  Escalation: {config.snapshot.limit_base} -> ... -> {config.snapshot.limit_max}")
    print(f"  Max replans: {config.retry.max_replans}")
    print(f"  Vision enabled: {config.vision.enabled}")


async def example_run_with_config() -> None:
    """Run agent with custom config."""
    print("\n--- Example 8: Run Agent with Custom Config ---")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("  Skipping (no OPENAI_API_KEY)")
        return

    predicate_api_key = os.getenv("PREDICATE_API_KEY")

    # Create config optimized for reliability
    config = PlannerExecutorConfig(
        snapshot=SnapshotEscalationConfig(
            enabled=True,
            limit_base=60,
            limit_step=30,
            limit_max=180,
        ),
        retry=RetryConfig(
            verify_timeout_s=10.0,
            max_replans=1,
        ),
    )

    planner = OpenAIProvider(model="gpt-4o")
    executor = OpenAIProvider(model="gpt-4o-mini")

    agent = PlannerExecutorAgent(
        planner=planner,
        executor=executor,
        config=config,
    )

    async with AsyncPredicateBrowser(
        api_key=predicate_api_key,
        headless=True,
    ) as browser:
        page = await browser.new_page()
        await page.goto("https://example.com")

        backend = PlaywrightBackend(page)
        runtime = AgentRuntime(backend=backend)

        result = await agent.run(
            runtime=runtime,
            task="Verify example.com is loaded",
        )

        print(f"  Success: {result.success}")
        print(f"  Steps: {result.steps_completed}/{result.steps_total}")


async def main() -> None:
    print("PlannerExecutorAgent Configuration Examples")
    print("=" * 50)

    await example_default_config()
    await example_disabled_escalation()
    await example_custom_step_size()
    await example_custom_limits()
    await example_retry_config()
    await example_vision_fallback()
    await example_full_custom()
    await example_run_with_config()

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
