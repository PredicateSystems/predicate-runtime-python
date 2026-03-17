#!/usr/bin/env python3
"""
Minimal PlannerExecutorAgent example with OpenAI models.

This example demonstrates basic usage of the two-tier agent:
- Planner: gpt-4o generates the execution plan
- Executor: gpt-4o-mini executes each step

Usage:
    export OPENAI_API_KEY="sk-..."
    export PREDICATE_API_KEY="sk_..."  # Optional, for cloud browser
    python minimal_example.py
"""

from __future__ import annotations

import asyncio
import os

from predicate import AsyncPredicateBrowser
from predicate.agent_runtime import AgentRuntime
from predicate.agents import PlannerExecutorAgent, PlannerExecutorConfig
from predicate.backends.playwright_backend import PlaywrightBackend
from predicate.llm_provider import OpenAIProvider


async def main() -> None:
    # Check for API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("Missing OPENAI_API_KEY environment variable")

    predicate_api_key = os.getenv("PREDICATE_API_KEY")

    # Create LLM providers
    # Planner: Use a capable model for plan generation
    planner = OpenAIProvider(model="gpt-4o")

    # Executor: Use a smaller/faster model for step execution
    executor = OpenAIProvider(model="gpt-4o-mini")

    # Create agent with default config
    agent = PlannerExecutorAgent(
        planner=planner,
        executor=executor,
        config=PlannerExecutorConfig(),
    )

    # Task to execute
    task = "Go to example.com and verify the page has a heading"

    print(f"Task: {task}")
    print("=" * 60)

    # Grant common permissions to avoid browser permission prompts.
    # This prevents dialogs like "Allow this site to access your location?"
    # from interrupting the automation.
    permission_policy = {
        "auto_grant": [
            "geolocation",      # Store locators, local inventory
            "notifications",    # Push notification prompts
            "clipboard-read",   # Paste coupon codes
            "clipboard-write",  # Copy product info
        ],
        "geolocation": {"latitude": 47.6762, "longitude": -122.2057},  # Kirkland, WA
    }

    async with AsyncPredicateBrowser(
        api_key=predicate_api_key,
        headless=False,
        permission_policy=permission_policy,
    ) as browser:
        page = await browser.new_page()
        await page.goto("https://example.com")
        await page.wait_for_load_state("networkidle")

        # Create runtime
        backend = PlaywrightBackend(page)
        runtime = AgentRuntime(backend=backend)

        # Run the agent
        result = await agent.run(
            runtime=runtime,
            task=task,
            start_url="https://example.com",
        )

        # Print results
        print("\n" + "=" * 60)
        print(f"Success: {result.success}")
        print(f"Steps completed: {result.steps_completed}/{result.steps_total}")
        print(f"Replans used: {result.replans_used}")
        print(f"Total duration: {result.total_duration_ms}ms")

        if result.error:
            print(f"Error: {result.error}")

        print("\nStep details:")
        for outcome in result.step_outcomes:
            status = "PASS" if outcome.status.value == "success" else "FAIL"
            print(f"  [{status}] Step {outcome.step_id}: {outcome.goal}")
            if outcome.action_taken:
                print(f"         Action: {outcome.action_taken}")
            if outcome.error:
                print(f"         Error: {outcome.error}")


if __name__ == "__main__":
    asyncio.run(main())
