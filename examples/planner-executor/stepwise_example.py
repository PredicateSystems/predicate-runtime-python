#!/usr/bin/env python3
"""
Stepwise (ReAct-style) planning example.

This example demonstrates the stepwise planning mode where the planner
decides one action at a time based on the current page state. This is
useful for unfamiliar sites where upfront planning may make wrong assumptions.

Features demonstrated:
- StepwisePlanningConfig for ReAct-style planning
- Permission policy to dismiss Chrome permission dialogs
- Action history tracking to avoid loops

Usage:
    export OPENAI_API_KEY="sk-..."
    export PREDICATE_API_KEY="sk_..."  # Optional, for cloud browser
    python stepwise_example.py
    python stepwise_example.py --url https://www.homedepot.com --goal "buy a hammer"
"""

from __future__ import annotations

import argparse
import asyncio
import os

from predicate import AsyncPredicateBrowser
from predicate.agent_runtime import AgentRuntime
from predicate.agents import (
    AutomationTask,
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    SnapshotEscalationConfig,
    StepwisePlanningConfig,
)
from predicate.backends.playwright_backend import PlaywrightBackend
from predicate.llm_provider import OpenAIProvider


async def main(
    goal: str = "buy a grass mower",
    starting_url: str = "https://www.acehardware.com",
) -> None:
    # Check for API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("Missing OPENAI_API_KEY environment variable")

    predicate_api_key = os.getenv("PREDICATE_API_KEY")

    # Create LLM providers
    planner = OpenAIProvider(model="gpt-4o")
    executor = OpenAIProvider(model="gpt-4o-mini")

    # Configure stepwise planning
    config = PlannerExecutorConfig(
        # Stepwise planning: plan one action at a time based on current page
        stepwise=StepwisePlanningConfig(
            max_steps=30,              # Maximum steps before stopping
            action_history_limit=5,    # Recent actions to include in context
            include_page_context=True, # Include page elements in planner prompt
        ),
        # Snapshot escalation for reliable element capture
        snapshot=SnapshotEscalationConfig(
            enabled=True,
            limit_base=60,
            limit_step=30,
            limit_max=200,
        ),
        # Verbose mode to see planner decisions
        verbose=True,
    )

    # Create agent
    agent = PlannerExecutorAgent(
        planner=planner,
        executor=executor,
        config=config,
    )

    # Create task
    task = AutomationTask(
        task_id="stepwise-demo",
        starting_url=starting_url,
        task=goal,
        enable_recovery=True,
        max_recovery_attempts=2,
    )

    print("=" * 60)
    print("Stepwise Planning Demo")
    print("=" * 60)
    print(f"Goal: {goal}")
    print(f"Starting URL: {starting_url}")
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
        # Mock geolocation coordinates (required when granting geolocation)
        "geolocation": {"latitude": 47.6762, "longitude": -122.2057},  # Kirkland, WA
    }

    async with AsyncPredicateBrowser(
        api_key=predicate_api_key,
        headless=False,
        permission_policy=permission_policy,
    ) as browser:
        page = browser.page
        await page.goto(starting_url)
        await page.wait_for_load_state("domcontentloaded")

        # Create runtime
        backend = PlaywrightBackend(page)
        runtime = AgentRuntime(
            backend=backend,
            predicate_api_key=predicate_api_key,
        )

        # Run stepwise planning
        # Unlike agent.run() which generates a full plan upfront,
        # run_stepwise() plans one action at a time based on the current page.
        result = await agent.run_stepwise(runtime, task)

        # Print results
        print("\n" + "=" * 60)
        print("Run Complete")
        print("=" * 60)
        print(f"Success: {result.success}")
        print(f"Steps completed: {result.steps_completed}/{result.steps_total}")
        print(f"Total duration: {result.total_duration_ms}ms")

        if result.error:
            print(f"Error: {result.error}")

        print("\nStep details:")
        for outcome in result.step_outcomes:
            status = "PASS" if outcome.verification_passed else "FAIL"
            print(f"  [{status}] Step {outcome.step_id}: {outcome.goal[:60]}...")
            if outcome.action_taken:
                print(f"         Action: {outcome.action_taken}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stepwise planning demo")
    parser.add_argument(
        "--goal",
        type=str,
        default="buy a grass mower",
        help="Goal for the automation",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://www.acehardware.com",
        help="Starting URL",
    )
    args = parser.parse_args()

    asyncio.run(main(goal=args.goal, starting_url=args.url))
