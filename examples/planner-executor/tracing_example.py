#!/usr/bin/env python3
"""
PlannerExecutorAgent example with full tracing for Predicate Studio.

This example demonstrates how to capture traces that can be visualized
in Predicate Studio, including:
- Run start/end events
- Plan generation events
- Step execution with screenshots
- Verification results
- Replan events on failure

Usage:
    export OPENAI_API_KEY="sk-..."
    export PREDICATE_API_KEY="sk_..."  # Required for cloud tracing
    python tracing_example.py

After running, view your trace at:
    https://studio.predicateapi.com/traces/<run_id>
"""

from __future__ import annotations

import asyncio
import os
import uuid

from predicate import AsyncPredicateBrowser
from predicate.agent_runtime import AgentRuntime
from predicate.agents import (
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    SnapshotEscalationConfig,
)
from predicate.backends.playwright_backend import PlaywrightBackend
from predicate.llm_provider import OpenAIProvider
from predicate.tracer_factory import create_tracer


async def main() -> None:
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("Missing OPENAI_API_KEY environment variable")

    predicate_api_key = os.getenv("PREDICATE_API_KEY")
    if not predicate_api_key:
        print("Warning: No PREDICATE_API_KEY - traces will be saved locally only")

    # Generate run ID for tracking
    run_id = str(uuid.uuid4())
    task = "Go to example.com and verify the page has loaded with a heading"

    print(f"Run ID: {run_id}")
    print(f"Task: {task}")
    print("=" * 60)

    # Create tracer for Predicate Studio
    # This will upload traces to the cloud if api_key is provided
    tracer = create_tracer(
        api_key=predicate_api_key,
        run_id=run_id,
        upload_trace=bool(predicate_api_key),  # Upload if key provided
        goal=task,
        agent_type="PlannerExecutorAgent",
        llm_model="gpt-4o / gpt-4o-mini",
        start_url="https://example.com",
    )

    # Create LLM providers
    planner = OpenAIProvider(model="gpt-4o")
    executor = OpenAIProvider(model="gpt-4o-mini")

    # Create config with screenshot tracing enabled
    config = PlannerExecutorConfig(
        snapshot=SnapshotEscalationConfig(
            enabled=True,
            limit_base=60,
            limit_step=30,
            limit_max=180,
        ),
        # Enable screenshot capture for Studio visualization
        trace_screenshots=True,
        trace_screenshot_format="jpeg",
        trace_screenshot_quality=80,
    )

    # Create agent WITH tracer
    agent = PlannerExecutorAgent(
        planner=planner,
        executor=executor,
        config=config,
        tracer=tracer,  # Pass tracer for Studio visualization
    )

    try:
        async with AsyncPredicateBrowser(
            api_key=predicate_api_key,
            headless=False,
        ) as browser:
            page = await browser.new_page()
            await page.goto("https://example.com")
            await page.wait_for_load_state("networkidle")

            # Create runtime
            backend = PlaywrightBackend(page)
            runtime = AgentRuntime(
                backend=backend,
                tracer=tracer,  # Also pass to runtime for step-level tracing
            )

            # Run the agent
            result = await agent.run(
                runtime=runtime,
                task=task,
                start_url="https://example.com",
                run_id=run_id,
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
                vision = " [VISION]" if outcome.used_vision else ""
                print(f"  [{status}]{vision} Step {outcome.step_id}: {outcome.goal}")
                print(f"         Duration: {outcome.duration_ms}ms")
                if outcome.action_taken:
                    print(f"         Action: {outcome.action_taken}")

    finally:
        # Close tracer to upload trace
        print("\n" + "=" * 60)
        print("Uploading trace to Predicate Studio...")
        tracer.close(blocking=True)

        if predicate_api_key:
            print(f"\nView your trace at:")
            print(f"  https://studio.predicateapi.com/traces/{run_id}")
        else:
            print(f"\nTrace saved locally to: traces/{run_id}.jsonl")


async def example_trace_events() -> None:
    """
    Demonstrates the trace events emitted by PlannerExecutorAgent.

    Events emitted (in order):
    1. run_start - Agent run begins
    2. plan_generated - Planner creates execution plan
    3. For each step:
       - step_start - Step execution begins
       - snapshot - Page snapshot with screenshot
       - step_end - Step execution completes
    4. replan (if needed) - Plan modified after failure
    5. run_end - Agent run completes
    """
    print("\nTrace Events Reference:")
    print("-" * 40)

    events = [
        ("run_start", "Agent run begins", "task, config, model names"),
        ("plan_generated", "Planner creates plan", "steps, raw LLM output"),
        ("step_start", "Step begins", "step_id, goal, pre_url"),
        ("snapshot", "Page captured", "screenshot_base64, elements, url"),
        ("step_end", "Step completes", "action, verification, post_url"),
        ("replan", "Plan modified", "failed_step, new_plan"),
        ("run_end", "Run completes", "success, duration, steps"),
    ]

    for event, description, data in events:
        print(f"  {event:20} - {description}")
        print(f"  {'':20}   Data: {data}")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(example_trace_events())
