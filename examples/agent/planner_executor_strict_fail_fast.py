"""
Example: PlannerExecutorAgent strict fail-fast behavior.

This demo runs the same failing required step in two modes:
- default mode (allows recovery/replan policy)
- strict fail-fast mode (abort immediately on required-step failure)

Why this example is deterministic:
- We inject a fixed single-step plan.
- We inject a fixed failed step outcome.
- We count whether recovery/replan hooks are reached.

Usage:
  python examples/agent/planner_executor_strict_fail_fast.py
"""

from __future__ import annotations

import asyncio

from predicate.agents import (
    Plan,
    PlanStep,
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    PredicateSpec,
    RetryConfig,
    StepOutcome,
    StepStatus,
)
from predicate.llm_provider import LLMProvider, LLMResponse


class FixedProvider(LLMProvider):
    """Minimal provider used only to satisfy agent construction."""

    def __init__(self) -> None:
        super().__init__(model="fixed-provider")

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        _ = system_prompt, user_prompt, kwargs
        return LLMResponse(content="{}", model_name=self.model_name)

    def supports_json_mode(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "fixed-provider"


class DemoRuntime:
    """Tiny runtime for the fail-fast demo."""

    def __init__(self, start_url: str = "https://shop.example.com/search") -> None:
        self._url = start_url

    async def get_url(self) -> str:
        return self._url

    async def goto(self, url: str) -> None:
        self._url = url

    async def stabilize(self) -> None:
        return None


async def run_demo(strict_fail_fast: bool) -> None:
    config = PlannerExecutorConfig(
        strict_fail_fast=strict_fail_fast,
        retry=RetryConfig(max_replans=1),
        auto_fallback_to_stepwise=False,
    )
    agent = PlannerExecutorAgent(
        planner=FixedProvider(),
        executor=FixedProvider(),
        config=config,
    )
    runtime = DemoRuntime()

    plan = Plan(
        task="Open a product details page",
        steps=[
            PlanStep(
                id=1,
                goal="Click a product link",
                action="CLICK",
                intent="product link",
                verify=[PredicateSpec(predicate="url_contains", args=["/product/"])],
                required=True,
            )
        ],
    )

    failed_step = StepOutcome(
        step_id=1,
        goal="Click a product link",
        status=StepStatus.FAILED,
        action_taken="CLICK(1)",
        verification_passed=False,
        error="verification_failed",
    )

    call_counts = {"recovery": 0, "replan": 0}

    async def fake_plan(*args, **kwargs) -> Plan:
        _ = args, kwargs
        return plan

    async def fake_execute_step(*args, **kwargs) -> StepOutcome:
        _ = args, kwargs
        return failed_step

    async def fake_attempt_recovery(*args, **kwargs) -> bool:
        _ = args, kwargs
        call_counts["recovery"] += 1
        return False

    async def fake_replan(*args, **kwargs) -> Plan:
        _ = args, kwargs
        call_counts["replan"] += 1
        # Mirror internal replan accounting so the loop exits after one replan.
        agent._replans_used += 1  # type: ignore[attr-defined]
        return plan

    agent.plan = fake_plan  # type: ignore[method-assign]
    agent._execute_step = fake_execute_step  # type: ignore[method-assign]
    agent._attempt_recovery = fake_attempt_recovery  # type: ignore[method-assign]
    agent.replan = fake_replan  # type: ignore[method-assign]

    result = await agent.run(
        runtime=runtime,
        task="Open a product details page",
        start_url="https://shop.example.com",
    )

    mode = "STRICT_FAIL_FAST" if strict_fail_fast else "DEFAULT"
    print(f"\n=== {mode} ===")
    print(f"success={result.success}")
    print(f"error={result.error}")
    print(f"steps_completed={result.steps_completed}")
    print(f"replans_used={result.replans_used}")
    print(f"recovery_calls={call_counts['recovery']}")
    print(f"replan_calls={call_counts['replan']}")


async def main() -> None:
    print("PlannerExecutorAgent strict fail-fast demo")
    await run_demo(strict_fail_fast=False)
    await run_demo(strict_fail_fast=True)


if __name__ == "__main__":
    asyncio.run(main())
