#!/usr/bin/env python3
"""
PlannerExecutorAgent example with local HuggingFace models.

This example demonstrates using local models instead of cloud APIs:
- Planner: DeepSeek-R1-Distill-Qwen-14B (reasoning model)
- Executor: Qwen2.5-7B-Instruct (fast instruction following)

Usage:
    export PREDICATE_API_KEY="sk_..."  # Optional, for cloud browser
    python local_models_example.py

Requirements:
    pip install torch transformers accelerate
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from predicate import AsyncPredicateBrowser
from predicate.agent_runtime import AgentRuntime
from predicate.agents import (
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    SnapshotEscalationConfig,
)
from predicate.backends.playwright_backend import PlaywrightBackend
from predicate.llm_provider import LLMProvider, LLMResponse


@dataclass
class LocalHFProvider(LLMProvider):
    """
    Local HuggingFace model provider.

    Loads a model from HuggingFace and runs inference locally.
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(model=model_name)
        self._model_name = model_name

        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"Model loaded: {model_name}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> LLMResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs.input_ids.shape[1]

        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.0)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        completion_tokens = outputs.shape[1] - prompt_tokens
        response_text = self.tokenizer.decode(
            outputs[0][prompt_tokens:],
            skip_special_tokens=True,
        )

        return LLMResponse(
            content=response_text,
            model_name=self._model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def supports_json_mode(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return self._model_name


async def main() -> None:
    predicate_api_key = os.getenv("PREDICATE_API_KEY")

    # Create local model providers
    # Use smaller models for demo; adjust based on your hardware
    planner_model = os.getenv(
        "PLANNER_MODEL",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    )
    executor_model = os.getenv(
        "EXECUTOR_MODEL",
        "Qwen/Qwen2.5-7B-Instruct",
    )

    planner = LocalHFProvider(planner_model)
    executor = LocalHFProvider(executor_model)

    # Create agent with custom config for local models
    config = PlannerExecutorConfig(
        # Slightly larger limits for local models
        snapshot=SnapshotEscalationConfig(
            limit_base=80,
            limit_step=40,
            limit_max=200,
        ),
        # Longer timeouts for local inference
        planner_max_tokens=2048,
        executor_max_tokens=128,
    )

    agent = PlannerExecutorAgent(
        planner=planner,
        executor=executor,
        config=config,
    )

    # Simple task
    task = "Navigate to example.com and find the main heading"

    print(f"Task: {task}")
    print(f"Planner: {planner_model}")
    print(f"Executor: {executor_model}")
    print("=" * 60)

    async with AsyncPredicateBrowser(
        api_key=predicate_api_key,
        headless=False,
    ) as browser:
        page = await browser.new_page()
        await page.goto("https://example.com")
        await page.wait_for_load_state("networkidle")

        backend = PlaywrightBackend(page)
        runtime = AgentRuntime(backend=backend)

        result = await agent.run(
            runtime=runtime,
            task=task,
            start_url="https://example.com",
        )

        print("\n" + "=" * 60)
        print(f"Success: {result.success}")
        print(f"Steps: {result.steps_completed}/{result.steps_total}")
        print(f"Duration: {result.total_duration_ms}ms")


if __name__ == "__main__":
    asyncio.run(main())
