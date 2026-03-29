"""
Agent factory for simplified agent creation.

Provides convenient factory functions to create PlannerExecutorAgent instances
with sensible defaults, auto-provider detection, and auto-tracer creation.

This module reduces boilerplate for common use cases:
- Local LLM via Ollama
- Cloud LLM via OpenAI/Anthropic
- Mixed configurations (cloud planner, local executor)
"""

from __future__ import annotations

import os
import uuid
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..llm_provider import AnthropicProvider, LLMProvider, OllamaProvider, OpenAIProvider
from ..tracer_factory import create_tracer
from ..tracing import JsonlTraceSink, Tracer
from .planner_executor_agent import (
    IntentHeuristics,
    PlannerExecutorAgent,
    PlannerExecutorConfig,
    RetryConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..models import Snapshot


# ---------------------------------------------------------------------------
# Config Presets
# ---------------------------------------------------------------------------


class ConfigPreset(str, Enum):
    """Pre-configured settings for common use cases."""

    DEFAULT = "default"
    LOCAL_SMALL_MODEL = "local_small"  # Optimized for 4B-8B local models
    CLOUD_HIGH_QUALITY = "cloud_high"  # Optimized for GPT-4/Claude
    FAST_ITERATION = "fast"  # Minimal retries for rapid development
    PRODUCTION = "production"  # Conservative settings for reliability


def get_config_preset(preset: ConfigPreset | str) -> PlannerExecutorConfig:
    """
    Get a pre-configured PlannerExecutorConfig for common use cases.

    Args:
        preset: Either a ConfigPreset enum value or string name

    Returns:
        PlannerExecutorConfig with preset values

    Example:
        >>> config = get_config_preset(ConfigPreset.LOCAL_SMALL_MODEL)
        >>> agent = create_planner_executor_agent(
        ...     planner_model="qwen3:8b",
        ...     executor_model="qwen3:4b",
        ...     config=config,
        ... )
    """
    if isinstance(preset, str):
        preset = ConfigPreset(preset)

    if preset == ConfigPreset.LOCAL_SMALL_MODEL:
        # Optimized for local 4B-8B models (Ollama)
        # - Tighter token limits work better with small models
        # - More lenient timeouts for slower local inference
        # - Verbose mode helpful for debugging local model behavior
        return PlannerExecutorConfig(
            planner_max_tokens=1024,
            executor_max_tokens=64,
            retry=RetryConfig(
                verify_timeout_s=15.0,
                verify_max_attempts=6,
            ),
            verbose=True,
        )

    elif preset == ConfigPreset.CLOUD_HIGH_QUALITY:
        # Optimized for high-capability cloud models (GPT-4, Claude)
        # - Higher token limits for more detailed plans
        # - Faster timeouts (cloud inference is quick)
        # - Verbose off for cleaner output
        return PlannerExecutorConfig(
            planner_max_tokens=2048,
            executor_max_tokens=128,
            retry=RetryConfig(
                verify_timeout_s=10.0,
                verify_max_attempts=4,
            ),
            verbose=False,
        )

    elif preset == ConfigPreset.FAST_ITERATION:
        # For rapid development and testing
        # - Minimal retries to fail fast
        # - Verbose for debugging
        return PlannerExecutorConfig(
            planner_max_tokens=1024,
            executor_max_tokens=64,
            retry=RetryConfig(
                verify_timeout_s=5.0,
                verify_max_attempts=2,
            ),
            verbose=True,
        )

    elif preset == ConfigPreset.PRODUCTION:
        # Conservative settings for production reliability
        # - More retries for robustness
        # - Longer timeouts for edge cases
        # - No verbose output
        return PlannerExecutorConfig(
            planner_max_tokens=2048,
            executor_max_tokens=128,
            retry=RetryConfig(
                verify_timeout_s=20.0,
                verify_max_attempts=8,
            ),
            verbose=False,
        )

    # Default
    return PlannerExecutorConfig()


# ---------------------------------------------------------------------------
# Provider Detection and Creation
# ---------------------------------------------------------------------------


def _detect_provider(model: str) -> str:
    """
    Auto-detect provider from model name.

    Args:
        model: Model name/identifier

    Returns:
        Provider name: "openai", "anthropic", or "ollama"
    """
    model_lower = model.lower()

    # OpenAI models
    if model_lower.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"

    # Anthropic models
    if model_lower.startswith("claude-"):
        return "anthropic"

    # Common Ollama model patterns
    if any(
        model_lower.startswith(p)
        for p in ("qwen", "llama", "phi", "mistral", "gemma", "deepseek", "codellama")
    ):
        return "ollama"

    # Ollama models typically have "model:tag" format
    if ":" in model:
        return "ollama"

    # Default to ollama for unknown models (assume local)
    return "ollama"


def _create_provider(
    model: str,
    provider: str,
    ollama_base_url: str,
    openai_api_key: str | None,
    anthropic_api_key: str | None,
) -> LLMProvider:
    """
    Create provider instance based on provider name.

    Args:
        model: Model name
        provider: Provider name ("auto", "ollama", "openai", "anthropic")
        ollama_base_url: Ollama server URL
        openai_api_key: OpenAI API key (can be None if using env var)
        anthropic_api_key: Anthropic API key (can be None if using env var)

    Returns:
        LLMProvider instance
    """
    if provider == "auto":
        provider = _detect_provider(model)

    if provider == "ollama":
        return OllamaProvider(model=model, base_url=ollama_base_url)

    elif provider == "openai":
        return OpenAIProvider(model=model, api_key=openai_api_key)

    elif provider == "anthropic":
        return AnthropicProvider(model=model, api_key=anthropic_api_key)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: 'auto', 'ollama', 'openai', 'anthropic'"
        )


def _create_auto_tracer(
    planner_model: str,
    executor_model: str,
    run_id: str | None = None,
) -> Tracer:
    """
    Create tracer based on environment configuration.

    If PREDICATE_API_KEY env var is set, creates a cloud tracer.
    Otherwise, creates a local JsonlTraceSink tracer.

    Args:
        planner_model: Planner model name (for metadata)
        executor_model: Executor model name (for metadata)
        run_id: Optional run ID (generates UUID if not provided)

    Returns:
        Configured Tracer instance
    """
    api_key = os.environ.get("PREDICATE_API_KEY")

    if run_id is None:
        run_id = f"run-{uuid.uuid4().hex[:8]}"

    if api_key:
        # Cloud tracing (auto-detected from env var)
        return create_tracer(
            api_key=api_key,
            run_id=run_id,
            llm_model=f"{planner_model}/{executor_model}",
            agent_type="planner-executor",
        )
    else:
        # Local file tracing
        trace_dir = Path("./traces")
        trace_dir.mkdir(exist_ok=True)
        trace_file = trace_dir / f"{run_id}.jsonl"
        sink = JsonlTraceSink(str(trace_file))
        return Tracer(run_id=run_id, sink=sink)


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def create_planner_executor_agent(
    *,
    planner_model: str,
    executor_model: str,
    planner_provider: str = "auto",
    executor_provider: str = "auto",
    ollama_base_url: str = "http://localhost:11434",
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    tracer: Tracer | Literal["auto"] | None = "auto",
    config: PlannerExecutorConfig | None = None,
    intent_heuristics: IntentHeuristics | None = None,
    context_formatter: Callable[[Snapshot, str], str] | None = None,
    run_id: str | None = None,
) -> PlannerExecutorAgent:
    """
    Create a PlannerExecutorAgent with sensible defaults and auto-detection.

    This factory function reduces boilerplate by:
    - Auto-detecting provider from model name (gpt-* -> OpenAI, claude-* -> Anthropic, etc.)
    - Auto-creating tracer (cloud if PREDICATE_API_KEY set, else local JSONL)
    - Providing sensible default configuration

    Args:
        planner_model: Model name for planning (e.g., "gpt-4o", "qwen3:8b")
        executor_model: Model name for execution (e.g., "gpt-4o-mini", "qwen3:4b")
        planner_provider: Provider for planner ("auto", "ollama", "openai", "anthropic")
        executor_provider: Provider for executor ("auto", "ollama", "openai", "anthropic")
        ollama_base_url: Ollama server URL (default: http://localhost:11434)
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        anthropic_api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        tracer: Tracer instance, "auto" to auto-create, or None to disable
        config: PlannerExecutorConfig (uses default if not provided)
        intent_heuristics: Optional domain-specific heuristics for element selection
        context_formatter: Optional custom context formatter for snapshots
        run_id: Optional run ID for tracing (generates UUID if not provided)

    Returns:
        Configured PlannerExecutorAgent instance

    Example (minimal - local Ollama):
        >>> agent = create_planner_executor_agent(
        ...     planner_model="qwen3:8b",
        ...     executor_model="qwen3:4b",
        ... )

    Example (cloud OpenAI):
        >>> agent = create_planner_executor_agent(
        ...     planner_model="gpt-4o",
        ...     executor_model="gpt-4o-mini",
        ...     openai_api_key="sk-...",  # or set OPENAI_API_KEY env var
        ... )

    Example (mixed - cloud planner, local executor):
        >>> agent = create_planner_executor_agent(
        ...     planner_model="gpt-4o",
        ...     planner_provider="openai",
        ...     executor_model="qwen3:4b",
        ...     executor_provider="ollama",
        ...     openai_api_key="sk-...",
        ... )

    Example (with config preset):
        >>> from predicate.agents import get_config_preset, ConfigPreset
        >>> agent = create_planner_executor_agent(
        ...     planner_model="qwen3:8b",
        ...     executor_model="qwen3:4b",
        ...     config=get_config_preset(ConfigPreset.LOCAL_SMALL_MODEL),
        ... )
    """
    # Create providers
    planner = _create_provider(
        model=planner_model,
        provider=planner_provider,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
    )

    executor = _create_provider(
        model=executor_model,
        provider=executor_provider,
        ollama_base_url=ollama_base_url,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
    )

    # Create tracer
    tracer_instance: Tracer | None = None
    if tracer == "auto":
        tracer_instance = _create_auto_tracer(
            planner_model=planner_model,
            executor_model=executor_model,
            run_id=run_id,
        )
    elif isinstance(tracer, Tracer):
        tracer_instance = tracer
    # else: tracer is None, leave tracer_instance as None

    # Use default config if not provided
    if config is None:
        config = PlannerExecutorConfig()

    return PlannerExecutorAgent(
        planner=planner,
        executor=executor,
        config=config,
        tracer=tracer_instance,
        intent_heuristics=intent_heuristics,
        context_formatter=context_formatter,
    )
