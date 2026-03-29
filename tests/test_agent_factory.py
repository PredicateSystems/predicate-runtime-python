"""Tests for agent_factory module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from predicate.agents.agent_factory import (
    ConfigPreset,
    _create_auto_tracer,
    _create_provider,
    _detect_provider,
    create_planner_executor_agent,
    get_config_preset,
)
from predicate.agents.planner_executor_agent import PlannerExecutorAgent, PlannerExecutorConfig
from predicate.llm_provider import OllamaProvider
from predicate.tracing import Tracer

# Optional imports for cloud providers
try:
    from predicate.llm_provider import OpenAIProvider

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from predicate.llm_provider import AnthropicProvider

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class TestDetectProvider:
    """Test provider auto-detection from model names."""

    def test_detect_openai_gpt4(self):
        """Should detect OpenAI for GPT-4 models."""
        assert _detect_provider("gpt-4o") == "openai"
        assert _detect_provider("gpt-4-turbo") == "openai"
        assert _detect_provider("gpt-4o-mini") == "openai"
        assert _detect_provider("GPT-4o") == "openai"  # Case insensitive

    def test_detect_openai_o1(self):
        """Should detect OpenAI for o1 reasoning models."""
        assert _detect_provider("o1-preview") == "openai"
        assert _detect_provider("o1-mini") == "openai"

    def test_detect_openai_o3(self):
        """Should detect OpenAI for o3 models."""
        assert _detect_provider("o3-mini") == "openai"

    def test_detect_anthropic_claude(self):
        """Should detect Anthropic for Claude models."""
        assert _detect_provider("claude-3-opus-20240229") == "anthropic"
        assert _detect_provider("claude-3-5-sonnet-20241022") == "anthropic"
        assert _detect_provider("claude-3-haiku-20240307") == "anthropic"
        assert _detect_provider("Claude-3-Opus") == "anthropic"  # Case insensitive

    def test_detect_ollama_qwen(self):
        """Should detect Ollama for Qwen models."""
        assert _detect_provider("qwen3:8b") == "ollama"
        assert _detect_provider("qwen2.5:7b-instruct") == "ollama"
        assert _detect_provider("Qwen3:4b") == "ollama"

    def test_detect_ollama_llama(self):
        """Should detect Ollama for Llama models."""
        assert _detect_provider("llama3:8b") == "ollama"
        assert _detect_provider("llama3.2:3b") == "ollama"
        assert _detect_provider("codellama:7b") == "ollama"

    def test_detect_ollama_other_local(self):
        """Should detect Ollama for other common local models."""
        assert _detect_provider("phi3:mini") == "ollama"
        assert _detect_provider("mistral:7b") == "ollama"
        assert _detect_provider("gemma:2b") == "ollama"
        assert _detect_provider("deepseek:6.7b") == "ollama"

    def test_detect_ollama_by_tag_format(self):
        """Should detect Ollama for model:tag format."""
        assert _detect_provider("custom-model:latest") == "ollama"
        assert _detect_provider("my-finetuned:v2") == "ollama"

    def test_detect_unknown_defaults_ollama(self):
        """Unknown models should default to Ollama."""
        assert _detect_provider("some-unknown-model") == "ollama"


class TestCreateProvider:
    """Test provider creation."""

    def test_create_ollama_provider(self):
        """Should create OllamaProvider for ollama."""
        provider = _create_provider(
            model="qwen3:8b",
            provider="ollama",
            ollama_base_url="http://localhost:11434",
            openai_api_key=None,
            anthropic_api_key=None,
        )
        assert isinstance(provider, OllamaProvider)
        assert provider.model_name == "qwen3:8b"

    @pytest.mark.skipif(not HAS_OPENAI, reason="openai package not installed")
    def test_create_openai_provider(self):
        """Should create OpenAIProvider for openai."""
        provider = _create_provider(
            model="gpt-4o",
            provider="openai",
            ollama_base_url="http://localhost:11434",
            openai_api_key="test-key",
            anthropic_api_key=None,
        )
        assert isinstance(provider, OpenAIProvider)
        assert provider.model_name == "gpt-4o"

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic package not installed")
    def test_create_anthropic_provider(self):
        """Should create AnthropicProvider for anthropic."""
        provider = _create_provider(
            model="claude-3-opus-20240229",
            provider="anthropic",
            ollama_base_url="http://localhost:11434",
            openai_api_key=None,
            anthropic_api_key="test-key",
        )
        assert isinstance(provider, AnthropicProvider)
        assert provider.model_name == "claude-3-opus-20240229"

    def test_create_provider_auto_detection(self):
        """Should auto-detect provider when 'auto' specified."""
        provider = _create_provider(
            model="qwen3:8b",
            provider="auto",
            ollama_base_url="http://localhost:11434",
            openai_api_key=None,
            anthropic_api_key=None,
        )
        assert isinstance(provider, OllamaProvider)

    def test_create_provider_invalid_raises(self):
        """Should raise ValueError for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            _create_provider(
                model="test",
                provider="invalid-provider",
                ollama_base_url="http://localhost:11434",
                openai_api_key=None,
                anthropic_api_key=None,
            )


class TestConfigPresets:
    """Test configuration presets."""

    def test_get_config_preset_default(self):
        """Should return default config for DEFAULT preset."""
        config = get_config_preset(ConfigPreset.DEFAULT)
        assert isinstance(config, PlannerExecutorConfig)

    def test_get_config_preset_local_small_model(self):
        """Should return optimized config for local small models."""
        config = get_config_preset(ConfigPreset.LOCAL_SMALL_MODEL)
        assert isinstance(config, PlannerExecutorConfig)
        # Check optimized settings
        assert config.planner_max_tokens == 1024
        assert config.executor_max_tokens == 64
        assert config.retry.verify_timeout_s == 15.0
        assert config.retry.verify_max_attempts == 6
        assert config.verbose is True

    def test_get_config_preset_cloud_high_quality(self):
        """Should return optimized config for cloud models."""
        config = get_config_preset(ConfigPreset.CLOUD_HIGH_QUALITY)
        assert isinstance(config, PlannerExecutorConfig)
        assert config.planner_max_tokens == 2048
        assert config.executor_max_tokens == 128
        assert config.retry.verify_timeout_s == 10.0
        assert config.verbose is False

    def test_get_config_preset_fast_iteration(self):
        """Should return fast iteration config."""
        config = get_config_preset(ConfigPreset.FAST_ITERATION)
        assert isinstance(config, PlannerExecutorConfig)
        assert config.retry.verify_max_attempts == 2
        assert config.verbose is True

    def test_get_config_preset_production(self):
        """Should return production config with conservative settings."""
        config = get_config_preset(ConfigPreset.PRODUCTION)
        assert isinstance(config, PlannerExecutorConfig)
        assert config.retry.verify_max_attempts == 8
        assert config.retry.verify_timeout_s == 20.0
        assert config.verbose is False

    def test_get_config_preset_by_string(self):
        """Should accept string preset names."""
        config = get_config_preset("local_small")
        assert isinstance(config, PlannerExecutorConfig)
        assert config.planner_max_tokens == 1024


class TestCreateAutoTracer:
    """Test automatic tracer creation."""

    def test_create_local_tracer_no_api_key(self):
        """Should create local tracer when no API key set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove PREDICATE_API_KEY if it exists
            os.environ.pop("PREDICATE_API_KEY", None)
            tracer = _create_auto_tracer(
                planner_model="qwen3:8b",
                executor_model="qwen3:4b",
                run_id="test-run",
            )
            assert isinstance(tracer, Tracer)
            assert tracer.run_id == "test-run"

    def test_create_tracer_generates_run_id(self):
        """Should generate run_id if not provided."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PREDICATE_API_KEY", None)
            tracer = _create_auto_tracer(
                planner_model="qwen3:8b",
                executor_model="qwen3:4b",
            )
            assert isinstance(tracer, Tracer)
            assert tracer.run_id.startswith("run-")


class TestCreatePlannerExecutorAgent:
    """Test the main factory function."""

    def test_create_agent_minimal_local(self):
        """Should create agent with minimal local config."""
        agent = create_planner_executor_agent(
            planner_model="qwen3:8b",
            executor_model="qwen3:4b",
            tracer=None,  # Disable tracer for test
        )
        assert isinstance(agent, PlannerExecutorAgent)

    def test_create_agent_with_explicit_providers(self):
        """Should respect explicit provider settings."""
        agent = create_planner_executor_agent(
            planner_model="qwen3:8b",
            executor_model="qwen3:4b",
            planner_provider="ollama",
            executor_provider="ollama",
            tracer=None,
        )
        assert isinstance(agent, PlannerExecutorAgent)

    def test_create_agent_with_custom_config(self):
        """Should use provided config."""
        custom_config = PlannerExecutorConfig(verbose=True, planner_max_tokens=512)
        agent = create_planner_executor_agent(
            planner_model="qwen3:8b",
            executor_model="qwen3:4b",
            config=custom_config,
            tracer=None,
        )
        assert isinstance(agent, PlannerExecutorAgent)

    def test_create_agent_with_preset(self):
        """Should work with config presets."""
        agent = create_planner_executor_agent(
            planner_model="qwen3:8b",
            executor_model="qwen3:4b",
            config=get_config_preset(ConfigPreset.LOCAL_SMALL_MODEL),
            tracer=None,
        )
        assert isinstance(agent, PlannerExecutorAgent)

    def test_create_agent_with_custom_tracer(self):
        """Should use provided tracer."""
        mock_tracer = MagicMock(spec=Tracer)
        agent = create_planner_executor_agent(
            planner_model="qwen3:8b",
            executor_model="qwen3:4b",
            tracer=mock_tracer,
        )
        assert isinstance(agent, PlannerExecutorAgent)

    @pytest.mark.skipif(not HAS_OPENAI, reason="openai package not installed")
    def test_create_agent_mixed_providers(self):
        """Should support mixed cloud/local configuration."""
        agent = create_planner_executor_agent(
            planner_model="gpt-4o",
            planner_provider="openai",
            executor_model="qwen3:4b",
            executor_provider="ollama",
            openai_api_key="test-key",
            tracer=None,
        )
        assert isinstance(agent, PlannerExecutorAgent)

    def test_create_agent_custom_ollama_base_url(self):
        """Should respect custom Ollama base URL."""
        agent = create_planner_executor_agent(
            planner_model="qwen3:8b",
            executor_model="qwen3:4b",
            ollama_base_url="http://192.168.1.100:11434",
            tracer=None,
        )
        assert isinstance(agent, PlannerExecutorAgent)


class TestAgentFactoryImports:
    """Test that factory is properly exported."""

    def test_import_from_agents_module(self):
        """Factory should be importable from predicate.agents."""
        from predicate.agents import (
            ConfigPreset,
            create_planner_executor_agent,
            get_config_preset,
        )

        assert create_planner_executor_agent is not None
        assert ConfigPreset is not None
        assert get_config_preset is not None
