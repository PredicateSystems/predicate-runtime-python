"""Tests for OllamaProvider."""

import pytest

from predicate.llm_provider import OllamaProvider, LLMProvider


class TestOllamaProvider:
    """Test suite for OllamaProvider."""

    def test_ollama_provider_is_subclass_of_llm_provider(self):
        """OllamaProvider should inherit from LLMProvider (not OpenAIProvider)."""
        assert issubclass(OllamaProvider, LLMProvider)

    def test_ollama_provider_default_base_url(self):
        """OllamaProvider should use default localhost:11434 base URL."""
        provider = OllamaProvider(model="qwen3:8b")
        # The internal base URL should be set correctly
        assert provider._ollama_base_url == "http://localhost:11434"
        assert provider.ollama_base_url == "http://localhost:11434"

    def test_ollama_provider_custom_base_url(self):
        """OllamaProvider should accept custom base URL."""
        provider = OllamaProvider(model="llama3:8b", base_url="http://192.168.1.100:11434")
        assert provider._ollama_base_url == "http://192.168.1.100:11434"
        assert provider.ollama_base_url == "http://192.168.1.100:11434"

    def test_ollama_provider_strips_trailing_slash(self):
        """OllamaProvider should strip trailing slash from base URL."""
        provider = OllamaProvider(model="mistral:7b", base_url="http://localhost:11434/")
        # The trailing slash should be stripped
        assert provider._ollama_base_url == "http://localhost:11434"
        # The API base URL should be properly formed
        assert provider._api_base_url == "http://localhost:11434/v1"

    def test_ollama_provider_is_local_property(self):
        """OllamaProvider.is_local should return True."""
        provider = OllamaProvider(model="qwen3:4b")
        assert provider.is_local is True

    def test_ollama_provider_name_property(self):
        """OllamaProvider.provider_name should return 'ollama'."""
        provider = OllamaProvider(model="phi3:mini")
        assert provider.provider_name == "ollama"

    def test_ollama_provider_model_name(self):
        """OllamaProvider should correctly report model name."""
        provider = OllamaProvider(model="qwen3:8b")
        assert provider.model_name == "qwen3:8b"

    def test_ollama_provider_supports_json_mode_false(self):
        """OllamaProvider should return False for supports_json_mode (conservative default)."""
        provider = OllamaProvider(model="qwen3:8b")
        assert provider.supports_json_mode() is False

    def test_ollama_provider_supports_vision_llava(self):
        """OllamaProvider should detect vision support for llava models."""
        provider = OllamaProvider(model="llava:7b")
        assert provider.supports_vision() is True

    def test_ollama_provider_supports_vision_bakllava(self):
        """OllamaProvider should detect vision support for bakllava models."""
        provider = OllamaProvider(model="bakllava:latest")
        assert provider.supports_vision() is True

    def test_ollama_provider_supports_vision_moondream(self):
        """OllamaProvider should detect vision support for moondream models."""
        provider = OllamaProvider(model="moondream:1.8b")
        assert provider.supports_vision() is True

    def test_ollama_provider_no_vision_for_text_models(self):
        """OllamaProvider should return False for non-vision models."""
        provider = OllamaProvider(model="qwen3:8b")
        assert provider.supports_vision() is False

        provider = OllamaProvider(model="llama3:8b")
        assert provider.supports_vision() is False

        provider = OllamaProvider(model="mistral:7b")
        assert provider.supports_vision() is False


class TestOllamaProviderImport:
    """Test that OllamaProvider is properly exported."""

    def test_import_from_llm_provider(self):
        """OllamaProvider should be importable from predicate.llm_provider."""
        from predicate.llm_provider import OllamaProvider

        assert OllamaProvider is not None

    def test_import_from_predicate(self):
        """OllamaProvider should be importable from predicate package root."""
        from predicate import OllamaProvider

        assert OllamaProvider is not None
