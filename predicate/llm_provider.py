"""
LLM Provider abstraction layer for Sentience SDK
Enables "Bring Your Own Brain" (BYOB) pattern - plug in any LLM provider
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .llm_provider_utils import get_api_key_from_env, handle_provider_error, require_package
from .llm_response_builder import LLMResponseBuilder


@dataclass
class LLMResponse:
    """Standardized LLM response across all providers"""

    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model_name: str | None = None
    finish_reason: str | None = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implement this interface to add support for any LLM:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Local models (Ollama, LlamaCpp)
    - Azure OpenAI
    - Any other completion API
    """

    def __init__(self, model: str):
        """
        Initialize LLM provider with model name.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-sonnet")
        """
        self._model_name = model

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response from the LLM

        Args:
            system_prompt: System instruction/context
            user_prompt: User query/request
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with content and token usage
        """
        pass

    async def generate_async(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        """
        Async wrapper around generate() for providers without native async support.
        """
        return await asyncio.to_thread(self.generate, system_prompt, user_prompt, **kwargs)

    @abstractmethod
    def supports_json_mode(self) -> bool:
        """
        Whether this provider supports structured JSON output

        Returns:
            True if provider has native JSON mode, False otherwise
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Model identifier (e.g., "gpt-4o", "claude-3-sonnet")

        Returns:
            Model name string
        """
        pass

    def supports_vision(self) -> bool:
        """
        Whether this provider supports image input for vision tasks.

        Override in subclasses that support vision-capable models.

        Returns:
            True if provider supports vision, False otherwise
        """
        return False

    def generate_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response with image input (for vision-capable models).

        This method is used for vision fallback in assertions and visual agents.
        Override in subclasses that support vision-capable models.

        Args:
            system_prompt: System instruction/context
            user_prompt: User query/request
            image_base64: Base64-encoded image (PNG or JPEG)
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with content and token usage

        Raises:
            NotImplementedError: If provider doesn't support vision
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support vision. "
            "Use a vision-capable provider like OpenAIProvider with GPT-4o "
            "or AnthropicProvider with Claude 3."
        )


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation (GPT-4, GPT-4o, GPT-3.5-turbo, etc.)

    Example:
        >>> from predicate.llm_provider import OpenAIProvider
        >>> llm = OpenAIProvider(api_key="sk-...", model="gpt-4o")
        >>> response = llm.generate("You are a helpful assistant", "Hello!")
        >>> print(response.content)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
        organization: str | None = None,
    ):
        """
        Initialize OpenAI provider

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.)
            base_url: Custom API base URL (for compatible APIs)
            organization: OpenAI organization ID
        """
        super().__init__(model)  # Initialize base class with model name

        OpenAI = require_package(
            "openai",
            "openai",
            "OpenAI",
            "pip install openai",
        )

        self.client = OpenAI(api_key=api_key, base_url=base_url, organization=organization)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response using OpenAI API

        Args:
            system_prompt: System instruction
            user_prompt: User query
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            json_mode: Enable JSON response format (requires model support)
            **kwargs: Additional OpenAI API parameters (max_new_tokens is mapped to max_tokens)

        Returns:
            LLMResponse object
        """
        # Handle max_new_tokens -> max_tokens mapping for cross-provider compatibility
        if "max_new_tokens" in kwargs:
            if max_tokens is None:
                max_tokens = kwargs.pop("max_new_tokens")
            else:
                kwargs.pop("max_new_tokens")  # max_tokens takes precedence

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Build API parameters
        api_params = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            api_params["max_tokens"] = max_tokens

        if json_mode and self.supports_json_mode():
            api_params["response_format"] = {"type": "json_object"}

        # Merge additional parameters
        api_params.update(kwargs)

        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(**api_params)
        except Exception as e:
            handle_provider_error(e, "OpenAI", "generate response")

        choice = response.choices[0]
        usage = response.usage

        return LLMResponseBuilder.from_openai_format(
            content=choice.message.content,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            model_name=response.model,
            finish_reason=choice.finish_reason,
        )

    def supports_json_mode(self) -> bool:
        """OpenAI models support JSON mode (GPT-4, GPT-3.5-turbo)"""
        model_lower = self._model_name.lower()
        return any(x in model_lower for x in ["gpt-4", "gpt-3.5"])

    def supports_vision(self) -> bool:
        """GPT-4o, GPT-4-turbo, and GPT-4-vision support vision."""
        model_lower = self._model_name.lower()
        return any(x in model_lower for x in ["gpt-4o", "gpt-4-turbo", "gpt-4-vision"])

    def generate_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response with image input using OpenAI Vision API.

        Args:
            system_prompt: System instruction
            user_prompt: User query
            image_base64: Base64-encoded image (PNG or JPEG)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI API parameters

        Returns:
            LLMResponse object

        Raises:
            NotImplementedError: If model doesn't support vision
        """
        if not self.supports_vision():
            raise NotImplementedError(
                f"Model {self._model_name} does not support vision. "
                "Use gpt-4o, gpt-4-turbo, or gpt-4-vision-preview."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Vision message format with image_url
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        )

        # Build API parameters
        api_params = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            api_params["max_tokens"] = max_tokens

        # Merge additional parameters
        api_params.update(kwargs)

        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(**api_params)
        except Exception as e:
            handle_provider_error(e, "OpenAI", "generate response with image")

        choice = response.choices[0]
        usage = response.usage

        return LLMResponseBuilder.from_openai_format(
            content=choice.message.content,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            model_name=response.model,
            finish_reason=choice.finish_reason,
        )

    @property
    def model_name(self) -> str:
        return self._model_name


class DeepInfraProvider(OpenAIProvider):
    """
    DeepInfra provider via OpenAI-compatible API.

    Uses DeepInfra's OpenAI-compatible endpoint:
    https://api.deepinfra.com/v1/openai

    API token is read from DEEPINFRA_TOKEN or DEEPINFRA_API_KEY if not provided.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        base_url: str = "https://api.deepinfra.com/v1/openai",
    ):
        api_key = get_api_key_from_env(["DEEPINFRA_TOKEN", "DEEPINFRA_API_KEY"], api_key)
        # IMPORTANT: If we pass api_key=None to the OpenAI SDK client, it may
        # implicitly fall back to OPENAI_API_KEY from the environment.
        # That leads to confusing 401s against DeepInfra with an OpenAI key.
        if not api_key:
            raise RuntimeError(
                "DeepInfra API key is missing. Set DEEPINFRA_API_KEY (or DEEPINFRA_TOKEN), "
                "or pass api_key=... to DeepInfraProvider."
            )
        super().__init__(api_key=api_key, model=model, base_url=base_url)

    def supports_vision(self) -> bool:
        """
        DeepInfra hosts many non-OpenAI multimodal models.

        Their OpenAI-compatible API supports the same `image_url` message format:
        `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`

        We therefore allow vision for common DeepInfra model naming patterns.
        """
        model_lower = self._model_name.lower()
        if any(x in model_lower for x in ["vision", "llava", "qvq", "ocr"]):
            return True
        # Preserve OpenAI-style vision detection for GPT models served via DeepInfra.
        return super().supports_vision()


class AnthropicProvider(LLMProvider):
    """
    Anthropic provider implementation (Claude 3 Opus, Sonnet, Haiku, etc.)

    Example:
        >>> from predicate.llm_provider import AnthropicProvider
        >>> llm = AnthropicProvider(api_key="sk-ant-...", model="claude-3-sonnet-20240229")
        >>> response = llm.generate("You are a helpful assistant", "Hello!")
        >>> print(response.content)
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize Anthropic provider

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model name (claude-3-opus, claude-3-sonnet, claude-3-haiku, etc.)
        """
        super().__init__(model)  # Initialize base class with model name

        Anthropic = require_package(
            "anthropic",
            "anthropic",
            "Anthropic",
            "pip install anthropic",
        )

        self.client = Anthropic(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response using Anthropic API

        Args:
            system_prompt: System instruction
            user_prompt: User query
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (required by Anthropic)
            **kwargs: Additional Anthropic API parameters (max_new_tokens is mapped to max_tokens)

        Returns:
            LLMResponse object
        """
        # Handle max_new_tokens -> max_tokens mapping for cross-provider compatibility
        if "max_new_tokens" in kwargs:
            # Use max_new_tokens value if max_tokens is still at default
            if max_tokens == 1024:
                max_tokens = kwargs.pop("max_new_tokens")
            else:
                kwargs.pop("max_new_tokens")  # Explicit max_tokens takes precedence

        # Build API parameters
        api_params = {
            "model": self._model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        if system_prompt:
            api_params["system"] = system_prompt

        # Merge additional parameters
        api_params.update(kwargs)

        # Call Anthropic API
        try:
            response = self.client.messages.create(**api_params)
        except Exception as e:
            handle_provider_error(e, "Anthropic", "generate response")

        content = response.content[0].text if response.content else ""

        return LLMResponseBuilder.from_anthropic_format(
            content=content,
            input_tokens=response.usage.input_tokens if hasattr(response, "usage") else None,
            output_tokens=response.usage.output_tokens if hasattr(response, "usage") else None,
            model_name=response.model,
            stop_reason=response.stop_reason,
        )

    def supports_json_mode(self) -> bool:
        """Anthropic doesn't have native JSON mode (requires prompt engineering)"""
        return False

    def supports_vision(self) -> bool:
        """Claude 3 models (Opus, Sonnet, Haiku) all support vision."""
        model_lower = self._model_name.lower()
        return any(x in model_lower for x in ["claude-3", "claude-3.5"])

    def generate_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response with image input using Anthropic Vision API.

        Args:
            system_prompt: System instruction
            user_prompt: User query
            image_base64: Base64-encoded image (PNG or JPEG)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (required by Anthropic)
            **kwargs: Additional Anthropic API parameters

        Returns:
            LLMResponse object

        Raises:
            NotImplementedError: If model doesn't support vision
        """
        if not self.supports_vision():
            raise NotImplementedError(
                f"Model {self._model_name} does not support vision. "
                "Use Claude 3 models (claude-3-opus, claude-3-sonnet, claude-3-haiku)."
            )

        # Anthropic vision message format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            }
        ]

        # Build API parameters
        api_params = {
            "model": self._model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system_prompt:
            api_params["system"] = system_prompt

        # Merge additional parameters
        api_params.update(kwargs)

        # Call Anthropic API
        try:
            response = self.client.messages.create(**api_params)
        except Exception as e:
            handle_provider_error(e, "Anthropic", "generate response with image")

        content = response.content[0].text if response.content else ""

        return LLMResponseBuilder.from_anthropic_format(
            content=content,
            input_tokens=response.usage.input_tokens if hasattr(response, "usage") else None,
            output_tokens=response.usage.output_tokens if hasattr(response, "usage") else None,
            model_name=response.model,
            stop_reason=response.stop_reason,
        )

    @property
    def model_name(self) -> str:
        return self._model_name


class GLMProvider(LLMProvider):
    """
    Zhipu AI GLM provider implementation (GLM-4, GLM-4-Plus, etc.)

    Requirements:
        pip install zhipuai

    Example:
        >>> from predicate.llm_provider import GLMProvider
        >>> llm = GLMProvider(api_key="your-api-key", model="glm-4-plus")
        >>> response = llm.generate("You are a helpful assistant", "Hello!")
        >>> print(response.content)
    """

    def __init__(self, api_key: str | None = None, model: str = "glm-4-plus"):
        """
        Initialize GLM provider

        Args:
            api_key: Zhipu AI API key (or set GLM_API_KEY env var)
            model: Model name (glm-4-plus, glm-4, glm-4-air, glm-4-flash, etc.)
        """
        super().__init__(model)  # Initialize base class with model name

        ZhipuAI = require_package(
            "zhipuai",
            "zhipuai",
            "ZhipuAI",
            "pip install zhipuai",
        )

        self.client = ZhipuAI(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response using GLM API

        Args:
            system_prompt: System instruction
            user_prompt: User query
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional GLM API parameters

        Returns:
            LLMResponse object
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Build API parameters
        api_params = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            api_params["max_tokens"] = max_tokens

        # Merge additional parameters
        api_params.update(kwargs)

        # Call GLM API
        try:
            response = self.client.chat.completions.create(**api_params)
        except Exception as e:
            handle_provider_error(e, "GLM", "generate response")

        choice = response.choices[0]
        usage = response.usage

        return LLMResponseBuilder.from_openai_format(
            content=choice.message.content,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            model_name=response.model,
            finish_reason=choice.finish_reason,
        )

    def supports_json_mode(self) -> bool:
        """GLM-4 models support JSON mode"""
        return "glm-4" in self._model_name.lower()

    @property
    def model_name(self) -> str:
        return self._model_name


class GeminiProvider(LLMProvider):
    """
    Google Gemini provider implementation (Gemini 2.0, Gemini 1.5 Pro, etc.)

    Requirements:
        pip install google-generativeai

    Example:
        >>> from predicate.llm_provider import GeminiProvider
        >>> llm = GeminiProvider(api_key="your-api-key", model="gemini-2.0-flash-exp")
        >>> response = llm.generate("You are a helpful assistant", "Hello!")
        >>> print(response.content)
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini provider

        Args:
            api_key: Google API key (or set GEMINI_API_KEY or GOOGLE_API_KEY env var)
            model: Model name (gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash, etc.)
        """
        super().__init__(model)  # Initialize base class with model name

        genai = require_package(
            "google-generativeai",
            "google.generativeai",
            install_command="pip install google-generativeai",
        )

        # Configure API key (check parameter first, then environment variables)
        api_key = get_api_key_from_env(["GEMINI_API_KEY", "GOOGLE_API_KEY"], api_key)
        if api_key:
            genai.configure(api_key=api_key)

        self.genai = genai
        self.model = genai.GenerativeModel(model)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response using Gemini API

        Args:
            system_prompt: System instruction
            user_prompt: User query
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very creative)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Gemini API parameters

        Returns:
            LLMResponse object
        """
        # Combine system and user prompts (Gemini doesn't have separate system role in all versions)
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

        # Build generation config
        generation_config = {
            "temperature": temperature,
        }

        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        # Merge additional parameters
        generation_config.update(kwargs)

        # Call Gemini API
        try:
            response = self.model.generate_content(full_prompt, generation_config=generation_config)
        except Exception as e:
            handle_provider_error(e, "Gemini", "generate response")

        # Extract content
        content = response.text if response.text else ""

        # Token usage (if available)
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count

        return LLMResponseBuilder.from_gemini_format(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model_name=self._model_name,
        )

    def supports_json_mode(self) -> bool:
        """Gemini 1.5+ models support JSON mode via response_mime_type"""
        model_lower = self._model_name.lower()
        return any(x in model_lower for x in ["gemini-1.5", "gemini-2.0"])

    @property
    def model_name(self) -> str:
        return self._model_name


class LocalLLMProvider(LLMProvider):
    """
    Local LLM provider using HuggingFace Transformers
    Supports Qwen, Llama, Gemma, Phi, and other instruction-tuned models

    Example:
        >>> from predicate.llm_provider import LocalLLMProvider
        >>> llm = LocalLLMProvider(model_name="Qwen/Qwen2.5-3B-Instruct")
        >>> response = llm.generate("You are helpful", "Hello!")
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str = "auto",
    ):
        """
        Initialize local LLM using HuggingFace Transformers

        Args:
            model_name: HuggingFace model identifier
                Popular options:
                - "Qwen/Qwen2.5-3B-Instruct" (recommended, 3B params)
                - "meta-llama/Llama-3.2-3B-Instruct" (3B params)
                - "google/gemma-2-2b-it" (2B params)
                - "microsoft/Phi-3-mini-4k-instruct" (3.8B params)
            device: Device to run on ("cpu", "cuda", "mps", "auto")
            load_in_4bit: Use 4-bit quantization (saves 75% memory)
            load_in_8bit: Use 8-bit quantization (saves 50% memory)
            torch_dtype: Data type ("auto", "float16", "bfloat16", "float32")
        """
        super().__init__(model_name)  # Initialize base class with model name

        # Import required packages with consistent error handling.
        # These are optional dependencies, so keep them out of module import-time.
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers and torch required for local LLM. "
                "Install with: pip install transformers torch"
            ) from exc

        self._torch = torch

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        device = (device or "auto").strip().lower()

        # Determine torch dtype
        if torch_dtype == "auto":
            dtype = torch.float16 if device not in {"cpu"} else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)

        # device_map is a Transformers concept (not a literal "cpu/mps/cuda" device string).
        # - "auto" enables Accelerate device mapping.
        # - Otherwise, we load normally and then move the model to the requested device.
        device_map: str | None = "auto" if device == "auto" else None

        def _load(*, device_map_override: str | None) -> Any:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=dtype if quantization_config is None else None,
                device_map=device_map_override,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        try:
            self.model = _load(device_map_override=device_map)
        except KeyError as e:
            # Some envs / accelerate versions can crash on auto mapping (e.g. KeyError: 'cpu').
            # Keep demo ergonomics: default stays "auto", but we gracefully fall back.
            if device == "auto" and ("cpu" in str(e).lower()):
                device = "cpu"
                dtype = torch.float32
                self.model = _load(device_map_override=None)
            else:
                raise

        # If we didn't use device_map, move model explicitly (only safe for non-quantized loads).
        if device_map is None and quantization_config is None and device in {"cpu", "cuda", "mps"}:
            self.model = self.model.to(device)
        self.model.eval()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response using local model

        Args:
            system_prompt: System instruction
            user_prompt: User query
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy, higher = more random)
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        torch = self._torch

        # Auto-determine sampling based on temperature
        do_sample = temperature > 0

        # Format prompt using model's chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Use model's native chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback formatting
            formatted_prompt = ""
            if system_prompt:
                formatted_prompt += f"System: {system_prompt}\n\n"
            formatted_prompt += f"User: {user_prompt}\n\nAssistant:"

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True).to(
            self.model.device
        )

        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode only the new tokens
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return LLMResponseBuilder.from_local_format(
            content=response_text,
            prompt_tokens=input_length,
            completion_tokens=len(generated_tokens),
            model_name=self._model_name,
        )

    def supports_json_mode(self) -> bool:
        """Local models typically need prompt engineering for JSON"""
        return False

    @property
    def model_name(self) -> str:
        return self._model_name


class LocalVisionLLMProvider(LLMProvider):
    """
    Local vision-language LLM provider using HuggingFace Transformers.

    Intended for models like:
    - Qwen/Qwen3-VL-8B-Instruct

    Notes on Mac (MPS) + quantization:
    - Transformers BitsAndBytes (4-bit/8-bit) typically requires CUDA and does NOT work on MPS.
    - If you want quantized local vision on Apple Silicon, you may prefer MLX-based stacks
      (e.g., mlx-vlm) or llama.cpp/gguf pipelines.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
    ):
        super().__init__(model_name)

        # Import required packages with consistent error handling
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoProcessor  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for LocalVisionLLMProvider. "
                "Install with: pip install transformers torch"
            ) from exc

        self._torch = torch

        # Resolve device
        if device == "auto":
            if (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        if device == "mps" and (load_in_4bit or load_in_8bit):
            raise ValueError(
                "Quantized (4-bit/8-bit) Transformers loading is typically not supported on Apple MPS. "
                "Set load_in_4bit/load_in_8bit to False for MPS, or use a different local runtime "
                "(e.g., MLX/llama.cpp) for quantized vision models."
            )

        # Determine torch dtype
        if torch_dtype == "auto":
            dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        # Load model (prefer vision2seq; fall back with guidance)
        try:
            import importlib

            transformers = importlib.import_module("transformers")
            AutoModelForVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
            if AutoModelForVision2Seq is None:
                raise AttributeError("transformers.AutoModelForVision2Seq is not available")

            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            # Some transformers versions/models don't expose AutoModelForVision2Seq.
            # We fail loudly with a helpful message rather than silently doing text-only.
            raise ImportError(
                "Failed to load a vision-capable Transformers model. "
                "Try upgrading transformers (vision models often require newer versions), "
                "or use a model class supported by your installed transformers build."
            ) from exc

        # Move to device
        self.device = device
        self.model.to(device)

        self.model.eval()

    def supports_json_mode(self) -> bool:
        return False

    def supports_vision(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs,
    ) -> LLMResponse:
        """
        Text-only generation (no image). Provided for interface completeness.
        """
        torch = self._torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if hasattr(self.processor, "apply_chat_template"):
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (system_prompt + "\n\n" if system_prompt else "") + user_prompt

        inputs = self.processor(text=[prompt], return_tensors="pt")
        inputs = {
            k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }

        do_sample = temperature > 0
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                **kwargs,
            )

        # Decode
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated = outputs[0][input_len:]
        if hasattr(self.processor, "batch_decode"):
            text = self.processor.batch_decode([generated], skip_special_tokens=True)[0].strip()
        else:
            text = str(generated)

        return LLMResponseBuilder.from_local_format(
            content=text,
            prompt_tokens=int(input_len) if input_len else None,
            completion_tokens=int(generated.shape[0]) if hasattr(generated, "shape") else None,
            model_name=self._model_name,
        )

    def generate_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        **kwargs,
    ) -> LLMResponse:
        """
        Vision generation using an image + prompt.

        This is used by vision fallback in assertions and by visual agents.
        """
        torch = self._torch

        # Lazy import PIL to avoid adding a hard dependency for text-only users.
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for LocalVisionLLMProvider image input. Install with: pip install pillow"
            ) from exc

        import base64
        import io

        img_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Prefer processor chat template if available (needed by many VL models).
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }
        )

        if hasattr(self.processor, "apply_chat_template"):
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            raise NotImplementedError(
                "This local vision model/processor does not expose apply_chat_template(). "
                "Install/upgrade to a Transformers version that supports your model's chat template."
            )

        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {
            k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }

        do_sample = temperature > 0
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                **kwargs,
            )

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated = outputs[0][input_len:]

        if hasattr(self.processor, "batch_decode"):
            text = self.processor.batch_decode([generated], skip_special_tokens=True)[0].strip()
        elif hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "decode"):
            text = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        else:
            text = ""

        return LLMResponseBuilder.from_local_format(
            content=text,
            prompt_tokens=int(input_len) if input_len else None,
            completion_tokens=int(generated.shape[0]) if hasattr(generated, "shape") else None,
            model_name=self._model_name,
        )


class MLXVLMProvider(LLMProvider):
    """
    Local vision-language provider using MLX-VLM (Apple Silicon optimized).

    Recommended for running *quantized* vision models on Mac (M1/M2/M3/M4), e.g.:
    - mlx-community/Qwen3-VL-8B-Instruct-3bit

    Optional dependencies:
    - mlx-vlm
    - pillow

    Notes:
    - MLX-VLM APIs can vary across versions; this provider tries a couple common call shapes.
    - For best results, use an MLX-converted model repo under `mlx-community/`.
    """

    def __init__(
        self,
        model: str = "mlx-community/Qwen3-VL-8B-Instruct-3bit",
        *,
        default_max_tokens: int = 256,
        default_temperature: float = 0.0,
        **kwargs,
    ):
        super().__init__(model)
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self._default_kwargs = dict(kwargs)

        # Lazy imports to keep base SDK light.
        try:
            import importlib

            self._mlx_vlm = importlib.import_module("mlx_vlm")
        except ImportError as exc:
            raise ImportError(
                "mlx-vlm is required for MLXVLMProvider. Install with: pip install mlx-vlm"
            ) from exc

        try:
            from PIL import Image  # type: ignore[import-not-found]

            self._PIL_Image = Image
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for MLXVLMProvider. Install with: pip install pillow"
            ) from exc

        # Some mlx_vlm versions expose load(model_id) -> (model, processor)
        self._model = None
        self._processor = None
        load_fn = getattr(self._mlx_vlm, "load", None)
        if callable(load_fn):
            try:
                loaded = load_fn(model)
                if isinstance(loaded, tuple) and len(loaded) >= 2:
                    self._model, self._processor = loaded[0], loaded[1]
            except Exception:
                # Keep it lazy; we'll try loading on demand during generate_with_image().
                self._model, self._processor = None, None

    def supports_json_mode(self) -> bool:
        return False

    def supports_vision(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        """
        Text-only generation is not a primary MLX-VLM use-case. We attempt it if the installed
        mlx_vlm exposes a compatible `generate()` signature; otherwise, raise a clear error.
        """
        generate_fn = getattr(self._mlx_vlm, "generate", None)
        if not callable(generate_fn):
            raise NotImplementedError("mlx_vlm.generate is not available in your mlx-vlm install.")

        prompt = (system_prompt + "\n\n" if system_prompt else "") + user_prompt
        max_tokens = kwargs.pop("max_tokens", self._default_max_tokens)
        temperature = kwargs.pop("temperature", self._default_temperature)
        merged_kwargs = {**self._default_kwargs, **kwargs}

        try:
            out = generate_fn(
                self._model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **merged_kwargs,
            )
        except TypeError as exc:
            if self._model is None or self._processor is None:
                raise NotImplementedError(
                    "Text-only generation is not supported by this mlx-vlm version without a loaded model."
                ) from exc
            out = generate_fn(
                self._model,
                self._processor,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **merged_kwargs,
            )

        text = getattr(out, "text", None) or getattr(out, "output", None) or str(out)
        return LLMResponseBuilder.from_local_format(
            content=str(text).strip(),
            prompt_tokens=None,
            completion_tokens=None,
            model_name=self._model_name,
        )

    def generate_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str,
        **kwargs,
    ) -> LLMResponse:
        import base64
        import io

        generate_fn = getattr(self._mlx_vlm, "generate", None)
        if not callable(generate_fn):
            raise NotImplementedError("mlx_vlm.generate is not available in your mlx-vlm install.")

        img_bytes = base64.b64decode(image_base64)
        image = self._PIL_Image.open(io.BytesIO(img_bytes)).convert("RGB")

        prompt = (system_prompt + "\n\n" if system_prompt else "") + user_prompt
        max_tokens = kwargs.pop("max_tokens", self._default_max_tokens)
        temperature = kwargs.pop("temperature", self._default_temperature)
        merged_kwargs = {**self._default_kwargs, **kwargs}

        # Try a couple common MLX-VLM call shapes.
        try:
            # 1) generate(model_id, image=..., prompt=...)
            out = generate_fn(
                self._model_name,
                image=image,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **merged_kwargs,
            )
        except TypeError as exc:
            # 2) generate(model, processor, prompt, image, ...)
            if self._model is None or self._processor is None:
                load_fn = getattr(self._mlx_vlm, "load", None)
                if callable(load_fn):
                    loaded = load_fn(self._model_name)
                    if isinstance(loaded, tuple) and len(loaded) >= 2:
                        self._model, self._processor = loaded[0], loaded[1]
            if self._model is None or self._processor is None:
                raise NotImplementedError(
                    "Unable to call mlx_vlm.generate with your installed mlx-vlm version. "
                    "Please upgrade mlx-vlm or use LocalVisionLLMProvider (Transformers backend)."
                ) from exc
            out = generate_fn(
                self._model,
                self._processor,
                prompt,
                image,
                max_tokens=max_tokens,
                temperature=temperature,
                **merged_kwargs,
            )

        text = getattr(out, "text", None) or getattr(out, "output", None) or str(out)
        return LLMResponseBuilder.from_local_format(
            content=str(text).strip(),
            prompt_tokens=None,
            completion_tokens=None,
            model_name=self._model_name,
        )
