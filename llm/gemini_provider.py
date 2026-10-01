"""Google Gemini concrete provider implementation using the official google-genai SDK.

Handles text and structured schema completions with latency tracking,
token count extraction, and categorized error mapping.
"""

import json
import re
import time
from typing import Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    APIError = Exception  # type: ignore
    GENAI_AVAILABLE = False

from config.settings import Settings, get_settings
from llm.base import (
    BaseLLMProvider,
    LLMAuthError,
    LLMError,
    LLMInvalidModelError,
    LLMRateLimitError,
    LLMResponse,
    LLMStructuredOutputError,
)
from utils.logging import get_logger

T = TypeVar("T", bound=BaseModel)
logger = get_logger("gemini_provider")


class GeminiProvider(BaseLLMProvider):
    """Concrete provider for Google Gemini models via google-genai SDK."""

    def __init__(self, settings: Optional[Settings] = None, api_key: Optional[str] = None):
        self.settings = settings or get_settings()
        self.api_key = api_key or self.settings.effective_api_key
        self._client: Optional[genai.Client] = None

        if not GENAI_AVAILABLE:
            logger.warning("google-genai SDK is not installed. Gemini provider will be unavailable.")

    def _get_client(self) -> genai.Client:
        """Instantiate or return initialized Gemini client."""
        if not GENAI_AVAILABLE:
            raise LLMAuthError(
                message="google-genai SDK is not installed. Install via: pip install google-genai",
                provider="gemini",
            )
        if not self.api_key:
            raise LLMAuthError(
                message="Gemini API Key is not configured. Please set GEMINI_API_KEY in .env or Streamlit secrets.",
                provider="gemini",
            )
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        """Check if Gemini SDK is installed and non-empty API key is configured."""
        return GENAI_AVAILABLE and bool(self.api_key and not self.api_key.startswith("your_"))

    def _map_api_error(self, err: Exception, model_name: str) -> LLMError:
        """Map generic or SDK API errors into categorized PromptPilot exceptions."""
        error_msg = str(err)
        lower_msg = error_msg.lower()

        # Scrub potential API keys from the error message for safety
        if self.api_key and self.api_key in error_msg:
            error_msg = error_msg.replace(self.api_key, "[API_KEY_REDACTED]")
        safe_msg = re.sub(r"(AIza[0-9A-Za-z-_]{20,}|AQ\.[0-9A-Za-z-_]{20,}|sk-[a-zA-Z0-9]{20,})", "[API_KEY_REDACTED]", error_msg)

        if "401" in safe_msg or "403" in safe_msg or "api_key" in lower_msg or "unauthenticated" in lower_msg:
            return LLMAuthError(f"Authentication failed: {safe_msg}", provider="gemini")
        if "429" in safe_msg or "quota" in lower_msg or "resource_exhausted" in lower_msg:
            # Extract retry delay if available
            delay_match = re.search(r"retry in (\d+(\.\d+)?)s", safe_msg) or re.search(r"retryDelay': '(\d+)s'", safe_msg)
            wait_text = f" (retry suggested in {delay_match.group(1)}s)" if delay_match else ""
            return LLMRateLimitError(
                f"Rate limit or daily quota reached for model '{model_name}' on Google AI Studio Free Tier{wait_text}. "
                f"Tips: 1) Switch to 'gemini-3.5-flash-lite' in Settings, 2) Decrease variants to 1 or 2, "
                f"or 3) Toggle 'Offline Demo Mode' in the sidebar for unlimited instant testing.",
                provider="gemini",
            )
        if "404" in safe_msg or "not found" in lower_msg or "model" in lower_msg:
            return LLMInvalidModelError(f"Model '{model_name}' not found or unsupported: {safe_msg}", provider="gemini")

        return LLMError(f"Gemini API interaction failed: {safe_msg}", provider="gemini")

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate textual completion from Gemini."""
        client = self._get_client()
        active_model = model or self.settings.llm_model

        config_params = {"temperature": temperature}
        if system_instruction:
            config_params["system_instruction"] = system_instruction
        if max_output_tokens:
            config_params["max_output_tokens"] = max_output_tokens

        config = types.GenerateContentConfig(**config_params)

        start_time = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=active_model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            logger.error("Error during Gemini text generation: %s", type(e).__name__)
            raise self._map_api_error(e, active_model) from None

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Safe extraction of response text
        content = response.text or ""

        # Safe extraction of token usage metrics
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        total_tokens = getattr(usage, "total_token_count", None) if usage else None

        finish_reason = None
        if hasattr(response, "candidates") and response.candidates:
            finish_reason = str(getattr(response.candidates[0], "finish_reason", "STOP"))

        return LLMResponse(
            content=content,
            model_name=active_model,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
        )

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> tuple[T, LLMResponse]:
        """Generate structured response validated against a Pydantic schema."""
        client = self._get_client()
        active_model = model or self.settings.llm_model

        config_params = {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_params)

        start_time = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=active_model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            logger.error("Error during Gemini structured generation: %s", type(e).__name__)
            raise self._map_api_error(e, active_model) from None

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        raw_text = response.text or "{}"

        # Clean markdown codeblocks if provider wraps JSON (```json ... ```)
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        try:
            parsed_object = response_schema.model_validate_json(cleaned_text)
        except (ValidationError, json.JSONDecodeError) as val_err:
            logger.warning("Structured response validation failed: %s", val_err)
            raise LLMStructuredOutputError(
                message=f"Failed to parse model response into schema {response_schema.__name__}: {val_err}",
                provider="gemini",
                details={"raw_response": raw_text[:500]},
            ) from val_err

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        total_tokens = getattr(usage, "total_token_count", None) if usage else None

        llm_response = LLMResponse(
            content=raw_text,
            model_name=active_model,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return parsed_object, llm_response
