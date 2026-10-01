"""Abstract LLM Provider interface and response contracts for PromptPilot.

Allows the application to decouple prompt analysis, generation, execution,
and evaluation from concrete provider SDKs (such as Google Gemini, OpenAI, or Ollama).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# --- Custom Exception Hierarchy ---
class LLMError(Exception):
    """Base exception for all LLM provider interactions."""

    def __init__(self, message: str, provider: str = "unknown", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.details = details or {}


class LLMAuthError(LLMError):
    """Raised when authentication fails (missing, expired, or invalid API key)."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when provider rate limits, quota limits, or TPM/RPM are exceeded."""
    pass


class LLMInvalidModelError(LLMError):
    """Raised when the specified model identifier is unsupported or deprecated."""
    pass


class LLMStructuredOutputError(LLMError):
    """Raised when provider fails to return valid schema-compliant structured data."""
    pass


# --- Response Models ---
@dataclass
class LLMResponse:
    """Standardized result returned by all LLM text and structured operations."""
    content: str
    model_name: str
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- Base Provider Contract ---
class BaseLLMProvider(ABC):
    """Protocol that every LLM provider backend must implement."""

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate unstructured textual completion.

        Args:
            prompt: Main prompt content.
            system_instruction: Optional system instruction guiding model persona/behavior.
            model: Specific model identifier (falls back to provider default).
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_output_tokens: Maximum tokens in response.

        Returns:
            LLMResponse containing response content and execution metrics.
        """
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> tuple[T, LLMResponse]:
        """Generate structured response validated against a Pydantic schema.

        Args:
            prompt: Main prompt or instructions.
            response_schema: The Pydantic model class to validate output against.
            system_instruction: Optional system level guidelines.
            model: Specific model identifier.
            temperature: Sampling temperature (default low for reliable schemas).

        Returns:
            Tuple of (parsed_pydantic_instance, raw_llm_response)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider credentials and endpoints are configured and ready."""
        pass
