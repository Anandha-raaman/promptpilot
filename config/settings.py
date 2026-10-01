"""Configuration management for PromptPilot.

Loads configuration from environment variables (.env) with strict validation,
safe defaults, and credential protection.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable loading and validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM Provider Configuration
    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
        description="Google Gemini API key",
        repr=False,
    )
    llm_api_key_fallback: Optional[str] = Field(
        default=None,
        validation_alias="LLM_API_KEY",
        description="Fallback API key variable",
        repr=False,
    )
    llm_model: str = Field(
        default="gemini-3.5-flash",
        validation_alias="LLM_MODEL",
        description="Default Gemini model identifier",
    )

    # Specialized Model Roles (fall back to llm_model if unspecified)
    prompt_analyzer_model: Optional[str] = Field(
        default=None,
        validation_alias="PROMPT_ANALYZER_MODEL",
        description="Specialized model for prompt health analysis",
    )
    prompt_generator_model: Optional[str] = Field(
        default=None,
        validation_alias="PROMPT_GENERATOR_MODEL",
        description="Specialized model for prompt variant generation",
    )
    response_generator_model: Optional[str] = Field(
        default=None,
        validation_alias="RESPONSE_GENERATOR_MODEL",
        description="Specialized model for executing prompt variants",
    )
    evaluator_model: Optional[str] = Field(
        default=None,
        validation_alias="EVALUATOR_MODEL",
        description="Specialized model for response evaluation",
    )

    # Database Configuration
    database_path: str = Field(
        default="data/promptpilot.db",
        validation_alias="DATABASE_PATH",
        description="Path to local SQLite database file",
    )

    # Cost & Abuse Safety Limits
    max_input_chars: int = Field(
        default=10000,
        validation_alias="MAX_INPUT_CHARS",
        description="Maximum characters allowed in user input context",
    )
    max_prompt_chars: int = Field(
        default=3000,
        validation_alias="MAX_PROMPT_CHARS",
        description="Maximum characters allowed in prompt text",
    )
    max_variants_per_run: int = Field(
        default=5,
        validation_alias="MAX_VARIANTS_PER_RUN",
        description="Maximum number of prompt variants allowed per run",
    )
    max_batch_test_cases: int = Field(
        default=10,
        validation_alias="MAX_BATCH_TEST_CASES",
        description="Maximum test cases allowed in a single batch evaluation",
    )
    rate_limit_cooldown_seconds: float = Field(
        default=2.0,
        validation_alias="RATE_LIMIT_COOLDOWN_SECONDS",
        description="Minimum seconds between automated API calls to avoid rate exhaustion",
    )

    # Diagnostics
    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    @field_validator("max_input_chars", "max_prompt_chars", "max_variants_per_run", "max_batch_test_cases")
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Configuration limit must be greater than zero.")
        return value

    @property
    def effective_api_key(self) -> Optional[str]:
        """Resolve active API key prioritizing GEMINI_API_KEY over fallback LLM_API_KEY, supporting env and Streamlit secrets."""
        key = self.gemini_api_key
        if not key or key.strip().startswith("your_"):
            key = _get_streamlit_secret("GEMINI_API_KEY")
        if not key or key.strip().startswith("your_"):
            key = self.llm_api_key_fallback
        if not key or key.strip().startswith("your_"):
            key = _get_streamlit_secret("LLM_API_KEY")

        if not key or key.strip().startswith("your_"):
            return None
        return key.strip()

    @property
    def has_valid_api_key(self) -> bool:
        """Check if a non-placeholder API key is configured."""
        return self.effective_api_key is not None and len(self.effective_api_key) > 5

    @property
    def analyzer_model(self) -> str:
        return self.prompt_analyzer_model or self.llm_model

    @property
    def generator_model(self) -> str:
        return self.prompt_generator_model or self.llm_model

    @property
    def response_model(self) -> str:
        return self.response_generator_model or self.llm_model

    @property
    def eval_model(self) -> str:
        return self.evaluator_model or self.llm_model

    def get_resolved_db_path(self, base_dir: Optional[Path] = None) -> Path:
        """Resolve database path and ensure its parent directory exists."""
        raw_path = Path(self.database_path)
        if raw_path.is_absolute():
            resolved = raw_path
        else:
            root = base_dir or Path(__file__).resolve().parent.parent
            resolved = root / raw_path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def __repr__(self) -> str:
        """Safe string representation without exposing secrets or partial keys."""
        return (
            f"Settings(model='{self.llm_model}', "
            f"api_key_configured={self.has_valid_api_key}, "
            f"db='{self.database_path}')"
        )

    def __str__(self) -> str:
        return self.__repr__()


def _get_streamlit_secret(key: str) -> Optional[str]:
    """Retrieve secret from Streamlit secrets dictionary if running within Streamlit."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if isinstance(val, str) and val.strip() and not val.strip().startswith("your_"):
                return val.strip()
    except Exception:
        pass
    return None


# Global singleton settings loader
def get_settings() -> Settings:
    """Return an initialized Settings instance, checking Streamlit secrets if env is unset."""
    settings = Settings()
    if not settings.gemini_api_key or settings.gemini_api_key.startswith("your_"):
        secret_key = _get_streamlit_secret("GEMINI_API_KEY") or _get_streamlit_secret("LLM_API_KEY")
        if secret_key:
            settings.gemini_api_key = secret_key
    return settings
