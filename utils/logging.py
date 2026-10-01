"""Sanitized structured logging utilities for PromptPilot.

Ensures that API keys, authorization tokens, and excessively large user
payloads are never printed to console logs or log files.
"""

import logging
import re
import sys
from typing import Optional


# Regex pattern to match potential API keys or tokens (AIza..., AQ...., sk-..., etc.)
API_KEY_REGEX = re.compile(
    r"(AIza[0-9A-Za-z-_]{20,}|AQ\.[0-9A-Za-z-_]{20,}|sk-[a-zA-Z0-9]{20,}|bearer\s+[a-zA-Z0-9_\-\.]{20,}|key=[a-zA-Z0-9_\-]{16,})",
    re.IGNORECASE,
)


def redact_secrets(text: str, custom_key: Optional[str] = None) -> str:
    """Redact sensitive API keys, tokens, and authorization headers from text without truncating."""
    if not text:
        return ""
    if custom_key and len(custom_key) > 5 and custom_key in text:
        text = text.replace(custom_key, "[REDACTED_SECRET]")
    return API_KEY_REGEX.sub("[REDACTED_SECRET]", text)


def format_safe_user_error(
    error: Optional[Exception],
    default_message: str = "Unable to complete the AI request. Please check your API configuration and try again.",
) -> str:
    """Format a safe, clean user-facing error message without internal paths, connection strings, or stack traces."""
    if error is None:
        return default_message

    err_str = redact_secrets(str(error))
    lower = err_str.lower()

    # 1. API key / authentication / quota issues
    if any(k in lower for k in ["api key", "unauthenticated", "401", "403", "quota", "resource_exhausted", "not configured", "invalid api key"]):
        return "Unable to complete the AI request. Please check your API configuration and try again."

    # 2. Rate limit issues
    if any(k in lower for k in ["429", "rate limit", "too many requests"]):
        return "API rate limit reached. Please wait a moment and try again."

    # 3. Network / connection issues
    if any(k in lower for k in ["connection error", "timeout", "timed out", "unreachable", "name resolution", "econnrefused"]):
        return "Network connection issue. Please verify your internet connection and try again."

    # 4. Filesystem paths, stack traces, or internal module references
    if re.search(r"[a-zA-Z]:\\[^\s]+|/(?:home|usr|var|tmp|Users|app)/[^\s]+|traceback|line \d+|file \"", err_str, re.IGNORECASE):
        return default_message

    # 5. Database internals
    if any(k in lower for k in ["sqlite3", "syntax error", "operationalerror", "database is locked", "table ", "column "]):
        return "A database operation could not be completed. Please try again."

    # 6. If the error message is clean, short, and safe (e.g. custom user validation error)
    if len(err_str) <= 100 and not any(c in err_str for c in ["{", "}", "\\", "/", "<", ">"]):
        return err_str

    return default_message



class SensitiveDataSanitizingFilter(logging.Filter):
    """Logging filter that redacts API keys and sensitive tokens from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.sanitize(record.msg)
        if record.args:
            sanitized_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    sanitized_args.append(self.sanitize(arg))
                else:
                    sanitized_args.append(arg)
            record.args = tuple(sanitized_args)
        return True

    @staticmethod
    def sanitize(text: str, max_length: Optional[int] = 1500) -> str:
        """Redact sensitive patterns and optionally truncate overly long payloads."""
        redacted = redact_secrets(text)
        if max_length and len(redacted) > max_length:
            return redacted[:max_length] + "... [TRUNCATED_FOR_SAFETY]"
        return redacted


_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with the sanitizing filter and safe format."""
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger("promptpilot")
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if re-executed
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(SensitiveDataSanitizingFilter())
        root_logger.addHandler(handler)

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger instance within the promptpilot namespace."""
    setup_logging()
    if name:
        return logging.getLogger(f"promptpilot.{name}")
    return logging.getLogger("promptpilot")
