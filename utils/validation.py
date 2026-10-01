"""Input validation, safety guardrails, and injection detection for PromptPilot."""

import re
from typing import List, Tuple

SUPPORTED_TASK_TYPES = [
    "Summarization",
    "Information Extraction",
    "Text Generation",
    "General Analysis",
]

# Heuristic patterns frequently observed in prompt injection attempts
INJECTION_SIGNALS = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "Instruction override attempt detected"),
    (r"disregard\s+(the\s+)?system\s+prompt", "System prompt disregard attempt detected"),
    (r"you\s+are\s+now\s+in\s+developer\s+mode", "Jailbreak mode attempt detected"),
    (r"reveal\s+(your\s+)?(api[_\s]key|system\s+prompt|credentials)", "Credential or system instruction exfiltration attempt"),
    (r"<system>.*?</system>", "Simulated system XML tags detected"),
]


class ValidationError(Exception):
    """Raised when user input fails structural or security constraints."""
    pass


def validate_task_type(task_type: str) -> str:
    """Validate task type against supported taxonomy."""
    clean = task_type.strip()
    if clean not in SUPPORTED_TASK_TYPES:
        raise ValidationError(f"Invalid task type '{task_type}'. Supported: {', '.join(SUPPORTED_TASK_TYPES)}")
    return clean


def validate_prompt_text(prompt: str, max_chars: int = 3000) -> str:
    """Validate prompt input text for presence, length, and null bytes."""
    clean = prompt.strip()
    if not clean:
        raise ValidationError("Prompt text cannot be empty.")
    if len(clean) > max_chars:
        raise ValidationError(f"Prompt length ({len(clean)} characters) exceeds maximum limit of {max_chars}.")
    if "\x00" in clean:
        raise ValidationError("Prompt contains illegal null-byte characters.")
    return clean


def validate_input_context(input_text: str, max_chars: int = 10000) -> str:
    """Validate source document context."""
    clean = input_text.strip()
    if not clean:
        raise ValidationError("Input context / source document cannot be empty.")
    if len(clean) > max_chars:
        raise ValidationError(f"Input context length ({len(clean)} characters) exceeds maximum limit of {max_chars}.")
    if "\x00" in clean:
        raise ValidationError("Input text contains illegal null-byte characters.")
    return clean


def check_for_injection_signals(text: str) -> List[str]:
    """Scan untrusted user text for heuristic prompt injection signals and return warnings.

    Note: This is an early defense-in-depth warning; true defense relies on boundary isolation
    and strict XML delimiter fencing inside LLM prompts.
    """
    detected_warnings: List[str] = []
    for pattern, warning_msg in INJECTION_SIGNALS:
        if re.search(pattern, text, re.IGNORECASE):
            detected_warnings.append(warning_msg)
    return detected_warnings


def estimate_api_calls(num_variants: int, num_test_cases: int = 1) -> dict:
    """Estimate total LLM API calls required for an experiment to ensure cost transparency.

    Formula:
    - 1 Analysis call
    - 1 Generator call
    - (1 Original + num_variants) * num_test_cases Execution calls
    - (1 Original + num_variants) * num_test_cases Evaluation calls
    """
    total_prompts = 1 + num_variants  # Original + variants
    execution_calls = total_prompts * num_test_cases
    evaluation_calls = total_prompts * num_test_cases
    total_calls = 1 + 1 + execution_calls + evaluation_calls

    return {
        "analysis_calls": 1,
        "generation_calls": 1,
        "execution_calls": execution_calls,
        "evaluation_calls": evaluation_calls,
        "total_api_calls": total_calls,
        "num_prompts": total_prompts,
        "num_test_cases": num_test_cases,
    }
