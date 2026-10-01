"""Response Evaluation scoring models and rubrics for PromptPilot.

Scores LLM-generated responses across 6 core criteria on a strict 0-10 scale:
1. Relevance: Focus on the user's intent without drifting or rambling.
2. Completeness: Coverage of critical key points or requested elements.
3. Instruction Following: Adherence to specified constraints, rules, and boundaries.
4. Format Compliance: Structural adherence (JSON, tables, bullet points, length).
5. Conciseness: Economy of language, density of information, avoiding filler.
6. Factual Consistency: Grounding against provided source data (no hallucinations).
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class EvaluationScore(BaseModel):
    """Model-based evaluation scores and granular feedback for a single response."""

    relevance: int = Field(
        default=7,
        ge=0, le=10,
        description="Relevance to the requested prompt and input (0-10).",
    )
    completeness: int = Field(
        default=7,
        ge=0, le=10,
        description="Coverage of necessary elements and comprehensiveness (0-10).",
    )
    instruction_following: int = Field(
        default=7,
        ge=0, le=10,
        description="Degree of adherence to explicit constraints, tone, and directives (0-10).",
    )
    format_compliance: Optional[int] = Field(
        default=None,
        ge=0, le=10,
        description="Adherence to requested formatting, sections, or schemas (0-10).",
    )
    conciseness: Optional[int] = Field(
        default=None,
        ge=0, le=10,
        description="Efficiency of expression; avoidance of repetitive filler (0-10).",
    )
    factual_consistency: Optional[int] = Field(
        default=None,
        ge=0, le=10,
        description="Factual consistency and faithfulness to the provided source text (0-10).",
    )
    raw_overall_score: Optional[float] = Field(
        default=None,
        alias="overall_score",
        description="Explicit evaluator overall score if directly supplied by the model.",
    )
    feedback: List[str] = Field(
        default_factory=list,
        description="Concise, evidence-based critique and justification for scores.",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Specific aspects done well in the generated response.",
    )
    is_available: bool = Field(
        default=True,
        description="Indicates whether the evaluation was successfully executed and parsed.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Safe error description if evaluation was unavailable or parsing failed.",
    )

    @field_validator("relevance", "completeness", "instruction_following", "format_compliance", "conciseness", "factual_consistency", mode="before")
    @classmethod
    def clamp_scores(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            val = int(round(float(v)))
            return max(0, min(10, val))
        except (ValueError, TypeError):
            return 7

    @property
    def effective_format_compliance(self) -> int:
        return self.format_compliance if self.format_compliance is not None else self.instruction_following

    @property
    def effective_conciseness(self) -> int:
        return self.conciseness if self.conciseness is not None else self.instruction_following

    @property
    def effective_factual_consistency(self) -> int:
        return self.factual_consistency if self.factual_consistency is not None else self.relevance

    @property
    def overall_score(self) -> float:
        """Weighted aggregate evaluation score on a 0-10 scale."""
        if not self.is_available:
            return 0.0

        calculated = (
            self.instruction_following * 0.25
            + self.relevance * 0.20
            + self.effective_factual_consistency * 0.20
            + self.completeness * 0.15
            + self.effective_format_compliance * 0.10
            + self.effective_conciseness * 0.10
        )

        if self.raw_overall_score is not None:
            raw = float(self.raw_overall_score)
            # Bound raw score to prevent inflation if core sub-scores (like factual consistency) are penalized
            max_allowed = max(calculated + 0.5, float(self.effective_factual_consistency))
            return round(max(0.0, min(max_allowed, raw, 10.0)), 1)

        return round(calculated, 1)

    def to_dict(self) -> dict:
        """Return dict representation including overall_score property and resolved criteria."""
        d = self.model_dump()
        d["overall_score"] = self.overall_score
        d["format_compliance"] = self.effective_format_compliance
        d["conciseness"] = self.effective_conciseness
        d["factual_consistency"] = self.effective_factual_consistency
        d["is_available"] = self.is_available
        d["error_message"] = self.error_message
        return d

    @classmethod
    def create_unavailable(cls, error_msg: str) -> "EvaluationScore":
        """Factory for safe evaluation failure representations without perfect-score fallbacks."""
        return cls(
            relevance=0,
            completeness=0,
            instruction_following=0,
            format_compliance=0,
            conciseness=0,
            factual_consistency=0,
            raw_overall_score=0.0,
            feedback=[f"Evaluation unavailable: {error_msg}"],
            strengths=[],
            is_available=False,
            error_message=error_msg,
        )

    @classmethod
    def from_raw_payload(cls, raw: Union[str, dict, BaseModel]) -> "EvaluationScore":
        """Robust parser mapping diverse LLM evaluator output structures into an EvaluationScore.

        Supports:
        - Strict 6-metric schemas
        - Partial 3 or 4 metric formats (e.g. relevance, completeness, instruction_following, overall_score)
        - Aliased keys (e.g. 'factual_accuracy' -> 'factual_consistency', 'score' -> 'overall_score')
        - JSON strings with markdown codeblocks
        - Safe error recording on malformed payloads without defaulting to 10
        """
        if isinstance(raw, EvaluationScore):
            return raw

        if isinstance(raw, BaseModel):
            raw = raw.model_dump()

        data: Dict[str, Any] = {}
        if isinstance(raw, str):
            cleaned = raw.strip()
            # Strip markdown formatting if present
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                # Try finding JSON block using regex
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                    except json.JSONDecodeError as err:
                        return cls.create_unavailable(f"Failed to parse evaluator JSON: {err}")
                else:
                    return cls.create_unavailable("Evaluator returned non-JSON output.")
        elif isinstance(raw, dict):
            data = raw
        else:
            return cls.create_unavailable(f"Unexpected evaluator payload type: {type(raw).__name__}")

        if not isinstance(data, dict):
            return cls.create_unavailable("Evaluator output did not resolve to a key-value object.")

        # Key normalization and aliasing map
        normalized: Dict[str, Any] = {}

        # 1. Relevance
        for k in ["relevance", "relevancy", "rel"]:
            if k in data:
                normalized["relevance"] = data[k]
                break

        # 2. Completeness
        for k in ["completeness", "coverage", "comp"]:
            if k in data:
                normalized["completeness"] = data[k]
                break

        # 3. Instruction following
        for k in ["instruction_following", "instruction", "following", "adherence", "constraint_adherence"]:
            if k in data:
                normalized["instruction_following"] = data[k]
                break

        # 4. Format compliance
        for k in ["format_compliance", "format", "formatting", "structure"]:
            if k in data:
                normalized["format_compliance"] = data[k]
                break

        # 5. Conciseness
        for k in ["conciseness", "brevity", "concise"]:
            if k in data:
                normalized["conciseness"] = data[k]
                break

        # 6. Factual consistency
        for k in ["factual_consistency", "factual_accuracy", "factuality", "grounding", "faithfulness"]:
            if k in data:
                normalized["factual_consistency"] = data[k]
                break

        # 7. Overall score
        for k in ["overall_score", "score", "rating", "overall"]:
            if k in data and data[k] is not None:
                try:
                    normalized["overall_score"] = float(data[k])
                except (ValueError, TypeError):
                    pass
                break

        # 8. Feedback & Strengths
        for k in ["feedback", "critique", "issues", "notes"]:
            if k in data:
                val = data[k]
                normalized["feedback"] = [val] if isinstance(val, str) else list(val)
                break

        for k in ["strengths", "positives", "highlights"]:
            if k in data:
                val = data[k]
                normalized["strengths"] = [val] if isinstance(val, str) else list(val)
                break

        try:
            return cls(**normalized)
        except Exception as e:
            return cls.create_unavailable(f"Evaluation schema initialization error: {e}")

