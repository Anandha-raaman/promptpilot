"""Data schemas and transfer objects for database records."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationRecord:
    id: str
    response_id: str
    relevance: int
    completeness: int
    instruction_following: int
    format_compliance: int
    conciseness: int
    factual_consistency: int
    overall_score: float
    feedback: List[str]
    strengths: List[str]
    created_at: str


@dataclass
class ResponseRecord:
    id: str
    prompt_version_id: str
    response_text: str
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    evaluation: Optional[EvaluationRecord] = None


@dataclass
class PromptVersionRecord:
    id: str
    experiment_id: str
    version_number: int
    strategy: str
    prompt_text: str
    expected_benefit: Optional[str] = None
    potential_limitations: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    response: Optional[ResponseRecord] = None


@dataclass
class ExperimentRecord:
    id: str
    task_type: str
    original_prompt: str
    input_text: str
    model_name: str
    created_at: str
    title: Optional[str] = None


@dataclass
class FullExperimentDetail:
    experiment: ExperimentRecord
    versions: List[PromptVersionRecord] = field(default_factory=list)


@dataclass
class DatasetRecord:
    id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    case_count: int = 0

