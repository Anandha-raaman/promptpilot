"""End-to-end mocked integration tests for ExperimentService."""

import tempfile
import unittest
from pathlib import Path
from typing import Type
from pydantic import BaseModel

from database.db import DatabaseManager
from database.repository import ExperimentRepository
from evaluation.scoring import EvaluationScore
from llm.base import BaseLLMProvider, LLMResponse
from prompts.analyzer import PromptAnalysisResult
from prompts.generator import GeneratedVariantsBundle, PromptVariant
from services.experiment_service import ExperimentService


class MockFullPipelineProvider(BaseLLMProvider):
    """Deterministic mock provider simulating analysis, variant generation, text execution, and evaluation."""

    def __init__(self):
        self.call_count = 0

    def generate_text(self, prompt: str, **kwargs) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            content=f"Mock response to prompt: {prompt[:30]}...",
            model_name="mock-model",
            latency_ms=120.0,
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
        )

    def generate_structured(self, prompt: str, response_schema: Type[BaseModel], **kwargs):
        self.call_count += 1
        if response_schema == PromptAnalysisResult:
            return PromptAnalysisResult(
                overall_analysis="Clear basic prompt with room for role and constraints.",
                clarity_score=7,
                context_score=6,
                constraint_score=5,
                output_format_score=4,
                specificity_score=6,
                issues=["Lacks explicit length constraints"],
                suggestions=["Add role persona", "Add bulleted format"],
            ), LLMResponse(content="{}", model_name="mock-model", latency_ms=100.0)

        elif response_schema == GeneratedVariantsBundle:
            return GeneratedVariantsBundle(
                variants=[
                    PromptVariant(
                        strategy_name="Role-Based",
                        strategy_description="Staff Analyst persona",
                        prompt_text="You are a Staff Analyst. Summarize the text accurately.",
                        expected_benefit="Technical authority and conciseness",
                        potential_limitations="Slightly formal tone",
                    ),
                    PromptVariant(
                        strategy_name="Constraint-Based",
                        strategy_description="Explicit boundary conditions",
                        prompt_text="Summarize the text in 3 bullet points. No commentary.",
                        expected_benefit="Strict length and bulleted output",
                        potential_limitations="Omits peripheral color",
                    ),
                ]
            ), LLMResponse(content="{}", model_name="mock-model", latency_ms=150.0)

        elif response_schema == EvaluationScore:
            return EvaluationScore(
                relevance=9,
                completeness=8,
                instruction_following=9,
                format_compliance=9,
                conciseness=8,
                factual_consistency=10,
                feedback=["Grounded, accurate summary."],
                strengths=["Followed all boundaries."],
            ), LLMResponse(content="{}", model_name="mock-model", latency_ms=90.0)

        raise ValueError(f"Unexpected schema: {response_schema}")

    def is_available(self) -> bool:
        return True


class TestExperimentService(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "service_test.db"
        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.repo = ExperimentRepository(self.db_manager)
        self.provider = MockFullPipelineProvider()
        from config.settings import Settings
        test_settings = Settings(rate_limit_cooldown_seconds=0.0)
        self.service = ExperimentService(provider=self.provider, repository=self.repo, settings=test_settings)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_experiment_pipeline(self):
        steps_recorded = []

        def progress_tracker(curr, total, msg):
            steps_recorded.append((curr, total, msg))

        detail, analysis, warnings = self.service.run_full_experiment(
            task_type="Summarization",
            original_prompt="Summarize this technical article.",
            input_text="Antigravity provides developer agent tools for rapid software creation.",
            desired_format="Bulleted list",
            num_variants=2,
            progress_cb=progress_tracker,
        )

        self.assertIsNotNone(detail)
        self.assertIsNotNone(analysis)
        self.assertGreater(len(steps_recorded), 0)

        # Baseline V0 + 2 variants = 3 total versions
        self.assertEqual(len(detail.versions), 3)

        for v in detail.versions:
            self.assertIsNotNone(v.response)
            self.assertIsNotNone(v.response.evaluation)
            self.assertGreater(v.response.evaluation.overall_score, 0)

        # Verify playground iteration
        v_next, eval_next = self.service.run_playground_iteration(
            experiment_id=detail.experiment.id,
            custom_prompt_text="You are a senior tech writer. Provide 2 bullet points.",
            version_number=3,
        )
        self.assertEqual(v_next.version_number, 3)
        self.assertGreater(eval_next.overall_score, 0)


if __name__ == "__main__":
    unittest.main()
