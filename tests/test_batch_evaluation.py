"""Unit tests for Batch Evaluation and EvaluationService."""

import tempfile
import unittest
from pathlib import Path
from typing import Type
from pydantic import BaseModel

from evaluation.scoring import EvaluationScore
from llm.base import BaseLLMProvider, LLMResponse
from services.evaluation_service import EvaluationService, TestCase


class MockBatchProvider(BaseLLMProvider):
    def generate_text(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(content="Generated test summary", model_name="mock-model", latency_ms=100.0)

    def generate_structured(self, prompt: str, response_schema: Type[BaseModel], **kwargs):
        score = EvaluationScore(
            relevance=9, completeness=8, instruction_following=9,
            format_compliance=8, conciseness=8, factual_consistency=10,
            feedback=["Consistent batch test output"],
        )
        return score, LLMResponse(content="{}", model_name="mock-model", latency_ms=80.0)

    def is_available(self) -> bool:
        return True


class TestBatchEvaluation(unittest.TestCase):
    def setUp(self):
        self.provider = MockBatchProvider()
        self.service = EvaluationService(provider=self.provider)

    def test_load_default_test_cases(self):
        cases = self.service.load_test_cases()
        self.assertGreater(len(cases), 0)

        sum_cases = self.service.load_test_cases(task_type="Summarization")
        self.assertGreater(len(sum_cases), 0)
        self.assertTrue(all(c.task_type == "Summarization" for c in sum_cases))

    def test_run_batch_evaluation(self):
        mock_cases = [
            TestCase(id="c1", task_type="Summarization", name="Test Case 1", description="", input_text="Input text 1"),
            TestCase(id="c2", task_type="Summarization", name="Test Case 2", description="", input_text="Input text 2"),
        ]
        report = self.service.run_batch_evaluation(
            prompt_text="Summarize the following document.",
            test_cases=mock_cases,
        )

        self.assertEqual(report.num_cases, 2)
        self.assertGreater(report.avg_overall_score, 0)
        self.assertEqual(len(report.case_results), 2)
        self.assertEqual(report.case_results[0].test_case_name, "Test Case 1")


if __name__ == "__main__":
    unittest.main()
