"""Unit tests for Prompt Analyzer."""

import unittest
from typing import Type
from pydantic import BaseModel

from llm.base import BaseLLMProvider, LLMResponse
from prompts.analyzer import PromptAnalyzer, PromptAnalysisResult


class MockAnalyzerProvider(BaseLLMProvider):
    """Deterministic mock provider for testing analyzer logic."""

    def __init__(self, mock_result: PromptAnalysisResult):
        self.mock_result = mock_result

    def generate_text(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(content="", model_name="mock-model", latency_ms=10.0)

    def generate_structured(self, prompt: str, response_schema: Type[BaseModel], **kwargs):
        llm_resp = LLMResponse(content="{}", model_name="mock-model", latency_ms=15.0)
        return self.mock_result, llm_resp

    def is_available(self) -> bool:
        return True


class TestPromptAnalyzer(unittest.TestCase):
    def test_analyzer_success(self):
        expected = PromptAnalysisResult(
            overall_analysis="The prompt is too generic.",
            clarity_score=6,
            context_score=4,
            constraint_score=3,
            output_format_score=2,
            specificity_score=5,
            issues=["Missing target audience", "No format specified"],
            suggestions=["Add persona", "Specify JSON format"],
            has_role=False,
            has_constraints=False,
            has_format_specification=False,
        )
        provider = MockAnalyzerProvider(expected)
        analyzer = PromptAnalyzer(provider=provider)

        result, resp = analyzer.analyze("Summarize this text.", task_type="Summarization")

        self.assertEqual(result.clarity_score, 6)
        self.assertEqual(len(result.issues), 2)
        self.assertGreater(result.overall_health_score, 0)
        self.assertEqual(resp.model_name, "mock-model")

    def test_analyzer_empty_prompt_validation(self):
        provider = MockAnalyzerProvider(
            PromptAnalysisResult(
                overall_analysis="", clarity_score=0, context_score=0,
                constraint_score=0, output_format_score=0, specificity_score=0
            )
        )
        analyzer = PromptAnalyzer(provider=provider)

        with self.assertRaises(ValueError):
            analyzer.analyze("   ")

    def test_analyzer_filters_missing_source_data_issues(self):
        # Verify that false-positive 'missing source text' complaints are purged
        raw_result = PromptAnalysisResult(
            overall_analysis="The target text to be summarized is missing entirely from the prompt. Instructions are vague.",
            clarity_score=5,
            context_score=2,
            constraint_score=3,
            output_format_score=2,
            specificity_score=4,
            issues=[
                "The target text to be summarized is missing entirely from the prompt.",
                "No output format specified (e.g. bullet points or length limit).",
                "No target audience specified.",
            ],
            suggestions=["Specify length and format."],
            has_role=False,
            has_constraints=False,
            has_format_specification=False,
        )
        provider = MockAnalyzerProvider(raw_result)
        analyzer = PromptAnalyzer(provider=provider)

        sanitized, _ = analyzer.analyze(
            user_prompt="Summarize this text.",
            task_type="Summarization",
            has_input_context=True,
        )

        # The 'target text is missing' issue must be purged
        self.assertNotIn("The target text to be summarized is missing entirely from the prompt.", sanitized.issues)
        # Legitimate prompt engineering issues must remain
        self.assertIn("No output format specified (e.g. bullet points or length limit).", sanitized.issues)
        self.assertIn("No target audience specified.", sanitized.issues)
        self.assertEqual(len(sanitized.issues), 2)
        # Context score should have been lifted from artificial 2
        self.assertGreaterEqual(sanitized.context_score, 4)


if __name__ == "__main__":
    unittest.main()
