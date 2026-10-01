"""Automated tests for Prompt Health vs Response Quality separation and instruction fidelity.

Covers:
1. Non-empty Source Data + short prompt ("summarize this")
2. Empty Source Data + short prompt
3. Weak prompt + high-quality response (low Prompt Health, high Response Quality)
4. Strong prompt + high-quality response (high Prompt Health, high Response Quality)
5. Evaluation feedback must not invent instructions that were not present in the prompt.
"""

import unittest
from typing import Type
from pydantic import BaseModel

from evaluation.scoring import EvaluationScore
from llm.base import BaseLLMProvider, LLMResponse
from prompts.analyzer import PromptAnalyzer, PromptAnalysisResult
from prompts.evaluator import ResponseEvaluator, sanitize_evaluation_score


class MockCustomLLMProvider(BaseLLMProvider):
    """Customizable mock provider for deterministic multi-case tests."""

    def __init__(self, structured_return: BaseModel):
        self.structured_return = structured_return

    def is_available(self) -> bool:
        return True

    def generate_text(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(content="", model_name="mock-test-engine", latency_ms=10.0)

    def generate_structured(self, prompt: str, response_schema: Type[BaseModel], **kwargs):
        return self.structured_return, LLMResponse(content="{}", model_name="mock-test-engine", latency_ms=15.0)


class TestEvaluationSeparation(unittest.TestCase):
    """Verifies strict separation between prompt health and response evaluation."""

    def setUp(self):
        self.sample_article = (
            "Artificial intelligence is transforming business operations across multiple dimensions: "
            "automating repetitive tasks, providing deep predictive data analytics, and improving 24/7 customer service. "
            "However, enterprises face significant implementation challenges including high upfront costs, data security risks, "
            "and the essential need for responsible AI governance and continuous human oversight."
        )

    def test_non_empty_source_data_short_prompt(self):
        """When source data is provided, analyzer must NOT report that source text is missing."""
        mock_raw = PromptAnalysisResult(
            overall_analysis="The prompt is missing the actual text to be summarized. Instructions are vague.",
            clarity_score=3,
            context_score=2,
            constraint_score=1,
            output_format_score=1,
            specificity_score=2,
            issues=[
                "The prompt is missing the actual text to be summarized.",
                "No target length or word count limit specified.",
                "No output format specified (e.g. bullet points or paragraph).",
            ],
            suggestions=[
                "Add the source text to the prompt.",
                "Specify target length and output structure.",
            ],
            has_role=False,
            has_constraints=False,
            has_format_specification=False,
        )

        provider = MockCustomLLMProvider(mock_raw)
        analyzer = PromptAnalyzer(provider=provider)

        result, _ = analyzer.analyze(
            user_prompt="summarize this",
            task_type="Summarization",
            has_input_context=True,
            input_context_summary="Enterprise AI business adoption article",
        )

        # Must not claim the actual text / source text is missing
        for issue in result.issues:
            self.assertNotIn("missing the actual text", issue.lower())
            self.assertNotIn("source text is missing", issue.lower())

        # Must not suggest adding source text to prompt
        for sugg in result.suggestions:
            self.assertNotIn("add the source text to the prompt", sugg.lower())

        # Overall analysis must be sanitized
        self.assertNotIn("missing the actual text", result.overall_analysis.lower())

        # Legitimate prompt engineering issues must remain
        self.assertTrue(any("length" in issue.lower() for issue in result.issues))
        self.assertTrue(any("format" in issue.lower() for issue in result.issues))

    def test_empty_source_data_short_prompt(self):
        """When source data is NOT provided, analyzer may legitimately report missing data."""
        mock_raw = PromptAnalysisResult(
            overall_analysis="The source data is missing entirely from both prompt and context.",
            clarity_score=2,
            context_score=1,
            constraint_score=1,
            output_format_score=1,
            specificity_score=1,
            issues=["The source text is missing."],
            suggestions=["Provide the input context text."],
            has_role=False,
            has_constraints=False,
            has_format_specification=False,
        )

        provider = MockCustomLLMProvider(mock_raw)
        analyzer = PromptAnalyzer(provider=provider)

        result, _ = analyzer.analyze(
            user_prompt="summarize this",
            task_type="Summarization",
            has_input_context=False,
        )

        # Since has_input_context is False, missing data issue is NOT filtered out
        self.assertIn("The source text is missing.", result.issues)

    def test_weak_prompt_high_quality_response(self):
        """A weak prompt (Prompt Health ~2/10) can still yield a high-quality response (9/10)."""
        weak_prompt = "summarize this"
        high_quality_response = (
            "Artificial intelligence provides major business benefits through automated workflows, predictive analytics, "
            "and 24/7 customer service. However, successful enterprise adoption requires addressing high implementation costs, "
            "data security risks, and maintaining human oversight with responsible governance."
        )

        # 1. Prompt Health evaluation
        analyzer_result = PromptAnalysisResult(
            overall_analysis="The prompt is understandable but vague. It lacks format, length limits, and negative constraints.",
            clarity_score=3,
            context_score=4,
            constraint_score=1,
            output_format_score=1,
            specificity_score=2,
            issues=[
                "No target length or ceiling specified.",
                "No output format specified.",
                "No prioritization criteria for key themes.",
            ],
            suggestions=["Add bullet point format", "Specify 100-word ceiling"],
            has_role=False,
            has_constraints=False,
            has_format_specification=False,
        )
        # Weak prompt health: 3*0.25 + 2*0.25 + 1*0.20 + 1*0.15 + 4*0.15 = 0.75 + 0.5 + 0.2 + 0.15 + 0.6 = 2.2
        self.assertLessEqual(analyzer_result.overall_health_score, 3.5)

        # 2. Response Quality evaluation
        eval_score = EvaluationScore(
            relevance=10,
            completeness=9,
            instruction_following=9,
            format_compliance=9,
            conciseness=9,
            factual_consistency=10,
            feedback=[
                "Response accurately captures the main benefits and challenges described in the source data and does not introduce unsupported claims."
            ],
            strengths=[
                "Strictly faithful to source text with high information density."
            ],
        )

        provider = MockCustomLLMProvider(eval_score)
        evaluator = ResponseEvaluator(provider=provider)

        score, _ = evaluator.evaluate(
            task_type="Summarization",
            prompt_text=weak_prompt,
            input_text=self.sample_article,
            response_text=high_quality_response,
        )

        # The response score should remain high (>= 9.0) independently of the weak prompt health
        self.assertGreaterEqual(score.overall_score, 9.0)
        self.assertGreaterEqual(score.factual_consistency, 9)
        self.assertLess(analyzer_result.overall_health_score, score.overall_score)

    def test_strong_prompt_high_quality_response(self):
        """A well-engineered prompt and high-quality response both yield high scores."""
        strong_prompt = (
            "You are a Senior Technology Strategist. Summarize the following AI enterprise report into exactly 3 bullet points:\n"
            "1. Primary operational benefits\n"
            "2. Critical implementation and security risks\n"
            "3. Governance and human oversight imperatives\n"
            "Do not include conversational filler or ungrounded claims."
        )

        analyzer_result = PromptAnalysisResult(
            overall_analysis="Exemplary prompt with expert persona, structural breakdown, and negative constraints.",
            clarity_score=10,
            context_score=9,
            constraint_score=9,
            output_format_score=10,
            specificity_score=10,
            issues=[],
            suggestions=[],
            has_role=True,
            has_constraints=True,
            has_format_specification=True,
        )
        self.assertGreaterEqual(analyzer_result.overall_health_score, 9.0)

        eval_score = EvaluationScore(
            relevance=10,
            completeness=10,
            instruction_following=10,
            format_compliance=10,
            conciseness=9,
            factual_consistency=10,
            feedback=["Perfectly followed all 3 requested categories and negative constraints."],
            strengths=["Highly structured, factual, and concise."],
        )

        self.assertGreaterEqual(eval_score.overall_score, 9.0)

    def test_evaluation_feedback_must_not_invent_instructions(self):
        """Evaluator sanitization must strip or correct hallucinated claims attributing unrequested rules to the prompt."""
        weak_prompt = "summarize this"

        raw_score = EvaluationScore(
            relevance=9,
            completeness=9,
            instruction_following=9,
            format_compliance=8,
            conciseness=9,
            factual_consistency=10,
            feedback=[
                "Response accurately captures key concepts from the source data.",
                "The prompt contained length instructions that were adhered to.",
                "The prompt specified negative constraints which were respected.",
            ],
            strengths=[
                "The prompt required bullet points and the response provided them.",
                "Strictly grounded in source text without unsupported claims.",
            ],
        )

        sanitized = sanitize_evaluation_score(raw_score, prompt_text=weak_prompt)

        # Ensure no feedback or strength claims the weak prompt contained length/negative/bullet instructions
        combined_text = " ".join(sanitized.feedback + sanitized.strengths).lower()
        self.assertNotIn("the prompt contained length", combined_text)
        self.assertNotIn("the prompt specified negative", combined_text)
        self.assertNotIn("the prompt required bullet", combined_text)


if __name__ == "__main__":
    unittest.main()
