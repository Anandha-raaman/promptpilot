"""Automated evaluation tests specifically for hallucinated and unsupported information."""

import unittest
from llm.mock_provider import MockLLMProvider
from prompts.evaluator import ResponseEvaluator


class TestFactualConsistencyAndHallucination(unittest.TestCase):
    """Test suite verifying factual consistency scoring against unsupported claims and contradictions."""

    def setUp(self):
        self.provider = MockLLMProvider()
        self.evaluator = ResponseEvaluator(provider=self.provider)

    def test_a_exact_source_supported_fact(self):
        """Test A: Source says revenue = $500M. Response says revenue = $500M -> High factual consistency."""
        source = "In Q3 2026, the company achieved total revenue of $500M."
        prompt = "Extract the reported revenue."
        response = "The company reported total revenue of $500M in Q3 2026."

        score, _ = self.evaluator.evaluate(
            task_type="Information Extraction",
            prompt_text=prompt,
            input_text=source,
            response_text=response,
        )

        self.assertGreaterEqual(
            score.factual_consistency, 9,
            f"Test A should have high factual consistency (>= 9), got {score.factual_consistency}",
        )
        self.assertGreaterEqual(score.overall_score, 8.5)

    def test_b_direct_numerical_contradiction(self):
        """Test B: Source says revenue = $500M. Response says revenue = $700M -> Low factual consistency."""
        source = "In Q3 2026, the company achieved total revenue of $500M."
        prompt = "Extract the reported revenue."
        response = "The company reported total revenue of $700M in Q3 2026."

        score, _ = self.evaluator.evaluate(
            task_type="Information Extraction",
            prompt_text=prompt,
            input_text=source,
            response_text=response,
        )

        self.assertLessEqual(
            score.factual_consistency, 4,
            f"Test B should have low factual consistency (<= 4), got {score.factual_consistency}",
        )
        self.assertTrue(
            any("contradict" in fb.lower() or "$700m" in fb.lower() for fb in score.feedback),
            "Feedback should explicitly mention the contradiction",
        )

    def test_c_unsupported_mandatory_requirement(self):
        """Test C: Source says '3.0 second latency threshold.' Response adds 'email alerts are mandatory.' -> Penalize."""
        source = "The alert threshold is 3.0 seconds."
        prompt = "State the alerting requirement."
        response = "The alert threshold is 3.0 seconds and email alerts are mandatory."

        score, _ = self.evaluator.evaluate(
            task_type="Text Generation",
            prompt_text=prompt,
            input_text=source,
            response_text=response,
        )

        self.assertLessEqual(
            score.factual_consistency, 6,
            f"Test C should be penalized for unsupported requirements (<= 6), got {score.factual_consistency}",
        )
        self.assertTrue(
            any("email" in fb.lower() or "unsupported" in fb.lower() for fb in score.feedback),
            "Feedback should cite the ungrounded email requirement",
        )

    def test_d_reasonable_explanation_of_purpose(self):
        """Test D: Source says '3.0 second latency threshold.' Response explains that this helps identify slow executions -> Do not penalize."""
        source = "The alert threshold is 3.0 seconds."
        prompt = "State the alerting requirement and explain its purpose."
        response = "The alert threshold is 3.0 seconds. This can help developers identify slow executions."

        score, _ = self.evaluator.evaluate(
            task_type="Text Generation",
            prompt_text=prompt,
            input_text=source,
            response_text=response,
        )

        self.assertGreaterEqual(
            score.factual_consistency, 9,
            f"Test D should have high factual consistency (>= 9), got {score.factual_consistency}",
        )

    def test_product_feature_acceptance_criteria_unsupported_specs(self):
        """Test Product Feature Acceptance Criteria case with invented specifications."""
        source = (
            "Product Requirement: We need an alerting feature in PromptPilot that notifies users "
            "when a model execution takes longer than 3.0 seconds, allowing developers to detect "
            "slow models or unoptimized token lengths during batch runs."
        )
        prompt = "Write formal product acceptance criteria for the latency alerting threshold in PromptPilot."
        
        # Hallucinated response with invented specifications
        hallucinated_response = (
            "Feature: Latency Alerting Threshold\n\n"
            "AC-01: User accesses the monitoring or alert settings configuration panel to configure thresholds.\n"
            "AC-02: Latency is calculated with millisecond precision.\n"
            "AC-03: When latency > 3.0s, alert is dispatched to configured notification channels (e.g., webhook, email).\n"
            "AC-04: Non-functional requirement: Latency monitoring must add no greater than 50ms of overhead.\n"
            "AC-05: If channel fails, alert must be logged in system audit trail."
        )

        score_hallucinated, _ = self.evaluator.evaluate(
            task_type="Text Generation",
            prompt_text=prompt,
            input_text=source,
            response_text=hallucinated_response,
        )

        self.assertLessEqual(
            score_hallucinated.factual_consistency, 6,
            f"Factual consistency must be penalized (<= 6) for invented specs, got {score_hallucinated.factual_consistency}",
        )
        self.assertTrue(
            any("unsupported" in fb.lower() or "webhook" in fb.lower() or "channel" in fb.lower() for fb in score_hallucinated.feedback),
            "Feedback must identify unsupported channels or specifications",
        )

        # Clean response strictly adhering to source without invented requirements
        clean_response = (
            "Feature: Execution Latency Alerting\n\n"
            "1. User Story: As a developer running batch executions in PromptPilot, I want notifications when model response duration exceeds 3.0 seconds.\n"
            "2. Acceptance Criteria:\n"
            "   - Given a batch test execution\n"
            "   - When model latency exceeds 3.0 seconds\n"
            "   - Then a latency alert notification is triggered to highlight slow models or unoptimized token lengths."
        )

        score_clean, _ = self.evaluator.evaluate(
            task_type="Text Generation",
            prompt_text=prompt,
            input_text=source,
            response_text=clean_response,
        )

        self.assertGreaterEqual(
            score_clean.factual_consistency, 9,
            f"Clean response must receive high factual consistency (>= 9), got {score_clean.factual_consistency}",
        )


if __name__ == "__main__":
    unittest.main()
