"""Automated test suite verifying Response Evaluator and scoring system behavior.

Verifies the 6 core evaluation criteria and behavioral separation rules:
TEST 1 - Excellent response (5 source facts covered, high scores)
TEST 2 - Incomplete response (5 source facts, only 1 covered, completeness penalized)
TEST 3 - Irrelevant response (Cloud migration source vs EV response, relevance penalized)
TEST 4 - Factual inconsistency (Revenue increased vs decreased, factual consistency penalized)
TEST 5 - Format violation (Prompt asks for exactly 3 bullets, response is paragraph)
TEST 6 - Good response with weak prompt (Separation between Prompt Health and Response Quality)
"""

import unittest
from config.settings import Settings
from evaluation.scoring import EvaluationScore
from llm.mock_provider import MockLLMProvider
from prompts.analyzer import PromptAnalyzer
from prompts.evaluator import ResponseEvaluator


class TestResponseEvaluatorScoring(unittest.TestCase):
    """Rigorous evaluation and scoring verification tests."""

    def setUp(self):
        self.settings = Settings(offline_demo_mode=True)
        self.provider = MockLLMProvider()
        self.evaluator = ResponseEvaluator(provider=self.provider, settings=self.settings)
        self.analyzer = PromptAnalyzer(provider=self.provider, settings=self.settings)

    def test_1_excellent_response(self):
        """TEST 1: Excellent response covering all 5 facts and adhering to all directives."""
        source_text = (
            "Fact 1: Serverless compute costs dropped by 28% across EU regions.\n"
            "Fact 2: Ingress bandwidth charges exceeded budget by $340,000.\n"
            "Fact 3: Non-peak cold start latency climbed to 480ms at p99.\n"
            "Fact 4: Provisioned concurrency was mandated for auth endpoints.\n"
            "Fact 5: Infrastructure migration completed during Q2."
        )
        prompt_text = "Provide an executive summary detailing all 5 facts from the cloud engineering report."
        response_text = (
            "Executive Cloud Migration Review:\n"
            "• Fact 1: Serverless migration achieved a 28% drop in compute costs.\n"
            "• Fact 2: Transition overshot budget by $340,000 due to ingress bandwidth miscalculations.\n"
            "• Fact 3: Cold start latency spiked to 480ms p99 on infrequently invoked microservices.\n"
            "• Fact 4: Provisioned concurrency has been mandated for auth endpoints.\n"
            "• Fact 5: Entire migration finished on schedule in Q2."
        )

        score, _ = self.evaluator.evaluate(
            task_type="Summarization",
            prompt_text=prompt_text,
            input_text=source_text,
            response_text=response_text,
        )

        self.assertIsInstance(score, EvaluationScore)
        self.assertGreaterEqual(score.completeness, 9, f"Completeness should be >= 9, got {score.completeness}")
        self.assertGreaterEqual(score.factual_consistency, 9, f"Factual consistency should be >= 9, got {score.factual_consistency}")
        self.assertGreaterEqual(score.instruction_following, 9, f"Instruction following should be >= 9, got {score.instruction_following}")
        self.assertGreaterEqual(score.relevance, 8, f"Relevance should be >= 8, got {score.relevance}")
        self.assertGreaterEqual(score.overall_score, 8.8, f"Overall score should be high (>=8.8), got {score.overall_score}")

    def test_2_incomplete_response(self):
        """TEST 2: Incomplete response covers only 1 of 5 important facts; completeness drops."""
        source_text = (
            "Fact 1: Serverless compute costs dropped by 28% across EU regions.\n"
            "Fact 2: Ingress bandwidth charges exceeded budget by $340,000.\n"
            "Fact 3: Non-peak cold start latency climbed to 480ms at p99.\n"
            "Fact 4: Provisioned concurrency was mandated for auth endpoints.\n"
            "Fact 5: Infrastructure migration completed during Q2."
        )
        prompt_text = "Summarize the key outcomes from the migration report."
        # Only mentions Fact 1, completely omitting Facts 2, 3, 4, 5
        response_text = "Fact 1: Serverless compute costs dropped by 28% across European regions during Q2."

        score, _ = self.evaluator.evaluate(
            task_type="Summarization",
            prompt_text=prompt_text,
            input_text=source_text,
            response_text=response_text,
        )

        self.assertLessEqual(score.completeness, 4, f"Completeness must decrease (<= 4), got {score.completeness}")
        self.assertLess(score.completeness, score.factual_consistency, "Completeness must be lower than factual consistency")
        self.assertLess(score.overall_score, 8.0, f"Overall score should decrease due to omissions, got {score.overall_score}")
        self.assertTrue(any("incomplete" in fb.lower() or "omit" in fb.lower() for fb in score.feedback))

    def test_3_irrelevant_response(self):
        """TEST 3: Irrelevant response discusses electric vehicles when source is cloud migration."""
        source_text = (
            "TechVanguard completed its migration to serverless compute across European regions during Q2. "
            "While operational compute expenses dropped by 28%, initial transition costs overshot the $1.5M budget by $340,000 "
            "due to data ingress bandwidth miscalculations. Infrequently invoked services experienced cold start increases up to 480ms p99."
        )
        prompt_text = "Summarize the technical outcomes of the migration."
        response_text = (
            "Electric vehicles (EVs) are transitioning to mainstream adoption, driven by regulatory targets "
            "and battery pack advancements. However, charging infrastructure bottlenecks remain a key obstacle."
        )

        score, _ = self.evaluator.evaluate(
            task_type="Summarization",
            prompt_text=prompt_text,
            input_text=source_text,
            response_text=response_text,
        )

        self.assertLessEqual(score.relevance, 3, f"Relevance must be very low (<= 3), got {score.relevance}")
        self.assertLessEqual(score.factual_consistency, 3, f"Factual consistency must be very low (<= 3), got {score.factual_consistency}")
        self.assertLessEqual(score.overall_score, 4.0, f"Overall score should be very low (<= 4.0), got {score.overall_score}")
        self.assertTrue(any("electric vehicle" in fb.lower() or "ungrounded" in fb.lower() for fb in score.feedback))

    def test_4_factual_inconsistency(self):
        """TEST 4: Factual inconsistency (source says increased by 8%, response says decreased by 8%)."""
        source_text = (
            "Apex Quantum Semiconductor reported Q1 fiscal revenues increased by 8% year-over-year. "
            "Net income expanded to $112 million, supported by cryogenic quantum chip demand."
        )
        prompt_text = "Extract the financial performance figures from the disclosure."
        response_text = (
            "According to the disclosure, Apex Quantum Semiconductor reported that Q1 fiscal revenues decreased by 8% year-over-year. "
            "Net income was reported at $112 million."
        )

        score, _ = self.evaluator.evaluate(
            task_type="Information Extraction",
            prompt_text=prompt_text,
            input_text=source_text,
            response_text=response_text,
        )

        self.assertLessEqual(score.factual_consistency, 4, f"Factual consistency must be penalized (<= 4), got {score.factual_consistency}")
        self.assertLessEqual(score.relevance, 5, f"Relevance should be penalized (<= 5), got {score.relevance}")
        self.assertLess(score.overall_score, 7.0, f"Overall score should reflect contradiction (< 7.0), got {score.overall_score}")
        self.assertTrue(any("contradict" in fb.lower() or "inconsistency" in fb.lower() or "decreased" in fb.lower() for fb in score.feedback))

    def test_5_format_violation(self):
        """TEST 5: Format violation (prompt requires exactly 3 bullet points, response is long paragraph)."""
        source_text = (
            "Security Alert SEC-2026-041: A remote code execution flaw (CVE-2026-19342) in OpenGateway proxy library "
            "allows unauthenticated remote attackers to inject arbitrary headers into internal RPC dispatchers. "
            "Advisory recommends patching to 3.4.2 or disabling WebSocket multiplexing."
        )
        prompt_text = "Provide an advisory summary with exactly 3 bullet points."
        # Continuous paragraph with 0 bullet points
        response_text = (
            "OpenGateway proxy library suffers from a severe remote code execution vulnerability designated as CVE-2026-19342. "
            "This critical flaw enables unauthenticated remote actors to inject malicious headers into internal RPC dispatchers. "
            "Security engineers must upgrade to version 3.4.2 immediately or disable WebSocket multiplexing as a temporary workaround."
        )

        score, _ = self.evaluator.evaluate(
            task_type="Summarization",
            prompt_text=prompt_text,
            input_text=source_text,
            response_text=response_text,
        )

        self.assertLessEqual(score.format_compliance, 4, f"Format compliance must decrease (<= 4), got {score.format_compliance}")
        self.assertLessEqual(score.instruction_following, 5, f"Instruction following must decrease (<= 5), got {score.instruction_following}")
        self.assertLess(score.format_compliance, score.factual_consistency, "Format compliance should be lower than factual accuracy")
        self.assertTrue(any("format" in fb.lower() or "bullet" in fb.lower() or "paragraph" in fb.lower() for fb in score.feedback))

    def test_6_good_response_with_weak_prompt(self):
        """TEST 6: Weak prompt ('Summarize this.') produces good response without automatic 10/10.

        Separation guarantee:
        - Prompt Health evaluates the prompt instruction quality (clarity, specificity, constraints).
        - Response Quality evaluates the generated completion against source and instructions.
        - Prompt Health can be low while Response Quality is high.
        - Neither should Response Quality be artificially dragged down by low Prompt Health,
          nor should it automatically receive 10.0/10.
        """
        weak_prompt = "Summarize this."
        detailed_source = (
            "TechVanguard completed its migration to serverless compute across European regions during Q2. "
            "While operational compute expenses dropped by 28%, initial transition costs overshot the $1.5M budget by $340,000 "
            "due to data ingress bandwidth miscalculations. Furthermore, cold start latency on infrequently invoked microservices "
            "increased p99 response times from 120ms to 480ms during non-peak hours. The infrastructure team has mandated "
            "provisioned concurrency for customer-facing authentication endpoints starting next sprint."
        )
        good_response = (
            "Cloud Migration Summary:\n"
            "TechVanguard completed its European serverless transition in Q2, cutting operational compute costs by 28%. "
            "However, ingress bandwidth miscalculations led to a $340,000 budget overrun. Infrequent services saw p99 cold starts "
            "increase to 480ms, prompting mandated provisioned concurrency for auth endpoints."
        )

        # 1. Evaluate Prompt Health
        analysis, _ = self.analyzer.analyze(
            user_prompt=weak_prompt,
            task_type="Summarization",
            has_input_context=True,
        )
        prompt_health = analysis.overall_health_score

        # 2. Evaluate Response Quality
        resp_eval, _ = self.evaluator.evaluate(
            task_type="Summarization",
            prompt_text=weak_prompt,
            input_text=detailed_source,
            response_text=good_response,
        )

        # Assertions proving separation and calibration
        # Prompt Health should be low for "Summarize this."
        self.assertLessEqual(prompt_health, 6.0, f"Prompt Health for weak prompt should be low/medium, got {prompt_health}")
        # Response Quality should be high because the response is accurate and grounded
        self.assertGreaterEqual(resp_eval.overall_score, 7.5, f"Response Quality should remain high (>= 7.5), got {resp_eval.overall_score}")
        self.assertGreaterEqual(resp_eval.factual_consistency, 8, f"Factual consistency should be high, got {resp_eval.factual_consistency}")

        # Response Quality should NOT automatically receive 10.0 / 10 / 10 / 10
        self.assertNotEqual(resp_eval.overall_score, 10.0, "Response to weak prompt must not automatically receive 10.0")
        self.assertNotEqual(resp_eval.relevance, 10, "Relevance to open-ended prompt should not be an automatic 10")

        # Proves Prompt Health and Response Quality are calculated independently
        self.assertNotEqual(resp_eval.overall_score, prompt_health, "Response Quality must not simply copy Prompt Health")


if __name__ == "__main__":
    unittest.main()
