"""Unit tests for response evaluation scoring rubric."""

import unittest
from evaluation.scoring import EvaluationScore


class TestScoring(unittest.TestCase):
    def test_score_calculation(self):
        score = EvaluationScore(
            relevance=10,
            completeness=8,
            instruction_following=9,
            format_compliance=10,
            conciseness=8,
            factual_consistency=10,
            feedback=["Clear, accurate summary"],
            strengths=["Zero hallucination"],
        )

        # Instruction following: 9 * 0.25 = 2.25
        # Relevance: 10 * 0.20 = 2.0
        # Factual consistency: 10 * 0.20 = 2.0
        # Completeness: 8 * 0.15 = 1.2
        # Format compliance: 10 * 0.10 = 1.0
        # Conciseness: 8 * 0.10 = 0.8
        # Total = 2.25 + 2.0 + 2.0 + 1.2 + 1.0 + 0.8 = 9.25 -> rounded to 9.2 or 9.3
        self.assertAlmostEqual(score.overall_score, 9.2, places=1)
        self.assertIn("overall_score", score.to_dict())

    def test_deterministic_scoring_layer(self):
        """Deterministic unit test for the scoring layer using mocked evaluator outputs.

        Existing weighted formula:
        instruction_following * 0.25 + relevance * 0.20 + factual_consistency * 0.20
        + completeness * 0.15 + format_compliance * 0.10 + conciseness * 0.10
        """
        # Mock Evaluation A: All high
        raw_a = {
            "relevance": 10,
            "completeness": 10,
            "instruction_following": 10,
        }
        score_a = EvaluationScore.from_raw_payload(raw_a)
        self.assertTrue(score_a.is_available)
        self.assertEqual(score_a.relevance, 10)
        self.assertEqual(score_a.completeness, 10)
        self.assertEqual(score_a.instruction_following, 10)
        self.assertGreaterEqual(score_a.overall_score, 9.5, f"Mock A overall score should be high (>= 9.5), got {score_a.overall_score}")
        self.assertEqual(score_a.overall_score, 10.0)

        # Mock Evaluation B: All low
        raw_b = {
            "relevance": 2,
            "completeness": 3,
            "instruction_following": 2,
        }
        score_b = EvaluationScore.from_raw_payload(raw_b)
        self.assertTrue(score_b.is_available)
        self.assertEqual(score_b.relevance, 2)
        self.assertEqual(score_b.completeness, 3)
        self.assertEqual(score_b.instruction_following, 2)
        self.assertLessEqual(score_b.overall_score, 3.5, f"Mock B overall score should be low (<= 3.5), got {score_b.overall_score}")
        self.assertAlmostEqual(score_b.overall_score, 2.15, delta=0.1)

        # Mock Evaluation C: Mixed / medium
        raw_c = {
            "relevance": 8,
            "completeness": 4,
            "instruction_following": 7,
        }
        score_c = EvaluationScore.from_raw_payload(raw_c)
        self.assertTrue(score_c.is_available)
        self.assertEqual(score_c.relevance, 8)
        self.assertEqual(score_c.completeness, 4)
        self.assertEqual(score_c.instruction_following, 7)
        # Expected overall: 7*0.25 + 8*0.20 + 8*0.20 + 4*0.15 + 7*0.10 + 7*0.10 = 6.95 -> 7.0
        self.assertGreaterEqual(score_c.overall_score, 6.5)
        self.assertLessEqual(score_c.overall_score, 7.5)
        self.assertEqual(score_c.overall_score, 7.0)

    def test_explicit_overall_score_mapping(self):
        """Verify that if the LLM evaluator supplies an explicit overall_score, it is preserved."""
        raw_payload = {
            "relevance": 7,
            "completeness": 6,
            "instruction_following": 8,
            "overall_score": 7,
        }
        score = EvaluationScore.from_raw_payload(raw_payload)
        self.assertTrue(score.is_available)
        self.assertEqual(score.relevance, 7)
        self.assertEqual(score.completeness, 6)
        self.assertEqual(score.instruction_following, 8)
        self.assertEqual(score.overall_score, 7.0)

    def test_failed_parsing_does_not_default_to_ten(self):
        """Verify that malformed evaluator output safely records an error and does NOT default to 10."""
        score = EvaluationScore.from_raw_payload("This is completely malformed text that cannot be parsed as JSON.")
        self.assertFalse(score.is_available)
        self.assertEqual(score.overall_score, 0.0)
        self.assertNotEqual(score.overall_score, 10.0)
        self.assertIsNotNone(score.error_message)


if __name__ == "__main__":
    unittest.main()
