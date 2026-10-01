"""Unit tests for regression detection and version comparison."""

import unittest
from evaluation.regression import RegressionDetector
from evaluation.scoring import EvaluationScore


class TestRegression(unittest.TestCase):
    def test_regression_detection(self):
        baseline = EvaluationScore(
            relevance=9, completeness=9, instruction_following=9,
            format_compliance=9, conciseness=9, factual_consistency=9,
        )
        # Drops significantly in instruction following and factual consistency
        degraded = EvaluationScore(
            relevance=7, completeness=6, instruction_following=5,
            format_compliance=8, conciseness=6, factual_consistency=5,
        )

        report = RegressionDetector.compare_evaluations(
            baseline_eval=baseline,
            current_eval=degraded,
            baseline_name="V1",
            current_name="V2",
        )

        self.assertTrue(report.is_regression)
        self.assertFalse(report.is_improvement)
        self.assertLess(report.overall_delta, -0.5)
        self.assertIn("Regression", report.status_label)
        self.assertIn("Instruction Following", report.regressed_criteria)
        self.assertIn("Factual Consistency", report.regressed_criteria)

    def test_improvement_detection(self):
        baseline = EvaluationScore(
            relevance=6, completeness=5, instruction_following=6,
            format_compliance=5, conciseness=5, factual_consistency=7,
        )
        improved = EvaluationScore(
            relevance=9, completeness=9, instruction_following=9,
            format_compliance=9, conciseness=8, factual_consistency=9,
        )

        report = RegressionDetector.compare_evaluations(
            baseline_eval=baseline,
            current_eval=improved,
            baseline_name="V1",
            current_name="V2",
        )

        self.assertTrue(report.is_improvement)
        self.assertFalse(report.is_regression)
        self.assertGreater(report.overall_delta, 0.5)
        self.assertIn("Improvement", report.status_label)


if __name__ == "__main__":
    unittest.main()
