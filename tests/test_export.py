"""Unit tests for sanitized export utilities."""

import json
import unittest
from database.models import (
    EvaluationRecord,
    ExperimentRecord,
    FullExperimentDetail,
    PromptVersionRecord,
    ResponseRecord,
)
from utils.export import ExperimentExporter


class TestExport(unittest.TestCase):
    def setUp(self):
        exp = ExperimentRecord(
            id="exp-123",
            task_type="Summarization",
            original_prompt="Summarize the report.",
            input_text="Financial performance exceeded expectations.",
            model_name="test-model",
            created_at="2026-09-30T12:00:00Z",
            title="Q3 Report",
        )
        ev = EvaluationRecord(
            id="ev-1",
            response_id="resp-1",
            relevance=9,
            completeness=8,
            instruction_following=9,
            format_compliance=9,
            conciseness=8,
            factual_consistency=10,
            overall_score=8.8,
            feedback=["Good conciseness"],
            strengths=["Faithful"],
            created_at="2026-09-30T12:01:00Z",
        )
        resp = ResponseRecord(
            id="resp-1",
            prompt_version_id="ver-1",
            response_text="Executive summary of finances.",
            latency_ms=180.0,
            prompt_tokens=40,
            completion_tokens=20,
            total_tokens=60,
            evaluation=ev,
        )
        ver = PromptVersionRecord(
            id="ver-1",
            experiment_id="exp-123",
            version_number=0,
            strategy="Original (Baseline)",
            prompt_text="Summarize the report.",
            expected_benefit="Baseline",
            potential_limitations="Generic",
            response=resp,
        )
        self.detail = FullExperimentDetail(experiment=exp, versions=[ver])

    def test_json_export(self):
        json_str = ExperimentExporter.to_json(self.detail)
        data = json.loads(json_str)
        self.assertEqual(data["experiment"]["id"], "exp-123")
        self.assertEqual(len(data["versions"]), 1)
        self.assertNotIn("API_KEY", json_str)
        self.assertNotIn("GEMINI", json_str)

    def test_csv_export(self):
        csv_str = ExperimentExporter.to_csv(self.detail)
        self.assertIn("experiment_id,task_type", csv_str)
        self.assertIn("exp-123", csv_str)
        self.assertIn("8.8", csv_str)

    def test_markdown_export(self):
        md_str = ExperimentExporter.to_markdown(self.detail)
        self.assertIn("# PromptPilot Experiment Report", md_str)
        self.assertIn("Q3 Report", md_str)
        self.assertIn("| V0 | Original (Baseline) | **8.8** |", md_str)


if __name__ == "__main__":
    unittest.main()
