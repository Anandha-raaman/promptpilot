"""Unit tests for SQLite database connection and repository."""

import tempfile
import unittest
from pathlib import Path

from database.db import DatabaseManager
from database.repository import ExperimentRepository


class TestExperimentRepository(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.repo = ExperimentRepository(self.db_manager)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_complete_crud_lifecycle(self):
        # 1. Create experiment
        exp_id = self.repo.create_experiment(
            task_type="Summarization",
            original_prompt="Summarize the quarterly report.",
            input_text="Revenue grew by 20% in Q3.",
            model_name="test-model",
            title="Q3 Summary Test",
        )
        self.assertIsNotNone(exp_id)

        # 2. Add prompt version
        v_id = self.repo.add_prompt_version(
            experiment_id=exp_id,
            version_number=1,
            strategy="Role-Based",
            prompt_text="You are a financial analyst. Summarize...",
            expected_benefit="Domain focus",
            potential_limitations="Too formal",
        )
        self.assertIsNotNone(v_id)

        # 3. Add response
        r_id = self.repo.add_response(
            prompt_version_id=v_id,
            response_text="Executive Summary: Q3 revenue showed 20% growth.",
            latency_ms=250.5,
            prompt_tokens=30,
            completion_tokens=15,
            total_tokens=45,
        )
        self.assertIsNotNone(r_id)

        # 4. Add evaluation
        e_id = self.repo.add_evaluation(
            response_id=r_id,
            relevance=9,
            completeness=8,
            instruction_following=9,
            format_compliance=9,
            conciseness=8,
            factual_consistency=10,
            overall_score=8.8,
            feedback=["Good focus on revenue metric"],
            strengths=["Faithful to source document"],
        )
        self.assertIsNotNone(e_id)

        # 5. Fetch full experiment tree
        detail = self.repo.get_experiment(exp_id)
        self.assertIsNotNone(detail)
        self.assertEqual(detail.experiment.title, "Q3 Summary Test")
        self.assertEqual(len(detail.versions), 1)

        v_record = detail.versions[0]
        self.assertEqual(v_record.strategy, "Role-Based")
        self.assertIsNotNone(v_record.response)
        self.assertEqual(v_record.response.latency_ms, 250.5)
        self.assertIsNotNone(v_record.response.evaluation)
        self.assertEqual(v_record.response.evaluation.overall_score, 8.8)
        self.assertEqual(v_record.response.evaluation.factual_consistency, 10)

        # 6. List experiments
        exp_list = self.repo.list_experiments()
        self.assertEqual(len(exp_list), 1)

        # 7. Delete experiment with cascading delete
        deleted = self.repo.delete_experiment(exp_id)
        self.assertTrue(deleted)
        self.assertIsNone(self.repo.get_experiment(exp_id))

        # Check cascading cleanup
        with self.db_manager.get_connection() as conn:
            v_count = conn.execute("SELECT COUNT(*) FROM prompt_versions WHERE experiment_id = ?", (exp_id,)).fetchone()[0]
            r_count = conn.execute("SELECT COUNT(*) FROM responses WHERE id = ?", (r_id,)).fetchone()[0]
            e_count = conn.execute("SELECT COUNT(*) FROM evaluations WHERE id = ?", (e_id,)).fetchone()[0]
            self.assertEqual(v_count, 0)
            self.assertEqual(r_count, 0)
            self.assertEqual(e_count, 0)


if __name__ == "__main__":
    unittest.main()
