"""Unit and Integration Tests for Test Dataset Management and Isolated Execution."""

import tempfile
import unittest
from pathlib import Path

from config.settings import Settings
from database.db import DatabaseManager
from database.repository import ExperimentRepository
from llm.mock_provider import MockLLMProvider
from services.evaluation_service import EvaluationService, TestCase


class TestDatasetFeatureSuite(unittest.TestCase):
    """Test suite verifying test case definition, independent execution, and result tracking."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_dataset_feature.db"
        self.settings = Settings(database_path=str(self.db_path))
        self.db_manager = DatabaseManager(db_path=self.db_path, settings=self.settings)
        self.repo = ExperimentRepository(self.db_manager)
        self.provider = MockLLMProvider()
        self.eval_service = EvaluationService(
            provider=self.provider,
            settings=self.settings,
            repository=self.repo,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_complete_test_case_definition_and_isolation(self):
        # 1. Create a custom dataset
        dataset_id = "validation-suite-01"
        self.repo.create_dataset(
            name="Validation Benchmark Suite",
            description="Verification suite for test case definition and isolated execution.",
            dataset_id=dataset_id,
        )

        datasets = self.repo.list_datasets()
        ds_ids = [d.id for d in datasets]
        self.assertIn("default-benchmark", ds_ids)
        self.assertIn(dataset_id, ds_ids)

        # 2. Define 3 distinct test cases: Good, Poor/Incomplete, Different Subject
        case1 = TestCase(
            id="val-good-01",
            dataset_id=dataset_id,
            name="Semiconductor Earnings Summary",
            task_type="Summarization",
            prompt="Summarize the financial highlights, revenue growth, and quarterly projections from the semiconductor report.",
            input_text=(
                "Apex Quantum Semiconductor (NASDAQ: APXQ) reported Q1 2026 fiscal revenues of $842 million, "
                "representing an 18% increase year-over-year. Net income expanded to $112 million, compared to "
                "$85 million in Q1 2025. CEO David Sterling cited heightened demand for cryogenic quantum control chips. "
                "For Q2 2026, CFO Maria Santos projected revenues between $890M and $915M."
            ),
            criteria="Must highlight the $842 million revenue (+18% YoY), $112 million net income, CEO David Sterling, and Q2 projection of $890M-$915M.",
            expected_output="Apex Quantum Semiconductor reported Q1 2026 revenue of $842M (+18% YoY) and net income of $112M. CFO Maria Santos projected Q2 revenues of $890M-$915M.",
            tags=["finance", "earnings", "semiconductor"],
        )

        case2 = TestCase(
            id="val-poor-02",
            dataset_id=dataset_id,
            name="Incomplete Cloud Outage Post-Mortem",
            task_type="Summarization",
            prompt="Produce an executive summary covering all five distinct failure root causes, total customer downtime duration, financial SLA penalties, and remediation steps.",
            input_text=(
                "On July 14, CloudStack experienced an outage lasting 4 hours and 12 minutes, resulting in $650,000 "
                "in customer SLA credits. Root causes included: (1) DNS zone corruption, (2) failure of automated DB failover, "
                "(3) expired internal mTLS certificates, (4) cascade queue saturation in message brokers, and "
                "(5) missing rate limiters on webhook retry workers. Remediation mandates immediate certificate auto-rotation."
            ),
            criteria="Must cover all five distinct root causes: DNS corruption, DB failover, expired mTLS certs, message broker queue saturation, and missing rate limiters, plus 4h 12m duration.",
            expected_output="CloudStack experienced a 4h 12m outage causing $650,000 in SLA credits. Root causes: DNS corruption, DB failover failure, expired mTLS certs, queue saturation, missing rate limiters.",
            tags=["cloud", "incident", "incomplete-test"],
        )

        case3 = TestCase(
            id="val-diff-03",
            dataset_id=dataset_id,
            name="Clinical AI Healthcare Regulatory Notice",
            task_type="Information Extraction",
            prompt="Extract the oversight effective date, audit requirements for Class II devices, and prospective trial requirements for emergency triage AI.",
            input_text=(
                "The Federal Health Authority released its final oversight framework for clinical machine learning algorithms. "
                "Effective January 2027, all Class II diagnostic aids must provide verifiable audit logs of training data distributions "
                "and demonstrate demographic parity across patient cohorts. Developers must submit post-market surveillance reports annually. "
                "Algorithms deployed in emergency triage settings are categorized under high-risk Class III and require randomized prospective clinical validation."
            ),
            criteria="Must extract: effective date January 2027, Class II training data audit logs and demographic parity, annual surveillance reports, and Class III emergency triage prospective clinical trials.",
            expected_output="Effective Date: January 2027\nClass II: Audit logs, demographic parity\nClass III (Triage): Prospective clinical trials",
            tags=["healthcare", "fda", "clinical-ai", "regulatory"],
        )

        # 3. Save test cases to repository
        self.eval_service.save_test_case(case1)
        self.eval_service.save_test_case(case2)
        self.eval_service.save_test_case(case3)

        # Verify test cases are retrieved correctly
        loaded_cases = self.eval_service.load_test_cases(dataset_id=dataset_id)
        self.assertEqual(len(loaded_cases), 3)
        self.assertEqual({c.id for c in loaded_cases}, {"val-good-01", "val-poor-02", "val-diff-03"})

        # 4. Verify individual test case execution
        res1 = self.eval_service.run_single_test_case(case1)
        self.assertEqual(res1.test_case_id, "val-good-01")
        self.assertEqual(res1.test_case_name, "Semiconductor Earnings Summary")
        self.assertIn("APXQ", res1.response_text + res1.source_data_preview)
        self.assertNotIn("CloudStack", res1.response_text)
        self.assertNotIn("Federal Health Authority", res1.response_text)
        self.assertNotEqual(res1.evaluation.overall_score, 10.0, "Score should not be hard-coded 10.0")

        res2 = self.eval_service.run_single_test_case(case2)
        self.assertEqual(res2.test_case_id, "val-poor-02")
        self.assertEqual(res2.test_case_name, "Incomplete Cloud Outage Post-Mortem")
        self.assertIn("CloudStack", res2.source_data_preview)
        self.assertNotIn("Apex Quantum", res2.response_text)
        self.assertNotEqual(res2.evaluation.overall_score, 10.0, "Score should not be hard-coded 10.0")

        res3 = self.eval_service.run_single_test_case(case3)
        self.assertEqual(res3.test_case_id, "val-diff-03")
        self.assertEqual(res3.test_case_name, "Clinical AI Healthcare Regulatory Notice")
        self.assertIn("Federal Health Authority", res3.source_data_preview)
        self.assertNotIn("APXQ", res3.response_text)

        # 5. Verify complete dataset execution (batch run)
        batch_report = self.eval_service.run_batch_evaluation(
            prompt_text=None,  # Each case must use its own prompt!
            dataset_id=dataset_id,
        )
        self.assertEqual(batch_report.num_cases, 3)
        self.assertEqual(batch_report.dataset_id, dataset_id)

        # Verify results linking
        case_ids_in_batch = [cr.test_case_id for cr in batch_report.case_results]
        self.assertEqual(case_ids_in_batch, ["val-good-01", "val-poor-02", "val-diff-03"])

        # Check that results are stored in SQLite dataset_case_results table
        db_results = self.repo.get_latest_case_results(dataset_id=dataset_id)
        db_case_ids = {r["test_case_id"] for r in db_results}
        self.assertIn("val-good-01", db_case_ids)
        self.assertIn("val-poor-02", db_case_ids)
        self.assertIn("val-diff-03", db_case_ids)

        # Verify each case used its own source data and prompt
        for cr in batch_report.case_results:
            self.assertTrue(len(cr.prompt_text) > 0)
            self.assertTrue(len(cr.source_data_preview) > 0)
            self.assertTrue(cr.evaluation.is_available)
            # Verify score is not hard-coded 10
            self.assertNotEqual(cr.evaluation.overall_score, 10.0)

        # 6. Verify Edit and Delete
        case1.name = "Updated Semiconductor Case"
        self.eval_service.save_test_case(case1)
        updated_case = self.repo.get_test_case("val-good-01")
        self.assertEqual(updated_case.name, "Updated Semiconductor Case")

        del_ok = self.eval_service.delete_test_case("val-poor-02")
        self.assertTrue(del_ok)
        cases_after_del = self.eval_service.load_test_cases(dataset_id=dataset_id)
        self.assertEqual(len(cases_after_del), 2)
        self.assertNotIn("val-poor-02", [c.id for c in cases_after_del])


if __name__ == "__main__":
    unittest.main()
