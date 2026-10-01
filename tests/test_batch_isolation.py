"""Automated Data Isolation & Anti-Contamination Tests for PromptPilot Test Datasets."""

import unittest
import uuid
from typing import List
import streamlit as st

from config.settings import Settings
from database.db import DatabaseManager
from database.models import ExperimentRecord, FullExperimentDetail, PromptVersionRecord, ResponseRecord
from database.repository import ExperimentRepository
from evaluation.scoring import EvaluationScore
from llm.mock_provider import MockLLMProvider
from services.evaluation_service import EvaluationService, TestCase


class TestBatchIsolation(unittest.TestCase):
    """Verifies that test cases in batch datasets execute in complete data isolation."""

    def setUp(self):
        self.provider = MockLLMProvider()
        self.settings = Settings(gemini_api_key="mock-key-for-test", max_batch_test_cases=10)
        self.service = EvaluationService(provider=self.provider, settings=self.settings)

    def tearDown(self):
        # Clean any test session state pollution
        keys_to_clear = [
            "playground_input", "playground_prompt", "playground_task",
            "prefill_input", "prefill_prompt", "prefill_task_type",
            "active_experiment", "active_analysis", "last_batch_report",
            "source_data", "input_text", "current_experiment",
        ]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]

    def test_two_different_test_cases_isolated(self):
        """Test 1: Run two different test cases with completely different source texts.
        Verify that Case A's response/evaluation never contains Case B's source.
        """
        sec_source = (
            "Security Alert SEC-2026-041: A critical remote code execution vulnerability "
            "(CVE-2026-19342, CVSS v3.1 score 9.6) was discovered in the OpenGateway proxy library."
        )
        health_source = (
            "The Federal Health Authority released its final oversight framework for clinical machine learning algorithms. "
            "Effective January 2027, all Class II diagnostic aids must provide verifiable audit logs."
        )

        cases = [
            TestCase(
                id="sec-001",
                task_type="Information Extraction",
                name="Security Vulnerability Case",
                description="Security advisory testing",
                input_text=sec_source,
                dataset_id="iso-test-suite",
            ),
            TestCase(
                id="health-001",
                task_type="Summarization",
                name="Clinical AI Regulation Case",
                description="Health guideline testing",
                input_text=health_source,
                dataset_id="iso-test-suite",
            ),
        ]

        report = self.service.run_batch_evaluation(
            prompt_text="Summarize the core findings and action points from the source data.",
            test_cases=cases,
            dataset_id="iso-test-suite",
        )

        self.assertEqual(len(report.case_results), 2)
        sec_res = report.case_results[0]
        health_res = report.case_results[1]

        # Verify Case A (Security) contains security entities and NEVER healthcare entities
        self.assertEqual(sec_res.test_case_id, "sec-001")
        self.assertIn("cve-2026-19342", sec_res.response_text.lower())
        self.assertNotIn("health authority", sec_res.response_text.lower())
        self.assertNotIn("clinical machine learning", sec_res.response_text.lower())
        self.assertNotIn("class ii", sec_res.response_text.lower())

        for fb in sec_res.evaluation.feedback:
            self.assertNotIn("clinical", fb.lower())
            self.assertNotIn("health authority", fb.lower())

        # Verify Case B (Health) contains clinical entities and NEVER security/CVE entities
        self.assertEqual(health_res.test_case_id, "health-001")
        self.assertIn("health authority", health_res.response_text.lower())
        self.assertNotIn("cve-2026-19342", health_res.response_text.lower())
        self.assertNotIn("opengateway", health_res.response_text.lower())

        for fb in health_res.evaluation.feedback:
            self.assertNotIn("cve", fb.lower())
            self.assertNotIn("opengateway", fb.lower())

    def test_ev_followed_by_financial_isolated(self):
        """Test 2: Run the EV test case followed by the financial test case.
        Verify the financial test does not use EV content.
        """
        ev_source = (
            "Electric vehicles (EVs) have moved from being a niche technology to an increasingly mainstream "
            "choice in the global automotive industry. Growth is driven by policy targets, battery costs, and range."
        )
        fin_source = (
            "Apex Quantum Semiconductor (NASDAQ: APXQ) reported Q1 2026 fiscal revenues of $842 million, "
            "representing an 18% increase year-over-year. Net income expanded to $112 million."
        )

        cases = [
            TestCase(
                id="ev-case",
                task_type="Summarization",
                name="Electric Mobility Article",
                description="EV industry overview",
                input_text=ev_source,
                dataset_id="ev-fin-benchmark",
            ),
            TestCase(
                id="fin-case",
                task_type="Information Extraction",
                name="Quarterly Financial Disclosure",
                description="Semiconductor earnings",
                input_text=fin_source,
                dataset_id="ev-fin-benchmark",
            ),
        ]

        report = self.service.run_batch_evaluation(
            prompt_text="Extract core insights and takeaways strictly from the provided source data.",
            test_cases=cases,
            dataset_id="ev-fin-benchmark",
        )

        self.assertEqual(len(report.case_results), 2)
        ev_res = report.case_results[0]
        fin_res = report.case_results[1]

        # EV test case checks
        self.assertEqual(ev_res.test_case_id, "ev-case")
        self.assertTrue(any(w in ev_res.response_text.lower() for w in ["electric vehicle", "ev"]))

        # Financial test case checks — MUST NOT USE EV CONTENT
        self.assertEqual(fin_res.test_case_id, "fin-case")
        lower_fin_resp = fin_res.response_text.lower()
        self.assertNotIn("electric vehicle", lower_fin_resp)
        self.assertNotIn("battery pack", lower_fin_resp)
        self.assertNotIn("automotive", lower_fin_resp)
        self.assertIn("apex quantum", lower_fin_resp)
        self.assertIn("$842 million", lower_fin_resp)

        # Evaluator feedback for financial case must NOT reference electric vehicles
        for fb in fin_res.evaluation.feedback:
            self.assertNotIn("electric vehicle", fb.lower())
            self.assertNotIn("evs", fb.lower())
            self.assertNotIn("automotive", fb.lower())

        # Financial case must score high on factual consistency with its own source
        self.assertGreaterEqual(fin_res.evaluation.factual_consistency, 8)

    def test_dataset_rerun_isolation(self):
        """Test 3: Run the same dataset twice.
        Verify results are generated from the correct test case inputs each time and IDs remain unique.
        """
        cases = self.service.load_test_cases()
        self.assertGreaterEqual(len(cases), 3)

        # Run 1
        report_1 = self.service.run_batch_evaluation(
            prompt_text="Summarize the provided source data concisely.",
            test_cases=cases[:3],
            dataset_id="run-1",
        )

        # Run 2
        report_2 = self.service.run_batch_evaluation(
            prompt_text="Summarize the provided source data concisely.",
            test_cases=cases[:3],
            dataset_id="run-2",
        )

        self.assertEqual(report_1.num_cases, report_2.num_cases)

        # Verify IDs are unique across runs
        run1_resp_ids = {cr.response_id for cr in report_1.case_results}
        run2_resp_ids = {cr.response_id for cr in report_2.case_results}
        self.assertEqual(len(run1_resp_ids.intersection(run2_resp_ids)), 0)

        run1_eval_ids = {cr.evaluation_id for cr in report_1.case_results}
        run2_eval_ids = {cr.evaluation_id for cr in report_2.case_results}
        self.assertEqual(len(run1_eval_ids.intersection(run2_eval_ids)), 0)

        # Verify each case maps strictly to its own input in both runs
        for r1, r2, c in zip(report_1.case_results, report_2.case_results, cases[:3]):
            self.assertEqual(r1.test_case_id, c.test_case_id)
            self.assertEqual(r2.test_case_id, c.test_case_id)
            self.assertEqual(r1.source_data_preview[:40], c.source_data[:40])
            self.assertEqual(r2.source_data_preview[:40], c.source_data[:40])

    def test_playground_state_does_not_leak(self):
        """Test 4: Run Test Dataset after using Prompt Playground.
        Verify Playground data does not leak into Dataset execution.
        """
        # Simulate active Prompt Playground state with alien content
        st.session_state["playground_input"] = (
            "Ancient Roman architecture made extensive use of concrete and aqueducts to supply metropolitan centers."
        )
        st.session_state["playground_prompt"] = "Analyze Roman architecture engineering."
        st.session_state["playground_task"] = "History"
        st.session_state["source_data"] = "Polluted global source_data"
        st.session_state["input_text"] = "Polluted global input_text"

        sec_case = TestCase(
            id="sec-check",
            task_type="Information Extraction",
            name="Security Advisory",
            description="Testing advisory",
            input_text="Security Alert SEC-2026-041: CVE-2026-19342 OpenGateway vulnerability.",
        )

        report = self.service.run_batch_evaluation(
            prompt_text="Extract the vulnerability identifier and mitigation.",
            test_cases=[sec_case],
        )

        res = report.case_results[0]
        # Assert Playground content never leaked into batch execution
        self.assertNotIn("roman", res.response_text.lower())
        self.assertNotIn("aqueduct", res.response_text.lower())
        self.assertNotIn("polluted", res.response_text.lower())
        self.assertIn("cve-2026-19342", res.response_text.lower())

        for fb in res.evaluation.feedback:
            self.assertNotIn("roman", fb.lower())
            self.assertNotIn("polluted", fb.lower())

    def test_new_experiment_state_does_not_leak(self):
        """Test 5: Run Test Dataset after creating a New Experiment.
        Verify New Experiment data does not leak into Dataset execution.
        """
        # Simulate an active New Experiment session with EV source data
        ev_exp_source = "Electric vehicle battery gigafactories are expanding across North America."
        st.session_state["prefill_input"] = ev_exp_source
        st.session_state["prefill_prompt"] = "Summarize EV battery production"
        st.session_state["active_experiment"] = FullExperimentDetail(
            experiment=ExperimentRecord(
                id="exp-ev-123",
                task_type="Summarization",
                original_prompt="Summarize EV battery production",
                input_text=ev_exp_source,
                model_name="mock-model",
                created_at="2026-10-01T00:00:00Z",
            ),
            versions=[],
        )

        fin_case = TestCase(
            id="fin-check",
            task_type="Information Extraction",
            name="Quarterly Financials",
            description="Semiconductor earnings",
            input_text="Apex Quantum Semiconductor (NASDAQ: APXQ) reported Q1 2026 fiscal revenues of $842 million.",
        )

        report = self.service.run_batch_evaluation(
            prompt_text="Extract revenue and company name from the source text.",
            test_cases=[fin_case],
        )

        res = report.case_results[0]
        # Assert Experiment EV state never leaked into Financial batch case
        self.assertNotIn("electric vehicle", res.response_text.lower())
        self.assertNotIn("gigafactory", res.response_text.lower())
        self.assertNotIn("battery", res.response_text.lower())
        self.assertIn("apex quantum", res.response_text.lower())
        self.assertIn("$842 million", res.response_text.lower())

        for fb in res.evaluation.feedback:
            self.assertNotIn("gigafactory", fb.lower())
            self.assertNotIn("electric", fb.lower())

    def test_evaluator_penalizes_contaminated_response(self):
        """Verify that if a response discusses electric vehicles when the source document
        is a security advisory, the evaluator penalizes factual_consistency and relevance.
        """
        sec_source = "Security Alert SEC-2026-041: CVE-2026-19342 OpenGateway vulnerability."
        ev_contaminated_response = (
            "Electric vehicles (EVs) have moved from being a niche technology to mainstream adoption, "
            "reducing carbon emissions through advanced battery chemistries."
        )

        score, _ = self.service.evaluator.evaluate(
            task_type="Information Extraction",
            prompt_text="Summarize this security advisory.",
            input_text=sec_source,
            response_text=ev_contaminated_response,
        )

        # Must be severely penalized for topic contamination / hallucination
        self.assertLessEqual(score.factual_consistency, 4)
        self.assertLessEqual(score.relevance, 4)
        self.assertLess(score.overall_score, 5.0)

        # Feedback must explain the topic mismatch
        feedback_str = " ".join(score.feedback).lower()
        self.assertTrue(
            "electric vehicles" in feedback_str or "ungrounded" in feedback_str or "absent" in feedback_str
        )


if __name__ == "__main__":
    unittest.main()
