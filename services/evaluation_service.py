"""Batch Evaluation and Regression Testing Service for PromptPilot."""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.settings import Settings, get_settings
from evaluation.regression import RegressionDetector, VersionComparisonReport
from evaluation.scoring import EvaluationScore
from llm.base import BaseLLMProvider
from prompts.evaluator import ResponseEvaluator
from utils.logging import get_logger

logger = get_logger("evaluation_service")


@dataclass
class TestCase:
    """Standardized test case with strict source data isolation and traceability."""
    id: str
    task_type: str
    name: str
    description: str = ""
    input_text: str = ""
    dataset_id: str = "default-benchmark"
    expected_output: Optional[str] = None
    criteria: Optional[str] = None
    prompt: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    ground_truth_key_points: Optional[List[str]] = None
    ground_truth_entities: Optional[Dict[str, Any]] = None

    @property
    def test_case_id(self) -> str:
        """Alias for id ensuring explicit test_case_id interface."""
        return self.id

    @property
    def source_data(self) -> str:
        """Alias for input_text ensuring explicit source_data interface."""
        return self.input_text


@dataclass
class BatchCaseResult:
    """Individual execution and evaluation result for a test case."""
    test_case_id: str
    test_case_name: str
    response_text: str
    latency_ms: float
    evaluation: EvaluationScore
    dataset_id: str = "default-benchmark"
    prompt_version_id: Optional[str] = None
    response_id: Optional[str] = None
    evaluation_id: Optional[str] = None
    source_data_preview: str = ""
    prompt_text: str = ""
    criteria: Optional[str] = None
    task_type: str = ""


@dataclass
class BatchEvaluationReport:
    prompt_text: str
    task_type: str
    num_cases: int
    case_results: List[BatchCaseResult]
    avg_relevance: float
    avg_completeness: float
    avg_instruction_following: float
    avg_format_compliance: float
    avg_conciseness: float
    avg_factual_consistency: float
    avg_overall_score: float
    avg_latency_ms: float
    dataset_id: str = "default-benchmark"

    def to_evaluation_score(self) -> EvaluationScore:
        """Convert aggregate averages into an EvaluationScore object for regression comparison."""
        return EvaluationScore(
            relevance=int(round(self.avg_relevance)),
            completeness=int(round(self.avg_completeness)),
            instruction_following=int(round(self.avg_instruction_following)),
            format_compliance=int(round(self.avg_format_compliance)),
            conciseness=int(round(self.avg_conciseness)),
            factual_consistency=int(round(self.avg_factual_consistency)),
            feedback=[f"Batch evaluated across {self.num_cases} test cases in dataset '{self.dataset_id}'. Composite score: {self.avg_overall_score}"],
        )


class EvaluationService:
    """Manages test dataset loading, batch test runs, and cross-dataset regression reporting."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        dataset_path: Optional[Path] = None,
        settings: Optional[Settings] = None,
        repository: Optional[Any] = None,
    ):
        self.provider = provider
        self.settings = settings or get_settings()
        self.dataset_path = dataset_path or (Path(__file__).resolve().parent.parent / "data" / "test_cases.json")
        self.evaluator = ResponseEvaluator(provider=self.provider, settings=self.settings)
        self.repository = repository

    def list_datasets(self) -> List[Any]:
        """List all datasets from repository or return default dataset."""
        if self.repository and hasattr(self.repository, "list_datasets"):
            try:
                ds_list = self.repository.list_datasets()
                if ds_list:
                    return ds_list
            except Exception as e:
                logger.warning("Error fetching datasets from repository: %s", e)
        # Default fallback
        from database.models import DatasetRecord
        return [
            DatasetRecord(
                id="default-benchmark",
                name="Default Benchmark Dataset",
                description="Standard enterprise benchmark cases covering summarization, extraction, and generation.",
                case_count=6,
            )
        ]

    def create_dataset(self, name: str, description: str = "", dataset_id: Optional[str] = None) -> str:
        """Create a new dataset container."""
        if self.repository and hasattr(self.repository, "create_dataset"):
            return self.repository.create_dataset(name=name, description=description, dataset_id=dataset_id)
        return dataset_id or f"ds-{uuid.uuid4().hex[:8]}"

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and its test cases."""
        if self.repository and hasattr(self.repository, "delete_dataset"):
            return self.repository.delete_dataset(dataset_id)
        return False

    def load_test_cases(self, task_type: Optional[str] = None, dataset_id: str = "default-benchmark") -> List[TestCase]:
        """Load test cases from repository (SQLite) or fallback to local JSON dataset."""
        if self.repository and hasattr(self.repository, "get_test_cases"):
            try:
                cases = self.repository.get_test_cases(dataset_id=dataset_id, task_type=task_type)
                if cases:
                    return cases
            except Exception as e:
                logger.warning("Failed to load test cases from repository: %s", e)

        if not self.dataset_path.exists():
            logger.warning("Test cases file not found at: %s", self.dataset_path)
            return []

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)

        cases: List[TestCase] = []
        for c in raw_cases:
            case_ds = c.get("dataset_id", "default-benchmark")
            if dataset_id and case_ds != dataset_id:
                continue
            if task_type and c.get("task_type", "").lower() != task_type.lower():
                continue
            cases.append(
                TestCase(
                    id=c["id"],
                    task_type=c["task_type"],
                    name=c["name"],
                    description=c.get("description", ""),
                    input_text=c["input_text"],
                    dataset_id=case_ds,
                    expected_output=c.get("expected_output"),
                    criteria=c.get("criteria"),
                    prompt=c.get("prompt"),
                    tags=c.get("tags", []),
                    ground_truth_key_points=c.get("ground_truth_key_points"),
                    ground_truth_entities=c.get("ground_truth_entities"),
                )
            )
        return cases

    def save_test_case(self, case: TestCase) -> str:
        """Persist a test case (insert or update) in repository."""
        if self.repository and hasattr(self.repository, "save_test_case"):
            return self.repository.save_test_case(case)
        return case.id

    def delete_test_case(self, case_id: str) -> bool:
        """Delete an individual test case."""
        if self.repository and hasattr(self.repository, "delete_test_case"):
            return self.repository.delete_test_case(case_id)
        return False

    def run_single_test_case(
        self,
        case: TestCase,
        prompt_override: Optional[str] = None,
        prompt_version_id: Optional[str] = None,
    ) -> BatchCaseResult:
        """Execute and evaluate a single test case independently with strict data isolation.

        Flow:
            case.prompt (or prompt_override) + case.source_data -> LLM -> Response -> Evaluator -> BatchCaseResult
        """
        assert case.test_case_id is not None, "TestCase has missing test_case_id"
        assert case.source_data is not None, f"TestCase '{case.test_case_id}' has missing source_data"
        assert len(case.source_data.strip()) > 0, f"TestCase '{case.test_case_id}' has empty source_data"

        # Prompt resolution: explicit prompt override (e.g. benchmarking custom prompt) else case's stored prompt
        case_prompt = (
            prompt_override.strip()
            if prompt_override and prompt_override.strip()
            else (case.prompt.strip() if case.prompt and case.prompt.strip() else "Summarize the key information from the provided source data.")
        )
        user_content = f"{case_prompt}\n\n<SOURCE_DATA>\n{case.source_data}\n</SOURCE_DATA>"

        case_dataset_id = case.dataset_id or "default-benchmark"
        case_resp_id = f"resp-{uuid.uuid4()}"
        case_eval_id = f"eval-{uuid.uuid4()}"
        case_ver_id = prompt_version_id or f"pv-{uuid.uuid4()}"

        logger.info(
            "Executing isolated test case: id=%s, dataset=%s, name=%s (source_chars=%d)",
            case.test_case_id,
            case_dataset_id,
            case.name,
            len(case.source_data),
        )

        resp = self.provider.generate_text(
            prompt=user_content,
            model=self.settings.response_model,
            temperature=0.2,
        )

        custom_criteria = case.criteria
        if not custom_criteria and case.ground_truth_key_points:
            custom_criteria = "Key points to cover: " + "; ".join(case.ground_truth_key_points)

        eval_score, eval_llm_resp = self.evaluator.evaluate(
            task_type=case.task_type,
            prompt_text=case_prompt,
            input_text=case.source_data,
            response_text=resp.content,
            custom_criteria=custom_criteria,
        )

        # DEVELOPMENT-ONLY diagnostic logging (traceable pipeline)
        logger.info(
            "\n" + "=" * 60 + "\n"
            "[EVALUATION DIAGNOSTIC PIPELINE]\n"
            "TEST_CASE_ID: %s (%s)\n"
            "PROMPT_ID: %s\n"
            "RESPONSE_ID: %s\n"
            "EVALUATION_ID: %s\n"
            "1. EVALUATED_RESPONSE: %s\n"
            "2. SOURCE_DATA: %s\n"
            "3. EVALUATOR_PROMPT_SENT: Prompt='%s' | Task='%s' | Criteria='%s'\n"
            "4. RAW_EVALUATOR_OUTPUT: %s\n"
            "5. PARSED_EVALUATION: Rel=%s, Comp=%s, Inst=%s, Fact=%s, Fmt=%s, Conc=%s, Available=%s\n"
            "6. SCORING_CALCULATION: Formula Weighted Composite = %s\n"
            "7. FINAL_SCORE: %s / 10.0\n"
            + "=" * 60,
            case.test_case_id,
            case.name,
            case_ver_id,
            case_resp_id,
            case_eval_id,
            resp.content.strip().replace("\n", " ")[:160] + "...",
            case.source_data.strip().replace("\n", " ")[:140] + "...",
            case_prompt,
            case.task_type,
            custom_criteria or "None",
            eval_llm_resp.content.strip().replace("\n", " ")[:200] + "...",
            eval_score.relevance,
            eval_score.completeness,
            eval_score.instruction_following,
            eval_score.effective_factual_consistency,
            eval_score.effective_format_compliance,
            eval_score.effective_conciseness,
            eval_score.is_available,
            eval_score.overall_score,
            eval_score.overall_score,
        )

        cr = BatchCaseResult(
            test_case_id=case.test_case_id,
            test_case_name=case.name,
            response_text=resp.content,
            latency_ms=resp.latency_ms,
            evaluation=eval_score,
            dataset_id=case_dataset_id,
            prompt_version_id=case_ver_id,
            response_id=case_resp_id,
            evaluation_id=case_eval_id,
            source_data_preview=case.source_data[:120] + ("..." if len(case.source_data) > 120 else ""),
            prompt_text=case_prompt,
            criteria=custom_criteria,
            task_type=case.task_type,
        )

        if self.repository and hasattr(self.repository, "save_case_result"):
            try:
                self.repository.save_case_result(cr)
            except Exception as e:
                logger.warning("Failed to auto-persist case result to repository: %s", e)

        return cr

    def run_batch_evaluation(
        self,
        prompt_text: Optional[str] = None,
        test_cases: Optional[List[TestCase]] = None,
        dataset_id: Optional[str] = None,
        prompt_version_id: Optional[str] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchEvaluationReport:
        """Run a test dataset batch with strict data isolation per case.

        Each test case is executed independently:
            test_case.prompt (or prompt_text) + test_case.source_data -> LLM -> Response -> Evaluator -> BatchCaseResult
        """
        cases_to_run = test_cases or self.load_test_cases(dataset_id=dataset_id or "default-benchmark")
        if not cases_to_run:
            raise ValueError("At least one test case must be provided for batch evaluation.")

        clamped_cases = cases_to_run[: self.settings.max_batch_test_cases]
        case_results: List[BatchCaseResult] = []
        total_steps = len(clamped_cases)
        active_dataset_id = dataset_id or (clamped_cases[0].dataset_id if clamped_cases else "default-benchmark")

        for idx, case in enumerate(clamped_cases, start=1):
            if progress_cb:
                progress_cb(idx, total_steps, f"Evaluating case {idx}/{total_steps}: {case.name}")

            cr = self.run_single_test_case(
                case=case,
                prompt_override=prompt_text,
                prompt_version_id=prompt_version_id,
            )
            case_results.append(cr)


        # Compute aggregate averages
        n = len(case_results)
        available_results = [r for r in case_results if r.evaluation.is_available]
        if available_results:
            n_avail = len(available_results)
            avg_rel = round(sum(r.evaluation.relevance for r in available_results) / n_avail, 2)
            avg_comp = round(sum(r.evaluation.completeness for r in available_results) / n_avail, 2)
            avg_inst = round(sum(r.evaluation.instruction_following for r in available_results) / n_avail, 2)
            avg_fmt = round(sum(r.evaluation.effective_format_compliance for r in available_results) / n_avail, 2)
            avg_conc = round(sum(r.evaluation.effective_conciseness for r in available_results) / n_avail, 2)
            avg_fact = round(sum(r.evaluation.effective_factual_consistency for r in available_results) / n_avail, 2)
            avg_overall = round(sum(r.evaluation.overall_score for r in available_results) / n_avail, 2)
        else:
            avg_rel = avg_comp = avg_inst = avg_fmt = avg_conc = avg_fact = avg_overall = 0.0

        avg_lat = round(sum(r.latency_ms for r in case_results) / n, 1)

        task_type = clamped_cases[0].task_type if clamped_cases else "General"

        return BatchEvaluationReport(
            prompt_text=prompt_text,
            task_type=task_type,
            num_cases=n,
            case_results=case_results,
            avg_relevance=avg_rel,
            avg_completeness=avg_comp,
            avg_instruction_following=avg_inst,
            avg_format_compliance=avg_fmt,
            avg_conciseness=avg_conc,
            avg_factual_consistency=avg_fact,
            avg_overall_score=avg_overall,
            avg_latency_ms=avg_lat,
            dataset_id=active_dataset_id,
        )

    def save_batch_report_to_db(self, report: BatchEvaluationReport) -> Optional[str]:
        """Persist batch report and all test case results with distinct relational IDs into database."""
        if not self.repository:
            return None

        exp_id = self.repository.create_experiment(
            task_type=report.task_type,
            original_prompt=report.prompt_text,
            input_text=f"Batch Dataset: {report.dataset_id} ({report.num_cases} isolated test cases)",
            model_name=self.settings.response_model,
            title=f"Batch Benchmark: {report.dataset_id}",
        )

        for idx, cr in enumerate(report.case_results, start=1):
            pv_id = self.repository.add_prompt_version(
                experiment_id=exp_id,
                version_number=idx,
                strategy=f"TestCase: {cr.test_case_name}",
                prompt_text=cr.prompt_text,
                expected_benefit=f"TestCase ID: {cr.test_case_id}",
                version_id=cr.prompt_version_id,
            )
            resp_id = self.repository.add_response(
                prompt_version_id=pv_id,
                response_text=cr.response_text,
                latency_ms=cr.latency_ms,
                response_id=cr.response_id,
            )
            self.repository.add_evaluation(
                response_id=resp_id,
                relevance=cr.evaluation.relevance,
                completeness=cr.evaluation.completeness,
                instruction_following=cr.evaluation.instruction_following,
                format_compliance=cr.evaluation.format_compliance,
                conciseness=cr.evaluation.conciseness,
                factual_consistency=cr.evaluation.factual_consistency,
                overall_score=cr.evaluation.overall_score,
                feedback=cr.evaluation.feedback,
                strengths=cr.evaluation.strengths,
                evaluation_id=cr.evaluation_id,
            )

        return exp_id

    @staticmethod
    def detect_batch_regression(
        baseline_report: BatchEvaluationReport,
        current_report: BatchEvaluationReport,
        baseline_label: str = "Baseline Prompt",
        current_label: str = "Modified Prompt",
    ) -> VersionComparisonReport:
        """Compare batch evaluation results between two prompt iterations to detect regressions."""
        return RegressionDetector.compare_evaluations(
            baseline_eval=baseline_report.to_evaluation_score(),
            current_eval=current_report.to_evaluation_score(),
            baseline_name=baseline_label,
            current_name=current_label,
            baseline_latency=baseline_report.avg_latency_ms,
            current_latency=current_report.avg_latency_ms,
        )
