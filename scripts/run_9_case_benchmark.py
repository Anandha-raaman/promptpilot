"""Execute the full 9-case benchmark and print detailed metrics table."""

from database.db import DatabaseManager
from database.repository import ExperimentRepository
from llm.mock_provider import MockLLMProvider
from services.evaluation_service import EvaluationService


def run_benchmark():
    db = DatabaseManager()
    repo = ExperimentRepository(db)
    provider = MockLLMProvider()
    eval_service = EvaluationService(provider=provider, repository=repo)

    cases = eval_service.load_test_cases(dataset_id="default-benchmark")
    print(f"Loaded {len(cases)} test cases from default-benchmark:")
    for c in cases:
        print(f" - [{c.id}] {c.name} ({c.task_type})")

    report = eval_service.run_batch_evaluation(
        prompt_text=None,  # Each case runs with its own prompt
        test_cases=cases,
        dataset_id="default-benchmark",
    )

    print("\n" + "=" * 110)
    print("PROMPTPILOT COMPLETE 9-CASE BENCHMARK RESULTS")
    print("=" * 110)
    print(f"Dataset ID: {report.dataset_id}")
    print(f"Tests Run: {report.num_cases}")
    print(f"Mean Composite Score: {report.avg_overall_score:.2f} / 10.0")
    print(f"Mean Latency: {(report.avg_latency_ms / 1000.0):.2f}s")
    print("-" * 110)

    # Print markdown table format for final report
    headers = [
        "Test Case ID",
        "Task",
        "Overall Score",
        "Relevance",
        "Completeness",
        "Instruction Following",
        "Factual Consistency",
        "Format Compliance",
        "Conciseness",
        "Latency",
    ]
    print(" | ".join(headers))
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for cr in report.case_results:
        ev = cr.evaluation
        row = [
            cr.test_case_id,
            cr.task_type,
            f"{ev.overall_score:.1f}",
            str(ev.relevance),
            str(ev.completeness),
            str(ev.instruction_following),
            str(ev.effective_factual_consistency),
            str(ev.effective_format_compliance),
            str(ev.effective_conciseness),
            f"{(cr.latency_ms / 1000.0):.2f}s",
        ]
        print(" | ".join(row))

    print("=" * 110)


if __name__ == "__main__":
    run_benchmark()
