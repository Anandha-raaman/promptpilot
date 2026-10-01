"""Script to run and verify the 8-test benchmark in default-benchmark."""

import sqlite3
from database.db import DatabaseManager
from database.repository import ExperimentRepository
from llm.mock_provider import MockLLMProvider
from services.evaluation_service import EvaluationService


def main():
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

    print("\n" + "=" * 70)
    print("8-TEST BENCHMARK EXECUTION RESULTS")
    print("=" * 70)
    print(f"Dataset ID: {report.dataset_id}")
    print(f"Tests Run: {report.num_cases}")
    print(f"Mean Composite Score: {report.avg_overall_score:.2f} / 10.0")
    print(f"Mean Latency: {(report.avg_latency_ms / 1000.0):.2f}s")
    print(f"Average Factual Consistency: {report.avg_factual_consistency:.2f} / 10.0")
    print(f"Average Relevance: {report.avg_relevance:.2f} / 10.0")
    print(f"Average Completeness: {report.avg_completeness:.2f} / 10.0")
    print(f"Average Instruction Following: {report.avg_instruction_following:.2f} / 10.0")
    print("-" * 70)

    for cr in report.case_results:
        ev = cr.evaluation
        print(f"Case: {cr.test_case_id:12} | Name: {cr.test_case_name[:35]:35} | Score: {ev.overall_score:4.1f}/10 | Fact: {ev.effective_factual_consistency:2}/10 | Rel: {ev.relevance:2}/10 | Comp: {ev.completeness:2}/10")
        if ev.effective_factual_consistency < 9:
            print(f"   -> Factual Consistency Note: {ev.feedback}")

    print("=" * 70)


if __name__ == "__main__":
    main()
