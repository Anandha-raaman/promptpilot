"""Execute the exact user test case and print full Prompt Health vs Response Quality metrics."""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from database.db import DatabaseManager
from database.repository import ExperimentRepository
from llm.gemini_provider import GeminiProvider
from llm.mock_provider import MockLLMProvider
from services.experiment_service import ExperimentService

def main():
    settings = get_settings()
    db_manager = DatabaseManager(settings=settings)
    repo = ExperimentRepository(db_manager)

    if settings.gemini_api_key:
        provider = GeminiProvider(settings=settings)
    else:
        provider = MockLLMProvider()

    print(f"Provider: {type(provider).__name__}")
    service = ExperimentService(provider=provider, repository=repo, settings=settings)

    test_prompt = "summarize this"
    test_article = (
        "Artificial intelligence is changing businesses across various sectors, including automation of routine operations, "
        "advanced data analysis, intelligent 24/7 customer service, and data-driven executive decision-making. "
        "However, organizations face major implementation challenges, including cybersecurity vulnerabilities, "
        "the necessity for responsible AI frameworks, and the critical importance of continuous human oversight."
    )

    print("\n" + "=" * 60)
    print("STEP 1: BASELINE PROMPT HEALTH AUDIT")
    print("=" * 60)

    analysis, _ = service.analyzer.analyze(
        user_prompt=test_prompt,
        task_type="Summarization",
        has_input_context=True,
        input_context_summary="Enterprise AI adoption article",
    )

    print(f"Overall Prompt Health Score: {analysis.overall_health_score}/10")
    print(f"Dimension Scores:")
    print(f"  - Clarity: {analysis.clarity_score}/10")
    print(f"  - Specificity: {analysis.specificity_score}/10")
    print(f"  - Format: {analysis.output_format_score}/10")
    print(f"  - Constraints: {analysis.constraint_score}/10")
    print(f"  - Context: {analysis.context_score}/10")
    print(f"\nDiagnostic Summary:\n  {analysis.overall_analysis}\n")
    print("Identified Weaknesses:")
    for issue in analysis.issues:
        print(f"  - {issue}")
    print("\nSuggestions:")
    for sugg in analysis.suggestions:
        print(f"  - {sugg}")

    # Verify no false-positive missing text
    combined_issues = " ".join(analysis.issues + [analysis.overall_analysis]).lower()
    assert "missing the actual text" not in combined_issues, "Found false-positive 'missing the actual text'!"
    assert "the source text is missing" not in combined_issues, "Found false-positive 'the source text is missing'!"

    print("\n" + "=" * 60)
    print("STEP 2: FULL EXPERIMENT (PROMPT HEALTH vs RESPONSE QUALITY)")
    print("=" * 60)

    detail, analysis_res, warnings = service.run_full_experiment(
        task_type="Summarization",
        original_prompt=test_prompt,
        input_text=test_article,
        num_variants=2,
    )

    print(f"Experiment ID: {detail.experiment.id}")
    print(f"Baseline Prompt Health: {analysis_res.overall_health_score}/10")

    for v in detail.versions:
        ev = v.response.evaluation if v.response else None
        print("\n" + "-" * 50)
        print(f"Version {v.version_number}: [{v.strategy}]")
        print(f"Prompt Text: {repr(v.prompt_text.strip())}")
        if v.response:
            print(f"Response: {repr(v.response.response_text.strip()[:120])}...")
            print(f"Response Quality Score: {ev.overall_score if ev else 0.0}/10")
            if ev:
                print(f"  - Instruction Following: {ev.instruction_following}/10")
                print(f"  - Relevance: {ev.relevance}/10")
                print(f"  - Factual Consistency: {ev.factual_consistency}/10")
                print(f"  - Completeness: {ev.completeness}/10")
                print(f"  - Format Compliance: {ev.format_compliance}/10")
                print(f"  - Conciseness: {ev.conciseness}/10")
                print(f"Feedback:")
                for fb in ev.feedback:
                    print(f"  * {fb}")
                print(f"Strengths:")
                for st_item in ev.strengths:
                    print(f"  * {st_item}")

                if v.version_number == 1:
                    # Verify for V1 ("summarize this") that feedback does NOT invent prompt constraints
                    comb_v1 = " ".join(ev.feedback + ev.strengths).lower()
                    assert "prompt contained length" not in comb_v1, "V1 feedback invented length instruction!"
                    assert "prompt specified negative" not in comb_v1, "V1 feedback invented negative constraints!"
                    assert "prompt required bullet" not in comb_v1, "V1 feedback invented bullet requirement!"

    print("\n" + "=" * 60)
    print("SUCCESS: Prompt Health and Response Quality correctly evaluated and separated!")
    print("=" * 60)

if __name__ == "__main__":
    main()
