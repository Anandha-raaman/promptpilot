"""Prompt regression analysis and version-to-version performance comparison."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from evaluation.scoring import EvaluationScore


@dataclass
class MetricDelta:
    criterion: str
    baseline_score: float
    current_score: float
    delta: float

    @property
    def percentage_change(self) -> float:
        if self.baseline_score == 0:
            return 0.0
        return round(((self.current_score - self.baseline_score) / self.baseline_score) * 100, 1)


@dataclass
class VersionComparisonReport:
    baseline_version: str
    current_version: str
    baseline_overall: float
    current_overall: float
    overall_delta: float
    is_regression: bool
    is_improvement: bool
    status_label: str
    narrative_summary: str
    metric_deltas: List[MetricDelta]
    regressed_criteria: List[str]
    improved_criteria: List[str]
    latency_delta_ms: Optional[float] = None


class RegressionDetector:
    """Analyzes score differences between two prompt iterations to detect regressions."""

    @staticmethod
    def compare_evaluations(
        baseline_eval: EvaluationScore,
        current_eval: EvaluationScore,
        baseline_name: str = "Baseline",
        current_name: str = "Current",
        baseline_latency: Optional[float] = None,
        current_latency: Optional[float] = None,
        regression_threshold: float = -0.5,
        improvement_threshold: float = 0.5,
    ) -> VersionComparisonReport:
        """Compare two evaluation records and identify regressions or improvements."""
        criteria_keys = [
            ("Instruction Following", baseline_eval.instruction_following, current_eval.instruction_following),
            ("Relevance", baseline_eval.relevance, current_eval.relevance),
            ("Factual Consistency", baseline_eval.factual_consistency, current_eval.factual_consistency),
            ("Completeness", baseline_eval.completeness, current_eval.completeness),
            ("Format Compliance", baseline_eval.format_compliance, current_eval.format_compliance),
            ("Conciseness", baseline_eval.conciseness, current_eval.conciseness),
        ]

        metric_deltas: List[MetricDelta] = []
        regressed_criteria: List[str] = []
        improved_criteria: List[str] = []

        for name, base_val, curr_val in criteria_keys:
            delta = round(curr_val - base_val, 1)
            metric_deltas.append(MetricDelta(criterion=name, baseline_score=float(base_val), current_score=float(curr_val), delta=delta))
            if delta <= -1.0:
                regressed_criteria.append(name)
            elif delta >= 1.0:
                improved_criteria.append(name)

        overall_delta = round(current_eval.overall_score - baseline_eval.overall_score, 1)
        is_regression = overall_delta <= regression_threshold
        is_improvement = overall_delta >= improvement_threshold

        latency_delta = None
        if baseline_latency is not None and current_latency is not None:
            latency_delta = round(current_latency - baseline_latency, 1)

        if is_regression:
            status_label = "Potential Regression Detected"
            narrative = (
                f"Performance decreased by {abs(overall_delta)} points according to the automated model-based evaluation rubric. "
                f"Notable drops occurred in: {', '.join(regressed_criteria) if regressed_criteria else 'composite scoring'}."
            )
        elif is_improvement:
            status_label = "Measurable Improvement Observed"
            narrative = (
                f"Performance improved by +{overall_delta} points across evaluated criteria. "
                f"Gains were observed in: {', '.join(improved_criteria) if improved_criteria else 'composite scoring'}."
            )
        else:
            status_label = "Comparable Performance"
            narrative = "Performance remained relatively steady within experimental margin of variance."

        return VersionComparisonReport(
            baseline_version=baseline_name,
            current_version=current_name,
            baseline_overall=baseline_eval.overall_score,
            current_overall=current_eval.overall_score,
            overall_delta=overall_delta,
            is_regression=is_regression,
            is_improvement=is_improvement,
            status_label=status_label,
            narrative_summary=narrative,
            metric_deltas=metric_deltas,
            regressed_criteria=regressed_criteria,
            improved_criteria=improved_criteria,
            latency_delta_ms=latency_delta,
        )
