"""Regression Testing View for PromptPilot.

Implements version-over-version prompt regression benchmarking with impartial,
evidence-based terminology according to defined evaluation criteria.
"""

from typing import List, Optional
import pandas as pd
import streamlit as st

from config.settings import Settings
from database.models import FullExperimentDetail
from database.repository import ExperimentRepository
from evaluation.regression import RegressionDetector, VersionComparisonReport
from evaluation.scoring import EvaluationScore
from ui.components import (
    render_empty_state,
    render_evaluation_disclaimer,
    render_page_header,
    render_score_badge,
    render_status_badge,
)


def render_regression_page(repo: ExperimentRepository, settings: Settings):
    """Render the Regression Testing interface."""
    render_page_header(
        title="Regression Testing Bench",
        subtitle="Benchmark prompt versions to detect performance degradation or verify intended improvements.",
    )

    experiments = repo.list_experiments(limit=100)
    if not experiments:
        if render_empty_state(
            title="No experiments available for regression testing",
            description="Run a prompt experiment with multiple versions to compare performance and detect regressions.",
            button_label="New Experiment",
        ):
            st.session_state["nav_page"] = "New Experiment"
            st.rerun()
        return

    # Select Experiment Context
    exp_options = {exp.id: f"{exp.title or exp.task_type} ({exp.created_at[:10]} - {exp.id[:8]})" for exp in experiments}
    selected_exp_id = st.selectbox(
        "Select Experiment Context",
        options=list(exp_options.keys()),
        format_func=lambda x: exp_options[x],
        index=0,
    )

    detail: Optional[FullExperimentDetail] = repo.get_experiment(selected_exp_id)
    if not detail or not detail.versions or len(detail.versions) < 2:
        st.info("The selected experiment contains fewer than 2 prompt versions. At least two versions are required for regression benchmarking.")
        return

    # Version Selectors
    v_col1, v_col2 = st.columns(2)
    version_map = {v.version_number: v for v in detail.versions}
    v_nums = sorted(list(version_map.keys()))

    with v_col1:
        base_num = st.selectbox("Baseline Version", v_nums, index=0)
    with v_col2:
        curr_index = 1 if len(v_nums) > 1 else 0
        curr_num = st.selectbox("Candidate / Current Version", v_nums, index=curr_index)

    baseline_v = version_map[base_num]
    current_v = version_map[curr_num]

    if not baseline_v.response or not baseline_v.response.evaluation:
        st.warning(f"Version V{base_num} does not have completed evaluation scores.")
        return
    if not current_v.response or not current_v.response.evaluation:
        st.warning(f"Version V{curr_num} does not have completed evaluation scores.")
        return

    base_eval_rec = baseline_v.response.evaluation
    curr_eval_rec = current_v.response.evaluation

    base_score = EvaluationScore(
        relevance=base_eval_rec.relevance,
        completeness=base_eval_rec.completeness,
        instruction_following=base_eval_rec.instruction_following,
        format_compliance=base_eval_rec.format_compliance,
        conciseness=base_eval_rec.conciseness,
        factual_consistency=base_eval_rec.factual_consistency,
        feedback=base_eval_rec.feedback,
        strengths=base_eval_rec.strengths,
    )
    curr_score = EvaluationScore(
        relevance=curr_eval_rec.relevance,
        completeness=curr_eval_rec.completeness,
        instruction_following=curr_eval_rec.instruction_following,
        format_compliance=curr_eval_rec.format_compliance,
        conciseness=curr_eval_rec.conciseness,
        factual_consistency=curr_eval_rec.factual_consistency,
        feedback=curr_eval_rec.feedback,
        strengths=curr_eval_rec.strengths,
    )

    report = RegressionDetector.compare_evaluations(
        baseline_eval=base_score,
        current_eval=curr_score,
        baseline_name=f"V{base_num} ({baseline_v.strategy})",
        current_name=f"V{curr_num} ({current_v.strategy})",
        baseline_latency=baseline_v.response.latency_ms,
        current_latency=current_v.response.latency_ms,
    )

    # 1. Summary Overview Banner with strict impartial terminology
    pct_change = round(((curr_score.overall_score - base_score.overall_score) / max(0.1, base_score.overall_score)) * 100, 1)
    pct_sign = "+" if pct_change > 0 else ""

    if report.is_regression:
        status_text = "Performance decreased according to selected evaluation criteria."
        st.warning(f"**Regression Indicator:** {status_text} (Delta: {report.overall_delta:+.1f} pts, {pct_sign}{pct_change}%)")
    elif report.is_improvement:
        status_text = "Performance increased according to selected evaluation criteria."
        st.success(f"**Improvement Indicator:** {status_text} (Delta: {report.overall_delta:+.1f} pts, {pct_sign}{pct_change}%)")
    else:
        status_text = "Comparable performance within margin of experimental variance."
        st.info(f"**Neutral Indicator:** {status_text} (Delta: {report.overall_delta:+.1f} pts, {pct_sign}{pct_change}%)")

    # 2. Executive Benchmark Table
    bench_data = [
        {
            "Test Context": detail.experiment.title or detail.experiment.task_type,
            "Baseline Version": f"V{base_num} ({baseline_v.strategy})",
            "Current Version": f"V{curr_num} ({current_v.strategy})",
            "Baseline Score": f"{base_score.overall_score:.1f} / 10",
            "Current Score": f"{curr_score.overall_score:.1f} / 10",
            "Performance Change": f"{pct_sign}{pct_change}%",
            "Status": "Performance Increased" if report.is_improvement else ("Performance Decreased" if report.is_regression else "Comparable"),
        }
    ]
    st.dataframe(pd.DataFrame(bench_data), use_container_width=True)

    # 3. Criterion-Level Delta Breakdown
    st.markdown("#### Criterion-Level Score Changes")
    delta_rows = []
    for d in report.metric_deltas:
        d_sign = "+" if d.delta > 0 else ""
        delta_rows.append({
            "Evaluation Criterion": d.criterion,
            "Baseline (V" + str(base_num) + ")": d.baseline_score,
            "Current (V" + str(curr_num) + ")": d.current_score,
            "Delta": f"{d_sign}{d.delta:.1f}",
            "% Change": f"{'+' if d.percentage_change > 0 else ''}{d.percentage_change}%",
        })

    st.dataframe(pd.DataFrame(delta_rows), use_container_width=True)

    # 4. Side-by-Side Prompt & Completion Comparison
    st.markdown("#### Side-by-Side Prompt & Response Inspection")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Baseline Prompt (V{base_num})**")
        st.code(baseline_v.prompt_text, language="text")
        st.markdown(f"**Baseline Completion** *(Latency: {baseline_v.response.latency_ms:.0f}ms)*")
        st.text_area("Baseline Response", value=baseline_v.response.response_text, height=200, disabled=True, key="reg_base_resp")
    with c2:
        st.markdown(f"**Current Prompt (V{curr_num})**")
        st.code(current_v.prompt_text, language="text")
        st.markdown(f"**Current Completion** *(Latency: {current_v.response.latency_ms:.0f}ms)*")
        st.text_area("Current Response", value=current_v.response.response_text, height=200, disabled=True, key="reg_curr_resp")

    render_evaluation_disclaimer()
