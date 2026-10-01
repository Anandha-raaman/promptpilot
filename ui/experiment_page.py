"""Professional Developer SaaS Experiment View for PromptPilot.

Implements a clean, technical two-section workspace:
- Section 1: Experiment Input & Ground Data Payload
- Section A: Prompt Health Audit
- Section B: Response Quality Analysis
- Section C: Version Performance Comparison Table
- Section D: Detailed Prompt Version Inspection
- Section E: Side-by-Side Prompt & Completion Comparison
"""

from typing import List, Optional
import pandas as pd
import streamlit as st

from config.settings import Settings
from database.models import FullExperimentDetail
from prompts.analyzer import PromptAnalysisResult
from services.experiment_service import ExperimentService
from ui.components import (
    render_api_warning,
    render_evaluation_disclaimer,
    render_horizontal_score_bar,
    render_page_header,
    render_score_badge,
    render_status_badge,
)
from utils.export import ExperimentExporter
from utils.logging import format_safe_user_error, get_logger, redact_secrets
from utils.validation import estimate_api_calls

logger = get_logger("experiment_page")


def render_new_experiment(service: ExperimentService, settings: Settings):
    """Render the primary prompt testing and optimization experiment workflow."""
    render_page_header(
        title="New Prompt Experiment",
        subtitle="Analyze prompt health, synthesize strategy-driven variants, execute against ground context, and evaluate.",
    )

    if not service.provider.is_available():
        render_api_warning()

    # Pre-populate fields if transferred from Template Library
    default_task = st.session_state.get("prefill_task_type", "Summarization")
    default_prompt = st.session_state.get("prefill_prompt", "")
    default_input = st.session_state.get("prefill_input", "")
    default_format = st.session_state.get("prefill_format", "")
    default_criteria = st.session_state.get("prefill_criteria", "")

    # SECTION 1 — EXPERIMENT INPUT
    st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A; margin: 0.5rem 0 1rem 0;'>1. Experiment Parameters & Data</h3>", unsafe_allow_html=True)

    with st.form("experiment_form"):
        task_types = ["Summarization", "Information Extraction", "Text Generation", "General Analysis"]
        task_index = task_types.index(default_task) if default_task in task_types else 0
        task_type = st.selectbox("Target Task Category", task_types, index=task_index)

        # Original Prompt Area
        original_prompt = st.text_area(
            "Original Prompt",
            value=default_prompt,
            height=130,
            placeholder="Enter the exact instruction you would send to an LLM (e.g. summarize this, or specific directives)...",
            help="Directives telling the AI what to do, how to format, and what boundaries to follow.",
        )
        prompt_char_count = len(original_prompt)
        st.markdown(
            f"<div style='font-size: 0.775rem; color: #64748B; margin-top: -0.4rem; margin-bottom: 0.8rem; font-family: ui-monospace, monospace;'>"
            f"Prompt length: {prompt_char_count} / {settings.max_prompt_chars} characters"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Source Data / Input Context Area
        st.markdown(
            "<div style='font-size: 0.825rem; color: #475569; margin-bottom: 0.25rem;'>"
            "<strong>Source Data / Input Context:</strong> The source data is the content that all prompt versions will be tested against."
            "</div>",
            unsafe_allow_html=True,
        )
        input_text = st.text_area(
            "Source Data / Input Context",
            value=default_input,
            height=160,
            placeholder="Paste the source document, article, dataset, or customer query here...",
            label_visibility="collapsed",
            help="This identical data payload will be provided to every prompt version during execution.",
        )
        input_char_count = len(input_text)
        st.markdown(
            f"<div style='font-size: 0.775rem; color: #64748B; margin-top: -0.4rem; margin-bottom: 0.8rem; font-family: ui-monospace, monospace;'>"
            f"Source data length: {input_char_count} / {settings.max_input_chars} characters"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Optional Configuration
        with st.expander("Configuration & Evaluation Constraints", expanded=False):
            desired_format = st.text_input(
                "Desired Output Format (Optional)",
                value=default_format,
                placeholder="e.g. 3 bullet points, Strict JSON schema, Executive memo",
            )
            custom_criteria = st.text_input(
                "Specific Evaluation Focus (Optional)",
                value=default_criteria,
                placeholder="e.g. Emphasize factual precision and avoidance of technical jargon",
            )
            num_variants = st.slider(
                "Number of Prompt Strategy Variants to Generate",
                min_value=1,
                max_value=settings.max_variants_per_run,
                value=min(3, settings.max_variants_per_run),
            )

        # Cost & Call Estimate
        calls_est = estimate_api_calls(num_variants=num_variants, num_test_cases=1)
        st.caption(
            f"Estimated API Calls: {calls_est['total_api_calls']} calls "
            f"(1 Analysis + 1 Generator + {calls_est['execution_calls']} Executions + {calls_est['evaluation_calls']} Evaluations)"
        )

        submitted = st.form_submit_button("Run Experiment", type="primary", use_container_width=True)

    if submitted:
        if not original_prompt.strip():
            st.error("Prompt cannot be empty. Please enter a prompt instruction to test.")
            return
        if not input_text.strip():
            st.error("Source data context is required. Please provide the document or payload for prompts to process.")
            return

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_progress(curr, total, msg):
            pct = min(1.0, float(curr) / float(total))
            progress_bar.progress(pct)
            status_text.markdown(f"<div style='font-size: 0.85rem; color: #334155; font-family: ui-monospace, monospace;'><strong>Stage {curr}/{total}:</strong> {msg}</div>", unsafe_allow_html=True)

        try:
            with st.spinner("Processing prompt experiment pipeline..."):
                detail, analysis, warnings = service.run_full_experiment(
                    task_type=task_type,
                    original_prompt=original_prompt,
                    input_text=input_text,
                    desired_format=desired_format if desired_format else None,
                    custom_criteria=custom_criteria if custom_criteria else None,
                    num_variants=num_variants,
                    progress_cb=on_progress,
                )

            progress_bar.progress(1.0)
            status_text.markdown("<div style='font-size: 0.85rem; color: #16A34A; font-weight: 600;'>Experiment executed and saved to history database.</div>", unsafe_allow_html=True)

            if warnings:
                with st.expander("Input Validation Warnings Detected", expanded=True):
                    for w in warnings:
                        st.warning(w)

            st.session_state["active_experiment"] = detail
            st.session_state["active_analysis"] = analysis

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            logger.error(f"Experiment execution failed: {e}", exc_info=True)
            st.error(format_safe_user_error(e, "The experiment could not be completed. Please try again."))
            return

    # Render results if active experiment exists in session
    if "active_experiment" in st.session_state:
        render_experiment_results(st.session_state["active_experiment"], st.session_state.get("active_analysis"))


def render_experiment_results(detail: FullExperimentDetail, analysis: Optional[PromptAnalysisResult]):
    """Render the full results: prompt health audit, response quality, comparison table, version details, and side-by-side."""
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    render_page_header(
        title="Experiment Results",
        subtitle="Compare prompt versions and evaluate response quality.",
    )

    # Core concept distinction callout
    st.markdown(
        """
        <div class='concept-callout'>
            <strong>Metric Independence Principle:</strong><br>
            • <strong>Prompt Health</strong> measures the structural quality and specificity of the instruction prompt.<br>
            • <strong>Response Quality</strong> measures how accurate, faithful, and comprehensive the AI's actual answer was against the source data.<br>
            A weak prompt (e.g. <code>summarize this</code>) can produce a high-quality response if the model accurately infers intent. The two scores are evaluated independently.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # SECTION A: PROMPT HEALTH
    if analysis:
        st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A; margin: 1.25rem 0 0.75rem 0;'>Section A: Prompt Health Audit</h3>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<div class='dev-panel'>", unsafe_allow_html=True)
            col_h1, col_h2 = st.columns([1, 2])

            with col_h1:
                st.markdown("<div class='stat-label'>Overall Health Score</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 2rem; font-weight: 700; color: #0F172A; font-family: ui-monospace, monospace;'>{analysis.overall_health_score:.1f} <span style='font-size: 1rem; color: #64748B;'>/ 10</span></div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size: 0.775rem; color: #64748B; margin-top: 0.25rem;'>Weighted prompt engineering index</div>", unsafe_allow_html=True)

            with col_h2:
                render_horizontal_score_bar("Clarity", float(analysis.clarity_score))
                render_horizontal_score_bar("Specificity", float(analysis.specificity_score))
                render_horizontal_score_bar("Format Specification", float(analysis.output_format_score))
                render_horizontal_score_bar("Constraints & Guardrails", float(analysis.constraint_score))
                render_horizontal_score_bar("Context & Framing", float(analysis.context_score))

            st.markdown("<div style='margin-top: 0.85rem; border-top: 1px solid #E2E8F0; padding-top: 0.75rem;'>", unsafe_allow_html=True)
            st.markdown(f"**Diagnostic Summary:** {analysis.overall_analysis}")

            if analysis.issues:
                st.markdown("<div style='margin-top: 0.5rem;'><strong>Identified Weaknesses:</strong></div>", unsafe_allow_html=True)
                for issue in analysis.issues:
                    st.markdown(f"- <span style='color: #991B1B;'>[Missing]</span> {issue}", unsafe_allow_html=True)

            if analysis.suggestions:
                st.markdown("<div style='margin-top: 0.5rem;'><strong>Optimization Opportunities:</strong></div>", unsafe_allow_html=True)
                for sugg in analysis.suggestions:
                    st.markdown(f"- <span style='color: #166534;'>[Recommendation]</span> {sugg}", unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

    if not detail.versions:
        st.warning("No prompt versions recorded for this experiment.")
        return

    # SECTION B & C: RESPONSE QUALITY & VERSION COMPARISON TABLE
    st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A; margin: 1.75rem 0 0.75rem 0;'>Section B: Version Comparison & Response Quality</h3>", unsafe_allow_html=True)

    summary_data = []
    for v in detail.versions:
        ev = v.response.evaluation if v.response else None
        latency_str = f"{(v.response.latency_ms / 1000.0):.1f}s" if v.response else "—"
        summary_data.append({
            "Version": f"V{v.version_number}",
            "Strategy": v.strategy,
            "Prompt": (v.prompt_text[:60] + "...") if len(v.prompt_text) > 60 else v.prompt_text,
            "Response Quality": f"{ev.overall_score:.1f}" if ev else "—",
            "Instruction Following": ev.instruction_following if ev else 0,
            "Factual Consistency": ev.factual_consistency if ev else 0,
            "Completeness": ev.completeness if ev else 0,
            "Format Compliance": ev.format_compliance if ev else 0,
            "Latency": latency_str,
            "_id": v.id,
            "_v_num": v.version_number,
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(
        summary_df.drop(columns=["_id", "_v_num"]),
        use_container_width=True,
        hide_index=True,
    )

    # Bar Chart for Response Quality comparison
    chart_df = summary_df.set_index("Version")[["Instruction Following", "Factual Consistency", "Completeness"]]
    st.bar_chart(chart_df, height=220)

    # SECTION D: PROMPT VERSION DETAIL
    st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A; margin: 1.75rem 0 0.75rem 0;'>Section C: Prompt Version Detail</h3>", unsafe_allow_html=True)

    version_tab_labels = [f"V{v.version_number}: {v.strategy}" for v in detail.versions]
    version_tabs = st.tabs(version_tab_labels)

    for idx, v in enumerate(detail.versions):
        with version_tabs[idx]:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"**Strategy:** `{v.strategy}`")
                if v.expected_benefit:
                    st.caption(f"**Rationale:** {v.expected_benefit}")
                if v.potential_limitations:
                    st.caption(f"**Trade-off:** {v.potential_limitations}")

                st.markdown("**Prompt Instruction:**")
                st.code(v.prompt_text, language="text")

                if st.button("Open in Playground to Tune", key=f"tune_btn_{v.id}"):
                    st.session_state["playground_prompt"] = v.prompt_text
                    st.session_state["playground_base_exp_id"] = detail.experiment.id
                    st.session_state["playground_base_version"] = v.version_number
                    st.session_state["nav_page"] = "Prompt Playground"
                    st.rerun()

            with c2:
                if v.response:
                    st.markdown(
                        f"**Generated Response** *(Latency: {(v.response.latency_ms / 1000.0):.2f}s | Tokens: {v.response.total_tokens or '—'})*"
                    )
                    st.text_area(
                        "Generated Response Text",
                        value=v.response.response_text,
                        height=200,
                        disabled=True,
                        key=f"resp_view_{v.id}",
                        label_visibility="collapsed",
                    )

                    if v.response.evaluation:
                        ev = v.response.evaluation
                        st.markdown(f"**Response Quality Score:** {render_score_badge(ev.overall_score)}", unsafe_allow_html=True)

                        if v.version_number == 0 and analysis:
                            st.caption(
                                f"Baseline Prompt Health: `{analysis.overall_health_score}/10` (Instruction Quality) vs. "
                                f"Response Quality: `{ev.overall_score}/10` (Output Factual Accuracy)"
                            )

                        if ev.feedback:
                            st.markdown("**Evaluator Feedback:**")
                            for fb in ev.feedback:
                                st.markdown(f"- {fb}")
                        if ev.strengths:
                            st.markdown("**Key Strengths:**")
                            for st_item in ev.strengths:
                                st.markdown(f"- {st_item}")

    # SECTION E: SIDE-BY-SIDE COMPARISON MODE
    if len(detail.versions) >= 2:
        st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A; margin: 1.75rem 0 0.75rem 0;'>Section D: Side-by-Side Comparison</h3>", unsafe_allow_html=True)
        st.caption("Compare any two prompt versions side-by-side to inspect prompt adjustments and evaluate response differences.")

        cmp_col1, cmp_col2 = st.columns(2)
        v_dict = {v.version_number: v for v in detail.versions}
        v_keys = sorted(list(v_dict.keys()))

        with cmp_col1:
            sel_a = st.selectbox("Version A", v_keys, index=0, key="cmp_sel_a")
        with cmp_col2:
            sel_b = st.selectbox("Version B", v_keys, index=1 if len(v_keys) > 1 else 0, key="cmp_sel_b")

        v_a = v_dict[sel_a]
        v_b = v_dict[sel_b]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Version {v_a.version_number} Prompt ({v_a.strategy})**")
            st.code(v_a.prompt_text, language="text")
            st.markdown(f"**Response A** *(Score: {v_a.response.evaluation.overall_score if v_a.response and v_a.response.evaluation else '—'}/10)*")
            st.text_area("Response A Text", value=v_a.response.response_text if v_a.response else "", height=180, disabled=True, key=f"side_a_{v_a.id}", label_visibility="collapsed")

        with col_b:
            st.markdown(f"**Version {v_b.version_number} Prompt ({v_b.strategy})**")
            st.code(v_b.prompt_text, language="text")
            st.markdown(f"**Response B** *(Score: {v_b.response.evaluation.overall_score if v_b.response and v_b.response.evaluation else '—'}/10)*")
            st.text_area("Response B Text", value=v_b.response.response_text if v_b.response else "", height=180, disabled=True, key=f"side_b_{v_b.id}", label_visibility="collapsed")

        # Metric Side-by-Side Table
        if v_a.response and v_a.response.evaluation and v_b.response and v_b.response.evaluation:
            ev_a = v_a.response.evaluation
            ev_b = v_b.response.evaluation

            criteria_metrics = [
                ("Response Quality (Overall)", ev_a.overall_score, ev_b.overall_score),
                ("Instruction Following", ev_a.instruction_following, ev_b.instruction_following),
                ("Factual Consistency", ev_a.factual_consistency, ev_b.factual_consistency),
                ("Completeness", ev_a.completeness, ev_b.completeness),
                ("Format Compliance", ev_a.format_compliance, ev_b.format_compliance),
                ("Conciseness", ev_a.conciseness, ev_b.conciseness),
            ]

            comp_rows = []
            for name, val_a, val_b in criteria_metrics:
                delta = round(val_b - val_a, 1)
                delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
                comp_rows.append({
                    "Evaluation Metric": name,
                    f"V{v_a.version_number}": f"{val_a:.1f}",
                    f"V{v_b.version_number}": f"{val_b:.1f}",
                    "Delta (B - A)": delta_str,
                })

            st.markdown("<div style='margin-top: 0.75rem;'><strong>Evaluation Comparison:</strong></div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # SECTION F: EXPORT EXPERIMENT ARTIFACTS
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h4 style='font-size: 0.95rem; font-weight: 600; color: #0F172A;'>Export Experiment Records</h4>", unsafe_allow_html=True)

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    json_data = ExperimentExporter.to_json(detail)
    csv_data = ExperimentExporter.to_csv(detail)
    md_data = ExperimentExporter.to_markdown(detail)

    with exp_col1:
        st.download_button(
            label="Export JSON Data",
            data=json_data,
            file_name=f"experiment_{detail.experiment.id[:8]}.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp_col2:
        st.download_button(
            label="Export CSV Metrics",
            data=csv_data,
            file_name=f"experiment_{detail.experiment.id[:8]}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_col3:
        st.download_button(
            label="Export Markdown Report",
            data=md_data,
            file_name=f"experiment_{detail.experiment.id[:8]}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    render_evaluation_disclaimer()
