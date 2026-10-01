"""Prompt Analyzer Standalone Audit View."""

import streamlit as st
from config.settings import Settings
from services.experiment_service import ExperimentService
from ui.components import render_api_warning, render_page_header, render_radar_or_bar_metrics
from utils.logging import format_safe_user_error, get_logger, redact_secrets

logger = get_logger("analyzer_page")


def render_prompt_analyzer_page(service: ExperimentService, settings: Settings):
    """Render the standalone prompt health and quality auditing view."""
    render_page_header(
        title="Prompt Health Analyzer",
        subtitle="Objective diagnostic review of prompt clarity, context, constraints, and ambiguity.",
    )

    if not service.provider.is_available():
        render_api_warning()

    with st.form("analyzer_form"):
        task_type = st.selectbox(
            "Task Category",
            ["Summarization", "Information Extraction", "Text Generation", "General Analysis"],
        )
        prompt_text = st.text_area(
            "Prompt to Analyze",
            height=160,
            placeholder="Paste any prompt here to audit its clarity, specificity, and weaknesses...",
        )
        has_separate_data = st.checkbox(
            "Source text/data will be supplied separately (Prompt contains instructions only)",
            value=True,
            help="When enabled, PromptPilot evaluates the instruction quality without penalizing you for not pasting the raw article into the prompt.",
        )
        submitted = st.form_submit_button("Run Prompt Audit", type="primary", use_container_width=True)

    if submitted:
        if not prompt_text.strip():
            st.error("Please provide prompt text to audit.")
            return

        with st.spinner("Auditing prompt quality and structure..."):
            try:
                analysis = service.analyze_only(
                    prompt=prompt_text,
                    task_type=task_type,
                    has_input_context=has_separate_data,
                )
            except Exception as e:
                logger.error(f"Analysis failed: {e}", exc_info=True)
                st.error(format_safe_user_error(e, "The prompt analysis could not be completed. Please try again."))
                return

        st.markdown("---")
        st.subheader("Diagnostic Report")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Health Score", f"{analysis.overall_health_score}/10")
        with c2:
            st.metric("Clarity", f"{analysis.clarity_score}/10")
        with c3:
            st.metric("Specificity", f"{analysis.specificity_score}/10")
        with c4:
            st.metric("Output Format", f"{analysis.output_format_score}/10")
        with c5:
            st.metric("Constraints", f"{analysis.constraint_score}/10")
        with c6:
            st.metric("Context", f"{analysis.context_score}/10")

        # Metric breakdown bar chart
        st.markdown("#### Rubric Dimension Breakdown")
        scores_dict = {
            "Clarity": analysis.clarity_score,
            "Specificity": analysis.specificity_score,
            "Format": analysis.output_format_score,
            "Constraints": analysis.constraint_score,
            "Context": analysis.context_score,
        }
        render_radar_or_bar_metrics(scores_dict)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Identified Defects & Weaknesses")
            if analysis.issues:
                for issue in analysis.issues:
                    st.markdown(f"- {issue}")
            else:
                st.info("No structural defects flagged.")

        with col_right:
            st.markdown("#### Recommended Improvements")
            if analysis.suggestions:
                for sugg in analysis.suggestions:
                    st.markdown(f"- {sugg}")
            else:
                st.info("No immediate recommendations.")

        st.markdown(f"**Detailed Qualitative Analysis:**\n\n{analysis.overall_analysis}")

        st.markdown("---")
        if st.button("Test this Prompt in an Experiment", type="primary"):
            st.session_state["prefill_prompt"] = prompt_text
            st.session_state["prefill_task_type"] = task_type
            st.session_state["nav_page"] = "New Experiment"
            st.rerun()
