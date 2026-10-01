"""Professional Developer SaaS Prompt Playground View for PromptPilot.

Provides an agile side-by-side prompt tuning interface:
- Left: Prompt Editor (code-editor style)
- Right: Test Input Data
- Bottom: Run Prompt & Retest
- Output: Generated Completion & Multi-Metric Response Evaluation
"""

import time
from typing import Optional
import pandas as pd
import streamlit as st

from config.settings import Settings
from database.models import FullExperimentDetail
from database.repository import ExperimentRepository
from evaluation.scoring import EvaluationScore
from services.experiment_service import ExperimentService
from ui.components import (
    render_api_warning,
    render_evaluation_disclaimer,
    render_page_header,
    render_score_badge,
)
from utils.logging import format_safe_user_error, get_logger, redact_secrets

logger = get_logger("playground_page")


def render_playground_page(service: ExperimentService, repo: ExperimentRepository, settings: Settings):
    """Render interactive prompt editor and testing bench."""
    render_page_header(
        title="Prompt Playground",
        subtitle="Iteratively edit prompts, execute against test inputs, and inspect response quality.",
    )

    if not service.provider.is_available():
        render_api_warning()

    # Pre-population options (from experiments or session state)
    experiments = repo.list_experiments(limit=50)

    with st.expander("Load Existing Prompt from Experiment", expanded=False):
        if experiments:
            exp_options = {exp.id: f"{exp.title or exp.task_type} ({exp.created_at[:10]} - {exp.id[:8]})" for exp in experiments}
            chosen_exp_id = st.selectbox(
                "Select Experiment to Load",
                options=list(exp_options.keys()),
                format_func=lambda x: exp_options[x],
                key="pg_load_exp_select",
            )
            exp_detail = repo.get_experiment(chosen_exp_id)
            if exp_detail and exp_detail.versions:
                v_options = {v.version_number: f"V{v.version_number}: {v.strategy}" for v in exp_detail.versions}
                chosen_v_num = st.selectbox("Select Version", options=list(v_options.keys()), format_func=lambda x: v_options[x], key="pg_load_v_select")
                if st.button("Load Prompt & Context into Playground"):
                    chosen_v = next(v for v in exp_detail.versions if v.version_number == chosen_v_num)
                    st.session_state["playground_prompt"] = chosen_v.prompt_text
                    st.session_state["playground_input"] = exp_detail.experiment.input_text
                    st.session_state["playground_task"] = exp_detail.experiment.task_type
                    st.rerun()
        else:
            st.caption("No historical experiments recorded yet.")

    # Current prompt and context state
    default_prompt = st.session_state.get("playground_prompt", "Summarize the key takeaways from the provided text into 3 concise bullet points.")
    default_input = st.session_state.get(
        "playground_input",
        "Artificial intelligence is transforming business operations through automated workflows, "
        "advanced predictive analytics, and 24/7 intelligent customer assistance. Enterprise adoption requires "
        "managing upfront capital investments, addressing data privacy concerns, and enforcing continuous human oversight."
    )
    default_task = st.session_state.get("playground_task", "Summarization")

    # 1. Side-by-Side Editor & Context Layout
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #0F172A; margin-bottom: 0.35rem;'>Prompt Editor</div>", unsafe_allow_html=True)
        edited_prompt = st.text_area(
            "Prompt Editor Text",
            value=default_prompt,
            height=240,
            label_visibility="collapsed",
            help="Write or refine prompt instructions.",
        )
        st.markdown(
            f"<div style='font-size: 0.775rem; color: #64748B; margin-top: -0.3rem; font-family: ui-monospace, monospace;'>"
            f"Prompt length: {len(edited_prompt)} characters"
            f"</div>",
            unsafe_allow_html=True,
        )

    with c_right:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #0F172A; margin-bottom: 0.35rem;'>Test Input Context</div>", unsafe_allow_html=True)
        edited_input = st.text_area(
            "Test Input Context Text",
            value=default_input,
            height=240,
            label_visibility="collapsed",
            help="Ground data context the prompt will operate on.",
        )
        st.markdown(
            f"<div style='font-size: 0.775rem; color: #64748B; margin-top: -0.3rem; font-family: ui-monospace, monospace;'>"
            f"Input length: {len(edited_input)} characters"
            f"</div>",
            unsafe_allow_html=True,
        )

    # 2. Controls & Run Action
    b_col1, b_col2, b_col3 = st.columns([1, 1, 2])
    with b_col1:
        task_types = ["Summarization", "Information Extraction", "Text Generation", "General Analysis"]
        selected_task = st.selectbox("Task Category", task_types, index=task_types.index(default_task) if default_task in task_types else 0)
    with b_col2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    with b_col3:
        st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
        run_clicked = st.button("Run Prompt", type="primary", use_container_width=True)

    if run_clicked:
        if not edited_prompt.strip():
            st.error("Prompt cannot be empty.")
            return

        with st.spinner("Executing prompt against test input..."):
            try:
                start_t = time.perf_counter()
                llm_response = service.provider.generate_text(
                    prompt=f"{edited_prompt.strip()}\n\nInput Context:\n{edited_input.strip()}",
                    temperature=temperature,
                )
                latency_ms = (time.perf_counter() - start_t) * 1000.0

                # Evaluate response
                score, _ = service.evaluator.evaluate(
                    task_type=selected_task,
                    prompt_text=edited_prompt,
                    input_text=edited_input,
                    response_text=llm_response.content,
                )

                st.session_state["pg_last_prompt"] = edited_prompt
                st.session_state["pg_last_input"] = edited_input
                st.session_state["pg_last_response"] = llm_response.content
                st.session_state["pg_last_score"] = score
                st.session_state["pg_last_latency"] = latency_ms

            except Exception as e:
                logger.error(f"Playground execution failed: {e}", exc_info=True)
                st.error(format_safe_user_error(e, "Unable to complete the AI request. Please check your API configuration and try again."))
                return

    # 3. Output Display: Response & Evaluation
    if "pg_last_response" in st.session_state and "pg_last_score" in st.session_state:
        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A; margin-bottom: 0.75rem;'>Output & Response Evaluation</h3>", unsafe_allow_html=True)

        resp_col, eval_col = st.columns([1, 1])

        with resp_col:
            lat = st.session_state.get("pg_last_latency", 0.0)
            st.markdown(f"**Generated Response** *(Latency: {lat:.0f}ms)*")
            st.text_area(
                "Playground Generated Response",
                value=st.session_state["pg_last_response"],
                height=220,
                disabled=True,
                label_visibility="collapsed",
            )

        with eval_col:
            sc: EvaluationScore = st.session_state["pg_last_score"]
            st.markdown(f"**Response Quality Composite:** {render_score_badge(sc.overall_score)}", unsafe_allow_html=True)

            score_table = [
                {"Metric": "Factual Consistency", "Score": f"{sc.factual_consistency} / 10"},
                {"Metric": "Instruction Following", "Score": f"{sc.instruction_following} / 10"},
                {"Metric": "Relevance", "Score": f"{sc.relevance} / 10"},
                {"Metric": "Completeness", "Score": f"{sc.completeness} / 10"},
                {"Metric": "Format Compliance", "Score": f"{sc.format_compliance} / 10"},
                {"Metric": "Conciseness", "Score": f"{sc.conciseness} / 10"},
            ]
            st.dataframe(pd.DataFrame(score_table), use_container_width=True, hide_index=True)

            if sc.feedback:
                st.markdown("**Evaluator Feedback:**")
                for fb in sc.feedback:
                    st.markdown(f"- {fb}")
            if sc.strengths:
                st.markdown("**Key Strengths:**")
                for st_item in sc.strengths:
                    st.markdown(f"- {st_item}")

        render_evaluation_disclaimer()
