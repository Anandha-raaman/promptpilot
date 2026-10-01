"""Professional Developer Dashboard for PromptPilot."""

import streamlit as st
import pandas as pd
from typing import List

from config.settings import Settings
from database.repository import ExperimentRepository
from ui.components import (
    render_empty_state,
    render_page_header,
    render_metric_card,
    render_score_badge,
    render_status_badge,
)


def render_dashboard(repo: ExperimentRepository, settings: Settings):
    """Render the high-level developer workbench overview and recent experiment records."""
    # 1. Header with primary action
    h_col1, h_col2 = st.columns([4, 1])
    with h_col1:
        render_page_header(
            title="PromptPilot",
            subtitle="Prompt testing and optimization workspace",
        )
    with h_col2:
        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        if st.button("+ New Experiment", type="primary", use_container_width=True):
            st.session_state["nav_page"] = "New Experiment"
            st.rerun()

    # 2. Overview Metrics
    all_experiments = repo.list_experiments(limit=100)
    total_experiments = len(all_experiments)

    # Compute aggregate metrics
    prompts_tested = 0
    scores_list: List[float] = []

    for exp in all_experiments:
        detail = repo.get_experiment(exp.id)
        if detail and detail.versions:
            prompts_tested += len(detail.versions)
            for v in detail.versions:
                if v.response and v.response.evaluation:
                    scores_list.append(v.response.evaluation.overall_score)

    avg_score = round(sum(scores_list) / len(scores_list), 1) if scores_list else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Total Experiments", str(total_experiments), "All recorded runs")
    with m2:
        render_metric_card("Prompts Tested", str(prompts_tested), "Generated & tuned iterations")
    with m3:
        avg_str = f"{avg_score:.1f} / 10" if scores_list else "—"
        render_metric_card("Average Response Score", avg_str, "Weighted multi-metric composite")
    with m4:
        reg_count = sum(1 for exp in all_experiments if repo.get_experiment(exp.id) and len(repo.get_experiment(exp.id).versions) >= 2)
        render_metric_card("Regression Tests", str(reg_count), "Multi-version benchmarks")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 3. Recent Experiments Section
    st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A; margin-bottom: 0.75rem;'>Recent Experiments</h3>", unsafe_allow_html=True)

    if not all_experiments:
        if render_empty_state(
            title="No experiments yet",
            description="Run your first prompt experiment to compare prompt versions and AI responses.",
            button_label="Create Experiment",
        ):
            st.session_state["nav_page"] = "New Experiment"
            st.rerun()
        return

    # Render clean table
    recent_rows = []
    for exp in all_experiments[:15]:
        detail = repo.get_experiment(exp.id)
        num_v = len(detail.versions) if detail and detail.versions else 1
        best_score = 0.0
        if detail and detail.versions:
            for v in detail.versions:
                if v.response and v.response.evaluation:
                    best_score = max(best_score, v.response.evaluation.overall_score)

        score_display = f"{best_score:.1f} / 10" if best_score > 0 else "—"
        status_text = "Completed" if best_score > 0 else "Pending"

        recent_rows.append({
            "Experiment": exp.title or exp.task_type,
            "Task": exp.task_type,
            "Prompt Versions": f"{num_v} {'version' if num_v == 1 else 'versions'}",
            "Best Score": score_display,
            "Created": exp.created_at[:10],
            "Status": status_text,
            "_id": exp.id,
        })

    recent_df = pd.DataFrame(recent_rows)

    # Display interactive selection table
    st.dataframe(
        recent_df.drop(columns=["_id"]),
        use_container_width=True,
        hide_index=True,
    )

    # Quick Inspection Drawer
    with st.expander("Inspect Recent Experiment", expanded=False):
        exp_select = st.selectbox(
            "Select Experiment to Open",
            options=[r["_id"] for r in recent_rows],
            format_func=lambda x: next((f"{r['Experiment']} ({r['Created']})" for r in recent_rows if r["_id"] == x), x),
        )
        if st.button("Open in History Inspector ➔", key="open_dash_exp"):
            st.session_state["selected_experiment_id"] = exp_select
            st.session_state["nav_page"] = "Experiment History"
            st.rerun()
