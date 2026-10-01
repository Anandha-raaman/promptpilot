"""Professional Developer SaaS Experiment History View for PromptPilot.

Provides structured data table browsing, search, task filtering, sorting,
and granular version-level audit inspection.
"""

from typing import List, Optional
import pandas as pd
import streamlit as st

from database.models import FullExperimentDetail
from database.repository import ExperimentRepository
from ui.components import (
    render_empty_state,
    render_evaluation_disclaimer,
    render_page_header,
    render_score_badge,
    render_status_badge,
)
from utils.export import ExperimentExporter


def render_history_page(repo: ExperimentRepository):
    """Render the experiment history browser and details view."""
    render_page_header(
        title="Experiment History",
        subtitle="Review, audit, compare, and export historical prompt engineering experiments.",
    )

    experiments = repo.list_experiments(limit=200)
    if not experiments:
        if render_empty_state(
            title="No experiments found in history",
            description="Run your first prompt experiment to generate and compare prompt iterations.",
            button_label="New Experiment",
        ):
            st.session_state["nav_page"] = "New Experiment"
            st.rerun()
        return

    # 1. Search, Filter, and Sorting Controls
    col_search, col_task, col_sort = st.columns([2, 1, 1])

    with col_search:
        search_query = st.text_input("Search experiments", placeholder="Filter by title, task, or ID...").strip().lower()

    with col_task:
        task_types = ["All Tasks"] + sorted(list(set(exp.task_type for exp in experiments)))
        selected_task = st.selectbox("Task Category", task_types)

    with col_sort:
        sort_option = st.selectbox("Sort By", ["Date (Newest first)", "Date (Oldest first)"])

    # 2. Filter & Assemble Rows
    filtered_experiments = []
    for exp in experiments:
        matches_task = (selected_task == "All Tasks" or exp.task_type == selected_task)
        matches_search = (
            not search_query
            or search_query in (exp.title or "").lower()
            or search_query in exp.task_type.lower()
            or search_query in exp.id.lower()
        )
        if matches_task and matches_search:
            filtered_experiments.append(exp)

    if sort_option == "Date (Oldest first)":
        filtered_experiments.reverse()

    if not filtered_experiments:
        st.info("No experiments match the specified search and filter criteria.")
        return

    table_data = []
    for exp in filtered_experiments:
        detail = repo.get_experiment(exp.id)
        num_v = len(detail.versions) if detail and detail.versions else 1
        best_score = 0.0
        if detail and detail.versions:
            for v in detail.versions:
                if v.response and v.response.evaluation:
                    best_score = max(best_score, v.response.evaluation.overall_score)

        score_display = f"{best_score:.1f} / 10" if best_score > 0 else "—"
        status_text = "Completed" if best_score > 0 else "Pending"

        table_data.append({
            "Date": exp.created_at[:10],
            "Experiment": exp.title or exp.task_type,
            "Task": exp.task_type,
            "Versions": f"{num_v}",
            "Best Score": score_display,
            "Model": exp.model_name,
            "Status": status_text,
            "_id": exp.id,
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df.drop(columns=["_id"]), use_container_width=True, hide_index=True)

    # 3. Experiment Inspector Section
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A;'>Experiment Inspector</h3>", unsafe_allow_html=True)

    options = {exp.id: f"{exp.title or exp.task_type} ({exp.created_at[:10]} - {exp.id[:8]})" for exp in filtered_experiments}
    default_selected = st.session_state.get("selected_experiment_id", filtered_experiments[0].id)
    if default_selected not in options:
        default_selected = filtered_experiments[0].id

    selected_id = st.selectbox(
        "Select Experiment to Inspect Details",
        options=list(options.keys()),
        format_func=lambda x: options[x],
        index=list(options.keys()).index(default_selected),
    )

    detail: Optional[FullExperimentDetail] = repo.get_experiment(selected_id)
    if not detail:
        st.warning("Selected experiment record could not be loaded from database.")
        return

    # Metadata Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='stat-label'>Task Type</div><div style='font-weight: 600;'>{detail.experiment.task_type}</div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='stat-label'>Model</div><div style='font-family: ui-monospace, monospace;'>{detail.experiment.model_name}</div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='stat-label'>Recorded Date</div><div>{detail.experiment.created_at[:10]}</div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='stat-label'>Versions Tested</div><div>{len(detail.versions)}</div>", unsafe_allow_html=True)

    # Ground Context Expander
    with st.expander("Ground Source Data Context", expanded=False):
        st.text_area("Source Data Payload", value=detail.experiment.input_text, height=130, disabled=True, label_visibility="collapsed")

    render_evaluation_disclaimer()

    # Detailed Version Inspections
    st.markdown("<h4 style='font-size: 0.95rem; font-weight: 600; color: #0F172A; margin-top: 1rem;'>Prompt Versions & Completions</h4>", unsafe_allow_html=True)
    if not detail.versions:
        st.info("This experiment has no recorded prompt iterations.")
    else:
        v_tabs = st.tabs([f"V{v.version_number}: {v.strategy}" for v in detail.versions])
        for idx, v in enumerate(detail.versions):
            with v_tabs[idx]:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Strategy:** `{v.strategy}`")
                    if v.expected_benefit:
                        st.caption(f"**Objective:** {v.expected_benefit}")
                    st.markdown("**Prompt Text:**")
                    st.code(v.prompt_text, language="text")

                with c2:
                    if v.response:
                        st.markdown(f"**Response** *(Latency: {v.response.latency_ms:.0f}ms | Tokens: {v.response.total_tokens or '—'})*")
                        st.text_area(
                            "Generated Response",
                            value=v.response.response_text,
                            height=170,
                            disabled=True,
                            key=f"hist_resp_{v.id}",
                            label_visibility="collapsed",
                        )
                        if v.response.evaluation:
                            ev = v.response.evaluation
                            st.markdown(f"**Response Quality Score:** {render_score_badge(ev.overall_score)}", unsafe_allow_html=True)
                            if ev.feedback:
                                st.markdown("**Feedback:**")
                                for fb in ev.feedback:
                                    st.markdown(f"- {fb}")
                            if ev.strengths:
                                st.markdown("**Strengths:**")
                                for st_item in ev.strengths:
                                    st.markdown(f"- {st_item}")

    # Actions Toolbar
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    act_col1, act_col2, act_col3, act_col4 = st.columns(4)

    with act_col1:
        st.download_button(
            "Export JSON",
            data=ExperimentExporter.to_json(detail),
            file_name=f"experiment_{detail.experiment.id[:8]}.json",
            mime="application/json",
            use_container_width=True,
        )
    with act_col2:
        st.download_button(
            "Export CSV",
            data=ExperimentExporter.to_csv(detail),
            file_name=f"experiment_{detail.experiment.id[:8]}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with act_col3:
        st.download_button(
            "Export Markdown",
            data=ExperimentExporter.to_markdown(detail),
            file_name=f"experiment_{detail.experiment.id[:8]}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with act_col4:
        if st.button("Delete Experiment", type="secondary", use_container_width=True):
            if repo.delete_experiment(detail.experiment.id):
                st.success("Experiment deleted from database.")
                st.session_state.pop("selected_experiment_id", None)
                st.rerun()
