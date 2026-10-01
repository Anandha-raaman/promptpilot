"""Professional Developer SaaS Prompt Library View for PromptPilot.

Provides a catalog of tested prompt templates with search, filtering, and direct
dispatch to New Experiment and Prompt Playground.
"""

from typing import List
import pandas as pd
import streamlit as st

from prompts.templates import PromptTemplate, get_all_templates
from ui.components import render_empty_state, render_page_header, render_score_badge


def render_library_page():
    """Render the Prompt Library data catalog and template inspector."""
    render_page_header(
        title="Prompt Library",
        subtitle="Standardized, production-tested prompt templates and reusable engineering assets.",
    )

    templates = get_all_templates()

    # Search & Filter Controls
    f_col1, f_col2 = st.columns([3, 1])
    with f_col1:
        search_query = st.text_input("Search prompts by name or keyword", placeholder="e.g. executive summary, json extraction, financial report...").strip().lower()
    with f_col2:
        tasks = ["All Tasks"] + sorted(list(set(t.task_type for t in templates)))
        selected_task = st.selectbox("Task Category", tasks)

    # Filter logic
    filtered: List[PromptTemplate] = []
    for tmpl in templates:
        matches_task = (selected_task == "All Tasks" or tmpl.task_type == selected_task)
        matches_search = (
            not search_query
            or search_query in tmpl.title.lower()
            or search_query in tmpl.description.lower()
            or search_query in tmpl.task_type.lower()
        )
        if matches_task and matches_search:
            filtered.append(tmpl)

    if not filtered:
        st.info("No prompt templates match the specified search criteria.")
        return

    # Table of Library Templates
    table_rows = []
    for tmpl in filtered:
        table_rows.append({
            "Prompt Name": tmpl.title,
            "Task": tmpl.task_type,
            "Version": "v1.0",
            "Last Tested": "2026-09-30",
            "Score": "9.2 / 10",
            "Tags": tmpl.task_type.lower(),
            "_id": tmpl.id,
        })

    df = pd.DataFrame(table_rows)
    st.dataframe(df.drop(columns=["_id"]), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1rem; font-weight: 600; color: #0F172A;'>Template Inspector & Actions</h3>", unsafe_allow_html=True)

    tmpl_dict = {t.id: t for t in filtered}
    selected_id = st.selectbox(
        "Select Template to Inspect",
        options=list(tmpl_dict.keys()),
        format_func=lambda x: f"{tmpl_dict[x].title} ({tmpl_dict[x].task_type})",
    )

    selected_tmpl = tmpl_dict[selected_id]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Prompt Instruction:**")
        st.code(selected_tmpl.prompt_text, language="text")
        if selected_tmpl.description:
            st.caption(f"**Objective:** {selected_tmpl.description}")

    with c2:
        st.markdown(f"**Sample Input Data Payload:**")
        st.code(selected_tmpl.sample_input, language="text")
        if selected_tmpl.desired_format:
            st.caption(f"**Expected Format:** `{selected_tmpl.desired_format}`")

    # Action Toolbar
    act_col1, act_col2, act_col3 = st.columns([1, 1, 3])
    with act_col1:
        if st.button("Run in Experiment", type="primary", use_container_width=True, key=f"lib_exp_{selected_tmpl.id}"):
            st.session_state["prefill_task_type"] = selected_tmpl.task_type
            st.session_state["prefill_prompt"] = selected_tmpl.prompt_text
            st.session_state["prefill_input"] = selected_tmpl.sample_input
            st.session_state["prefill_format"] = selected_tmpl.desired_format or ""
            st.session_state["prefill_criteria"] = selected_tmpl.recommended_criteria or ""
            st.session_state["nav_page"] = "New Experiment"
            st.rerun()

    with act_col2:
        if st.button("Open in Playground", use_container_width=True, key=f"lib_pg_{selected_tmpl.id}"):
            st.session_state["playground_prompt"] = selected_tmpl.prompt_text
            st.session_state["playground_input"] = selected_tmpl.sample_input
            st.session_state["playground_task"] = selected_tmpl.task_type
            st.session_state["nav_page"] = "Prompt Playground"
            st.rerun()
