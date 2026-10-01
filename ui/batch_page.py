"""Professional Developer SaaS Test Datasets & Batch Evaluation View for PromptPilot."""

import uuid
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from config.settings import Settings
from services.evaluation_service import (
    BatchCaseResult,
    BatchEvaluationReport,
    EvaluationService,
    TestCase,
)
from ui.components import (
    render_api_warning,
    render_evaluation_disclaimer,
    render_metric_card,
    render_page_header,
    render_score_badge,
    render_status_badge,
)
from utils.logging import format_safe_user_error, get_logger, redact_secrets

logger = get_logger("ui.batch_page")

TASK_CATEGORIES = [
    "Summarization",
    "Information Extraction",
    "Text Generation",
    "Classification",
    "Question Answering",
    "Code Generation",
    "Custom",
]


def render_batch_page(eval_service: EvaluationService, settings: Settings):
    """Render the comprehensive Test Datasets & Test Case Management interface."""
    render_page_header(
        title="Test Datasets & Evaluation",
        subtitle="Define, manage, and benchmark test cases with strict isolated execution and automated rubric evaluation.",
    )

    if not eval_service.provider.is_available():
        render_api_warning()

    # Session State Initialization for Datasets & Case Editor
    if "active_dataset_id" not in st.session_state:
        st.session_state["active_dataset_id"] = "default-benchmark"
    if "editing_case" not in st.session_state:
        st.session_state["editing_case"] = None  # None, or dict with case fields
    if "single_case_results" not in st.session_state:
        st.session_state["single_case_results"] = {}  # case_id -> BatchCaseResult
    if "dataset_reports" not in st.session_state:
        st.session_state["dataset_reports"] = {}  # dataset_id -> BatchEvaluationReport
    if "show_new_dataset" not in st.session_state:
        st.session_state["show_new_dataset"] = False

    # 1. Dataset Selection & Management Bar
    datasets = eval_service.list_datasets()
    dataset_ids = [d.id for d in datasets]
    if st.session_state["active_dataset_id"] not in dataset_ids and dataset_ids:
        st.session_state["active_dataset_id"] = dataset_ids[0]

    # Dataset display labels: "Name (id) - N cases"
    ds_lookup = {d.id: d for d in datasets}
    active_ds = ds_lookup.get(st.session_state["active_dataset_id"])

    col_ds_select, col_ds_actions = st.columns([3, 2])
    with col_ds_select:
        selected_ds_id = st.selectbox(
            "Select Test Dataset",
            options=dataset_ids,
            index=dataset_ids.index(st.session_state["active_dataset_id"]) if st.session_state["active_dataset_id"] in dataset_ids else 0,
            format_func=lambda did: f"{ds_lookup[did].name} ({did}) — {ds_lookup[did].case_count} cases" if did in ds_lookup else did,
            key="dataset_select_box",
        )
        if selected_ds_id != st.session_state["active_dataset_id"]:
            st.session_state["active_dataset_id"] = selected_ds_id
            st.session_state["editing_case"] = None
            st.rerun()

    with col_ds_actions:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        col_btn_new, col_btn_del = st.columns([1, 1])
        with col_btn_new:
            if st.button("+ New Dataset", use_container_width=True, key="toggle_new_ds_btn"):
                st.session_state["show_new_dataset"] = not st.session_state["show_new_dataset"]
                st.rerun()
        with col_btn_del:
            if st.session_state["active_dataset_id"] != "default-benchmark":
                if st.button("Delete Dataset", type="secondary", use_container_width=True, key="del_ds_btn"):
                    eval_service.delete_dataset(st.session_state["active_dataset_id"])
                    st.session_state["active_dataset_id"] = "default-benchmark"
                    st.session_state["editing_case"] = None
                    st.success("Dataset deleted.")
                    st.rerun()

    # Form: Create New Dataset
    if st.session_state["show_new_dataset"]:
        with st.expander("Create New Dataset Container", expanded=True):
            col_d1, col_d2 = st.columns([1, 2])
            with col_d1:
                new_ds_id = st.text_input("Dataset ID", value=f"ds-{uuid.uuid4().hex[:6]}", help="Unique slug identifier (e.g. finance-eval)")
            with col_d2:
                new_ds_name = st.text_input("Dataset Name", placeholder="e.g., Enterprise Financial Benchmarks")
            new_ds_desc = st.text_area("Description (Optional)", placeholder="Describe the purpose, domain, or evaluation goals of this dataset...", height=70)

            col_sub_d1, col_sub_d2 = st.columns([1, 4])
            with col_sub_d1:
                if st.button("Save Dataset", type="primary", use_container_width=True, key="create_ds_confirm_btn"):
                    if not new_ds_name.strip():
                        st.error("Dataset name is required.")
                    else:
                        created_id = eval_service.create_dataset(
                            name=new_ds_name.strip(),
                            description=new_ds_desc.strip(),
                            dataset_id=new_ds_id.strip() if new_ds_id.strip() else None,
                        )
                        st.session_state["active_dataset_id"] = created_id
                        st.session_state["show_new_dataset"] = False
                        st.success(f"Dataset '{new_ds_name}' created.")
                        st.rerun()
            with col_sub_d2:
                if st.button("Cancel", key="cancel_new_ds_btn"):
                    st.session_state["show_new_dataset"] = False
                    st.rerun()

    current_dataset_id = st.session_state["active_dataset_id"]
    test_cases: List[TestCase] = eval_service.load_test_cases(dataset_id=current_dataset_id)

    # 2. Dataset Overview Stats & Top Action Bar
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns([2, 1, 2])
    with m1:
        st.markdown(
            f"<div style='font-size: 0.95rem; font-weight: 600; color: #0F172A;'>{active_ds.name if active_ds else current_dataset_id}</div>"
            f"<div style='font-size: 0.8rem; color: #64748B;'>{active_ds.description if active_ds and active_ds.description else 'Dataset ID: ' + current_dataset_id}</div>",
            unsafe_allow_html=True,
        )
    with m2:
        render_metric_card("Total Test Cases", str(len(test_cases)))
    with m3:
        st.markdown("<div style='height: 0.2rem;'></div>", unsafe_allow_html=True)
        col_act1, col_act2 = st.columns([1, 1])
        with col_act1:
            if st.button("+ Add Test Case", type="secondary", use_container_width=True, key="add_test_case_btn"):
                st.session_state["editing_case"] = {
                    "id": f"tc-{uuid.uuid4().hex[:6]}",
                    "name": "",
                    "task_type": "Summarization",
                    "prompt": "",
                    "source_data": "",
                    "criteria": "",
                    "expected_output": "",
                    "tags": "",
                    "is_new": True,
                }
                st.rerun()
        with col_act2:
            run_all_disabled = len(test_cases) == 0
            if st.button("Run All Tests", type="primary", use_container_width=True, disabled=run_all_disabled, key="run_all_tests_btn"):
                _execute_dataset_all_tests(eval_service, test_cases, current_dataset_id)
                st.rerun()

    st.markdown("---")

    # 3. Test Case Editor (Add or Edit)
    if st.session_state["editing_case"] is not None:
        _render_test_case_editor(eval_service, current_dataset_id)
        st.markdown("---")

    # 4. Filter & Test Cases List
    st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A; margin-bottom: 0.5rem;'>Dataset Test Cases</h3>", unsafe_allow_html=True)

    if not test_cases:
        st.markdown(
            """
            <div class='empty-state-box'>
                <div class='empty-state-title'>No Test Cases in Dataset</div>
                <div class='empty-state-desc'>Click <strong>'+ Add Test Case'</strong> above to create your first isolated test case with prompt, source data, and evaluation criteria.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Task filter row
    task_types = ["All Tasks"] + sorted(list(set(c.task_type for c in test_cases)))
    col_f1, col_f2 = st.columns([2, 4])
    with col_f1:
        task_filter = st.selectbox("Filter by Task Category", task_types, key="case_task_filter")

    visible_cases = [c for c in test_cases if task_filter == "All Tasks" or c.task_type == task_filter]

    # Render each test case card
    for case in visible_cases:
        _render_test_case_card(eval_service, case, current_dataset_id)

    # 5. Full Dataset Batch Report (if available)
    if current_dataset_id in st.session_state["dataset_reports"]:
        _render_batch_evaluation_report(st.session_state["dataset_reports"][current_dataset_id])

    render_evaluation_disclaimer()


def _render_test_case_editor(eval_service: EvaluationService, dataset_id: str):
    """Render the full test case definition editor with validation and execution buttons."""
    ed = st.session_state["editing_case"]
    is_new = ed.get("is_new", False)
    title_text = "Add New Test Case" if is_new else f"Edit Test Case: {ed.get('id', '')}"

    with st.container():
        st.markdown(
            f"""
            <div style='background-color: #FFFFFF; border: 2px solid #2563EB; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;'>
                <div style='font-size: 1.05rem; font-weight: 600; color: #1E293B; margin-bottom: 0.25rem;'>{title_text}</div>
                <div style='font-size: 0.825rem; color: #64748B; margin-bottom: 1rem;'>
                    Each test case binds its own prompt, source data, and evaluation criteria for strict isolated LLM execution.
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Fields: ID, Name, Category
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            case_id = st.text_input(
                "Test Case ID*",
                value=ed.get("id", f"tc-{uuid.uuid4().hex[:6]}"),
                disabled=not is_new,
                help="Unique identifier for test traceability across runs.",
                key="editor_case_id",
            )
        with c2:
            case_name = st.text_input(
                "Test Case Name*",
                value=ed.get("name", ""),
                placeholder="e.g. Semiconductor Q1 Earnings Extraction",
                key="editor_case_name",
            )
        with c3:
            curr_cat = ed.get("task_type", "Summarization")
            cat_idx = TASK_CATEGORIES.index(curr_cat) if curr_cat in TASK_CATEGORIES else 0
            task_type = st.selectbox(
                "Task Category*",
                TASK_CATEGORIES,
                index=cat_idx,
                key="editor_task_type",
            )

        # Fields: Prompt / Instructions, Source Data / Input
        col_p, col_s = st.columns([1, 1])
        with col_p:
            prompt_text = st.text_area(
                "Prompt / Instructions*",
                value=ed.get("prompt", ""),
                height=130,
                placeholder="Enter the specific instruction for this test case (e.g., 'Summarize key financial metrics into bullet points')...",
                help="The prompt instruction to execute. This prompt is executed strictly against this test case's source data.",
                key="editor_prompt",
            )
        with col_s:
            source_data = st.text_area(
                "Source Data / Input*",
                value=ed.get("source_data", ""),
                height=130,
                placeholder="Paste the raw text, document, article, or payload for this test case...",
                help="The isolated input content for this test case. Never shared with other cases.",
                key="editor_source_data",
            )

        # Fields: Evaluation Criteria, Expected Output, Tags
        criteria_text = st.text_area(
            "Evaluation Criteria*",
            value=ed.get("criteria", ""),
            height=80,
            placeholder="e.g. Must include exact revenue growth (+18%), CEO name (David Sterling), and cold start p99 metrics...",
            help="Specific standards and key points the automated evaluator must verify against the generated output.",
            key="editor_criteria",
        )

        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            expected_output = st.text_area(
                "Expected Output (Optional)",
                value=ed.get("expected_output", ""),
                height=70,
                placeholder="Reference golden answer or expected completion structure...",
                key="editor_expected_output",
            )
        with col_opt2:
            tags_str = st.text_input(
                "Tags (Optional, comma-separated)",
                value=ed.get("tags", ""),
                placeholder="e.g. finance, quarterly, earnings",
                key="editor_tags",
            )

        # Editor Action Buttons
        col_act1, col_act2, col_act3, col_spacer = st.columns([1.2, 1.2, 1, 3])
        with col_act1:
            if st.button("Save Test Case", type="primary", use_container_width=True, key="save_case_btn"):
                errs = _validate_test_case_fields(case_id, case_name, prompt_text, source_data, criteria_text)
                if errs:
                    for err in errs:
                        st.error(err)
                else:
                    tag_list = [t.strip() for t in tags_str.split(",") if t.strip()]
                    tc = TestCase(
                        id=case_id.strip(),
                        dataset_id=dataset_id,
                        name=case_name.strip(),
                        task_type=task_type,
                        prompt=prompt_text.strip(),
                        input_text=source_data.strip(),
                        criteria=criteria_text.strip(),
                        expected_output=expected_output.strip() if expected_output else None,
                        tags=tag_list,
                    )
                    eval_service.save_test_case(tc)
                    st.session_state["editing_case"] = None
                    st.success(f"Test case '{case_id}' saved successfully!")
                    st.rerun()

        with col_act2:
            if st.button("Run Test Case", type="secondary", use_container_width=True, key="run_case_from_editor_btn"):
                errs = _validate_test_case_fields(case_id, case_name, prompt_text, source_data, criteria_text)
                if errs:
                    for err in errs:
                        st.error(err)
                else:
                    tag_list = [t.strip() for t in tags_str.split(",") if t.strip()]
                    tc = TestCase(
                        id=case_id.strip(),
                        dataset_id=dataset_id,
                        name=case_name.strip(),
                        task_type=task_type,
                        prompt=prompt_text.strip(),
                        input_text=source_data.strip(),
                        criteria=criteria_text.strip(),
                        expected_output=expected_output.strip() if expected_output else None,
                        tags=tag_list,
                    )
                    eval_service.save_test_case(tc)
                    with st.spinner(f"Executing isolated test case '{tc.name}'..."):
                        res = eval_service.run_single_test_case(tc)
                        st.session_state["single_case_results"][tc.id] = res
                    st.success(f"Execution complete for '{tc.id}'. Overall Score: {res.evaluation.overall_score:.1f}/10")
                    # Keep editor open so user can inspect or edit

        with col_act3:
            if st.button("Cancel", use_container_width=True, key="cancel_editor_btn"):
                st.session_state["editing_case"] = None
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def _render_test_case_card(eval_service: EvaluationService, case: TestCase, dataset_id: str):
    """Render an individual test case with independent execution, editing, and results view."""
    # Check if there is an active execution result for this case
    res: Optional[BatchCaseResult] = st.session_state["single_case_results"].get(case.id)

    header_score = ""
    if res and res.evaluation.is_available:
        header_score = f"— Score: {res.evaluation.overall_score:.1f}/10"

    with st.expander(f"{case.name} ({case.id}) [{case.task_type}] {header_score}", expanded=False):
        # Card Action Buttons
        col_hdr1, col_hdr2, col_hdr3, col_hdr_space = st.columns([1.2, 1, 1, 3])
        with col_hdr1:
            if st.button("Run Test Case", key=f"run_btn_{case.id}", type="primary"):
                with st.spinner(f"Executing test case '{case.name}'..."):
                    try:
                        result = eval_service.run_single_test_case(case)
                        st.session_state["single_case_results"][case.id] = result
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Single test execution failed: {e}", exc_info=True)
                        st.error(format_safe_user_error(e, "Unable to complete the AI request. Please check your API configuration and try again."))
        with col_hdr2:
            if st.button("Edit", key=f"edit_btn_{case.id}"):
                st.session_state["editing_case"] = {
                    "id": case.id,
                    "name": case.name,
                    "task_type": case.task_type,
                    "prompt": case.prompt or "",
                    "source_data": case.source_data or "",
                    "criteria": case.criteria or "",
                    "expected_output": case.expected_output or "",
                    "tags": ", ".join(case.tags) if case.tags else "",
                    "is_new": False,
                }
                st.rerun()
        with col_hdr3:
            if st.button("Delete", key=f"del_btn_{case.id}", type="secondary"):
                eval_service.delete_test_case(case.id)
                if case.id in st.session_state["single_case_results"]:
                    del st.session_state["single_case_results"][case.id]
                st.success(f"Deleted test case '{case.id}'.")
                st.rerun()

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        # Definitions
        col_c_in, col_c_meta = st.columns([1, 1])
        with col_c_in:
            st.markdown("**Test Case Prompt / Instructions**")
            st.markdown(
                f"<div class='dev-code-box' style='margin-bottom: 0.75rem; color: #1E293B;'>"
                f"{case.prompt if case.prompt else '<em style=\"color:#94A3B8;\">No prompt defined</em>'}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("**Test Case Source Data / Input**")
            st.markdown(
                f"<div class='dev-code-box' style='max-height: 160px; overflow-y: auto; color: #334155;'>"
                f"{case.source_data}"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_c_meta:
            st.markdown("**Evaluation Criteria**")
            st.markdown(
                f"<div class='dev-code-box' style='margin-bottom: 0.75rem; color: #1E293B;'>"
                f"{case.criteria if case.criteria else '<em style=\"color:#94A3B8;\">Default rubric criteria</em>'}"
                f"</div>",
                unsafe_allow_html=True,
            )
            if case.expected_output:
                st.markdown("**Expected Output**")
                st.markdown(
                    f"<div class='dev-code-box' style='max-height: 120px; overflow-y: auto; color: #334155;'>"
                    f"{case.expected_output}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if case.tags:
                tag_badges = " ".join([f"<span class='badge-base badge-status-completed'>#{t}</span>" for t in case.tags])
                st.markdown(f"<div style='margin-top: 0.5rem;'>{tag_badges}</div>", unsafe_allow_html=True)

        # Results Section (if executed)
        if res:
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='border-top: 1px solid #E2E8F0; padding-top: 0.75rem;'></div>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-size: 0.95rem; font-weight: 600; color: #0F172A;'>Latest Execution & Evaluation</h4>", unsafe_allow_html=True)

            col_res_l, col_res_r = st.columns([1, 1])
            with col_res_l:
                st.markdown("**Generated LLM Response**")
                st.markdown(
                    f"<div class='dev-code-box' style='background-color: #FFFFFF; border: 1px solid #CBD5E1; color: #0F172A; max-height: 220px; overflow-y: auto; white-space: pre-wrap;'>"
                    f"{res.response_text}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size: 0.75rem; color: #64748B; font-family: ui-monospace, monospace; margin-top: 0.25rem;'>"
                    f"Latency: {(res.latency_ms / 1000.0):.2f}s | Response ID: {res.response_id or 'isolated'}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col_res_r:
                if res.evaluation.is_available:
                    score_badge = render_score_badge(res.evaluation.overall_score)
                    st.markdown(f"**Overall Evaluation:** {score_badge}", unsafe_allow_html=True)

                    st.markdown(
                        f"<div style='font-size: 0.825rem; font-family: ui-monospace, monospace; color: #334155; margin: 0.5rem 0; line-height: 1.6;'>"
                        f"• Relevance: <strong>{res.evaluation.relevance}/10</strong><br>"
                        f"• Completeness: <strong>{res.evaluation.completeness}/10</strong><br>"
                        f"• Instruction Following: <strong>{res.evaluation.instruction_following}/10</strong><br>"
                        f"• Factual Consistency: <strong>{res.evaluation.effective_factual_consistency}/10</strong><br>"
                        f"• Format Compliance: <strong>{res.evaluation.effective_format_compliance}/10</strong><br>"
                        f"• Conciseness: <strong>{res.evaluation.effective_conciseness}/10</strong>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    if res.evaluation.feedback:
                        st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #475569;'>Critique:</div>", unsafe_allow_html=True)
                        for fb in res.evaluation.feedback:
                            st.markdown(f"<div style='font-size: 0.8rem; color: #334155;'>• {fb}</div>", unsafe_allow_html=True)
                else:
                    st.warning(f"Evaluation could not be completed: {res.evaluation.error_message}")


def _execute_dataset_all_tests(eval_service: EvaluationService, cases: List[TestCase], dataset_id: str):
    """Execute complete dataset with each test case using its own isolated prompt and source data."""
    prog = st.progress(0.0)
    status = st.empty()

    def update_cb(curr: int, total: int, msg: str):
        prog.progress(float(curr) / float(total))
        status.markdown(
            f"<div style='font-family: ui-monospace, monospace; font-size: 0.825rem;'><strong>Case {curr}/{total}:</strong> {msg}</div>",
            unsafe_allow_html=True,
        )

    try:
        report = eval_service.run_batch_evaluation(
            prompt_text=None,  # None means each test case uses its own stored prompt!
            test_cases=cases,
            dataset_id=dataset_id,
            progress_cb=update_cb,
        )
        prog.progress(1.0)
        status.markdown(
            f"<div style='color: #16A34A; font-weight: 600; font-size: 0.85rem;'>Completed all {report.num_cases} tests across dataset '{dataset_id}'.</div>",
            unsafe_allow_html=True,
        )
        st.session_state["dataset_reports"][dataset_id] = report
        # Also sync single case results
        for cr in report.case_results:
            st.session_state["single_case_results"][cr.test_case_id] = cr
    except Exception as e:
        status.empty()
        logger.error(f"Dataset execution failed: {e}", exc_info=True)
        st.error(format_safe_user_error(e, "The dataset execution could not be completed. Please try again."))


def _render_batch_evaluation_report(rep: BatchEvaluationReport):
    """Render aggregate results and performance table for the full dataset run."""
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A;'>Complete Dataset Benchmark Results</h3>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Mean Composite Score", f"{rep.avg_overall_score:.1f} / 10")
    with m2:
        render_metric_card("Mean Latency", f"{(rep.avg_latency_ms / 1000.0):.2f}s")
    with m3:
        render_metric_card("Total Tests Executed", str(rep.num_cases))
    with m4:
        render_metric_card("Dataset ID", rep.dataset_id)

    # Performance Table
    table_rows = []
    for cr in rep.case_results:
        ev = cr.evaluation
        table_rows.append(
            {
                "Case ID": cr.test_case_id,
                "Case Name": cr.test_case_name,
                "Overall Score": f"{ev.overall_score:.1f}" if ev.is_available else "Unavailable",
                "Relevance": f"{ev.relevance}/10" if ev.is_available else "N/A",
                "Completeness": f"{ev.completeness}/10" if ev.is_available else "N/A",
                "Instruction Following": f"{ev.instruction_following}/10" if ev.is_available else "N/A",
                "Factual Consistency": f"{ev.effective_factual_consistency}/10" if ev.is_available else "N/A",
                "Latency": f"{(cr.latency_ms / 1000.0):.2f}s",
                "Completion Preview": cr.response_text[:75] + ("..." if len(cr.response_text) > 75 else ""),
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


def _validate_test_case_fields(
    case_id: str,
    name: str,
    prompt: str,
    source_data: str,
    criteria: str,
) -> List[str]:
    """Validate mandatory test case fields."""
    errors = []
    if not case_id.strip():
        errors.append("Test Case ID is required.")
    if not name.strip():
        errors.append("Test Case Name is required.")
    if not prompt.strip():
        errors.append("Prompt / Instructions is required.")
    if not source_data.strip():
        errors.append("Source Data / Input is required.")
    if not criteria.strip():
        errors.append("Evaluation Criteria is required.")
    return errors
