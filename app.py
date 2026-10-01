"""PromptPilot — AI Prompt Testing & Optimization Workbench.

Streamlit main application entry point and router.
"""

import streamlit as st

from config.settings import Settings, get_settings
from database.db import DatabaseManager
from database.repository import ExperimentRepository
from llm.gemini_provider import GeminiProvider
from llm.mock_provider import MockLLMProvider
from services.evaluation_service import EvaluationService
from services.experiment_service import ExperimentService
from ui.analyzer_page import render_prompt_analyzer_page
from ui.batch_page import render_batch_page
from ui.components import inject_custom_css
from ui.dashboard import render_dashboard
from ui.experiment_page import render_new_experiment
from ui.history_page import render_history_page
from ui.library_page import render_library_page
from ui.playground_page import render_playground_page
from ui.regression_page import render_regression_page
from ui.settings_page import render_settings_page
from utils.logging import get_logger

logger = get_logger("app")

# Page Configuration
st.set_page_config(
    page_title="PromptPilot — Prompt Testing & Optimization Workbench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()


def get_or_create_app_state():
    """Initialize application dependencies and singleton services in Streamlit session state."""
    if "settings" not in st.session_state:
        st.session_state["settings"] = get_settings()

    settings: Settings = st.session_state["settings"]

    if "db_manager" not in st.session_state:
        st.session_state["db_manager"] = DatabaseManager(settings=settings)

    if "repo" not in st.session_state:
        st.session_state["repo"] = ExperimentRepository(st.session_state["db_manager"])

    # Engine Mode: Live vs Mock Demo
    use_mock = st.session_state.get("use_mock_mode", False)
    if use_mock:
        active_provider = MockLLMProvider()
    else:
        active_provider = GeminiProvider(settings=settings)

    st.session_state["provider"] = active_provider

    # Services
    st.session_state["experiment_service"] = ExperimentService(
        provider=active_provider,
        repository=st.session_state["repo"],
        settings=settings,
    )

    st.session_state["evaluation_service"] = EvaluationService(
        provider=active_provider,
        settings=settings,
        repository=st.session_state["repo"],
    )


def main():
    get_or_create_app_state()

    settings: Settings = st.session_state["settings"]
    repo: ExperimentRepository = st.session_state["repo"]
    provider = st.session_state["provider"]
    experiment_service: ExperimentService = st.session_state["experiment_service"]
    evaluation_service: EvaluationService = st.session_state["evaluation_service"]

    # Sidebar Navigation & System Diagnostics
    with st.sidebar:
        st.markdown(
            """
            <div style='padding: 0.2rem 0 0.8rem 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 0.8rem;'>
                <div style='font-size: 1.15rem; font-weight: 700; color: #0F172A; letter-spacing: -0.02em;'>
                    PromptPilot
                </div>
                <div style='font-size: 0.775rem; color: #64748B; margin-top: 0.1rem;'>
                    Prompt Workbench
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_options = [
            "Dashboard",
            "Prompt Analyzer",
            "New Experiment",
            "Prompt Playground",
            "Prompt Library",
            "Test Datasets",
            "Experiment History",
            "Regression Tests",
            "Settings",
        ]

        # Handle programmatic navigation redirects
        default_nav = st.session_state.get("nav_page", "Dashboard")
        if default_nav not in nav_options:
            default_nav = "Dashboard"

        nav_index = nav_options.index(default_nav)
        selected_page = st.radio(
            "Navigation",
            nav_options,
            index=nav_index,
            label_visibility="collapsed",
        )
        st.session_state["nav_page"] = selected_page

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")

        # Engine Mode toggle
        st.caption("EXECUTION ENGINE")
        mode_choice = st.radio(
            "Execution Engine",
            ["Live Gemini API", "Offline Demo Mode"],
            index=1 if st.session_state.get("use_mock_mode", False) else 0,
            label_visibility="collapsed",
            help="Switch to Demo Mode to test the workbench without consuming API quota.",
        )
        is_mock_chosen = (mode_choice == "Offline Demo Mode")
        if is_mock_chosen != st.session_state.get("use_mock_mode", False):
            st.session_state["use_mock_mode"] = is_mock_chosen
            st.rerun()

        # System Diagnostics Footer
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        if is_mock_chosen:
            st.markdown("<div style='font-size: 0.8rem; color: #334155;'>● <strong>API Status:</strong> <span style='color: #6366F1;'>Offline Demo</span></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.775rem; color: #64748B;'>Model: <code>mock-demo-engine</code></div>", unsafe_allow_html=True)
        else:
            if provider.is_available():
                st.markdown("<div style='font-size: 0.8rem; color: #334155;'>● <strong>API Status:</strong> <span style='color: #16A34A;'>Connected</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 0.8rem; color: #334155;'>● <strong>API Status:</strong> <span style='color: #DC2626;'>Key Required</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.775rem; color: #64748B;'>Model: <code>{settings.llm_model}</code></div>", unsafe_allow_html=True)

        st.markdown("<div style='font-size: 0.75rem; color: #94A3B8; margin-top: 0.4rem;'>Version: <code>v1.2.0-prod</code></div>", unsafe_allow_html=True)

    # Page Routing
    if selected_page == "Dashboard":
        render_dashboard(repo=repo, settings=settings)
    elif selected_page == "Prompt Analyzer":
        render_prompt_analyzer_page(service=experiment_service, settings=settings)
    elif selected_page == "New Experiment":
        render_new_experiment(service=experiment_service, settings=settings)
    elif selected_page == "Prompt Playground":
        render_playground_page(service=experiment_service, repo=repo, settings=settings)
    elif selected_page == "Prompt Library":
        render_library_page()
    elif selected_page == "Test Datasets":
        render_batch_page(eval_service=evaluation_service, settings=settings)
    elif selected_page == "Experiment History":
        render_history_page(repo=repo)
    elif selected_page == "Regression Tests":
        render_regression_page(repo=repo, settings=settings)
    elif selected_page == "Settings":
        render_settings_page(provider=provider, settings=settings)


if __name__ == "__main__":
    main()
