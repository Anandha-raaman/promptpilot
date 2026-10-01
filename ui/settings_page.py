"""Professional Developer SaaS Settings and Diagnostics View for PromptPilot."""

import streamlit as st

from config.settings import Settings
from llm.base import BaseLLMProvider
from ui.components import render_page_header


def render_settings_page(provider: BaseLLMProvider, settings: Settings):
    """Render configuration, model selectors, safety boundaries, and API connectivity testing."""
    render_page_header(
        title="Settings & System Diagnostics",
        subtitle="Manage model parameters, API credentials, safety boundaries, and local database storage.",
    )

    # 1. API Configuration
    st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A;'>1. API Configuration</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        is_conn = provider.is_available()
        status_text = "Configured" if is_conn else "Not configured"
        status_color = "#16A34A" if is_conn else "#DC2626"
        provider_name = "Google Gemini" if "Gemini" in type(provider).__name__ else "Offline Mock Engine" if "Mock" in type(provider).__name__ else type(provider).__name__
        st.markdown(f"**API status:** <span style='color: {status_color}; font-weight: 600;'>{status_text}</span>", unsafe_allow_html=True)
        st.markdown(f"**Provider name:** {provider_name}")
        st.markdown(f"**Model name:** <code>{settings.llm_model}</code>", unsafe_allow_html=True)

    with col2:
        if st.button("Test API Handshake", use_container_width=True):
            if not provider.is_available():
                st.error("Cannot test: API is not configured. Configure GEMINI_API_KEY in .env, Streamlit secrets, or below.")
            else:
                with st.spinner("Testing API handshake with Gemini..."):
                    try:
                        resp = provider.generate_text("Reply with exactly: 'OK'", max_output_tokens=10)
                        st.success(f"Handshake successful (Latency: {resp.latency_ms:.0f} ms)")
                    except Exception as e:
                        from utils.logging import format_safe_user_error, get_logger
                        get_logger("settings_page").error(f"Handshake failed: {e}", exc_info=True)
                        clean_err = format_safe_user_error(e, "Unable to complete the AI request. Please check your API configuration and try again.")
                        st.error(f"Handshake failed: {clean_err}")

    st.markdown("---")

    # 2. Runtime Credential Configuration
    st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A;'>2. Runtime API Credentials</h3>", unsafe_allow_html=True)
    st.caption("API keys entered here apply to the active runtime session. For persistence, specify GEMINI_API_KEY in your local .env file or Streamlit secrets.")

    if provider.is_available() or settings.has_valid_api_key:
        st.success("API key configured")

    session_key = st.text_input(
        "Gemini API Key",
        type="password",
        value="",
        placeholder="Enter API key to update..." if (provider.is_available() or settings.has_valid_api_key) else "Paste API key here...",
        help="Obtain key from Google AI Studio (https://aistudio.google.com/)",
    )
    if st.button("Apply Key to Session"):
        if session_key.strip():
            clean_key = session_key.strip()
            settings.gemini_api_key = clean_key
            if hasattr(provider, "api_key"):
                provider.api_key = clean_key
                provider._client = None
            st.success("API Key applied to current session.")
            st.rerun()
        else:
            st.warning("Please enter an API key.")

    st.markdown("---")

    # 3. Model Role Allocations
    st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A;'>3. Model Architecture & Roles</h3>", unsafe_allow_html=True)
    available_models = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]
    curr_idx = available_models.index(settings.llm_model) if settings.llm_model in available_models else 0
    new_model = st.selectbox("Active Primary Model", available_models, index=curr_idx)
    if new_model != settings.llm_model:
        settings.llm_model = new_model
        st.success(f"Active model switched to {new_model}")
        st.rerun()

    st.caption("PromptPilot modularly routes pipeline stages. By default, all stages leverage your primary model.")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.text_input("Prompt Analyzer Model", value=settings.analyzer_model, disabled=True)
        st.text_input("Response Generator Model", value=settings.response_model, disabled=True)
    with m_col2:
        st.text_input("Prompt Generator Model", value=settings.generator_model, disabled=True)
        st.text_input("Response Evaluator Model", value=settings.eval_model, disabled=True)

    st.markdown("---")

    # 4. Safety & Cost Controls
    st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A;'>4. Safety Boundaries & Guardrails</h3>", unsafe_allow_html=True)
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.metric("Max Input Length", f"{settings.max_input_chars:,} chars")
    with s_col2:
        st.metric("Max Prompt Length", f"{settings.max_prompt_chars:,} chars")
    with s_col3:
        st.metric("Rate Limit Cooldown", f"{settings.rate_limit_cooldown_seconds}s")

    st.markdown("---")

    # 5. Database Diagnostics
    st.markdown("<h3 style='font-size: 1.05rem; font-weight: 600; color: #0F172A;'>5. Local SQLite Storage</h3>", unsafe_allow_html=True)
    db_path = settings.get_resolved_db_path()
    st.markdown(f"**Database Location:** <code>{db_path}</code>", unsafe_allow_html=True)
    st.markdown(f"**Persistence Status:** {'Ready & Active' if db_path.exists() else 'Not yet created'}")
