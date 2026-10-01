"""Professional Developer SaaS UI Components and Design System for PromptPilot.

Provides clean, minimal, technical styles inspired by modern developer platforms
(Linear, Datadog, Vercel, LangSmith) with zero flashy AI effects.
"""

from typing import Dict, List, Optional
import pandas as pd
import streamlit as st


def inject_custom_css():
    """Inject restrained, professional developer SaaS stylesheet."""
    st.markdown(
        """
        <style>
        /* Base typography & structural refinement */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #0F172A;
            -webkit-font-smoothing: antialiased;
        }

        /* Page Headers */
        .page-header-container {
            padding-bottom: 0.85rem;
            margin-bottom: 1.25rem;
            border-bottom: 1px solid #E2E8F0;
        }
        .page-title {
            font-size: 1.45rem;
            font-weight: 600;
            color: #0F172A;
            letter-spacing: -0.02em;
            margin: 0 0 0.2rem 0;
            line-height: 1.25;
        }
        .page-subtitle {
            font-size: 0.875rem;
            color: #64748B;
            margin: 0;
            line-height: 1.4;
        }

        /* Technical Developer Cards / Panels */
        .dev-panel {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .dev-panel-header {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #475569;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Metric Stat Box */
        .stat-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
        }
        .stat-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-bottom: 0.25rem;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #0F172A;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        .stat-caption {
            font-size: 0.75rem;
            color: #94A3B8;
            margin-top: 0.25rem;
        }

        /* Badges & Tags */
        .badge-base {
            display: inline-flex;
            align-items: center;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.15rem 0.55rem;
            border-radius: 4px;
            letter-spacing: 0.02em;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            border: 1px solid transparent;
        }
        .badge-version {
            background-color: #F1F5F9;
            color: #0F172A;
            border-color: #CBD5E1;
        }
        .badge-strategy {
            background-color: #EFF6FF;
            color: #1D4ED8;
            border-color: #BFDBFE;
        }
        .badge-score-high {
            background-color: #F0FDF4;
            color: #166534;
            border-color: #BBF7D0;
        }
        .badge-score-med {
            background-color: #FFFBEB;
            color: #92400E;
            border-color: #FDE68A;
        }
        .badge-score-low {
            background-color: #FEF2F2;
            color: #991B1B;
            border-color: #FECACA;
        }
        .badge-status-completed {
            background-color: #F0FDF4;
            color: #166534;
            border-color: #BBF7D0;
        }
        .badge-status-running {
            background-color: #EFF6FF;
            color: #1E40AF;
            border-color: #BFDBFE;
        }
        .badge-status-failed {
            background-color: #FEF2F2;
            color: #991B1B;
            border-color: #FECACA;
        }

        /* Code & Monospace Containers */
        .dev-code-box {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.825rem;
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 4px;
            padding: 0.75rem 1rem;
            color: #1E293B;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }

        /* Concept Callouts & Disclaimers */
        .concept-callout {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 3px solid #2563EB;
            border-radius: 4px;
            padding: 0.75rem 1rem;
            margin: 1rem 0;
            font-size: 0.85rem;
            color: #334155;
            line-height: 1.45;
        }
        .disclaimer-box {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 3px solid #94A3B8;
            border-radius: 4px;
            padding: 0.65rem 0.9rem;
            margin: 0.85rem 0;
            font-size: 0.8rem;
            color: #64748B;
            line-height: 1.4;
        }

        /* Compact Score Indicators */
        .score-bar-wrapper {
            margin-bottom: 0.5rem;
        }
        .score-bar-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            margin-bottom: 0.2rem;
            color: #334155;
        }
        .score-bar-track {
            height: 6px;
            background-color: #E2E8F0;
            border-radius: 3px;
            overflow: hidden;
        }
        .score-bar-fill {
            height: 100%;
            border-radius: 3px;
            background-color: #2563EB;
        }

        /* Clean Empty States */
        .empty-state-box {
            text-align: center;
            padding: 2.5rem 1.5rem;
            border: 1px dashed #CBD5E1;
            border-radius: 6px;
            background-color: #F8FAFC;
            margin: 1.5rem 0;
        }
        .empty-state-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #1E293B;
            margin-bottom: 0.35rem;
        }
        .empty-state-desc {
            font-size: 0.85rem;
            color: #64748B;
            max-width: 460px;
            margin: 0 auto 1rem auto;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str,
    action_button: Optional[dict] = None,
    icon: Optional[str] = None,
    **kwargs,
):
    """Render standard professional header across all developer views without emojis."""
    st.markdown(
        f"""
        <div class='page-header-container'>
            <h1 class='page-title'>{title}</h1>
            <p class='page-subtitle'>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evaluation_disclaimer():
    """Render technical non-misleading evaluation notice."""
    st.markdown(
        """
        <div class='disclaimer-box'>
            <strong>Automated Evaluation Note:</strong> Metrics and qualitative critiques are computed using an automated
            rubric model. Automated evaluations provide rapid directional benchmarks; critical tasks should always be
            audited with human-in-the-loop validation.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, caption: Optional[str] = None):
    """Render clean, compact technical metric card."""
    caption_html = f"<div class='stat-caption'>{caption}</div>" if caption else ""
    st.markdown(
        f"""
        <div class='stat-card'>
            <div class='stat-label'>{label}</div>
            <div class='stat-value'>{value}</div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_badge(score: float, max_score: float = 10.0) -> str:
    """Format numeric score into a restrained, high-contrast technical badge."""
    if score >= 8.0:
        cls = "badge-score-high"
    elif score >= 5.5:
        cls = "badge-score-med"
    else:
        cls = "badge-score-low"
    return f"<span class='badge-base {cls}'>{score:.1f} / {max_score:.1f}</span>"


def render_status_badge(status: str) -> str:
    """Render compact status badge (Completed, Running, Failed, Draft)."""
    s_lower = status.lower()
    if "complete" in s_lower or "success" in s_lower:
        cls = "badge-status-completed"
    elif "run" in s_lower or "evaluat" in s_lower:
        cls = "badge-status-running"
    elif "fail" in s_lower or "error" in s_lower or "regress" in s_lower:
        cls = "badge-status-failed"
    else:
        cls = "badge-version"
    return f"<span class='badge-base {cls}'>{status}</span>"


def render_horizontal_score_bar(label: str, score: float, max_score: float = 10.0):
    """Render compact horizontal progress bar for score dimensions."""
    pct = min(100.0, max(0.0, (score / max_score) * 100))
    fill_color = "#16A34A" if score >= 8.0 else "#D97706" if score >= 5.0 else "#DC2626"
    st.markdown(
        f"""
        <div class='score-bar-wrapper'>
            <div class='score-bar-header'>
                <span>{label}</span>
                <span style='font-weight: 600; font-family: ui-monospace, monospace;'>{score:.1f}/{max_score:.0f}</span>
            </div>
            <div class='score-bar-track'>
                <div class='score-bar-fill' style='width: {pct}%; background-color: {fill_color};'></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_radar_or_bar_metrics(criteria_scores: Dict[str, float], max_score: float = 10.0):
    """Render clean, responsive technical metric bars and comparative chart for criteria dimensions."""
    if not criteria_scores:
        return

    col_chart, col_bars = st.columns([3, 2])
    with col_chart:
        df = pd.DataFrame(
            [{"Dimension": k, "Score": float(v)} for k, v in criteria_scores.items()]
        )
        st.bar_chart(
            df.set_index("Dimension"),
            y="Score",
            color="#2563EB",
            height=230,
            use_container_width=True,
        )
    with col_bars:
        st.markdown("<div style='padding-top: 0.35rem;'>", unsafe_allow_html=True)
        for label, score in criteria_scores.items():
            render_horizontal_score_bar(label, float(score), max_score=max_score)
        st.markdown("</div>", unsafe_allow_html=True)



def render_empty_state(title: str, description: str, button_label: Optional[str] = None) -> bool:
    """Render clean, non-decorative empty state with optional action button."""
    st.markdown(
        f"""
        <div class='empty-state-box'>
            <div class='empty-state-title'>{title}</div>
            <div class='empty-state-desc'>{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if button_label:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            return st.button(button_label, type="primary", use_container_width=True)
    return False


def render_api_warning():
    """Display calm, informative API configuration notice."""
    st.warning(
        "**API Configuration Notice:** Gemini API Key is not configured. "
        "Prompts can be tested using the **Offline Demo Mode** (zero quota required) or configured in **Settings**."
    )
