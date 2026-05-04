"""
Page 4 — AI Insights

Generates natural-language business insights from the AnalysisResult
using the OpenAI API (or dry-run mode if no key is configured).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
from dotenv import load_dotenv

# Load .env BEFORE importing pipeline modules so os.getenv() picks up the key
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

import streamlit as st
from app import state
from ai.insight_generator import InsightGenerator
from privacy.config import PrivacyConfig

st.set_page_config(page_title="AI Insights", page_icon="🤖", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Sales Insights")
st.sidebar.markdown("---")
st.sidebar.markdown("**Step 4 of 4** — Generate AI-powered business insights.")

# ── Guard: need analysis result ───────────────────────────────────────────────
if not state.has(state.ANALYSIS_RESULT):
    st.warning("No analysis available yet. Please complete **🔧 Schema Setup** first.")
    if st.button("← Go to Schema Setup"):
        st.switch_page("pages/2_Schema_Setup.py")
    st.stop()

result = state.get(state.ANALYSIS_RESULT)

st.title("🤖 AI Insights")
st.markdown(
    "Generate a written business report from your analysis. "
    "The AI never sees your raw data — only anonymised summaries."
)
st.markdown("---")

# ── API key status ────────────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY", "")
has_key = bool(api_key and not api_key.startswith("sk-your"))

col_key, col_status = st.columns([3, 1])
with col_key:
    if not has_key:
        st.warning(
            "No OpenAI API key found. Running in **dry-run mode** — "
            "a placeholder report will be shown instead of a real AI response. "
            "Add your key to the `.env` file to enable live generation."
        )
    else:
        st.success("OpenAI API key detected — live generation enabled.")

# ── Privacy settings ──────────────────────────────────────────────────────────
with st.expander("Privacy settings", expanded=False):
    st.markdown(
        "These settings control what data is anonymised before being sent "
        "to the AI. Your original data is never modified."
    )
    c1, c2 = st.columns(2)
    mask_names      = c1.checkbox("Mask rep / product names", value=True)
    mask_regions    = c1.checkbox("Mask region names", value=False)
    round_rev       = c1.checkbox("Round revenue figures", value=True)
    no_training     = c2.checkbox("Request no AI training on this data", value=True)
    enable_audit    = c2.checkbox("Enable audit log", value=True)

    privacy_config = PrivacyConfig(
        mask_rep_names      = mask_names,
        mask_product_names  = mask_names,
        mask_region_names   = mask_regions,
        round_revenue_to    = 1000 if round_rev else 0,
        request_no_training = no_training,
        enable_audit_log    = enable_audit,
    )

    st.markdown("**Active protections:**")
    for line in privacy_config.summary().split("\n"):
        if line.strip():
            st.caption(line)

st.markdown("---")

# ── Insight template selector ─────────────────────────────────────────────────
st.subheader("Choose a report template")
TEMPLATES = {
    "Full Report":          "full_report",
    "Executive Summary":    "executive_summary",
    "Recommendations":      "recommendations",
    "Anomaly Detection":    "anomalies",
}
template_label = st.radio(
    "Template",
    list(TEMPLATES.keys()),
    horizontal = True,
    help = (
        "**Full Report** — comprehensive analysis with all sections | "
        "**Executive Summary** — 3–5 bullet points for leadership | "
        "**Recommendations** — actionable next steps | "
        "**Anomaly Detection** — unusual patterns worth investigating"
    ),
)
template = TEMPLATES[template_label]

# ── Generate button ───────────────────────────────────────────────────────────
st.markdown("---")

previous_report = state.get(state.INSIGHT_REPORT)

generate_label = (
    "Generate Insights (dry-run)" if not has_key
    else "Generate Insights with AI"
)

if st.button(generate_label, type="primary"):
    # Re-read env inside the button handler to guarantee the key is fresh
    from dotenv import load_dotenv as _ld
    _ld(dotenv_path=_env_path, override=True)
    live_key     = os.getenv("OPENAI_API_KEY", "")
    live_has_key = bool(live_key and not live_key.startswith("sk-your"))

    with st.spinner("Generating insights… this may take 10–20 seconds."):
        try:
            from ai.llm_client import LLMClient
            # Pass the key explicitly — bypasses any stale cached module default
            client = LLMClient(api_key=live_key) if live_has_key else None
            generator = InsightGenerator(
                client         = client,
                privacy_config = privacy_config,
                dry_run        = not live_has_key,
            )
            report = generator.generate(result, template=template)
            state.set(state.INSIGHT_REPORT, report)
            st.rerun()
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

# ── Display report ────────────────────────────────────────────────────────────
if state.has(state.INSIGHT_REPORT):
    report = state.get(state.INSIGHT_REPORT)
    st.markdown("---")
    st.subheader("Generated Report")

    # Status badges
    badge_col1, badge_col2, badge_col3 = st.columns(3)
    badge_col1.metric("Template", template_label)
    badge_col2.metric("Mode", "Dry-run" if report.is_dry_run else "Live AI")
    if report.total_tokens:
        badge_col3.metric("Tokens used", f"{report.total_tokens:,}")

    st.markdown("---")

    if report.is_dry_run:
        st.info(
            "This is a **dry-run preview** — add your OpenAI API key to "
            "`.env` and regenerate for a real AI response."
        )

    # Render the report content
    st.markdown(report.insights_text)

    st.markdown("---")

    # Download options
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label     = "Download as Markdown",
            data      = report.to_markdown(),
            file_name = "ai_insights.md",
            mime      = "text/markdown",
        )
    with dl2:
        st.download_button(
            label     = "Download as JSON",
            data      = report.to_json(),
            file_name = "ai_insights.json",
            mime      = "application/json",
        )

# ── Full report metadata (collapsible) ───────────────────────────────────────
if state.has(state.INSIGHT_REPORT):
    report = state.get(state.INSIGHT_REPORT)
    with st.expander("Report metadata"):
        col_m1, col_m2 = st.columns(2)
        col_m1.markdown(f"**Model:** {report.model or 'n/a'}")
        col_m1.markdown(f"**Latency:** {report.latency_ms:.0f} ms")
        col_m1.markdown(f"**Rows analysed:** {report.row_count:,}")
        col_m2.markdown(f"**Prompt tokens:** {report.prompt_tokens:,}")
        col_m2.markdown(f"**Completion tokens:** {report.completion_tokens:,}")
        col_m2.markdown(f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M')}")

st.markdown("---")
st.caption(
    "Privacy notice: only anonymised statistical summaries are sent to the AI. "
    "Individual transaction records, customer names, and exact figures are "
    "never transmitted. An audit log is written locally for compliance."
)
