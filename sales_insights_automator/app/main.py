"""
Sales Insights Automator — Streamlit App

Entry point.  Run with:
    streamlit run app/main.py

From the project root (sales_insights_automator/).
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so all modules resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from app import state

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Sales Insights Automator",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Sidebar progress tracker ──────────────────────────────────────────────────
def sidebar_progress() -> None:
    st.sidebar.title("📊 Sales Insights")
    st.sidebar.markdown("---")

    progress = state.progress_status()
    steps = [
        ("📂 Upload & Profile",  progress["uploaded"]),
        ("🔧 Schema Setup",      progress["schema"]),
        ("📈 Dashboard",         progress["analysed"]),
        ("🤖 AI Insights",       progress["insights"]),
    ]
    for label, done in steps:
        icon = "✅" if done else "⬜"
        st.sidebar.markdown(f"{icon} {label}")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start Over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


sidebar_progress()

# ── Home page content ─────────────────────────────────────────────────────────
st.title("Sales Insights Automator")
st.markdown(
    "**AI-powered sales analytics** — upload any sales dataset, map your "
    "columns, explore interactive charts, and generate natural-language "
    "business insights in minutes."
)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 📂 Upload & Profile")
    st.markdown(
        "Upload a CSV file or connect a SQLite database. "
        "The tool immediately runs a **data quality report** — "
        "missing values, duplicates, outliers, and a quality score."
    )

with col2:
    st.markdown("### 🔧 Schema Setup")
    st.markdown(
        "Your columns are **auto-detected** and mapped to semantic roles "
        "(revenue, date, region…). Confirm with dropdowns — "
        "no configuration files needed."
    )

with col3:
    st.markdown("### 📈 Dashboard")
    st.markdown(
        "Interactive charts: revenue trends, regional breakdowns, "
        "top products, sales rep performance, discount impact, "
        "and day-of-week patterns."
    )

with col4:
    st.markdown("### 🤖 AI Insights")
    st.markdown(
        "GPT-powered natural-language summaries of your data. "
        "Choose a template — full report, executive summary, "
        "recommendations, or anomaly detection. "
        "Privacy-protected before sending."
    )

st.markdown("---")

# ── Quick start guide ─────────────────────────────────────────────────────────
with st.expander("How to get started", expanded=True):
    st.markdown("""
1. Click **📂 Upload & Profile** in the sidebar to upload your data
2. Click **🔧 Schema Setup** — confirm which column is revenue, date, etc.
3. Click **📈 Dashboard** to see all charts
4. Click **🤖 AI Insights** to generate a written business report

*No data file yet? The sample dataset is pre-loaded on the Upload page.*
""")

# ── Sample data shortcut ──────────────────────────────────────────────────────
if not state.has(state.RAW_DF):
    st.info(
        "👋 No data loaded yet. Head to **📂 Upload & Profile** in the sidebar "
        "to get started, or click the button below to load the built-in sample."
    )
    if st.button("Load sample dataset (500 rows)", type="primary"):
        import pandas as pd
        sample_path = Path(__file__).parent.parent / "data" / "samples" / "sample_sales.csv"
        df = pd.read_csv(sample_path)
        state.set(state.RAW_DF, df)
        state.set(state.FILE_NAME, "sample_sales.csv")
        st.success("Sample data loaded! Navigate to **📂 Upload & Profile** to see the quality report.")
        st.rerun()
else:
    fname = state.get(state.FILE_NAME, "your dataset")
    progress = state.progress_status()
    completed = sum(progress.values())
    st.success(
        f"**{fname}** is loaded. "
        f"{completed}/4 pipeline stages complete."
    )
    if not progress["analysed"]:
        st.info("Continue from where you left off in the sidebar.")
