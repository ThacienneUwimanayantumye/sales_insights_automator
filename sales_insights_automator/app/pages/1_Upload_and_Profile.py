"""
Page 1 — Upload & Profile

Lets the user upload a CSV file or use the built-in sample dataset.
Immediately runs the DataProfiler and shows a full quality report.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import io
import pandas as pd
import streamlit as st

from app import state
from profiling.profiler import DataProfiler
from app.components.profile_table import (
    render_quality_score,
    render_flags,
    render_column_table,
    render_numeric_stats,
    render_recommendations,
)

st.set_page_config(page_title="Upload & Profile", page_icon="📂", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Sales Insights")
st.sidebar.markdown("---")
st.sidebar.markdown("**Step 1 of 4** — Upload your data and inspect its quality before cleaning.")

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📂 Upload & Profile")
st.markdown(
    "Upload a CSV file to begin. The profiler will immediately scan every "
    "column for missing values, duplicates, outliers, and data types."
)
st.markdown("---")

# ── File upload ───────────────────────────────────────────────────────────────
col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv", "tsv"],
        help="Drag and drop or click to browse. Max 200 MB.",
    )

with col_sample:
    st.markdown("**Or use the built-in sample:**")
    if st.button("Load sample dataset (500 rows)", use_container_width=True):
        sample_path = (
            Path(__file__).resolve().parent.parent.parent
            / "data" / "samples" / "sample_sales.csv"
        )
        df = pd.read_csv(sample_path)
        state.set(state.RAW_DF, df)
        state.set(state.FILE_NAME, "sample_sales.csv")
        state.clear_downstream(state.RAW_DF)
        st.rerun()

# ── Process upload ────────────────────────────────────────────────────────────
if uploaded is not None:
    try:
        sep = "\t" if uploaded.name.endswith(".tsv") else ","
        df  = pd.read_csv(uploaded, sep=sep)
        if df.empty:
            st.error("The uploaded file appears to be empty.")
            st.stop()
        prev_name = state.get(state.FILE_NAME)
        if prev_name != uploaded.name:
            state.clear_downstream(state.RAW_DF)
        state.set(state.RAW_DF, df)
        state.set(state.FILE_NAME, uploaded.name)
        st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

# ── Show profile ──────────────────────────────────────────────────────────────
if not state.has(state.RAW_DF):
    st.info("Upload a CSV file above or load the sample dataset to see the quality report.")
    st.stop()

raw_df    = state.get(state.RAW_DF)
file_name = state.get(state.FILE_NAME, "your dataset")

# Run profiler (cache by file name to avoid re-running on every interaction)
if not state.has(state.PROFILE):
    with st.spinner("Running data quality analysis…"):
        profiler = DataProfiler()
        profile  = profiler.profile(raw_df)
        state.set(state.PROFILE, profile)

profile = state.get(state.PROFILE)

# ── Quality score + key metrics ───────────────────────────────────────────────
st.subheader(f"Quality Report — {file_name}")
render_quality_score(profile)

st.markdown("---")

# ── Flags / warnings ─────────────────────────────────────────────────────────
render_flags(profile)

st.markdown("---")

# ── Column breakdown ──────────────────────────────────────────────────────────
st.subheader("Column Overview")
st.caption(
    f"{profile.total_columns} columns · "
    f"{len(profile.numeric_columns)} numeric · "
    f"{len(profile.categorical_columns)} categorical · "
    f"{len(profile.datetime_columns)} datetime"
)
render_column_table(profile)

# ── Numeric stats (collapsible) ───────────────────────────────────────────────
if profile.numeric_columns:
    with st.expander("Numeric column statistics"):
        render_numeric_stats(profile)

# ── Raw data preview (collapsible) ────────────────────────────────────────────
with st.expander("Raw data preview (first 50 rows)"):
    st.dataframe(raw_df.head(50), width="stretch")

st.markdown("---")

# ── Cleaning recommendations ──────────────────────────────────────────────────
st.subheader("Cleaning Recommendations")
render_recommendations(profile)

st.markdown("---")

# ── Navigation ────────────────────────────────────────────────────────────────
st.success("Profile complete. Proceed to **🔧 Schema Setup** to map your columns.")
if st.button("Next → Schema Setup", type="primary"):
    st.switch_page("pages/2_Schema_Setup.py")
