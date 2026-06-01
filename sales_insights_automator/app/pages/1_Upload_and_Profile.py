"""
Page 1 — Upload & Profile

Lets the user upload a CSV or TSV file or use the built-in sample dataset.
Immediately runs the DataProfiler and shows a full quality report.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    _env = _PROJECT_ROOT / ".env"
    if _env.is_file():
        load_dotenv(_env, override=False)
except ImportError:
    pass

import pandas as pd
import streamlit as st

from app import state
from app.theme import apply_page_theme
from cleaning.functions import dedupe_column_names
from config.settings import GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH
from ingestion.base import DataSourceError
from ingestion.google_drive_source import GoogleDriveSource
from profiling.profiler import DataProfiler
from app.components.profile_table import (
    render_quality_score,
    render_flags,
    render_column_table,
    render_numeric_stats,
    render_recommendations,
)

st.set_page_config(page_title="Upload & Profile", page_icon="📂", layout="wide")

apply_page_theme()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Sales Insights")
st.sidebar.markdown("---")
st.sidebar.markdown("**Step 1 of 4** — Upload your data and inspect its quality before cleaning.")

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📂 Upload & Profile")
st.markdown(
    "Upload a **CSV** or **TSV** file to begin. The profiler scans every column "
    "for missing values, duplicates, outliers, and data types."
)
st.markdown("---")

# ── File upload ───────────────────────────────────────────────────────────────
col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Upload a CSV or TSV file",
        type=["csv", "tsv"],
        help="Comma- or tab-separated text with a header row. Max 200 MB.",
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

# ── Google Drive ──────────────────────────────────────────────────────────────
with st.expander("Load from Google Drive"):
    st.caption(
        "Loads a shared **CSV/TSV** file or **Google Sheet** (first sheet as CSV). "
        "Service account: share the file with the SA email from your JSON. "
        "OAuth: put Desktop client secrets at the credentials path and run "
        "`python scripts/setup_gdrive_oauth.py` once."
    )
    gdrive_id = st.text_input(
        "File ID",
        placeholder="from link …/file/d/THIS_PART/view…",
        key="gdrive_file_id",
    )
    cred_path = st.text_input(
        "Credentials JSON path",
        value=GDRIVE_CREDENTIALS_PATH,
        key="gdrive_cred_path",
    )
    tok_path = st.text_input(
        "OAuth token path (service accounts can leave default)",
        value=GDRIVE_TOKEN_PATH,
        key="gdrive_token_path",
    )
    load_gdrive = st.button("Load from Google Drive", key="gdrive_load_btn")

if load_gdrive:
    fid = (gdrive_id or "").strip()
    if not fid:
        st.warning("Enter a Google Drive file ID.")
    else:
        try:
            tok = (tok_path or "").strip()
            src = GoogleDriveSource(
                file_id=fid,
                credentials_path=(cred_path or "").strip() or None,
                token_path=tok or None,
            )
            with st.spinner("Downloading from Google Drive…"):
                df = dedupe_column_names(src.load_validated())
            if df.empty:
                st.error("The file appears to be empty.")
                st.stop()
            meta_name = None
            if "_source_file" in df.columns and len(df):
                meta_name = str(df["_source_file"].iloc[0])
            display_name = meta_name or f"gdrive_{fid}.csv"
            prev_name = state.get(state.FILE_NAME)
            if prev_name != display_name:
                state.clear_downstream(state.RAW_DF)
            state.set(state.RAW_DF, df)
            state.set(state.FILE_NAME, display_name)
            st.success(
                f"Loaded **{display_name}** from Drive — {len(df):,} rows × {len(df.columns)} columns"
            )
            st.rerun()
        except DataSourceError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Could not load from Google Drive: {e}")
            st.stop()

# ── Process upload ────────────────────────────────────────────────────────────
if uploaded is not None:
    try:
        name_lower = uploaded.name.lower()
        sep = "\t" if name_lower.endswith(".tsv") else ","
        df = dedupe_column_names(pd.read_csv(uploaded, sep=sep))
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
    st.info("Upload a CSV or TSV file above or load the sample dataset to see the quality report.")
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
