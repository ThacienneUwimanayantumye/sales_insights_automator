"""
Page 2 — Schema Setup

Auto-detects which column plays which semantic role (revenue, date, region…)
and lets the user confirm or correct the mapping using dropdowns.
Then runs cleaning + analysis so the Dashboard is ready immediately.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from app import state
from config.schema import SchemaConfig, ALL_ROLES, REQUIRED_ROLES, ROLE_DESCRIPTIONS
from profiling.schema_wizard import SchemaWizard
from cleaning.cleaner import DataCleaner
from cleaning.config import CleaningConfig
from analysis.analyzer import SalesAnalyzer

st.set_page_config(page_title="Schema Setup", page_icon="🔧", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Sales Insights")
st.sidebar.markdown("---")
st.sidebar.markdown("**Step 2 of 4** — Confirm which column plays each role.")

# ── Guard: need data ──────────────────────────────────────────────────────────
if not state.has(state.RAW_DF):
    st.warning("No data loaded yet. Please go to **📂 Upload & Profile** first.")
    if st.button("← Go to Upload"):
        st.switch_page("pages/1_Upload_and_Profile.py")
    st.stop()

raw_df    = state.get(state.RAW_DF)
file_name = state.get(state.FILE_NAME, "your dataset")
columns   = list(raw_df.columns)
none_opt  = "— not in this dataset —"

st.title("🔧 Schema Setup")
st.markdown(
    "The wizard has **auto-detected** which column likely plays each role. "
    "Use the dropdowns to confirm or correct any mapping. "
    "Required roles (marked ✱) must be assigned before analysis can run."
)
st.markdown("---")

# ── Auto-detect on first visit ────────────────────────────────────────────────
if "wizard_mapping" not in st.session_state:
    with st.spinner("Auto-detecting column roles…"):
        detected = SchemaWizard().detect(raw_df)
    st.session_state["wizard_mapping"] = detected.to_dict()

mapping: dict = st.session_state["wizard_mapping"]

# ── Column overview mini-table ────────────────────────────────────────────────
with st.expander("Your dataset columns", expanded=False):
    preview_rows = []
    for col in columns:
        s = raw_df[col].dropna()
        preview_rows.append({
            "Column":  col,
            "Type":    str(raw_df[col].dtype),
            "Unique":  int(s.nunique()),
            "Sample":  ", ".join(str(v) for v in s.unique()[:3]),
        })
    st.dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)

st.markdown("---")

# ── Role assignment dropdowns ─────────────────────────────────────────────────
st.subheader("Assign Roles to Columns")

col_left, col_right = st.columns(2)
role_list     = list(ALL_ROLES)
half          = len(role_list) // 2
left_roles    = role_list[:half]
right_roles   = role_list[half:]

def role_selector(role: str, container) -> None:
    required  = role in REQUIRED_ROLES
    label     = f"{'✱ ' if required else ''}{role}"
    help_text = ROLE_DESCRIPTIONS[role]
    current   = mapping.get(role)

    options = [none_opt] + columns
    default_idx = columns.index(current) + 1 if current and current in columns else 0

    chosen = container.selectbox(
        label,
        options     = options,
        index       = default_idx,
        key         = f"role_{role}",
        help        = help_text,
    )
    mapping[role] = None if chosen == none_opt else chosen

with col_left:
    for role in left_roles:
        role_selector(role, col_left)

with col_right:
    for role in right_roles:
        role_selector(role, col_right)

st.session_state["wizard_mapping"] = mapping

# ── Validation feedback ───────────────────────────────────────────────────────
st.markdown("---")
schema_preview = SchemaConfig.from_dict(mapping)
validation_errors = schema_preview.validate(raw_df)

if validation_errors:
    for err in validation_errors:
        st.error(f"**Missing required role:** {err}")
else:
    st.success("✓ All required roles are mapped. Ready to run analysis.")

# ── Mapped roles summary ──────────────────────────────────────────────────────
with st.expander("Current mapping summary"):
    rows = []
    for role in ALL_ROLES:
        actual = mapping.get(role)
        rows.append({
            "Role":     ("✱ " if role in REQUIRED_ROLES else "  ") + role,
            "Maps to":  actual if actual else "— not mapped —",
            "Required": "Yes" if role in REQUIRED_ROLES else "No",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ── Apply & run analysis ──────────────────────────────────────────────────────
st.markdown("---")

apply_disabled = bool(validation_errors)

if st.button("Apply Schema & Run Analysis", type="primary", disabled=apply_disabled):
    schema = SchemaConfig.from_dict(mapping)
    state.set(state.SCHEMA, schema)
    state.clear_downstream(state.SCHEMA)

    with st.spinner("Cleaning data…"):
        # Build type_conversions: convert the mapped date column to datetime
        date_col    = mapping.get("date")
        type_convs  = {date_col: "datetime"} if date_col else {}

        config  = CleaningConfig(
            normalize_columns = True,
            drop_duplicates   = True,
            type_conversions  = type_convs,
        )
        cleaner  = DataCleaner(config)
        clean_df = cleaner.clean(raw_df)
        # Apply schema rename so standard column names are used
        clean_df = schema.rename_to_standard(clean_df)
        state.set(state.CLEAN_DF, clean_df)
        state.set(state.CLEANING_REPORT, cleaner.report)

    with st.spinner("Running analysis…"):
        # Schema already applied during cleaning — pass no schema to analyzer
        analyzer = SalesAnalyzer()
        try:
            result = analyzer.analyze(clean_df)
            state.set(state.ANALYSIS_RESULT, result)
            st.success(
                f"Analysis complete — "
                f"{result.row_count:,} rows, "
                f"${result.summary_stats['total_revenue']:,.0f} total revenue"
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    st.info("Navigate to **📈 Dashboard** to explore the charts.")
    if st.button("Next → Dashboard", type="primary"):
        st.switch_page("pages/3_Dashboard.py")

# ── Show cleaning report if already done ─────────────────────────────────────
if state.has(state.CLEANING_REPORT):
    report = state.get(state.CLEANING_REPORT)
    with st.expander("Cleaning report"):
        st.text(report.summary())
