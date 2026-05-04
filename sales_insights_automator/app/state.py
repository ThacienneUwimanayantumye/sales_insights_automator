"""
Session state management for the Streamlit app.

All pages share data through st.session_state.  This module defines the
canonical key names and helper functions so no page hard-codes a string.

State lifecycle
---------------
  Page 1 (Upload)    → sets RAW_DF, PROFILE
  Page 2 (Schema)    → sets SCHEMA, CLEAN_DF, ANALYSIS_RESULT
  Page 3 (Dashboard) → reads ANALYSIS_RESULT
  Page 4 (AI)        → reads ANALYSIS_RESULT, sets INSIGHT_REPORT
"""

import streamlit as st
from typing import Any, Optional

# ── Key names ─────────────────────────────────────────────────────────────────
RAW_DF          = "raw_df"           # pd.DataFrame — original uploaded data
PROFILE         = "profile"          # DataProfile   — quality report
SCHEMA          = "schema"           # SchemaConfig  — confirmed column mapping
CLEAN_DF        = "clean_df"         # pd.DataFrame  — cleaned data
CLEANING_REPORT = "cleaning_report"  # CleaningReport
ANALYSIS_RESULT = "analysis_result"  # AnalysisResult
INSIGHT_REPORT  = "insight_report"   # InsightReport
FILE_NAME       = "file_name"        # str — display name of uploaded file


# ── Helpers ───────────────────────────────────────────────────────────────────

def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set(key: str, value: Any) -> None:
    st.session_state[key] = value


def has(key: str) -> bool:
    return key in st.session_state and st.session_state[key] is not None


def clear_downstream(from_key: str) -> None:
    """Clear all state that depends on a key that has changed.

    Call this when the user re-uploads a file or changes the schema,
    so stale analysis results are not shown on later pages.
    """
    cascade = {
        RAW_DF:          [PROFILE, SCHEMA, CLEAN_DF, CLEANING_REPORT,
                          ANALYSIS_RESULT, INSIGHT_REPORT, "wizard_mapping"],
        SCHEMA:          [CLEAN_DF, CLEANING_REPORT,
                          ANALYSIS_RESULT, INSIGHT_REPORT],
        ANALYSIS_RESULT: [INSIGHT_REPORT],
    }
    for key in cascade.get(from_key, []):
        if key in st.session_state:
            del st.session_state[key]


def progress_status() -> dict:
    """Return which pipeline stages have been completed."""
    return {
        "uploaded":  has(RAW_DF),
        "profiled":  has(PROFILE),
        "schema":    has(SCHEMA),
        "analysed":  has(ANALYSIS_RESULT),
        "insights":  has(INSIGHT_REPORT),
    }
