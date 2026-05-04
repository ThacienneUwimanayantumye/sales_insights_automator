"""
Renders the DataProfile report as Streamlit UI components.
"""

import streamlit as st
import pandas as pd
from profiling.profiler import DataProfile, ColumnProfile


def render_quality_score(profile: DataProfile) -> None:
    """Big quality score card at the top of the profile page."""
    score = profile.quality_score
    color = "#10B981" if score >= 80 else ("#F59E0B" if score >= 60 else "#EF4444")
    label = "Good" if score >= 80 else ("Fair" if score >= 60 else "Poor")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data Quality Score", f"{score}/100", label)
    col2.metric("Total Rows", f"{profile.total_rows:,}")
    col3.metric("Duplicate Rows",
                f"{profile.duplicate_rows:,}",
                f"{profile.duplicate_pct:.1f}% of data",
                delta_color="inverse")
    col4.metric("Null Cells",
                f"{profile.total_null_cells:,}",
                f"{profile.null_density_pct:.1f}% density",
                delta_color="inverse")


def render_flags(profile: DataProfile) -> None:
    """Warning / info banners for data quality issues."""
    if profile.duplicate_rows > 0:
        st.warning(
            f"**{profile.duplicate_rows:,} duplicate rows** detected "
            f"({profile.duplicate_pct:.1f}% of data). "
            f"The cleaning layer will remove them automatically."
        )
    if profile.high_null_columns:
        st.warning(
            f"**High-null columns** (>20% missing): "
            f"{', '.join(f'`{c}`' for c in profile.high_null_columns)}"
        )
    if profile.constant_columns:
        st.info(
            f"**Constant columns** (zero variance, will be dropped): "
            f"{', '.join(f'`{c}`' for c in profile.constant_columns)}"
        )
    if profile.outlier_columns:
        st.warning(
            f"**Outliers detected** in: "
            f"{', '.join(f'`{c}`' for c in profile.outlier_columns)}"
        )
    if not (profile.duplicate_rows or profile.high_null_columns
            or profile.constant_columns or profile.outlier_columns):
        st.success("No critical data quality issues found.")


def render_column_table(profile: DataProfile) -> None:
    """Interactive table of all column profiles with colour-coded null severity."""
    rows = []
    for cp in profile.columns:
        null_severity = (
            "🔴 HIGH"  if cp.null_pct > 20
            else "🟡 MED" if cp.null_pct > 5
            else ("🟢 none" if cp.null_count == 0 else "🟢 LOW")
        )
        flags = []
        if cp.is_constant:   flags.append("constant")
        if cp.is_likely_id:  flags.append("ID col")
        if cp.has_outliers:  flags.append("outliers")

        rows.append({
            "Column":      cp.name,
            "Type":        cp.dtype,
            "Kind":        cp.inferred_kind,
            "Nulls":       cp.null_count,
            "Null %":      f"{cp.null_pct:.1f}%",
            "Severity":    null_severity,
            "Unique":      f"{cp.unique_count:,}",
            "Cardinality": f"{cp.cardinality_pct:.1f}%",
            "Sample":      ", ".join(cp.sample_values[:2]),
            "Flags":       ", ".join(flags) if flags else "—",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Column":      st.column_config.TextColumn(width="medium"),
            "Null %":      st.column_config.TextColumn(width="small"),
            "Severity":    st.column_config.TextColumn(width="small"),
            "Cardinality": st.column_config.TextColumn(width="small"),
            "Sample":      st.column_config.TextColumn(width="large"),
        },
    )


def render_numeric_stats(profile: DataProfile) -> None:
    """Expander with a summary table for all numeric columns."""
    numeric = profile.numeric_columns
    if not numeric:
        return

    rows = []
    for cp in numeric:
        rows.append({
            "Column":    cp.name,
            "Min":       f"{cp.min:,.2f}",
            "Max":       f"{cp.max:,.2f}",
            "Mean":      f"{cp.mean:,.2f}",
            "Median":    f"{cp.median:,.2f}",
            "Std Dev":   f"{cp.std:,.2f}",
            "Skewness":  f"{cp.skewness:+.2f}",
            "Outliers":  f"{cp.outlier_count} ({cp.outlier_pct:.1f}%)" if cp.outlier_count else "None",
        })

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_recommendations(profile: DataProfile) -> None:
    """Numbered list of auto-generated cleaning recommendations."""
    recs = profile._generate_recommendations()
    if not recs:
        st.success("Data looks ready for analysis — no cleaning required.")
        return
    for i, rec in enumerate(recs, 1):
        st.markdown(f"**{i}.** {rec}")
