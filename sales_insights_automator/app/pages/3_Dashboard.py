"""
Page 3 — Analysis Dashboard

Interactive exploration of sales data.  Sidebar filters (date range + category)
re-compute all KPIs and charts live from the stored clean DataFrame so the user
can slice and drill down freely without re-running the full pipeline.

Chart variety:
  Revenue trend  — bar/line combo  OR  cumulative area  (tab toggle)
  Breakdowns     — horizontal bar  OR  donut  OR  treemap  (radio toggle)
  Patterns       — polar day-of-week  +  transaction histogram
  Correlation    — scatter: quantity vs revenue
"""

import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from app import state
from app import dashboard_export as dex
from app.dashboard_export import DashboardPdfError
from config.schema import ROLE_LABELS as _ROLE_LABELS
from app.components.charts import (
    # existing
    revenue_trend,
    revenue_by_dimension,
    sales_rep_performance,
    regional_trend,
    discount_gauge,
    category_heatmap,
    # added in redesign
    revenue_donut,
    revenue_treemap,
    revenue_area_cumulative,
    revenue_histogram,
    weekday_polar,
    scatter_qty_revenue,
    # quarterly + demographic
    revenue_quarterly,
    category_by_group,
    category_group_heatmap,
    # extra-column discovery
    metric_by_dimension,
    metric_distribution,
    _label,
)
from analysis import metrics as m
from analysis import trends as t

st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

# ── Guard: need analysis result ───────────────────────────────────────────────
if not state.has(state.ANALYSIS_RESULT):
    st.warning("No analysis available yet. Please complete **🔧 Schema Setup** first.")
    if st.button("← Go to Schema Setup"):
        st.switch_page("pages/2_Schema_Setup.py")
    st.stop()

result        = state.get(state.ANALYSIS_RESULT)
clean_df      = state.get(state.CLEAN_DF)
extra_dims    = state.get(state.EXTRA_DIMS,    []) or []
extra_metrics = state.get(state.EXTRA_METRICS, []) or []

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Sales Insights")
st.sidebar.markdown("---")
st.sidebar.markdown("**Step 3 of 4** — Explore your data interactively.")
st.sidebar.markdown("### Filters")

# ── Date range filter ─────────────────────────────────────────────────────────
has_dates = (
    clean_df is not None
    and "date" in clean_df.columns
    and pd.api.types.is_datetime64_any_dtype(clean_df["date"])
)

if has_dates:
    d_min = clean_df["date"].min().date()
    d_max = clean_df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value     = (d_min, d_max),
        min_value = d_min,
        max_value = d_max,
        key       = "dash_date_range",
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        sel_from, sel_to = date_range
    else:
        sel_from, sel_to = d_min, d_max
else:
    sel_from = sel_to = None

# ── Dynamic dimension filters (fully data-driven, zero hardcoding) ────────────
# Discover every column that is useful as a filter:
#   • categorical dtype  AND  2–100 unique values  (not free-text or IDs)
#   • tiny-range numerics (≤ 10 unique values, e.g. 0/1 flags, 1–5 ratings)
# Date and high-cardinality ID columns are explicitly excluded.
_FILTER_EXCLUDE = {"date", "order_id", "customer_id", "age"}

def _filterable_cols(df: pd.DataFrame) -> list:
    if df is None:
        return []
    cols = []
    for col in df.columns:
        if col in _FILTER_EXCLUDE:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        n_unique = int(df[col].nunique())
        if pd.api.types.is_numeric_dtype(df[col]):
            if 2 <= n_unique <= 10:        # binary flags, star ratings …
                cols.append(col)
        else:
            if 2 <= n_unique <= 100:       # categorical with manageable cardinality
                cols.append(col)
    return cols

dim_filter_cols = _filterable_cols(clean_df)

# ── Per-dimension filter: "All" checkbox + individual multiselect ─────────────
# Default state is "All selected" (clean sidebar).  Unchecking "All" reveals
# a multiselect so the user picks specific values without having to manually
# deselect a long list.
col_filters: dict = {}

if dim_filter_cols:
    # "Reset all filters" button — clears every filter back to All
    if st.sidebar.button("↺ Reset all filters", key="reset_all_filters"):
        for _c in dim_filter_cols:
            st.session_state.pop(f"use_all_{_c}", None)
            st.session_state.pop(f"filter_{_c}", None)
        st.rerun()

    for _col in dim_filter_cols:
        _all_vals = sorted(clean_df[_col].dropna().astype(str).unique().tolist())
        n_unique  = len(_all_vals)

        # Header row: label + value count badge
        st.sidebar.markdown(
            f"<span style='font-weight:600'>{_label(_col)}</span>"
            f"<span style='color:grey; font-size:0.8em'> &nbsp;{n_unique} values</span>",
            unsafe_allow_html=True,
        )

        _use_all = st.sidebar.checkbox(
            "All",
            value = st.session_state.get(f"use_all_{_col}", True),
            key   = f"use_all_{_col}",
        )

        if _use_all:
            col_filters[_col] = _all_vals
        else:
            _sel = st.sidebar.multiselect(
                _label(_col),
                options          = _all_vals,
                default          = st.session_state.get(f"filter_{_col}", _all_vals[:1]),
                key              = f"filter_{_col}",
                label_visibility = "collapsed",
            )
            col_filters[_col] = _sel if _sel else _all_vals

st.sidebar.markdown("---")
_active = sum(1 for _c in col_filters if not st.session_state.get(f"use_all_{_c}", True))
st.sidebar.caption(
    (f"**{_active} filter(s) active.** " if _active else "No filters active. ")
    + "Filters apply to all KPIs and charts on this page."
)

# Shortcut used throughout the page to check category presence (data-driven)
has_category = clean_df is not None and "category" in clean_df.columns

# ── Apply filters to clean_df ─────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    if has_dates and sel_from and sel_to:
        out = out[
            (out["date"].dt.date >= sel_from) &
            (out["date"].dt.date <= sel_to)
        ]
    for _col, _sel_vals in col_filters.items():
        if _col in out.columns:
            out = out[out[_col].astype(str).isin(_sel_vals)]
    return out

fdf = apply_filters(clean_df)   # filtered DataFrame for live re-computation

# ── Compute live KPIs from filtered df ────────────────────────────────────────
def safe_summary(df: pd.DataFrame) -> dict:
    """Compute summary stats; return zeros on error."""
    if df is None or df.empty:
        return {"total_revenue": 0, "total_orders": 0, "average_order_value": 0,
                "total_units_sold": 0, "average_discount_pct": 0}
    try:
        return m.compute_summary_stats(df)
    except Exception:
        return {"total_revenue": 0, "total_orders": 0, "average_order_value": 0,
                "total_units_sold": 0, "average_discount_pct": 0}

live_stats = safe_summary(fdf)

# Day-of-week aggregate (polar chart + export bundle)
dow_chart = result.revenue_by_weekday
if fdf is not None and not fdf.empty:
    try:
        dow_chart = t.revenue_by_day_of_week(fdf)
    except Exception:
        dow_chart = result.revenue_by_weekday

# Compare against full-dataset stats for delta indicators
full_stats = result.summary_stats

def _delta(key: str) -> float | None:
    full = full_stats.get(key, 0) or 0
    live = live_stats.get(key, 0) or 0
    if full and live != full:
        return live - full
    return None

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📈 Analysis Dashboard")

# Period label
if has_dates and sel_from != sel_to:
    st.caption(
        f"Showing: **{sel_from}** → **{sel_to}** · "
        f"{len(fdf):,} transactions"
        + (f" · {_active} filter(s) active" if _active else "")
    )
else:
    dr = result.date_range
    st.caption(f"Period: **{dr.get('from','?')}** → **{dr.get('to','?')}** · {result.row_count:,} transactions")

st.markdown("---")

# ── Section 1: KPI cards ──────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Total Revenue",
    f"${live_stats['total_revenue']:,.0f}",
    delta = f"${_delta('total_revenue'):+,.0f}" if _delta("total_revenue") is not None else None,
)
k2.metric(
    "Total Orders",
    f"{live_stats['total_orders']:,}",
    delta = f"{_delta('total_orders'):+,.0f}" if _delta("total_orders") is not None else None,
)
k3.metric(
    "Avg Order Value",
    f"${live_stats['average_order_value']:,.0f}",
    delta = f"${_delta('average_order_value'):+,.0f}" if _delta("average_order_value") is not None else None,
)
k4.metric(
    "Units Sold",
    f"{live_stats.get('total_units_sold', 0):,}",
)
disc = live_stats.get("average_discount_pct")
k5.metric(
    "Avg Discount",
    f"{disc:.1f}%" if disc else "N/A",
)

st.markdown("---")

# ── Section 2: Revenue trend ──────────────────────────────────────────────────
st.subheader("Revenue Over Time")

# Recompute monthly trend from filtered df if filters are active
def get_monthly(df: pd.DataFrame):
    if df is None or df.empty or "date" not in df.columns:
        return None
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        return None
    try:
        monthly = t.monthly_revenue(df)
        monthly = t.compute_growth_rates(monthly)
        monthly = t.rolling_revenue(monthly, window=3)
        return monthly
    except Exception:
        return None

monthly = get_monthly(fdf)
if monthly is None:
    monthly = result.monthly_trend   # fall back to pre-computed

# Compute quarterly data from filtered df
def get_quarterly(df: pd.DataFrame):
    if df is None or df.empty or "date" not in df.columns:
        return None
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        return None
    try:
        return t.quarterly_revenue(df)
    except Exception:
        return None

quarterly = get_quarterly(fdf)

tab_monthly, tab_quarterly, tab_cumul = st.tabs(
    ["Monthly Breakdown", "Quarterly Seasons", "Cumulative Growth"]
)

with tab_monthly:
    if monthly is not None and not monthly.empty:
        st.plotly_chart(revenue_trend(monthly), width="stretch")
        trend_s = t.trend_summary(monthly) if monthly is not None else {}
        if trend_s:
            t1, t2, t3 = st.columns(3)
            t1.metric("Best Month",       trend_s.get("best_month", "—"),
                      f"${trend_s.get('best_month_revenue', 0):,.0f}")
            t2.metric("Overall Growth",   f"{trend_s.get('overall_growth_pct', 0):+.1f}%")
            t3.metric("Avg Monthly Rev.", f"${trend_s.get('avg_monthly_revenue', 0):,.0f}")
    else:
        st.info("Monthly trend not available — date column may not be datetime format.")

with tab_quarterly:
    if quarterly is not None and not quarterly.empty:
        st.plotly_chart(revenue_quarterly(quarterly), width="stretch")
        st.caption(
            "**Multi-year datasets:** each year gets its own colour so you can compare "
            "Q1-2023 vs Q1-2024 side by side to spot seasonal patterns. "
            "Green bars = quarter grew vs previous quarter; red = declined."
        )

        # Quarter-by-quarter summary table
        display_cols = ["quarter", "total_revenue", "order_count", "qoq_growth_pct"]
        display_cols = [c for c in display_cols if c in quarterly.columns]
        with st.expander("Quarterly breakdown table"):
            st.dataframe(quarterly[display_cols], width="stretch", hide_index=True)

        # Best and worst quarter
        best_q  = quarterly.loc[quarterly["total_revenue"].idxmax()]
        worst_q = quarterly.loc[quarterly["total_revenue"].idxmin()]
        q1, q2, q3 = st.columns(3)
        q1.metric("Best Quarter",  best_q["quarter"],  f"${best_q['total_revenue']:,.0f}")
        q2.metric("Worst Quarter", worst_q["quarter"], f"${worst_q['total_revenue']:,.0f}")
        if "qoq_growth_pct" in quarterly.columns:
            avg_qoq = quarterly["qoq_growth_pct"].dropna().mean()
            q3.metric("Avg QoQ Growth", f"{avg_qoq:+.1f}%")
    else:
        st.info("Quarterly data not available for this date range.")

with tab_cumul:
    if monthly is not None and not monthly.empty:
        st.plotly_chart(revenue_area_cumulative(monthly), width="stretch")
        st.caption(
            "A steeper slope means faster growth. "
            "A flattening curve means revenue is slowing down in that period."
        )
    else:
        st.info("Cumulative chart not available.")

st.markdown("---")

# ── Section 3: Breakdown by dimension ────────────────────────────────────────
st.subheader("Revenue Breakdown")
st.caption(
    "Use the **chart type** toggle to switch between views. "
    "Donut is best for seeing shares; Treemap for comparing sizes visually; "
    "Bar for precise ranking."
)

# Determine which dimensions are available in the filtered df
def dim_data(df: pd.DataFrame, col: str):
    """Compute revenue breakdown for a dimension from the filtered df."""
    if df is None or df.empty or col not in df.columns:
        return None
    try:
        return m.revenue_by_dimension(df, col)
    except Exception:
        return None

dims = {}
if "category"  in fdf.columns: dims["Product Category"] = ("category",  dim_data(fdf, "category"))
if "region"    in fdf.columns: dims["Region"]            = ("region",    dim_data(fdf, "region"))
if "product"   in fdf.columns: dims["Product"]           = ("product",   dim_data(fdf, "product"))
if "sales_rep" in fdf.columns: dims[_ROLE_LABELS["sales_rep"]] = ("sales_rep", dim_data(fdf, "sales_rep"))
# New Layer 1 roles
if "channel"          in fdf.columns: dims[_label("channel")]          = ("channel",          dim_data(fdf, "channel"))
if "payment_method"   in fdf.columns: dims[_label("payment_method")]   = ("payment_method",   dim_data(fdf, "payment_method"))
if "customer_segment" in fdf.columns: dims[_label("customer_segment")] = ("customer_segment", dim_data(fdf, "customer_segment"))
if "return_flag"      in fdf.columns: dims[_label("return_flag")]      = ("return_flag",      dim_data(fdf, "return_flag"))
# Extra auto-discovered dimensions (Layer 2)
for _edim in extra_dims:
    if _edim in fdf.columns and _label(_edim) not in dims:
        dims[_label(_edim)] = (_edim, dim_data(fdf, _edim))

# Shared aggregates for charts + export (avoid duplicate metric calls)
rep_perf_df = None
if fdf is not None and not fdf.empty and "sales_rep" in fdf.columns:
    try:
        rep_perf_df = m.sales_rep_performance(fdf)
    except Exception:
        rep_perf_df = None

crosstab_df = None
if fdf is not None and not fdf.empty and "category" in fdf.columns and "region" in fdf.columns:
    try:
        crosstab_df = m.category_region_crosstab(fdf)
    except Exception:
        crosstab_df = None

if dims:
    dim_tabs = st.tabs(list(dims.keys()))
    for tab, (label, (col, data)) in zip(dim_tabs, dims.items()):
        with tab:
            if data is None or data.empty:
                st.info(f"No {label.lower()} data available for the current filter selection.")
                continue

            chart_type = st.radio(
                "Chart type",
                ["Bar", "Donut", "Treemap"],
                horizontal = True,
                key        = f"chart_type_{label}",
            )

            if chart_type == "Bar":
                st.plotly_chart(
                    revenue_by_dimension(data, col, f"Revenue by {label}"),
                    width="stretch",
                )
            elif chart_type == "Donut":
                st.plotly_chart(
                    revenue_donut(data, col, f"Revenue Share by {label}"),
                    width="stretch",
                )
            else:  # Treemap
                st.plotly_chart(
                    revenue_treemap(data, col, f"Revenue by {label} — Treemap"),
                    width="stretch",
                )

            with st.expander(f"Full {label} table"):
                st.dataframe(data, width="stretch", hide_index=True)

    # Sales rep — special dual-axis chart needs avg_order_value, so compute
    # it fresh from the filtered df using the dedicated metrics function
    # (the generic dim_data result doesn't include avg_order_value)
    if rep_perf_df is not None and not rep_perf_df.empty:
        st.markdown("#### Salesperson Performance (revenue + avg order value)")
        st.plotly_chart(sales_rep_performance(rep_perf_df), width="stretch")

    # Category × Region heatmap when both are present
    if "Product Category" in dims and "Region" in dims and crosstab_df is not None:
        try:
            st.markdown("#### Category × Region Revenue Heatmap")
            st.plotly_chart(category_heatmap(crosstab_df), width="stretch")
        except Exception:
            pass
else:
    st.info(
        "No breakdown dimensions available for this dataset. "
        "Map **Product Category**, **Region**, **Product**, or **Salesperson** in Schema Setup to enable these charts."
    )

st.markdown("---")

# ── Section 4: Transaction patterns ──────────────────────────────────────────
st.subheader("Transaction Patterns")
st.caption(
    "The polar chart shows which days drive the most revenue. "
    "The histogram shows how transaction sizes are distributed."
)

col_polar, col_hist = st.columns([1, 1])

with col_polar:
    if dow_chart is not None and not dow_chart.empty:
        st.plotly_chart(weekday_polar(dow_chart), width="stretch")
    else:
        st.info("Day-of-week data not available.")

with col_hist:
    if fdf is not None and not fdf.empty:
        st.plotly_chart(revenue_histogram(fdf, "revenue"), width="stretch")
        st.caption(
            "A right-skewed histogram (most bars on the left) means most "
            "transactions are small with a few very large ones. "
            "A bell shape means transactions are consistently sized."
        )
    else:
        st.info("Transaction histogram not available.")

st.markdown("---")

# ── Section 5: Correlation explorer ──────────────────────────────────────────
st.subheader("Quantity vs Revenue Correlation")
st.caption(
    "Each dot is one transaction. "
    "Dots trending upward (bottom-left → top-right) mean buying more units = more revenue. "
    "A flat cloud means price per unit drops as quantity increases (bulk discounts). "
    "Outlier dots far above the trend are high-value single-unit orders."
)

if fdf is not None and not fdf.empty:
    st.plotly_chart(scatter_qty_revenue(fdf), width="stretch")
else:
    st.info("Scatter chart not available.")

# Regional trend (when region is present)
if result.regional_trend is not None and not result.regional_trend.empty:
    st.markdown("---")
    st.subheader("Regional Revenue Trend")
    st.plotly_chart(regional_trend(result.regional_trend), width="stretch")

# Discount analysis (when discount is present)
if result.discount_stats:
    st.markdown("---")
    st.subheader("Discount Analysis")
    col_gauge, col_disc = st.columns([1, 2])
    d = result.discount_stats
    with col_gauge:
        st.plotly_chart(
            discount_gauge(d.get("avg_discount_pct", 0)),
            width="stretch",
        )
    with col_disc:
        st.metric("Max Discount",          f"{d.get('max_discount_pct', 0):.1f}%")
        st.metric("Orders with Discount",  f"{d.get('orders_with_discount', 0):,}")
        st.metric("Revenue Lost",          f"${d.get('revenue_lost_to_discount', 0):,.0f}")

st.markdown("---")

# ── Section 6: Demographic breakdown ─────────────────────────────────────────
# Auto-detect gender and age columns; show only if both category and at least
# one demographic dimension are present in the filtered DataFrame.

# Use schema-mapped column names when available, fall back to standard names.
# After rename_to_standard(), the columns are the standard names defined in
# config/schema.py (e.g. "gender", "age", "category"), so a simple presence
# check is sufficient.
has_cat_col = fdf is not None and "category" in fdf.columns
has_gender  = fdf is not None and "gender"   in fdf.columns
has_age     = fdf is not None and "age"      in fdf.columns

if has_cat_col and (has_gender or has_age):
    st.subheader("Demographic Breakdown")
    st.caption(
        "Compare how different customer groups spend across product categories. "
        "Use this to spot which segment to target for each category."
    )

    # Prepare age groups (bin continuous age into labelled brackets)
    demo_df = fdf.copy()
    if has_age:
        bins   = [0, 25, 35, 45, 55, 120]
        labels = ["18–25", "26–35", "36–45", "46–55", "55+"]
        demo_df["age_group"] = pd.cut(
            pd.to_numeric(demo_df["age"], errors="coerce"),
            bins   = bins,
            labels = labels,
            right  = True,
        )

    demo_tabs = []
    if has_gender:  demo_tabs.append("Gender")
    if has_age:     demo_tabs.append("Age Group")

    dtabs = st.tabs(demo_tabs)
    tab_idx = 0

    if has_gender:
        with dtabs[tab_idx]:
            tab_idx += 1
            st.markdown(
                "Revenue per **product category**, split by gender. "
                "Taller bars for a gender = that group spends more in that product category."
            )

            chart_style = st.radio(
                "Chart type",
                ["Grouped Bar", "Heatmap"],
                horizontal = True,
                key        = "demo_gender_chart",
            )

            if chart_style == "Grouped Bar":
                st.plotly_chart(
                    category_by_group(
                        demo_df, "category", "gender", "revenue",
                        "Revenue by Product Category & Gender",
                    ),
                    width="stretch",
                )
            else:
                st.plotly_chart(
                    category_group_heatmap(
                        demo_df, "category", "gender", "revenue",
                        "Revenue Heatmap: Product Category × Gender",
                    ),
                    width="stretch",
                )

            # Gender share summary
            try:
                gender_totals = (
                    demo_df.groupby("gender")["revenue"]
                    .sum()
                    .reset_index()
                    .rename(columns={"revenue": "total_revenue"})
                )
                total = gender_totals["total_revenue"].sum()
                gender_totals["share_%"] = (
                    gender_totals["total_revenue"] / total * 100
                ).round(1)
                with st.expander("Gender revenue summary"):
                    st.dataframe(gender_totals, width="stretch", hide_index=True)
            except Exception:
                pass

    if has_age:
        with dtabs[tab_idx]:
            st.markdown(
                "Revenue per **product category**, split by age group. "
                "Taller bars for an age group = that cohort spends more in that product category. "
                "Useful for targeting promotions at the right age segment."
            )

            chart_style_age = st.radio(
                "Chart type",
                ["Grouped Bar", "Heatmap"],
                horizontal = True,
                key        = "demo_age_chart",
            )

            if chart_style_age == "Grouped Bar":
                st.plotly_chart(
                    category_by_group(
                        demo_df.dropna(subset=["age_group"]),
                        "category", "age_group", "revenue",
                        "Revenue by Product Category & Age Group",
                    ),
                    width="stretch",
                )
            else:
                st.plotly_chart(
                    category_group_heatmap(
                        demo_df.dropna(subset=["age_group"]),
                        "category", "age_group", "revenue",
                        "Revenue Heatmap: Product Category × Age Group",
                    ),
                    width="stretch",
                )

            # Age group share
            try:
                age_totals = (
                    demo_df.dropna(subset=["age_group"])
                    .groupby("age_group", observed=True)["revenue"]
                    .sum()
                    .reset_index()
                    .rename(columns={"revenue": "total_revenue"})
                )
                total = age_totals["total_revenue"].sum()
                age_totals["share_%"] = (
                    age_totals["total_revenue"] / total * 100
                ).round(1)
                with st.expander("Age group revenue summary"):
                    st.dataframe(age_totals, width="stretch", hide_index=True)
            except Exception:
                pass

st.markdown("---")

# ── Section 7: Additional Metrics (auto-discovered) ──────────────────────────
# Show a section only when the dataset contains extra numeric columns beyond
# what the fixed schema roles cover (e.g. profit, cost, rating).

# Which new Layer-1 metric roles are present in the filtered df?
_l1_metric_roles = ["profit", "cost", "rating"]
_l1_metrics_present = [r for r in _l1_metric_roles if r in fdf.columns] if fdf is not None else []
all_extra_metric_cols = _l1_metrics_present + [
    m for m in extra_metrics if m not in _l1_metrics_present
]

if all_extra_metric_cols and fdf is not None and not fdf.empty:
    st.subheader("Additional Metrics")
    st.caption(
        "Numeric columns in your dataset beyond the core revenue figure. "
        "Toggle between **Summary** (KPI cards + breakdown by category) "
        "and **Distribution** (histogram) for each metric."
    )

    for mcol in all_extra_metric_cols:
        if mcol not in fdf.columns:
            continue
        series  = pd.to_numeric(fdf[mcol], errors="coerce").dropna()
        if series.empty:
            continue

        col_label = _label(mcol)
        is_monetary = "($)" in col_label
        fmt = "${:,.2f}" if is_monetary else "{:,.2f}"

        with st.expander(f"**{col_label}**", expanded=True):
            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total",   fmt.format(series.sum()))
            k2.metric("Average", fmt.format(series.mean()))
            k3.metric("Minimum", fmt.format(series.min()))
            k4.metric("Maximum", fmt.format(series.max()))

            # Charts side by side
            view = st.radio(
                "View",
                ["By Product Category", "By another dimension", "Distribution"],
                horizontal = True,
                key        = f"extra_metric_view_{mcol}",
            )

            if view == "By Product Category" and "category" in fdf.columns:
                agg_fn = "mean" if mcol == "rating" else "sum"
                st.plotly_chart(
                    metric_by_dimension(fdf, mcol, "category", agg=agg_fn),
                    width="stretch",
                )
            elif view == "By another dimension":
                available_dims = (
                    [c for c in ["category", "region", "channel",
                                  "payment_method", "customer_segment",
                                  "gender", "sales_rep"] if c in fdf.columns]
                    + [d for d in extra_dims if d in fdf.columns]
                )
                if available_dims:
                    chosen_dim = st.selectbox(
                        "Group by",
                        available_dims,
                        format_func=_label,
                        key=f"extra_metric_dim_{mcol}",
                    )
                    agg_fn = "mean" if mcol == "rating" else "sum"
                    st.plotly_chart(
                        metric_by_dimension(fdf, mcol, chosen_dim, agg=agg_fn),
                        width="stretch",
                    )
                else:
                    st.info("No dimension columns available for breakdown.")
            else:
                st.plotly_chart(metric_distribution(fdf, mcol), width="stretch")

st.markdown("---")

# ── Section 8: Export ─────────────────────────────────────────────────────────
st.subheader("Export")
st.caption(
    "Exports use the **current view** (sidebar date range and dimension filters). "
    "**JSON** is the full programmatic analysis result. "
    "**CSV** is row-level filtered data with readable column headers. "
    "**ZIP** bundles KPIs, transactions, and summary tables (monthly, breakdowns, day-of-week, etc.). "
    "**PDF** groups **related charts on the same page** (e.g. monthly + cumulative revenue; weekday + histogram). "
    "Each image uses a **full print title** and a high-contrast style. "
    "Click **Generate** (uses Kaleido; parallel rendering). After filters change, generate again. "
    "Requires: `pip install kaleido`."
)

export_base = dex.safe_export_basename(state.get(state.FILE_NAME, "sales_dashboard"))

_note_parts: list[str] = []
if has_dates and sel_from and sel_to:
    _note_parts.append(f"Date range: {sel_from} → {sel_to}")
if _active:
    _note_parts.append(f"{_active} active sidebar filter(s)")
_export_filter_note = " · ".join(_note_parts) if _note_parts else "No filters applied — full cleaned dataset."


def _fdf_fingerprint(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


_dash_pdf_sig = (
    _fdf_fingerprint(fdf),
    round(float(live_stats.get("total_revenue") or 0), 4),
    int(live_stats.get("total_orders") or 0),
    _export_filter_note,
    export_base,
    tuple(extra_dims),
    tuple(extra_metrics),
)
if st.session_state.get("_dash_pdf_sig") != _dash_pdf_sig:
    st.session_state.pop("_dash_pdf_bytes", None)
    st.session_state["_dash_pdf_sig"] = _dash_pdf_sig

row_a, row_b = st.columns(2)
row_c, row_d = st.columns(2)

with row_a:
    st.download_button(
        label     = "Full analysis (JSON)",
        data      = result.to_json(),
        file_name = f"{export_base}_analysis.json",
        mime      = "application/json",
    )

with row_b:
    if fdf is not None and not fdf.empty:
        st.download_button(
            label     = "Filtered transactions (CSV)",
            data      = dex.filtered_transactions_csv_bytes(fdf),
            file_name = f"{export_base}_transactions.csv",
            mime      = "text/csv",
        )
    else:
        st.caption("No rows in the current filter — adjust filters to export CSV.")

with row_c:
    if fdf is not None and not fdf.empty:
        _zip_data = dex.dashboard_zip_bytes(
            live_stats       = live_stats,
            fdf              = fdf,
            dims             = dims,
            monthly          = monthly,
            quarterly        = quarterly,
            dow              = dow_chart,
            rep_perf         = rep_perf_df,
            crosstab         = crosstab_df,
            filter_note      = _export_filter_note,
            base_name        = export_base,
        )
        st.download_button(
            label     = "Dashboard tables (ZIP)",
            data      = _zip_data,
            file_name = f"{export_base}_dashboard_tables.zip",
            mime      = "application/zip",
        )

with row_d:
    if fdf is not None and not fdf.empty:
        if st.button("Generate dashboard PDF", type="secondary", key="gen_dash_pdf"):
            try:
                with st.spinner("Rendering grouped charts for PDF…"):
                    st.session_state["_dash_pdf_bytes"] = dex.dashboard_pdf_bytes(
                        live_stats     = live_stats,
                        fdf            = fdf,
                        result         = result,
                        dims           = dims,
                        monthly        = monthly,
                        quarterly      = quarterly,
                        dow            = dow_chart,
                        rep_perf       = rep_perf_df,
                        crosstab       = crosstab_df,
                        extra_dims     = extra_dims,
                        extra_metrics  = extra_metrics,
                        filter_note    = _export_filter_note,
                        base_name      = export_base,
                    )
            except DashboardPdfError as e:
                st.session_state.pop("_dash_pdf_bytes", None)
                st.error(str(e))
        if st.session_state.get("_dash_pdf_bytes"):
            st.download_button(
                label     = "Download dashboard PDF",
                data      = st.session_state["_dash_pdf_bytes"],
                file_name = f"{export_base}_dashboard_charts.pdf",
                mime      = "application/pdf",
                key       = "dl_dash_pdf",
            )

st.markdown("---")
st.success("Dashboard ready. Generate a written business report on the next page.")
if st.button("Next → AI Insights", type="primary"):
    st.switch_page("pages/4_AI_Insights.py")
