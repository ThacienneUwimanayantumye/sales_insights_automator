"""
Page 3 — Analysis Dashboard

Interactive charts powered by the AnalysisResult from SalesAnalyzer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app import state
from app.components.charts import (
    revenue_trend,
    revenue_by_dimension,
    sales_rep_performance,
    revenue_by_weekday,
    regional_trend,
    discount_gauge,
    category_heatmap,
)

st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Sales Insights")
st.sidebar.markdown("---")
st.sidebar.markdown("**Step 3 of 4** — Explore your sales data interactively.")

# ── Guard: need analysis result ───────────────────────────────────────────────
if not state.has(state.ANALYSIS_RESULT):
    st.warning("No analysis available yet. Please complete **🔧 Schema Setup** first.")
    if st.button("← Go to Schema Setup"):
        st.switch_page("pages/2_Schema_Setup.py")
    st.stop()

result    = state.get(state.ANALYSIS_RESULT)
stats     = result.summary_stats
date_from = result.date_range.get("from", "?")
date_to   = result.date_range.get("to",   "?")

st.title("📈 Analysis Dashboard")
st.caption(f"Period: **{date_from}** → **{date_to}** · {result.row_count:,} transactions")
st.markdown("---")

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue",      f"${stats['total_revenue']:,.0f}")
k2.metric("Total Orders",       f"{stats['total_orders']:,}")
k3.metric("Avg Order Value",    f"${stats['average_order_value']:,.0f}")
k4.metric("Units Sold",         f"{stats['total_units_sold']:,}")
k5.metric("Avg Discount",       f"{stats['average_discount_pct']:.1f}%")

st.markdown("---")

# ── Revenue trend ─────────────────────────────────────────────────────────────
st.subheader("Revenue Trend")
if result.monthly_trend is not None and not result.monthly_trend.empty:
    st.plotly_chart(
        revenue_trend(result.monthly_trend),
        width="stretch",
    )

    trend = result.trend_summary
    if trend:
        t1, t2, t3 = st.columns(3)
        t1.metric("Best Month",  trend.get("best_month", "—"),
                  f"${trend.get('best_month_revenue', 0):,.0f}")
        t2.metric("Overall Growth",
                  f"{trend.get('overall_growth_pct', 0):+.1f}%")
        t3.metric("Avg Monthly Revenue",
                  f"${trend.get('avg_monthly_revenue', 0):,.0f}")
else:
    st.info("Monthly trend not available — date column may not be datetime format.")

st.markdown("---")

# ── Breakdown charts ──────────────────────────────────────────────────────────
tab_region, tab_product, tab_category, tab_rep = st.tabs(
    ["By Region", "By Product", "By Category", "By Sales Rep"]
)

with tab_region:
    if result.revenue_by_region is not None and not result.revenue_by_region.empty:
        col = result.revenue_by_region.columns[0]
        st.plotly_chart(
            revenue_by_dimension(result.revenue_by_region, col,
                                 "Revenue by Region", "#2563EB"),
            width="stretch",
        )
        st.dataframe(result.revenue_by_region, width="stretch", hide_index=True)
    else:
        st.info("Region data not available.")

with tab_product:
    if result.revenue_by_product is not None and not result.revenue_by_product.empty:
        col = result.revenue_by_product.columns[0]
        st.plotly_chart(
            revenue_by_dimension(result.revenue_by_product, col,
                                 "Revenue by Product", "#10B981"),
            width="stretch",
        )
        st.dataframe(result.revenue_by_product, width="stretch", hide_index=True)
    else:
        st.info("Product data not available.")

with tab_category:
    if result.revenue_by_category is not None and not result.revenue_by_category.empty:
        col = result.revenue_by_category.columns[0]
        st.plotly_chart(
            revenue_by_dimension(result.revenue_by_category, col,
                                 "Revenue by Category", "#8B5CF6"),
            width="stretch",
        )
        # Heatmap if both category and region are available
        if (result.revenue_by_region is not None and
                not result.revenue_by_region.empty and
                hasattr(result, "category_region_crosstab") and
                result.category_region_crosstab is not None):
            st.markdown("#### Category × Region Heatmap")
            try:
                st.plotly_chart(
                    category_heatmap(result.category_region_crosstab),
                    width="stretch",
                )
            except Exception:
                pass
    else:
        st.info("Category data not available.")

with tab_rep:
    if result.revenue_by_sales_rep is not None and not result.revenue_by_sales_rep.empty:
        st.plotly_chart(
            sales_rep_performance(result.revenue_by_sales_rep),
            width="stretch",
        )
        st.dataframe(result.revenue_by_sales_rep, width="stretch", hide_index=True)
    else:
        st.info("Sales rep data not available.")

st.markdown("---")

# ── Day-of-week + discount ────────────────────────────────────────────────────
col_dow, col_disc = st.columns([2, 1])

with col_dow:
    st.subheader("Revenue by Day of Week")
    if result.revenue_by_weekday is not None and not result.revenue_by_weekday.empty:
        st.plotly_chart(
            revenue_by_weekday(result.revenue_by_weekday),
            width="stretch",
        )
    else:
        st.info("Day-of-week data not available.")

with col_disc:
    st.subheader("Discount Analysis")
    d = result.discount_stats
    if d:
        st.plotly_chart(
            discount_gauge(d.get("avg_discount_pct", 0)),
            width="stretch",
        )
        st.metric("Max Discount",         f"{d.get('max_discount_pct', 0):.1f}%")
        st.metric("Orders with Discount", f"{d.get('orders_with_discount', 0):,}")
        st.metric("Revenue Lost",         f"${d.get('revenue_lost_to_discount', 0):,.0f}")
    else:
        st.info("Discount data not available.")

st.markdown("---")

# ── Regional trend ────────────────────────────────────────────────────────────
st.subheader("Regional Revenue Trend")
if result.regional_trend is not None and not result.regional_trend.empty:
    st.plotly_chart(
        regional_trend(result.regional_trend),
        width="stretch",
    )
else:
    st.info("Regional trend not available.")

st.markdown("---")

# ── Download analysis as JSON ─────────────────────────────────────────────────
st.subheader("Export")
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    if st.download_button(
        label     = "Download analysis (JSON)",
        data      = result.to_json(),
        file_name = "analysis_result.json",
        mime      = "application/json",
    ):
        pass

# ── Navigate to AI ────────────────────────────────────────────────────────────
st.markdown("---")
st.success("Dashboard ready. Generate a written business report on the next page.")
if st.button("Next → AI Insights", type="primary"):
    st.switch_page("pages/4_AI_Insights.py")
