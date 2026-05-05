"""
Build the same Plotly figures as the Analysis Dashboard for full PDF export.

Mirrors chart visibility rules in ``3_Dashboard.py`` so the PDF matches the
on-screen dashboard for the same filtered data.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go

from app.components import charts as ch
from app.components.charts import _label


def collect_dashboard_figures(
    fdf: Optional[pd.DataFrame],
    result: Any,
    dims: dict[str, tuple[str, Optional[pd.DataFrame]]],
    monthly: Optional[pd.DataFrame],
    quarterly: Optional[pd.DataFrame],
    dow_chart: Optional[pd.DataFrame],
    rep_perf_df: Optional[pd.DataFrame],
    crosstab_df: Optional[pd.DataFrame],
    extra_dims: list[str],
    extra_metrics: list[str],
) -> list[tuple[str, str, go.Figure]]:
    """Return ``(section heading, chart subtitle, figure)`` in dashboard order."""

    items: list[tuple[str, str, go.Figure]] = []
    fdf = fdf if fdf is not None else pd.DataFrame()

    # ── Revenue over time ─────────────────────────────────────────────────────
    if monthly is not None and not monthly.empty:
        items.append(("Revenue Over Time", "Monthly breakdown", ch.revenue_trend(monthly)))
    if quarterly is not None and not quarterly.empty:
        items.append(("Revenue Over Time", "Quarterly seasons", ch.revenue_quarterly(quarterly)))
    if monthly is not None and not monthly.empty:
        items.append(
            ("Revenue Over Time", "Cumulative growth", ch.revenue_area_cumulative(monthly)),
        )

    # ── Breakdowns: bar + donut + treemap per dimension ───────────────────────
    if dims:
        for label, (col, data) in dims.items():
            if data is None or data.empty:
                continue
            sec = "Revenue Breakdown"
            items.append(
                (sec, f"{label} — bar", ch.revenue_by_dimension(data, col, f"Revenue by {label}")),
            )
            items.append(
                (sec, f"{label} — donut", ch.revenue_donut(data, col, f"Revenue Share by {label}")),
            )
            items.append(
                (sec, f"{label} — treemap", ch.revenue_treemap(data, col, f"Revenue by {label} — Treemap")),
            )

        if rep_perf_df is not None and not rep_perf_df.empty:
            items.append(
                (
                    "Salesperson Performance",
                    "Revenue and average order value",
                    ch.sales_rep_performance(rep_perf_df),
                ),
            )

        if crosstab_df is not None and not crosstab_df.empty:
            items.append(
                ("Category × Region", "Revenue heatmap", ch.category_heatmap(crosstab_df)),
            )

    # ── Transaction patterns ────────────────────────────────────────────────────
    if dow_chart is not None and not dow_chart.empty:
        items.append(("Transaction Patterns", "Revenue by day of week", ch.weekday_polar(dow_chart)))
    if not fdf.empty and "revenue" in fdf.columns:
        items.append(
            ("Transaction Patterns", "Transaction size distribution", ch.revenue_histogram(fdf, "revenue")),
        )

    # ── Quantity vs revenue ─────────────────────────────────────────────────────
    if not fdf.empty:
        items.append(("Quantity vs Revenue", "Scatter plot", ch.scatter_qty_revenue(fdf)))

    # ── Regional trend (same source as dashboard: full analysis result) ───────
    rt = getattr(result, "regional_trend", None)
    if rt is not None and hasattr(rt, "empty") and not rt.empty:
        items.append(
            ("Regional Revenue Trend", "Monthly revenue by region", ch.regional_trend(rt)),
        )

    # ── Discount ────────────────────────────────────────────────────────────────
    ds = getattr(result, "discount_stats", None) or {}
    if ds:
        avg_d = float(ds.get("avg_discount_pct", 0) or 0)
        items.append(("Discount Analysis", "Average discount gauge", ch.discount_gauge(avg_d)))

    # ── Demographics ────────────────────────────────────────────────────────────
    has_cat = not fdf.empty and "category" in fdf.columns
    has_gender = has_cat and "gender" in fdf.columns
    has_age = has_cat and "age" in fdf.columns
    if has_cat and (has_gender or has_age):
        demo_df = fdf.copy()
        if has_age:
            bins = [0, 25, 35, 45, 55, 120]
            labels = ["18–25", "26–35", "36–45", "46–55", "55+"]
            demo_df["age_group"] = pd.cut(
                pd.to_numeric(demo_df["age"], errors="coerce"),
                bins=bins,
                labels=labels,
                right=True,
            )
        if has_gender:
            items.append(
                (
                    "Demographic Breakdown",
                    "Product category × gender (grouped bar)",
                    ch.category_by_group(
                        demo_df, "category", "gender", "revenue",
                        "Revenue by Product Category & Gender",
                    ),
                ),
            )
            items.append(
                (
                    "Demographic Breakdown",
                    "Product category × gender (heatmap)",
                    ch.category_group_heatmap(
                        demo_df, "category", "gender", "revenue",
                        "Revenue Heatmap: Product Category × Gender",
                    ),
                ),
            )
        if has_age:
            dfa = demo_df.dropna(subset=["age_group"])
            if not dfa.empty:
                items.append(
                    (
                        "Demographic Breakdown",
                        "Product category × age group (grouped bar)",
                        ch.category_by_group(
                            dfa, "category", "age_group", "revenue",
                            "Revenue by Product Category & Age Group",
                        ),
                    ),
                )
                items.append(
                    (
                        "Demographic Breakdown",
                        "Product category × age group (heatmap)",
                        ch.category_group_heatmap(
                            dfa, "category", "age_group", "revenue",
                            "Revenue Heatmap: Product Category × Age Group",
                        ),
                    ),
                )

    # ── Additional metrics ──────────────────────────────────────────────────────
    _l1_roles = ("profit", "cost", "rating")
    l1_present = [r for r in _l1_roles if r in fdf.columns]
    all_extra = l1_present + [c for c in extra_metrics if c not in l1_present]
    if all_extra and not fdf.empty:
        dims_list = [
            c
            for c in (
                "category",
                "region",
                "channel",
                "payment_method",
                "customer_segment",
                "gender",
                "sales_rep",
            )
            if c in fdf.columns
        ]
        for d in extra_dims:
            if d in fdf.columns and d not in dims_list:
                dims_list.append(d)

        for mcol in all_extra:
            if mcol not in fdf.columns:
                continue
            series = pd.to_numeric(fdf[mcol], errors="coerce").dropna()
            if series.empty:
                continue
            label = _label(mcol)
            sec = "Additional Metrics"
            agg_fn = "mean" if mcol == "rating" else "sum"
            if "category" in fdf.columns:
                items.append(
                    (
                        sec,
                        f"{label} by product category",
                        ch.metric_by_dimension(fdf, mcol, "category", agg=agg_fn),
                    ),
                )
            dim2 = next((c for c in dims_list if c != "category"), None)
            if dim2:
                items.append(
                    (
                        sec,
                        f"{label} by {_label(dim2)}",
                        ch.metric_by_dimension(fdf, mcol, dim2, agg=agg_fn),
                    ),
                )
            elif "category" not in fdf.columns and dims_list:
                c0 = dims_list[0]
                items.append(
                    (
                        sec,
                        f"{label} by {_label(c0)}",
                        ch.metric_by_dimension(fdf, mcol, c0, agg=agg_fn),
                    ),
                )
            items.append((sec, f"{label} distribution", ch.metric_distribution(fdf, mcol)))

    return items
