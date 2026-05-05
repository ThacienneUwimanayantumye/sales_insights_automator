"""
Build grouped Plotly figures for dashboard PDF export.

Charts that answer the same business question are placed in the same *group*
so the PDF renderer can stack them on one page with a single section header.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go

from app.components import charts as ch
from app.components.charts import _label


@dataclass(frozen=True)
class PdfChartItem:
    """One chart with a self-contained title for static export."""

    export_title: str
    figure: go.Figure


@dataclass
class PdfChartGroup:
    """Logical bundle of charts (one PDF page, stacked)."""

    section_heading: str
    blurb: str
    charts: list[PdfChartItem]


def collect_dashboard_pdf_groups(
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
) -> list[PdfChartGroup]:
    """Return ordered groups for PDF assembly (mirrors ``3_Dashboard.py`` visibility)."""

    groups: list[PdfChartGroup] = []
    fdf = fdf if fdf is not None else pd.DataFrame()

    # ── 1. Revenue over time (single insight: trend + seasonality + run-rate) ─
    time_charts: list[PdfChartItem] = []
    if monthly is not None and not monthly.empty:
        time_charts.append(
            PdfChartItem(
                "Revenue over time — Monthly revenue, rolling average, and month-over-month growth",
                ch.revenue_trend(monthly),
            )
        )
    if quarterly is not None and not quarterly.empty:
        time_charts.append(
            PdfChartItem(
                "Revenue over time — Quarterly revenue (compare seasons across years)",
                ch.revenue_quarterly(quarterly),
            )
        )
    if monthly is not None and not monthly.empty:
        time_charts.append(
            PdfChartItem(
                "Revenue over time — Cumulative revenue vs monthly contribution",
                ch.revenue_area_cumulative(monthly),
            )
        )
    if time_charts:
        groups.append(
            PdfChartGroup(
                section_heading="Revenue over time",
                blurb="How sales evolve by month and quarter, and how fast total revenue is accumulating.",
                charts=time_charts,
            )
        )

    # ── 2. Revenue breakdown (per dimension: ranking + composition) ───────────
    if dims:
        for label, (col, data) in dims.items():
            if data is None or data.empty:
                continue
            dim_charts = [
                PdfChartItem(
                    f"Revenue breakdown — {label} — Ranked bars (share of total shown on hover)",
                    ch.revenue_by_dimension(data, col, f"Revenue by {label}"),
                ),
                PdfChartItem(
                    f"Revenue breakdown — {label} — Share of total revenue (donut)",
                    ch.revenue_donut(data, col, f"Revenue share by {label}"),
                ),
            ]
            groups.append(
                PdfChartGroup(
                    section_heading=f"Revenue breakdown — {label}",
                    blurb="Bar chart ranks segments; donut shows each segment’s share of the filtered total.",
                    charts=dim_charts,
                )
            )

        if rep_perf_df is not None and not rep_perf_df.empty:
            groups.append(
                PdfChartGroup(
                    section_heading="Salesperson performance",
                    blurb="Total revenue vs average order value per salesperson for the current filter.",
                    charts=[
                        PdfChartItem(
                            "Salesperson performance — Revenue and average order value by rep",
                            ch.sales_rep_performance(rep_perf_df),
                        ),
                    ],
                )
            )

        if crosstab_df is not None and not crosstab_df.empty:
            groups.append(
                PdfChartGroup(
                    section_heading="Category × region",
                    blurb="Where each product category sells strongest by territory.",
                    charts=[
                        PdfChartItem(
                            "Category × region — Revenue heatmap (category rows, region columns)",
                            ch.category_heatmap(crosstab_df),
                        ),
                    ],
                )
            )

    # ── 3. Transaction patterns (calendar rhythm + deal sizes) ────────────────────
    pattern_charts: list[PdfChartItem] = []
    if dow_chart is not None and not dow_chart.empty:
        pattern_charts.append(
            PdfChartItem(
                "Transaction patterns — Revenue by weekday (polar view)",
                ch.weekday_polar(dow_chart),
            )
        )
    if not fdf.empty and "revenue" in fdf.columns:
        pattern_charts.append(
            PdfChartItem(
                "Transaction patterns — Distribution of individual transaction amounts",
                ch.revenue_histogram(fdf, "revenue"),
            )
        )
    if pattern_charts:
        groups.append(
            PdfChartGroup(
                section_heading="Transaction patterns",
                blurb="Which weekdays drive revenue, and how transaction sizes are spread.",
                charts=pattern_charts,
            )
        )

    # ── 4. Quantity vs revenue (single chart) ───────────────────────────────────
    if not fdf.empty:
        groups.append(
            PdfChartGroup(
                section_heading="Order size vs quantity",
                blurb="Each point is one transaction: units purchased vs revenue (colour = product category when available).",
                charts=[
                    PdfChartItem(
                        "Order size vs quantity — Scatter of units vs revenue per transaction",
                        ch.scatter_qty_revenue(fdf),
                    ),
                ],
            )
        )

    # ── 5. Regional trend ───────────────────────────────────────────────────────
    rt = getattr(result, "regional_trend", None)
    if rt is not None and hasattr(rt, "empty") and not rt.empty:
        groups.append(
            PdfChartGroup(
                section_heading="Regional revenue trend",
                blurb="Monthly revenue lines by region (same scope as the on-screen dashboard chart).",
                charts=[
                    PdfChartItem(
                        "Regional revenue trend — Monthly revenue by sales region",
                        ch.regional_trend(rt),
                    ),
                ],
            )
        )

    # ── 6. Discount ─────────────────────────────────────────────────────────────
    ds = getattr(result, "discount_stats", None) or {}
    if ds:
        avg_d = float(ds.get("avg_discount_pct", 0) or 0)
        groups.append(
            PdfChartGroup(
                section_heading="Discount analysis",
                blurb="Average discount level on the analysed dataset (gauge).",
                charts=[
                    PdfChartItem(
                        "Discount analysis — Average discount percentage (gauge)",
                        ch.discount_gauge(avg_d),
                    ),
                ],
            )
        )

    # ── 7. Demographics ─────────────────────────────────────────────────────────
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
            groups.append(
                PdfChartGroup(
                    section_heading="Demographics — Gender",
                    blurb="How revenue splits across product categories for each gender.",
                    charts=[
                        PdfChartItem(
                            "Demographics — Gender — Grouped bars (revenue by category and gender)",
                            ch.category_by_group(
                                demo_df, "category", "gender", "revenue",
                                "Revenue by Product Category & Gender",
                            ),
                        ),
                        PdfChartItem(
                            "Demographics — Gender — Heatmap (category × gender)",
                            ch.category_group_heatmap(
                                demo_df, "category", "gender", "revenue",
                                "Revenue Heatmap: Product Category × Gender",
                            ),
                        ),
                    ],
                )
            )
        if has_age:
            dfa = demo_df.dropna(subset=["age_group"])
            if not dfa.empty:
                groups.append(
                    PdfChartGroup(
                        section_heading="Demographics — Age group",
                        blurb="How revenue splits across product categories for each age band.",
                        charts=[
                            PdfChartItem(
                                "Demographics — Age group — Grouped bars (revenue by category and age)",
                                ch.category_by_group(
                                    dfa, "category", "age_group", "revenue",
                                    "Revenue by Product Category & Age Group",
                                ),
                            ),
                            PdfChartItem(
                                "Demographics — Age group — Heatmap (category × age group)",
                                ch.category_group_heatmap(
                                    dfa, "category", "age_group", "revenue",
                                    "Revenue Heatmap: Product Category × Age Group",
                                ),
                            ),
                        ],
                    )
                )

    # ── 8. Additional metrics (per metric: breakdown + distribution) ────────────
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
            ml = _label(mcol)
            agg_fn = "mean" if mcol == "rating" else "sum"
            m_charts: list[PdfChartItem] = []
            if "category" in fdf.columns:
                m_charts.append(
                    PdfChartItem(
                        f"Additional metric — {ml} — Total or average by product category",
                        ch.metric_by_dimension(fdf, mcol, "category", agg=agg_fn),
                    )
                )
            dim2 = next((c for c in dims_list if c != "category"), None)
            if dim2:
                m_charts.append(
                    PdfChartItem(
                        f"Additional metric — {ml} — By {_label(dim2)}",
                        ch.metric_by_dimension(fdf, mcol, dim2, agg=agg_fn),
                    )
                )
            elif "category" not in fdf.columns and dims_list:
                c0 = dims_list[0]
                m_charts.append(
                    PdfChartItem(
                        f"Additional metric — {ml} — By {_label(c0)}",
                        ch.metric_by_dimension(fdf, mcol, c0, agg=agg_fn),
                    )
                )
            m_charts.append(
                PdfChartItem(
                    f"Additional metric — {ml} — Distribution across transactions",
                    ch.metric_distribution(fdf, mcol),
                )
            )
            groups.append(
                PdfChartGroup(
                    section_heading=f"Additional metric — {ml}",
                    blurb="Breakdown by key dimensions and the shape of the distribution.",
                    charts=m_charts,
                )
            )

    return groups
