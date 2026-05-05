"""
Reusable Plotly chart components for the Sales Insights dashboard.

Every function takes a DataFrame (or dict) and returns a plotly Figure.
Pages call these functions and display results with st.plotly_chart().
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional

# ── Colour palette (consistent across all charts) ────────────────────────────
PRIMARY   = "#2563EB"   # blue
SECONDARY = "#10B981"   # green
ACCENT    = "#F59E0B"   # amber
DANGER    = "#EF4444"   # red
GREY      = "#6B7280"
# Opaque page + plot backgrounds so Plotly's "Download plot as PNG" (and other
# raster exports) stay readable: transparent paper inherits a dark canvas and
# dark titles disappear.
PAPER_BG  = "#FFFFFF"
PLOT_BG   = "#F9FAFB"
TITLE_COLOR = "#111827"   # WCAG-friendly on white
AXIS_COLOR  = "#1F2937"

# Legacy alias — charts previously used transparent "BG"; keep name for diffs.
BG = PAPER_BG

PALETTE = [PRIMARY, SECONDARY, ACCENT, "#8B5CF6", "#EC4899",
           "#14B8A6", "#F97316", "#06B6D4"]

# Maps internal column names → human-readable axis / legend labels.
# "category" is the standard internal name for the product-category column,
# but it must always appear as "Product Category" in charts so it is never
# confused with demographic groupings such as gender or age group.
_LABEL: dict = {
    # required roles
    "order_id":         "Order ID",
    "date":             "Date",
    "revenue":          "Revenue ($)",
    # sales dimensions
    "category":         "Product Category",
    "region":           "Region",
    "product":          "Product",
    "sales_rep":        "Salesperson",
    "quantity":         "Units Sold",
    "unit_price":       "Price per Unit ($)",
    "discount_pct":     "Discount (%)",
    # customer demographics
    "customer_id":      "Customer ID",
    "gender":           "Gender",
    "age":              "Age",
    "age_group":        "Age Group",
    # financial enrichment
    "profit":           "Profit ($)",
    "cost":             "Cost ($)",
    # transaction attributes
    "channel":          "Sales Channel",
    "payment_method":   "Payment Method",
    # customer attributes
    "customer_segment": "Customer Segment",
    "return_flag":      "Returned",
    "rating":           "Rating",
}

def _label(col: str) -> str:
    """Return the display label for an internal column name."""
    return _LABEL.get(col, col.replace("_", " ").title())


def _base_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=TITLE_COLOR)),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, sans-serif", color=AXIS_COLOR, size=12),
        margin=dict(l=12, r=12, t=52, b=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=AXIS_COLOR, size=11),
        ),
        hoverlabel=dict(bgcolor="white", font_size=13, font_color=TITLE_COLOR),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        title_font=dict(color=TITLE_COLOR, size=13),
        tickfont=dict(color="#4B5563", size=11),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        title_font=dict(color=TITLE_COLOR, size=13),
        tickfont=dict(color="#4B5563", size=11),
    )
    return fig


# ── Static export (PNG / PDF) — high contrast, readable on white paper ───────

_EXPORT_PAPER = "#FFFFFF"
_EXPORT_PLOT  = "#F3F4F6"
_EXPORT_TEXT  = "#111827"
_EXPORT_GRID  = "#D1D5DB"
_EXPORT_MUTED = "#4B5563"


def apply_static_export_style(fig: go.Figure, *, document_title: str) -> go.Figure:
    """Restyle a figure for print/PDF: white background, large title, legible axes.

    Call on a **copy** of the figure (e.g. ``go.Figure(fig)``) before ``to_image``;
    the interactive dashboard figure is left unchanged.
    """
    fig.update_layout(
        title=dict(
            text=document_title,
            font=dict(size=18, color=_EXPORT_TEXT, family="Arial, Helvetica, sans-serif"),
            x=0.02,
            xanchor="left",
            y=0.97,
            yanchor="top",
        ),
        paper_bgcolor=_EXPORT_PAPER,
        plot_bgcolor=_EXPORT_PLOT,
        font=dict(color=_EXPORT_TEXT, size=12, family="Arial, Helvetica, sans-serif"),
        margin=dict(t=76, l=56, r=40, b=52),
        legend=dict(
            font=dict(size=11, color=_EXPORT_TEXT),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=_EXPORT_GRID,
            borderwidth=1,
            orientation="h",
            yanchor="top",
            y=-0.22,
            x=0.5,
            xanchor="center",
        ),
        hovermode="closest",
    )
    fig.update_xaxes(
        title_font=dict(size=13, color=_EXPORT_TEXT),
        tickfont=dict(size=11, color=_EXPORT_MUTED),
        gridcolor=_EXPORT_GRID,
        showgrid=True,
        zeroline=False,
    )
    fig.update_yaxes(
        title_font=dict(size=13, color=_EXPORT_TEXT),
        tickfont=dict(size=11, color=_EXPORT_MUTED),
        gridcolor=_EXPORT_GRID,
        showgrid=True,
        zeroline=False,
    )
    # Remove outside bar labels (often crowded or clipped in static export)
    fig.update_traces(
        patch=dict(text=None, textposition="none"),
        selector=dict(type="bar"),
    )
    # Pie / donut: slightly smaller slice labels for export
    fig.update_traces(
        patch=dict(textfont=dict(size=11, color=_EXPORT_TEXT)),
        selector=dict(type="pie"),
    )
    # Polar charts
    if getattr(fig.layout, "polar", None) is not None:
        fig.update_layout(
            polar=dict(
                bgcolor=_EXPORT_PLOT,
                radialaxis=dict(
                    gridcolor=_EXPORT_GRID,
                    tickfont=dict(size=10, color=_EXPORT_MUTED),
                ),
                angularaxis=dict(tickfont=dict(size=11, color=_EXPORT_MUTED)),
            ),
        )
    return fig


# ── 1. Monthly revenue trend ──────────────────────────────────────────────────

def revenue_trend(monthly_df: pd.DataFrame) -> go.Figure:
    """Line chart of monthly revenue with rolling average overlay."""
    df = monthly_df.copy()

    rolling_col = next(
        (c for c in df.columns if c.startswith("rolling_")), None
    )

    fig = go.Figure()

    # Actual revenue bars
    fig.add_trace(go.Bar(
        x    = df["month"],
        y    = df["total_revenue"],
        name = "Monthly Revenue",
        marker_color = PRIMARY,
        opacity      = 0.7,
    ))

    # Rolling average line
    if rolling_col:
        fig.add_trace(go.Scatter(
            x    = df["month"],
            y    = df[rolling_col],
            name = "Rolling Avg",
            mode = "lines",
            line = dict(color=ACCENT, width=2.5, dash="dash"),
        ))

    # MoM growth annotation on line
    if "mom_growth_pct" in df.columns:
        growth = df["mom_growth_pct"].dropna()
        colours = [SECONDARY if v >= 0 else DANGER for v in growth]
        fig.add_trace(go.Scatter(
            x    = df["month"].iloc[1:],
            y    = df["total_revenue"].iloc[1:],
            mode = "markers",
            name = "MoM Growth",
            marker = dict(color=colours, size=8, symbol="diamond"),
            customdata = growth,
            hovertemplate = "%{x}<br>Revenue: $%{y:,.0f}<br>MoM: %{customdata:+.1f}%<extra></extra>",
        ))

    fig.update_layout(barmode="overlay")
    return _base_layout(fig, "Monthly Revenue Trend")


# ── 2. Revenue by dimension (horizontal bar) ──────────────────────────────────

def revenue_by_dimension(
    df: pd.DataFrame,
    dimension: str,
    title: str,
    color: str = PRIMARY,
) -> go.Figure:
    """Horizontal bar chart ranked by total revenue."""
    df = df.sort_values("total_revenue", ascending=True)

    fig = go.Figure(go.Bar(
        x           = df["total_revenue"],
        y           = df[dimension],
        orientation = "h",
        marker_color= color,
        text        = df["revenue_share_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition= "outside",
        customdata  = df[["total_revenue", "order_count"]].values,
        hovertemplate = (
            "<b>%{y}</b><br>"
            "Revenue: $%{customdata[0]:,.0f}<br>"
            "Orders: %{customdata[1]:,}<extra></extra>"
        ),
    ))

    fig.update_layout(
        xaxis_tickprefix = "$",
        xaxis_tickformat = ",.0f",
        yaxis_title      = _label(dimension),
    )
    return _base_layout(fig, title)


# ── 3. Sales rep performance ──────────────────────────────────────────────────

def sales_rep_performance(df: pd.DataFrame) -> go.Figure:
    """Grouped bar: revenue + avg order value per rep."""
    df = df.sort_values("total_revenue", ascending=False)
    reps = df["sales_rep"] if "sales_rep" in df.columns else df.iloc[:, 0]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x    = reps,
        y    = df["total_revenue"],
        name = "Total Revenue",
        marker_color = PRIMARY,
        yaxis = "y",
    ))
    fig.add_trace(go.Scatter(
        x    = reps,
        y    = df["avg_order_value"],
        name = "Avg Order Value",
        mode = "lines+markers",
        line = dict(color=ACCENT, width=2),
        marker = dict(size=8),
        yaxis = "y2",
    ))

    fig.update_layout(
        yaxis  = dict(title="Total Revenue ($)", tickprefix="$", tickformat=",.0f"),
        yaxis2 = dict(title="Avg Order ($)", overlaying="y", side="right",
                      tickprefix="$", tickformat=",.0f", showgrid=False),
    )
    return _base_layout(fig, "Salesperson Performance")


# ── 4. Revenue by weekday ─────────────────────────────────────────────────────

def revenue_by_weekday(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing which day of the week generates most revenue."""
    fig = go.Figure(go.Bar(
        x            = df["day_of_week"],
        y            = df["total_revenue"],
        marker_color = [PRIMARY if v == df["total_revenue"].max()
                        else GREY for v in df["total_revenue"]],
        text         = df["total_revenue"].apply(lambda v: f"${v:,.0f}"),
        textposition = "outside",
    ))
    fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    return _base_layout(fig, "Revenue by Day of Week")


# ── 5. Regional trend (multi-line) ────────────────────────────────────────────

def regional_trend(df: pd.DataFrame) -> go.Figure:
    """Multi-line chart: one line per region over time."""
    region_cols = [c for c in df.columns if c != "month"]
    fig = go.Figure()
    for i, region in enumerate(region_cols):
        fig.add_trace(go.Scatter(
            x    = df["month"],
            y    = df[region],
            name = region,
            mode = "lines+markers",
            line = dict(color=PALETTE[i % len(PALETTE)], width=2),
            marker = dict(size=5),
        ))
    fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    return _base_layout(fig, "Monthly Revenue by Region")


# ── 6. Discount impact ────────────────────────────────────────────────────────

def discount_gauge(avg_discount_pct: float) -> go.Figure:
    """Gauge chart showing average discount level."""
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = avg_discount_pct,
        number = dict(suffix="%", font=dict(size=28, color=TITLE_COLOR)),
        gauge = dict(
            axis  = dict(range=[0, 30], tickfont=dict(color=AXIS_COLOR)),
            bar   = dict(color=PRIMARY),
            steps = [
                dict(range=[0, 5],   color="#D1FAE5"),
                dict(range=[5, 15],  color="#FEF3C7"),
                dict(range=[15, 30], color="#FEE2E2"),
            ],
            threshold = dict(
                line  = dict(color=DANGER, width=3),
                value = 20,
            ),
        ),
        title = dict(text="Avg Discount", font=dict(size=14, color=TITLE_COLOR)),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=0),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
    )
    return fig


# ── 7. Category × Region heatmap ─────────────────────────────────────────────

def category_heatmap(pivot_df: pd.DataFrame) -> go.Figure:
    """Heatmap of revenue by category (rows) and region (columns)."""
    df = pivot_df.copy()
    if "Total" in df.index:
        df = df.drop("Total")
    if "Total" in df.columns:
        df = df.drop(columns=["Total"])

    fig = px.imshow(
        df,
        color_continuous_scale = [[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, PRIMARY]],
        text_auto              = ",.0f",
        aspect                 = "auto",
    )
    fig.update_traces(textfont=dict(size=11))
    fig.update_layout(
        coloraxis_showscale = False,
        xaxis_title         = "Region",
        yaxis_title         = "Category",
    )
    return _base_layout(fig, "Revenue by Category & Region ($)")


# ── 8. Donut / pie chart ──────────────────────────────────────────────────────

def revenue_donut(
    df: pd.DataFrame,
    dimension_col: str,
    title: str,
) -> go.Figure:
    """Donut chart showing each segment's share of total revenue.

    Non-technical users read composition questions ("which category dominates?")
    faster from a donut than from a bar chart.
    """
    df = df.sort_values("total_revenue", ascending=False)

    fig = go.Figure(go.Pie(
        labels       = df[dimension_col],
        values       = df["total_revenue"],
        hole         = 0.55,
        marker       = dict(colors=PALETTE),
        textinfo     = "label+percent",
        hovertemplate = (
            "<b>%{label}</b><br>"
            "Revenue: $%{value:,.0f}<br>"
            "Share: %{percent}<extra></extra>"
        ),
    ))
    fig.update_layout(
        showlegend = True,
        legend     = dict(orientation="v", x=1.0, y=0.5),
        margin     = dict(l=10, r=120, t=40, b=10),
    )
    return _base_layout(fig, title)


# ── 9. Treemap ────────────────────────────────────────────────────────────────

def revenue_treemap(
    df: pd.DataFrame,
    dimension_col: str,
    title: str,
) -> go.Figure:
    """Treemap where rectangle area is proportional to revenue.

    Makes large vs small differences visually obvious at a glance —
    much stronger than reading bar lengths for non-technical audiences.
    """
    fig = px.treemap(
        df,
        path   = [dimension_col],
        values = "total_revenue",
        color  = "total_revenue",
        color_continuous_scale = [[0, "#DBEAFE"], [1, PRIMARY]],
        custom_data            = ["revenue_share_pct", "order_count"],
    )
    fig.update_traces(
        hovertemplate = (
            "<b>%{label}</b><br>"
            "Revenue: $%{value:,.0f}<br>"
            "Share: %{customdata[0]:.1f}%<br>"
            "Orders: %{customdata[1]:,}<extra></extra>"
        ),
        texttemplate  = "<b>%{label}</b><br>$%{value:,.0f}",
        textfont      = dict(size=13),
    )
    fig.update_layout(
        coloraxis_showscale = False,
        margin              = dict(l=10, r=10, t=40, b=10),
        paper_bgcolor       = PAPER_BG,
        plot_bgcolor        = PLOT_BG,
    )
    fig.update_layout(title=dict(text=title, font=dict(size=17, color=TITLE_COLOR)))
    return fig


# ── 10. Cumulative revenue area chart ────────────────────────────────────────

def revenue_area_cumulative(monthly_df: pd.DataFrame) -> go.Figure:
    """Filled area chart of cumulative revenue over time.

    The slope shows acceleration or slowdown — steeper = faster growth.
    Answers "how much have we made in total so far?" in one glance.
    """
    df = monthly_df.copy()
    df["cumulative_revenue"] = df["total_revenue"].cumsum()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x            = df["month"],
        y            = df["cumulative_revenue"],
        mode         = "lines",
        fill         = "tozeroy",
        fillcolor    = f"rgba(37,99,235,0.15)",
        line         = dict(color=PRIMARY, width=2.5),
        name         = "Cumulative Revenue",
        hovertemplate = "%{x}<br>Total so far: $%{y:,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x            = df["month"],
        y            = df["total_revenue"],
        name         = "Monthly Revenue",
        marker_color = SECONDARY,
        opacity      = 0.5,
        yaxis        = "y2",
        hovertemplate = "%{x}<br>This month: $%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        yaxis  = dict(
            title        = "Cumulative Revenue ($)",
            tickprefix   = "$",
            tickformat   = ",.0f",
        ),
        yaxis2 = dict(
            title      = "Monthly ($)",
            overlaying = "y",
            side       = "right",
            showgrid   = False,
            tickprefix = "$",
            tickformat = ",.0f",
        ),
        barmode = "overlay",
    )
    return _base_layout(fig, "Cumulative Revenue Over Time")


# ── 11. Transaction amount histogram ─────────────────────────────────────────

def revenue_histogram(clean_df: pd.DataFrame, revenue_col: str = "revenue") -> go.Figure:
    """Distribution of individual transaction amounts.

    Shows whether most transactions are small (right-skewed distribution)
    or spread evenly — immediately actionable for pricing strategy.
    """
    if revenue_col not in clean_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Revenue column not available",
                           showarrow=False, font=dict(size=14, color=TITLE_COLOR))
        return _base_layout(fig, "Transaction Amount Distribution")

    fig = px.histogram(
        clean_df,
        x         = revenue_col,
        nbins     = 30,
        color_discrete_sequence = [PRIMARY],
        labels    = {revenue_col: "Transaction Amount ($)"},
        opacity   = 0.8,
    )
    fig.update_traces(
        hovertemplate = "Range: $%{x}<br>Transactions: %{y:,}<extra></extra>",
    )
    fig.update_layout(
        xaxis_tickprefix = "$",
        xaxis_tickformat = ",.0f",
        yaxis_title      = "Number of Transactions",
        bargap           = 0.05,
    )
    return _base_layout(fig, "Transaction Amount Distribution")


# ── 12. Day-of-week polar bar ─────────────────────────────────────────────────

def weekday_polar(weekday_df: pd.DataFrame) -> go.Figure:
    """Polar bar chart for day-of-week revenue pattern.

    Cyclical patterns are more intuitive on a circular axis for
    non-technical users than a standard bar chart.
    """
    df = weekday_df.copy()

    fig = go.Figure(go.Barpolar(
        r           = df["total_revenue"],
        theta       = df["day_of_week"],
        marker_color= PALETTE[:len(df)],
        marker_line_color = "white",
        marker_line_width = 1.5,
        opacity     = 0.85,
        hovertemplate = "<b>%{theta}</b><br>Revenue: $%{r:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        polar = dict(
            radialaxis = dict(
                visible    = True,
                tickprefix = "$",
                tickformat = ",.0f",
                gridcolor  = "#E5E7EB",
                tickfont=dict(color="#4B5563", size=10),
            ),
            angularaxis = dict(
                direction = "clockwise",
                rotation  = 90,
                tickfont=dict(color="#4B5563", size=11),
            ),
            bgcolor = PLOT_BG,
        ),
        showlegend    = False,
        paper_bgcolor = PAPER_BG,
        margin        = dict(l=40, r=40, t=50, b=40),
    )
    fig.update_layout(title=dict(
        text="Revenue by Day of Week",
        font=dict(size=17, color=TITLE_COLOR),
    ))
    return fig


# ── 13. Quarterly revenue bar ────────────────────────────────────────────────

def revenue_quarterly(quarterly_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of quarterly revenue.

    If multiple years are present each year gets its own colour group so
    seasons can be compared across years (Q1-2023 vs Q1-2024, etc.).
    """
    df = quarterly_df.copy()
    years = sorted(df["year"].unique())
    multi_year = len(years) > 1

    fig = go.Figure()

    if multi_year:
        for i, yr in enumerate(years):
            sub = df[df["year"] == yr]
            colour = PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Bar(
                x            = sub["q_label"],
                y            = sub["total_revenue"],
                name         = str(yr),
                marker_color = colour,
                text         = sub["total_revenue"].apply(lambda v: f"${v:,.0f}"),
                textposition = "outside",
                customdata   = sub[["qoq_growth_pct", "order_count"]].values,
                hovertemplate = (
                    "<b>%{x} " + str(yr) + "</b><br>"
                    "Revenue: $%{y:,.0f}<br>"
                    "QoQ: %{customdata[0]:+.1f}%<br>"
                    "Orders: %{customdata[1]:,}<extra></extra>"
                ),
            ))
        fig.update_layout(barmode="group")
    else:
        colours = [
            SECONDARY if v >= 0 else DANGER
            for v in df["qoq_growth_pct"].fillna(0)
        ]
        fig.add_trace(go.Bar(
            x            = df["quarter"],
            y            = df["total_revenue"],
            marker_color = colours,
            text         = df["total_revenue"].apply(lambda v: f"${v:,.0f}"),
            textposition = "outside",
            customdata   = df[["qoq_growth_pct", "order_count"]].values,
            hovertemplate = (
                "<b>%{x}</b><br>"
                "Revenue: $%{y:,.0f}<br>"
                "QoQ growth: %{customdata[0]:+.1f}%<br>"
                "Orders: %{customdata[1]:,}<extra></extra>"
            ),
        ))

    fig.update_layout(
        yaxis_tickprefix = "$",
        yaxis_tickformat = ",.0f",
        xaxis_title      = "Quarter",
        yaxis_title      = "Revenue ($)",
    )
    return _base_layout(fig, "Quarterly Revenue")


# ── 14. Category × demographic grouped bar ───────────────────────────────────

def category_by_group(
    df: pd.DataFrame,
    category_col: str,
    group_col: str,
    revenue_col: str = "revenue",
    title: str = "",
) -> go.Figure:
    """Grouped bar chart: revenue per category split by a demographic group.

    Used for category × gender or category × age_group comparisons.
    Each group (Male/Female or 18-25/26-35…) gets its own colour bar,
    making cross-group differences immediately visible.
    """
    pivot = (
        df.groupby([category_col, group_col])[revenue_col]
        .sum()
        .reset_index()
        .rename(columns={revenue_col: "total_revenue"})
    )

    groups = sorted(pivot[group_col].dropna().unique().tolist())
    fig = go.Figure()

    for i, grp in enumerate(groups):
        sub = pivot[pivot[group_col] == grp]
        fig.add_trace(go.Bar(
            x            = sub[category_col],
            y            = sub["total_revenue"],
            name         = str(grp),
            marker_color = PALETTE[i % len(PALETTE)],
            hovertemplate = (
                f"<b>%{{x}}</b> — {grp}<br>"
                "Revenue: $%{y:,.0f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode          = "group",
        xaxis_title      = _label(category_col),
        yaxis_tickprefix = "$",
        yaxis_tickformat = ",.0f",
        legend_title     = _label(group_col),
    )
    return _base_layout(fig, title or f"Revenue by {category_col.title()} & {group_col.title()}")


# ── 15. Category × demographic heatmap ───────────────────────────────────────

def category_group_heatmap(
    df: pd.DataFrame,
    category_col: str,
    group_col: str,
    revenue_col: str = "revenue",
    title: str = "",
) -> go.Figure:
    """Heatmap: categories as rows, demographic groups as columns.

    Colour intensity shows which combination generates the most revenue.
    Great for spotting which customer segment dominates each category.
    """
    pivot = (
        df.groupby([category_col, group_col])[revenue_col]
        .sum()
        .round(2)
        .unstack(fill_value=0.0)
    )

    fig = px.imshow(
        pivot,
        color_continuous_scale = [[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, PRIMARY]],
        text_auto              = ",.0f",
        aspect                 = "auto",
    )
    fig.update_traces(textfont=dict(size=12))
    fig.update_layout(
        coloraxis_showscale = False,
        xaxis_title         = _label(group_col),
        yaxis_title         = _label(category_col),
        margin              = dict(l=10, r=10, t=40, b=10),
        paper_bgcolor       = PAPER_BG,
        plot_bgcolor        = PLOT_BG,
    )
    fig.update_layout(title=dict(
        text=title or f"Revenue Heatmap: {category_col.title()} × {group_col.title()}",
        font=dict(size=17, color=TITLE_COLOR),
    ))
    return fig


# ── 16. Quantity vs Revenue scatter ──────────────────────────────────────────

_SCATTER_SYMBOLS = [
    "circle", "square", "diamond", "cross", "x",
    "triangle-up", "star", "pentagon",
]


def scatter_qty_revenue(
    clean_df: pd.DataFrame,
    qty_col: str      = "quantity",
    revenue_col: str  = "revenue",
    category_col: str = "category",
) -> go.Figure:
    """Scatter plot: units purchased vs transaction revenue, coloured by category.

    Reveals whether high-quantity orders also yield high revenue (healthy)
    or whether large orders are heavily discounted (margin risk).

    Quantity is an integer (e.g. 1-4), so without jitter every category stacks
    at the same x-position and only the top-rendered trace is visible.  We add
    a small reproducible horizontal jitter and assign distinct marker symbols
    per category so every group is clearly distinguishable.
    """
    if qty_col not in clean_df.columns or revenue_col not in clean_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Quantity and/or revenue column not available",
                           showarrow=False, font=dict(size=14, color=TITLE_COLOR))
        return _base_layout(fig, "Quantity vs Revenue")

    has_category = category_col in clean_df.columns

    # ── Build a plotting copy with jittered quantity ──────────────────────────
    rng = np.random.default_rng(seed=42)          # reproducible across rerenders
    df_plot = clean_df[[qty_col, revenue_col]
                       + ([category_col] if has_category else [])].copy()
    df_plot["_qty_jitter"] = (
        df_plot[qty_col].astype(float)
        + rng.uniform(-0.35, 0.35, size=len(df_plot))
    )

    # ── Symbol map: one distinct shape per category ───────────────────────────
    if has_category:
        cats = sorted(df_plot[category_col].dropna().unique().tolist())
        symbol_map = {
            cat: _SCATTER_SYMBOLS[i % len(_SCATTER_SYMBOLS)]
            for i, cat in enumerate(cats)
        }
        color_arg   = category_col
        symbol_arg  = category_col
    else:
        cats        = []
        symbol_map  = {}
        color_arg   = None
        symbol_arg  = None

    qty_ticks = sorted(df_plot[qty_col].dropna().unique().tolist())

    fig = px.scatter(
        df_plot,
        x                      = "_qty_jitter",
        y                      = revenue_col,
        color                  = color_arg,
        symbol                 = symbol_arg,
        symbol_map             = symbol_map or None,
        color_discrete_sequence = PALETTE,
        category_orders        = {category_col: cats} if cats else None,
        opacity                = 0.55,
        labels                 = {
            "_qty_jitter": "Units Purchased",
            revenue_col:   "Transaction Revenue ($)",
            category_col:  _label(category_col),
        },
        hover_data             = {
            "_qty_jitter": False,          # hide jittered value
            qty_col:       True,           # show original integer
            revenue_col:   ":$,.0f",
        },
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.4, color="white")))
    fig.update_layout(
        xaxis = dict(
            tickmode  = "array",
            tickvals  = qty_ticks,
            ticktext  = [str(int(v)) for v in qty_ticks],
            title     = "Units Purchased",
        ),
        yaxis_tickprefix = "$",
        yaxis_tickformat = ",.0f",
        legend_title     = _label(category_col) if has_category else "",
    )
    return _base_layout(fig, "Quantity vs Revenue per Transaction")


def metric_by_dimension(
    df: pd.DataFrame,
    metric_col: str,
    dim_col: str,
    agg: str = "sum",
    top_n: int = 20,
) -> go.Figure:
    """Horizontal bar chart for any numeric metric broken down by any dimension.

    Used by the dashboard's Additional Metrics section and the extra-dimension
    tabs in Revenue Breakdown.  Works with any (metric, dimension) combination
    so the dashboard doesn't need to know what columns a dataset has in advance.

    Parameters
    ----------
    df : pd.DataFrame
    metric_col : str
        Numeric column to aggregate (e.g. ``"profit"``, ``"rating"``).
    dim_col : str
        Categorical column to group by.
    agg : str
        Aggregation function — ``"sum"`` or ``"mean"``.
    top_n : int
        Keep the top-N groups by aggregated value.
    """
    if metric_col not in df.columns or dim_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{metric_col}' or '{dim_col}' not found",
                           showarrow=False, font=dict(size=13, color=TITLE_COLOR))
        return _base_layout(fig, f"{_label(metric_col)} by {_label(dim_col)}")

    grouped = (
        df.groupby(dim_col, observed=True)[metric_col]
        .agg(agg)
        .reset_index()
        .rename(columns={metric_col: "value"})
        .sort_values("value", ascending=True)
        .tail(top_n)
    )

    is_monetary = "($)" in _label(metric_col)
    tick_prefix = "$" if is_monetary else ""
    agg_label   = "Total" if agg == "sum" else "Average"
    y_title     = f"{agg_label} {_label(metric_col)}"

    fig = px.bar(
        grouped,
        x                      = "value",
        y                      = dim_col,
        orientation            = "h",
        color                  = "value",
        color_continuous_scale = ["#DBEAFE", PRIMARY],
        labels                 = {"value": y_title, dim_col: _label(dim_col)},
        text                   = "value",
    )
    fig.update_traces(
        texttemplate = (
            f"{tick_prefix}%{{text:,.0f}}" if is_monetary else "%{text:,.2f}"
        ),
        textposition = "outside",
    )
    fig.update_layout(
        xaxis_tickprefix = tick_prefix,
        xaxis_tickformat = ",.0f",
        coloraxis_showscale = False,
    )
    title = f"{agg_label} {_label(metric_col)} by {_label(dim_col)}"
    return _base_layout(fig, title)


def metric_distribution(
    df: pd.DataFrame,
    metric_col: str,
    n_bins: int = 30,
) -> go.Figure:
    """Histogram showing the distribution of any numeric metric.

    A generalisation of ``revenue_histogram`` that works on any column.
    """
    if metric_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{metric_col}' not available",
                           showarrow=False, font=dict(size=13, color=TITLE_COLOR))
        return _base_layout(fig, _label(metric_col))

    series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    is_monetary = "($)" in _label(metric_col)

    fig = go.Figure(go.Histogram(
        x       = series,
        nbinsx  = n_bins,
        marker  = dict(color=PRIMARY, line=dict(color="white", width=0.5)),
        opacity = 0.85,
    ))
    fig.update_layout(
        xaxis_title      = _label(metric_col),
        yaxis_title      = "Number of Transactions",
        xaxis_tickprefix = "$" if is_monetary else "",
        xaxis_tickformat = ",.0f",
        bargap           = 0.05,
    )
    return _base_layout(fig, f"Distribution of {_label(metric_col)}")
