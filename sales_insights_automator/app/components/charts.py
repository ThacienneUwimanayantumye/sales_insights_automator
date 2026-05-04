"""
Reusable Plotly chart components for the Sales Insights dashboard.

Every function takes a DataFrame (or dict) and returns a plotly Figure.
Pages call these functions and display results with st.plotly_chart().
"""

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
BG        = "rgba(0,0,0,0)"   # transparent background

PALETTE = [PRIMARY, SECONDARY, ACCENT, "#8B5CF6", "#EC4899",
           "#14B8A6", "#F97316", "#06B6D4"]


def _base_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title      = dict(text=title, font=dict(size=16, color="#1F2937")),
        paper_bgcolor = BG,
        plot_bgcolor  = BG,
        font          = dict(family="Inter, sans-serif", color="#374151"),
        margin        = dict(l=10, r=10, t=40, b=10),
        legend        = dict(orientation="h", yanchor="bottom", y=1.02,
                             xanchor="right", x=1),
        hoverlabel    = dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6", zeroline=False)
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
        xaxis_tickprefix="$",
        xaxis_tickformat=",.0f",
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
    return _base_layout(fig, "Sales Rep Performance")


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
        number = dict(suffix="%", font=dict(size=28)),
        gauge = dict(
            axis  = dict(range=[0, 30]),
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
        title = dict(text="Avg Discount", font=dict(size=14)),
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=0),
                      paper_bgcolor=BG)
    return fig


# ── 7. Category × Region heatmap ─────────────────────────────────────────────

def category_heatmap(pivot_df: pd.DataFrame) -> go.Figure:
    """Heatmap of revenue by category (rows) and region (columns)."""
    # Drop the "Total" row and column if present
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
