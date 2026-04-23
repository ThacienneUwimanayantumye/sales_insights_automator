"""
Descriptive statistics and business KPIs.

Every function here is a pure function:
  - Input:  a cleaned pandas DataFrame
  - Output: a dict of scalars, or a grouped DataFrame
  - No side effects, no class state

This makes each metric independently testable and composable.  The
SalesAnalyzer calls these functions and hands the results to insight_builder
to assemble into a single AnalysisResult.

Assumed column names (post-cleaning):
  order_id, date, product, category, region, sales_rep,
  quantity, unit_price, discount_pct, revenue
"""

from typing import Dict, List

import pandas as pd


# ── Column name constants ─────────────────────────────────────────────────────
# Centralised here so if column names ever change, only this file needs editing.

COL_ORDER_ID    = "order_id"
COL_DATE        = "date"
COL_PRODUCT     = "product"
COL_CATEGORY    = "category"
COL_REGION      = "region"
COL_SALES_REP   = "sales_rep"
COL_QUANTITY    = "quantity"
COL_UNIT_PRICE  = "unit_price"
COL_DISCOUNT    = "discount_pct"
COL_REVENUE     = "revenue"


# ── 1. Top-level summary statistics ──────────────────────────────────────────

def compute_summary_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute a single-row summary of the entire sales dataset.

    Returns
    -------
    dict with keys:
      total_revenue       — sum of all revenue
      total_orders        — number of distinct orders
      total_units_sold    — sum of all quantities
      average_order_value — total_revenue / total_orders
      average_unit_price  — mean unit price across all line items
      average_discount_pct— mean discount percentage
      median_order_value  — median revenue per order
      min_order_value     — smallest single-order revenue
      max_order_value     — largest single-order revenue
    """
    total_revenue = float(df[COL_REVENUE].sum())
    total_orders  = int(df[COL_ORDER_ID].nunique())
    total_units   = int(df[COL_QUANTITY].sum())

    return {
        "total_revenue":        round(total_revenue, 2),
        "total_orders":         total_orders,
        "total_units_sold":     total_units,
        "average_order_value":  round(total_revenue / total_orders, 2) if total_orders else 0.0,
        "average_unit_price":   round(float(df[COL_UNIT_PRICE].mean()), 2),
        "average_discount_pct": round(float(df[COL_DISCOUNT].mean()) * 100, 2),
        "median_order_value":   round(float(df[COL_REVENUE].median()), 2),
        "min_order_value":      round(float(df[COL_REVENUE].min()), 2),
        "max_order_value":      round(float(df[COL_REVENUE].max()), 2),
    }


# ── 2. Revenue breakdowns by dimension ───────────────────────────────────────

def revenue_by_dimension(
    df: pd.DataFrame,
    dimension: str,
    include_units: bool = True,
) -> pd.DataFrame:
    """Aggregate revenue (and optionally units sold) grouped by any column.

    Parameters
    ----------
    df : pd.DataFrame
    dimension : str
        Column name to group by (e.g. ``"region"``, ``"product"``,
        ``"category"``, ``"sales_rep"``).
    include_units : bool
        If True, also include total units sold and order count.

    Returns
    -------
    pd.DataFrame
        Columns: ``[dimension, "total_revenue", "revenue_share_pct",
                    "total_units", "order_count"]``
        Sorted by ``total_revenue`` descending.
    """
    # Named aggregation: keys are output column names, values are (source_col, func)
    agg: Dict[str, object] = {
        "total_revenue": (COL_REVENUE,  "sum"),
        "order_count":   (COL_ORDER_ID, "count"),
    }
    if include_units:
        agg["total_units"] = (COL_QUANTITY, "sum")

    result = (
        df.groupby(dimension)
        .agg(**agg)
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )

    total = result["total_revenue"].sum()
    result["revenue_share_pct"] = (result["total_revenue"] / total * 100).round(2)
    result["total_revenue"] = result["total_revenue"].round(2)

    return result


def revenue_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue breakdown by sales region."""
    return revenue_by_dimension(df, COL_REGION)


def revenue_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue breakdown by product."""
    return revenue_by_dimension(df, COL_PRODUCT)


def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue breakdown by product category."""
    return revenue_by_dimension(df, COL_CATEGORY)


def revenue_by_sales_rep(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue breakdown by sales representative."""
    return revenue_by_dimension(df, COL_SALES_REP)


# ── 3. Top-N performers ───────────────────────────────────────────────────────

def top_n(
    df: pd.DataFrame,
    dimension: str,
    n: int = 5,
    metric: str = "total_revenue",
) -> pd.DataFrame:
    """Return the top-N rows from a dimension breakdown.

    Parameters
    ----------
    df : pd.DataFrame
    dimension : str
        Column to group by.
    n : int
        Number of top performers to return.
    metric : str
        Column in the grouped result to rank by.
        Defaults to ``"total_revenue"``.

    Returns
    -------
    pd.DataFrame
    """
    breakdown = revenue_by_dimension(df, dimension)
    return breakdown.nlargest(n, metric).reset_index(drop=True)


# ── 4. Discount analysis ──────────────────────────────────────────────────────

def discount_analysis(df: pd.DataFrame) -> Dict[str, float]:
    """Quantify the impact of discounts on revenue.

    Returns
    -------
    dict with keys:
      avg_discount_pct      — average discount across all orders
      max_discount_pct      — highest discount given
      orders_with_discount  — number of orders that had any discount
      discount_rate         — share of orders with a discount (0–1)
      revenue_lost_to_discount — estimated revenue lost due to discounting
    """
    total_orders = len(df)
    orders_with_discount = int((df[COL_DISCOUNT] > 0).sum())

    # Gross revenue = what would have been earned without any discount
    gross_revenue = float((df[COL_UNIT_PRICE] * df[COL_QUANTITY]).sum())
    actual_revenue = float(df[COL_REVENUE].sum())

    return {
        "avg_discount_pct":         round(float(df[COL_DISCOUNT].mean()) * 100, 2),
        "max_discount_pct":         round(float(df[COL_DISCOUNT].max()) * 100, 2),
        "orders_with_discount":     orders_with_discount,
        "discount_rate":            round(orders_with_discount / total_orders, 4) if total_orders else 0,
        "revenue_lost_to_discount": round(gross_revenue - actual_revenue, 2),
    }


# ── 5. Sales rep performance ──────────────────────────────────────────────────

def sales_rep_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Detailed per-rep breakdown including average deal size and discount.

    Returns
    -------
    pd.DataFrame
        Columns: sales_rep, total_revenue, order_count, avg_order_value,
                 avg_discount_pct, total_units, revenue_share_pct
        Sorted by total_revenue descending.
    """
    result = (
        df.groupby(COL_SALES_REP)
        .agg(
            total_revenue   = (COL_REVENUE,  "sum"),
            order_count     = (COL_ORDER_ID, "count"),
            avg_order_value = (COL_REVENUE,  "mean"),
            avg_discount_pct= (COL_DISCOUNT, "mean"),
            total_units     = (COL_QUANTITY, "sum"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )

    total = result["total_revenue"].sum()
    result["revenue_share_pct"] = (result["total_revenue"] / total * 100).round(2)
    result["avg_discount_pct"]  = (result["avg_discount_pct"] * 100).round(2)
    result["total_revenue"]     = result["total_revenue"].round(2)
    result["avg_order_value"]   = result["avg_order_value"].round(2)
    return result


# ── 6. Category × region cross-tab ───────────────────────────────────────────

def category_region_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue pivot: categories as rows, regions as columns.

    Useful for quickly spotting which category performs best in each region.

    Returns
    -------
    pd.DataFrame
        Index = category, columns = regions, values = total_revenue.
        A ``"Total"`` column and row are appended.
    """
    pivot = (
        df.pivot_table(
            index=COL_CATEGORY,
            columns=COL_REGION,
            values=COL_REVENUE,
            aggfunc="sum",
            fill_value=0.0,
        )
        .round(2)
    )
    pivot["Total"] = pivot.sum(axis=1)
    pivot.loc["Total"] = pivot.sum(axis=0)
    return pivot
