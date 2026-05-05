"""
Time-series and trend analysis functions.

All functions here are pure — they take a DataFrame, return a DataFrame
or dict, and never mutate their input or hold state.

Prerequisites
-------------
The ``date`` column must be datetime dtype before calling these functions.
The cleaning layer's ``type_conversions: {"date": "datetime"}`` handles this.

Assumed column names (post-cleaning):
  date, revenue, quantity, order_id, region, product, category
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from analysis.metrics import (
    COL_DATE, COL_ORDER_ID, COL_QUANTITY, COL_REVENUE,
    COL_REGION, COL_PRODUCT, COL_CATEGORY,
)


# ── 1. Monthly revenue trend ──────────────────────────────────────────────────

def monthly_revenue(
    df: pd.DataFrame,
    date_col: str = COL_DATE,
    freq: str = "ME",
) -> pd.DataFrame:
    """Aggregate revenue by calendar month.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
        Name of the datetime column.
    freq : str
        Resampling frequency.  Defaults to ``"ME"`` (month-end).
        Use ``"W"`` for weekly or ``"QE"`` for quarterly.

    Returns
    -------
    pd.DataFrame
        Columns: ``month`` (period string), ``total_revenue``,
                 ``total_units``, ``order_count``
        Sorted chronologically.
    """
    df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)

    # Build agg dict defensively — only include columns that actually exist
    _count_col = COL_ORDER_ID if COL_ORDER_ID in df.columns else COL_REVENUE
    _monthly_agg: dict = {
        "total_revenue": (COL_REVENUE, "sum"),
        "order_count":   (_count_col,  "count"),
    }
    if COL_QUANTITY in df.columns:
        _monthly_agg["total_units"] = (COL_QUANTITY, "sum")

    monthly = df.resample(freq).agg(**_monthly_agg).reset_index()

    monthly.rename(columns={date_col: "month"}, inplace=True)
    monthly["month"] = monthly["month"].dt.to_period("M").astype(str)
    monthly["total_revenue"] = monthly["total_revenue"].round(2)
    return monthly


# ── 2. Month-over-month growth rates ─────────────────────────────────────────

def compute_growth_rates(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Add month-over-month revenue growth rate to a monthly trend DataFrame.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Output of ``monthly_revenue()``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two new columns:
          ``mom_growth_pct``  — percentage change vs prior month (rounded, 2dp)
          ``mom_growth_abs``  — absolute revenue change vs prior month
    """
    df = monthly_df.copy()
    df["mom_growth_abs"] = df["total_revenue"].diff().round(2)
    df["mom_growth_pct"] = (df["total_revenue"].pct_change() * 100).round(2)
    return df


# ── 3. Rolling average ────────────────────────────────────────────────────────

def rolling_revenue(
    monthly_df: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """Add a rolling mean of revenue to smooth out month-to-month noise.

    A 3-month rolling average is the standard way to spot underlying trends
    while filtering out seasonal one-off spikes.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Output of ``monthly_revenue()``.
    window : int
        Number of periods for the rolling window.  Defaults to 3.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with a new ``rolling_avg_revenue`` column.
    """
    df = monthly_df.copy()
    df[f"rolling_{window}m_avg_revenue"] = (
        df["total_revenue"]
        .rolling(window=window, min_periods=1)
        .mean()
        .round(2)
    )
    return df


# ── 4. Peak and trough identification ────────────────────────────────────────

def best_and_worst_periods(
    monthly_df: pd.DataFrame,
    n: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the top-N and bottom-N months by revenue.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Output of ``monthly_revenue()``.
    n : int
        Number of months to include in each list.

    Returns
    -------
    best : pd.DataFrame
        Top-N months sorted by revenue descending.
    worst : pd.DataFrame
        Bottom-N months sorted by revenue ascending.
    """
    best  = monthly_df.nlargest(n,  "total_revenue").reset_index(drop=True)
    worst = monthly_df.nsmallest(n, "total_revenue").reset_index(drop=True)
    return best, worst


# ── 5. Revenue by day of week ─────────────────────────────────────────────────

def revenue_by_day_of_week(
    df: pd.DataFrame,
    date_col: str = COL_DATE,
) -> pd.DataFrame:
    """Aggregate revenue by day of the week.

    Useful for spotting whether certain days consistently produce more revenue
    — a common question in business reviews.

    Returns
    -------
    pd.DataFrame
        Columns: ``day_of_week`` (Mon–Sun), ``total_revenue``,
                 ``order_count``, ``avg_order_value``
        Sorted Monday → Sunday.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_of_week"] = df[date_col].dt.day_name()

    _count_col = COL_ORDER_ID if COL_ORDER_ID in df.columns else COL_REVENUE
    result = (
        df.groupby("day_of_week")
        .agg(**{
            "total_revenue":   (COL_REVENUE, "sum"),
            "order_count":     (_count_col,  "count"),
            "avg_order_value": (COL_REVENUE, "mean"),
        })
        .reindex(day_order)
        .reset_index()
    )
    result["total_revenue"]   = result["total_revenue"].round(2)
    result["avg_order_value"] = result["avg_order_value"].round(2)
    return result


# ── 6. Regional trend comparison ─────────────────────────────────────────────

def monthly_revenue_by_region(
    df: pd.DataFrame,
    date_col: str = COL_DATE,
) -> pd.DataFrame:
    """Monthly revenue broken down by region — a multi-series trend.

    Returns a wide-format DataFrame (months as rows, regions as columns)
    suitable for line charts or multi-series comparisons.

    Returns
    -------
    pd.DataFrame
        Index = month string, one column per region.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df["month"] = df[date_col].dt.to_period("M").astype(str)

    pivot = (
        df.groupby(["month", COL_REGION])[COL_REVENUE]
        .sum()
        .round(2)
        .unstack(fill_value=0.0)
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


# ── 7. Quarterly revenue trend ───────────────────────────────────────────────

def quarterly_revenue(
    df: pd.DataFrame,
    date_col: str = COL_DATE,
) -> pd.DataFrame:
    """Aggregate revenue by calendar quarter.

    Returns
    -------
    pd.DataFrame
        Columns: ``quarter`` (e.g. "2024Q1"), ``year`` (int), ``q_label``
                 (e.g. "Q1"), ``total_revenue``, ``order_count``,
                 ``total_units``, ``qoq_growth_pct``.
        Sorted chronologically.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)

    _count_col = COL_ORDER_ID if COL_ORDER_ID in df.columns else COL_REVENUE
    _q_agg: dict = {
        "total_revenue": (COL_REVENUE, "sum"),
        "order_count":   (_count_col,  "count"),
    }
    if COL_QUANTITY in df.columns:
        _q_agg["total_units"] = (COL_QUANTITY, "sum")

    quarterly = df.resample("QE").agg(**_q_agg).reset_index()
    quarterly.rename(columns={date_col: "_qdate"}, inplace=True)
    quarterly["quarter"] = quarterly["_qdate"].dt.to_period("Q").astype(str)
    quarterly["year"]    = quarterly["_qdate"].dt.year
    quarterly["q_label"] = "Q" + quarterly["_qdate"].dt.quarter.astype(str)
    quarterly.drop(columns=["_qdate"], inplace=True)

    quarterly["total_revenue"] = quarterly["total_revenue"].round(2)
    quarterly["qoq_growth_pct"] = (
        quarterly["total_revenue"].pct_change() * 100
    ).round(2)

    return quarterly


# ── 8. Summary stats for the trend data ──────────────────────────────────────

def trend_summary(monthly_df: pd.DataFrame) -> Dict[str, object]:
    """Compute high-level trend statistics from a monthly revenue DataFrame.

    Returns
    -------
    dict with keys:
      best_month            — month string with highest revenue
      worst_month           — month string with lowest revenue
      best_month_revenue    — peak revenue value
      worst_month_revenue   — trough revenue value
      overall_growth_pct    — revenue change from first to last month (%)
      avg_monthly_revenue   — mean monthly revenue
      months_with_growth    — count of months where MoM growth was positive
      months_with_decline   — count of months where MoM growth was negative
    """
    if monthly_df.empty:
        return {}

    df = compute_growth_rates(monthly_df)
    best_row  = df.loc[df["total_revenue"].idxmax()]
    worst_row = df.loc[df["total_revenue"].idxmin()]

    first_revenue = float(df["total_revenue"].iloc[0])
    last_revenue  = float(df["total_revenue"].iloc[-1])
    overall_growth = (
        round((last_revenue - first_revenue) / first_revenue * 100, 2)
        if first_revenue != 0 else 0.0
    )

    growth_series = df["mom_growth_pct"].dropna()

    return {
        "best_month":           str(best_row["month"]),
        "worst_month":          str(worst_row["month"]),
        "best_month_revenue":   round(float(best_row["total_revenue"]), 2),
        "worst_month_revenue":  round(float(worst_row["total_revenue"]), 2),
        "overall_growth_pct":   overall_growth,
        "avg_monthly_revenue":  round(float(df["total_revenue"].mean()), 2),
        "months_with_growth":   int((growth_series > 0).sum()),
        "months_with_decline":  int((growth_series < 0).sum()),
    }
