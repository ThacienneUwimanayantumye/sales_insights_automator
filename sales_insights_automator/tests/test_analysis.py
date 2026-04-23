"""
Unit tests for the analysis layer.

Tests are grouped by module:
  - TestMetrics         (metrics.py)
  - TestTrends          (trends.py)
  - TestAnalysisResult  (insight_builder.py)
  - TestSalesAnalyzer   (analyzer.py — integration)

Run with:
    pytest tests/test_analysis.py -v
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis import metrics as m
from analysis import trends  as t
from analysis.insight_builder import AnalysisResult
from analysis.analyzer import SalesAnalyzer


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def sales_df() -> pd.DataFrame:
    """Realistic, cleaned 20-row sales DataFrame."""
    import random
    random.seed(0)

    products  = ["Laptop", "Keyboard", "Monitor", "Webcam", "Headphones"]
    regions   = ["North", "South", "East", "West"]
    reps      = ["Alice", "Bob", "Carla"]
    categories= {"Laptop": "Computers", "Keyboard": "Peripherals",
                 "Monitor": "Displays", "Webcam": "Peripherals",
                 "Headphones": "Audio"}
    prices    = {"Laptop": 1299.0, "Keyboard": 89.99, "Monitor": 449.0,
                 "Webcam": 129.0, "Headphones": 349.0}

    start = date(2024, 1, 1)
    rows  = []
    for i in range(1, 21):
        p   = random.choice(products)
        qty = random.randint(1, 5)
        disc= round(random.uniform(0, 0.1), 2)
        rev = round(prices[p] * qty * (1 - disc), 2)
        rows.append({
            "order_id":    f"ORD-{i:03d}",
            "date":        pd.Timestamp(start + timedelta(days=random.randint(0, 364))),
            "product":     p,
            "category":    categories[p],
            "region":      random.choice(regions),
            "sales_rep":   random.choice(reps),
            "quantity":    qty,
            "unit_price":  prices[p],
            "discount_pct":disc,
            "revenue":     rev,
        })

    return pd.DataFrame(rows)


@pytest.fixture()
def monthly_df(sales_df) -> pd.DataFrame:
    return t.monthly_revenue(sales_df)


# ── TestMetrics ───────────────────────────────────────────────────────────────

class TestMetrics:

    def test_summary_stats_keys(self, sales_df):
        stats = m.compute_summary_stats(sales_df)
        expected = {
            "total_revenue", "total_orders", "total_units_sold",
            "average_order_value", "average_unit_price",
            "average_discount_pct", "median_order_value",
            "min_order_value", "max_order_value",
        }
        assert expected == set(stats.keys())

    def test_total_revenue_matches_sum(self, sales_df):
        stats = m.compute_summary_stats(sales_df)
        assert stats["total_revenue"] == pytest.approx(sales_df["revenue"].sum(), rel=1e-3)

    def test_total_orders_is_unique_count(self, sales_df):
        stats = m.compute_summary_stats(sales_df)
        assert stats["total_orders"] == sales_df["order_id"].nunique()

    def test_revenue_by_dimension_returns_dataframe(self, sales_df):
        df = m.revenue_by_dimension(sales_df, "region")
        assert isinstance(df, pd.DataFrame)
        assert "total_revenue" in df.columns
        assert "revenue_share_pct" in df.columns

    def test_revenue_share_sums_to_100(self, sales_df):
        df = m.revenue_by_region(sales_df)
        assert df["revenue_share_pct"].sum() == pytest.approx(100.0, rel=1e-2)

    def test_revenue_by_product_sorted_descending(self, sales_df):
        df = m.revenue_by_product(sales_df)
        revenues = df["total_revenue"].tolist()
        assert revenues == sorted(revenues, reverse=True)

    def test_top_n_returns_n_rows(self, sales_df):
        df = m.top_n(sales_df, "product", n=3)
        assert len(df) == 3

    def test_discount_analysis_keys(self, sales_df):
        d = m.discount_analysis(sales_df)
        assert "avg_discount_pct" in d
        assert "revenue_lost_to_discount" in d
        assert "discount_rate" in d

    def test_discount_rate_between_0_and_1(self, sales_df):
        d = m.discount_analysis(sales_df)
        assert 0.0 <= d["discount_rate"] <= 1.0

    def test_sales_rep_performance_has_rep_column(self, sales_df):
        df = m.sales_rep_performance(sales_df)
        assert "sales_rep" in df.columns
        assert "avg_order_value" in df.columns

    def test_category_region_crosstab_has_total_row(self, sales_df):
        pivot = m.category_region_crosstab(sales_df)
        assert "Total" in pivot.index
        assert "Total" in pivot.columns


# ── TestTrends ────────────────────────────────────────────────────────────────

class TestTrends:

    def test_monthly_revenue_returns_dataframe(self, sales_df):
        df = t.monthly_revenue(sales_df)
        assert isinstance(df, pd.DataFrame)
        assert "total_revenue" in df.columns
        assert "month" in df.columns

    def test_monthly_revenue_total_matches_overall(self, sales_df):
        monthly = t.monthly_revenue(sales_df)
        assert monthly["total_revenue"].sum() == pytest.approx(
            sales_df["revenue"].sum(), rel=1e-3
        )

    def test_compute_growth_rates_adds_columns(self, monthly_df):
        df = t.compute_growth_rates(monthly_df)
        assert "mom_growth_pct" in df.columns
        assert "mom_growth_abs" in df.columns

    def test_first_month_growth_is_nan(self, monthly_df):
        df = t.compute_growth_rates(monthly_df)
        assert pd.isna(df["mom_growth_pct"].iloc[0])

    def test_rolling_revenue_adds_column(self, monthly_df):
        df = t.rolling_revenue(monthly_df, window=3)
        assert "rolling_3m_avg_revenue" in df.columns

    def test_rolling_revenue_length_unchanged(self, monthly_df):
        df = t.rolling_revenue(monthly_df, window=3)
        assert len(df) == len(monthly_df)

    def test_best_and_worst_periods_returns_two_dfs(self, monthly_df):
        best, worst = t.best_and_worst_periods(monthly_df, n=2)
        assert len(best)  == 2
        assert len(worst) == 2

    def test_best_revenue_gte_worst_revenue(self, monthly_df):
        best, worst = t.best_and_worst_periods(monthly_df, n=1)
        assert best["total_revenue"].iloc[0] >= worst["total_revenue"].iloc[0]

    def test_revenue_by_day_of_week_has_7_rows(self, sales_df):
        df = t.revenue_by_day_of_week(sales_df)
        assert len(df) == 7

    def test_revenue_by_day_of_week_starts_monday(self, sales_df):
        df = t.revenue_by_day_of_week(sales_df)
        assert df["day_of_week"].iloc[0] == "Monday"

    def test_monthly_revenue_by_region_wide_format(self, sales_df):
        df = t.monthly_revenue_by_region(sales_df)
        assert "month" in df.columns
        # Should have at least one region column beyond 'month'
        assert len(df.columns) > 1

    def test_trend_summary_keys(self, monthly_df):
        summary = t.trend_summary(monthly_df)
        assert "best_month"          in summary
        assert "worst_month"         in summary
        assert "overall_growth_pct"  in summary
        assert "avg_monthly_revenue" in summary

    def test_trend_summary_empty_df(self):
        empty = pd.DataFrame(columns=["month", "total_revenue"])
        assert t.trend_summary(empty) == {}


# ── TestAnalysisResult ────────────────────────────────────────────────────────

class TestAnalysisResult:

    @pytest.fixture()
    def result(self, sales_df) -> AnalysisResult:
        return SalesAnalyzer().analyze(sales_df)

    def test_text_summary_is_string(self, result):
        assert isinstance(result.text_summary(), str)

    def test_text_summary_contains_kpi_header(self, result):
        assert "KEY PERFORMANCE INDICATORS" in result.text_summary()

    def test_text_summary_contains_revenue(self, result):
        assert "Total Revenue" in result.text_summary()

    def test_to_dict_keys(self, result):
        d = result.to_dict()
        assert "summary_stats" in d
        assert "monthly_trend" in d
        assert "revenue_by_region" in d

    def test_to_json_is_valid_json(self, result):
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert "summary_stats" in parsed

    def test_to_json_writes_file(self, result, tmp_path):
        path = str(tmp_path / "result.json")
        result.to_json(path)
        with open(path) as f:
            parsed = json.load(f)
        assert "row_count" in parsed

    def test_repr_contains_row_count(self, result):
        assert str(result.row_count) in repr(result)


# ── TestSalesAnalyzer (integration) ──────────────────────────────────────────

class TestSalesAnalyzer:

    def test_analyze_returns_analysis_result(self, sales_df):
        analyzer = SalesAnalyzer()
        result = analyzer.analyze(sales_df)
        assert isinstance(result, AnalysisResult)

    def test_result_unavailable_before_analyze(self):
        with pytest.raises(RuntimeError):
            _ = SalesAnalyzer().result

    def test_result_available_after_analyze(self, sales_df):
        analyzer = SalesAnalyzer()
        analyzer.analyze(sales_df)
        assert analyzer.result is not None

    def test_row_count_matches_input(self, sales_df):
        result = SalesAnalyzer().analyze(sales_df)
        assert result.row_count == len(sales_df)

    def test_date_range_populated(self, sales_df):
        result = SalesAnalyzer().analyze(sales_df)
        assert "from" in result.date_range
        assert "to"   in result.date_range
        assert result.date_range["from"] != "unknown"

    def test_summary_stats_non_zero(self, sales_df):
        result = SalesAnalyzer().analyze(sales_df)
        assert result.summary_stats["total_revenue"] > 0
        assert result.summary_stats["total_orders"]  > 0

    def test_all_dataframe_fields_non_empty(self, sales_df):
        result = SalesAnalyzer().analyze(sales_df)
        assert not result.revenue_by_region.empty
        assert not result.revenue_by_product.empty
        assert not result.revenue_by_category.empty
        assert not result.revenue_by_sales_rep.empty
        assert not result.monthly_trend.empty

    def test_raises_on_empty_dataframe(self):
        df = pd.DataFrame(columns=["order_id", "date", "revenue"])
        with pytest.raises(ValueError, match="empty"):
            SalesAnalyzer().analyze(df)

    def test_custom_rolling_window(self, sales_df):
        result = SalesAnalyzer(rolling_window=2).analyze(sales_df)
        assert "rolling_2m_avg_revenue" in result.monthly_trend.columns
