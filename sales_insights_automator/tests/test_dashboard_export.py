"""Tests for app.dashboard_export helpers."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _path():
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))


def test_safe_export_basename():
    from app.dashboard_export import safe_export_basename

    assert safe_export_basename("My Sales Q1.csv") == "My_Sales_Q1"
    assert safe_export_basename("weird name!!.xlsx") == "weird_name"


def test_filtered_transactions_csv_uses_display_headers():
    from app.dashboard_export import filtered_transactions_csv_bytes

    df = pd.DataFrame({"revenue": [10.0], "sales_rep": ["Ann"], "category": ["A"]})
    raw = filtered_transactions_csv_bytes(df).decode("utf-8")
    assert "Revenue ($)" in raw
    assert "Salesperson" in raw
    assert "Product Category" in raw


def test_dashboard_zip_contains_expected_members():
    from app.dashboard_export import dashboard_zip_bytes

    fdf = pd.DataFrame({"revenue": [1.0, 2.0], "order_id": ["a", "b"]})
    dims = {
        "Region": ("region", pd.DataFrame({"region": ["East"], "total_revenue": [3.0]})),
    }
    live = {"total_revenue": 3.0, "total_orders": 2}
    zbytes = dashboard_zip_bytes(
        live_stats=live,
        fdf=fdf,
        dims=dims,
        monthly=pd.DataFrame({"month": ["2024-01"], "total_revenue": [3.0]}),
        quarterly=None,
        dow=pd.DataFrame({"day": ["Mon"], "total_revenue": [1.0]}),
        rep_perf=None,
        crosstab=None,
        filter_note="Test export",
        base_name="fixture",
    )
    zf = zipfile.ZipFile(io.BytesIO(zbytes))
    names = set(zf.namelist())
    assert "README.txt" in names
    assert "kpis.csv" in names
    assert "transactions.csv" in names
    assert "monthly_revenue.csv" in names
    assert "revenue_by_day_of_week.csv" in names
    assert any(n.startswith("revenue_by_") and n.endswith(".csv") for n in names)


def test_dashboard_pdf_is_non_empty():
    from app.dashboard_export import dashboard_pdf_bytes

    live = {
        "total_revenue": 100.0,
        "total_orders": 5,
        "average_order_value": 20.0,
        "total_units_sold": 10,
        "average_discount_pct": 0.0,
        "median_order_value": 18.0,
    }
    pdf_b = dashboard_pdf_bytes(
        live_stats=live,
        dims={},
        monthly=pd.DataFrame({"m": ["2024-01"], "total_revenue": [100.0]}),
        quarterly=None,
        dow=None,
        filter_note="No filters",
        base_name="test",
    )
    assert pdf_b.startswith(b"%PDF")
