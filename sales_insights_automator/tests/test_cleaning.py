"""
Unit tests for the cleaning layer.

Tests are grouped by module:
  - TestCleaningConfig     (config.py)
  - TestCleaningReport     (report.py)
  - TestNormalizeColumns   (functions.py)
  - TestDropDuplicates     (functions.py)
  - TestHandleMissing      (functions.py)
  - TestConvertDtypes      (functions.py)
  - TestDropColumns        (functions.py)
  - TestDataCleaner        (cleaner.py — integration)

Run with:
    pytest tests/test_cleaning.py -v
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cleaning.config import CleaningConfig
from cleaning.report import CleaningReport
from cleaning.cleaner import DataCleaner
from cleaning import functions as fn


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Clean, well-formed sales DataFrame for baseline tests."""
    return pd.DataFrame({
        "order_id":    ["ORD-001", "ORD-002", "ORD-003"],
        "date":        ["2024-01-15", "2024-02-10", "2024-03-22"],
        "product":     ["Laptop", "Keyboard", "Monitor"],
        "region":      ["North", "South", "West"],
        "sales_rep":   ["Alice", "Bob", "Carla"],
        "quantity":    [2, 5, 1],
        "unit_price":  [1299.0, 89.99, 449.0],
        "revenue":     [2598.0, 449.95, 449.0],
    })


@pytest.fixture()
def dirty_df() -> pd.DataFrame:
    """DataFrame with duplicates, nulls, and type issues."""
    return pd.DataFrame({
        "Order ID":  ["ORD-001", "ORD-002", "ORD-002", "ORD-003"],
        "Date":      ["2024-01-15", "2024-02-10", "2024-02-10", None],
        "Product":   ["Laptop", "Keyboard", "Keyboard", "Monitor"],
        "Region":    ["North", None, None, "South"],
        "Quantity":  [2, "five", "five", 1],
        "Revenue":   [2598.0, 449.95, 449.95, None],
    })


# ── CleaningConfig ────────────────────────────────────────────────────────────

class TestCleaningConfig:

    def test_default_values(self):
        config = CleaningConfig()
        assert config.normalize_columns is True
        assert config.drop_duplicates is True
        assert config.fill_missing == {}
        assert config.type_conversions == {}
        assert config.drop_columns == []

    def test_from_dict(self):
        data = {
            "normalize_columns": False,
            "drop_duplicates": False,
            "fill_missing": {"region": "Unknown"},
        }
        config = CleaningConfig.from_dict(data)
        assert config.normalize_columns is False
        assert config.fill_missing == {"region": "Unknown"}

    def test_from_dict_ignores_unknown_keys(self):
        data = {"unknown_key": "should_be_ignored", "drop_duplicates": True}
        config = CleaningConfig.from_dict(data)
        assert config.drop_duplicates is True

    def test_from_json_roundtrip(self, tmp_path):
        config = CleaningConfig(
            fill_missing={"revenue": "median"},
            type_conversions={"date": "datetime"},
        )
        json_path = str(tmp_path / "config.json")
        config.to_json(json_path)

        loaded = CleaningConfig.from_json(json_path)
        assert loaded.fill_missing == config.fill_missing
        assert loaded.type_conversions == config.type_conversions

    def test_to_dict_is_json_serialisable(self):
        config = CleaningConfig(
            fill_missing={"region": "Unknown"},
            type_conversions={"date": "datetime"},
        )
        json.dumps(config.to_dict())  # must not raise


# ── CleaningReport ────────────────────────────────────────────────────────────

class TestCleaningReport:

    def test_rows_removed(self):
        report = CleaningReport(original_shape=(100, 5), final_shape=(90, 5))
        assert report.rows_removed == 10

    def test_retention_rate(self):
        report = CleaningReport(original_shape=(100, 5), final_shape=(75, 5))
        assert report.retention_rate == pytest.approx(0.75)

    def test_retention_rate_empty_input(self):
        report = CleaningReport(original_shape=(0, 5), final_shape=(0, 5))
        assert report.retention_rate == 1.0

    def test_total_nulls_filled(self):
        report = CleaningReport(
            null_counts_before={"region": 3, "revenue": 5},
            null_counts_after={"region": 0, "revenue": 0},
            null_fills={"region": "Unknown", "revenue": 450.0},
        )
        assert report.total_nulls_filled == 8

    def test_summary_is_string(self):
        report = CleaningReport(original_shape=(100, 5), final_shape=(98, 5))
        assert isinstance(report.summary(), str)
        assert "100" in report.summary()

    def test_to_dict_keys(self):
        report = CleaningReport(original_shape=(50, 4), final_shape=(48, 4))
        d = report.to_dict()
        assert "original_shape" in d
        assert "retention_rate" in d
        assert "rows_removed" in d

    def test_to_json_roundtrip(self, tmp_path):
        report = CleaningReport(original_shape=(50, 4), final_shape=(48, 4))
        path = str(tmp_path / "report.json")
        report.to_json(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["rows_removed"] == 2


# ── normalize_column_names ────────────────────────────────────────────────────

class TestNormalizeColumns:

    def test_lowercase(self):
        df = pd.DataFrame({"OrderID": [1], "ProductName": [2]})
        df, renamed = fn.normalize_column_names(df)
        assert "orderid" in df.columns
        assert "productname" in df.columns

    def test_spaces_to_underscores(self):
        df = pd.DataFrame({"Sales Rep": [1], "Unit Price": [2]})
        df, _ = fn.normalize_column_names(df)
        assert "sales_rep" in df.columns
        assert "unit_price" in df.columns

    def test_hyphens_and_dots(self):
        df = pd.DataFrame({"sales-rep": [1], "unit.price": [2]})
        df, _ = fn.normalize_column_names(df)
        assert "sales_rep" in df.columns
        assert "unit_price" in df.columns

    def test_already_normalised_not_in_renamed(self):
        df = pd.DataFrame({"order_id": [1], "revenue": [2]})
        _, renamed = fn.normalize_column_names(df)
        assert len(renamed) == 0

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"Order ID": [1]})
        original_cols = list(df.columns)
        fn.normalize_column_names(df)
        assert list(df.columns) == original_cols


# ── drop_duplicate_rows ───────────────────────────────────────────────────────

class TestDropDuplicates:

    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
        df, n_dropped = fn.drop_duplicate_rows(df)
        assert n_dropped == 1
        assert len(df) == 2

    def test_subset_dedup(self):
        df = pd.DataFrame({"id": [1, 1, 2], "value": [10, 99, 20]})
        df, n_dropped = fn.drop_duplicate_rows(df, subset=["id"])
        assert n_dropped == 1

    def test_no_duplicates(self, sample_df):
        df, n_dropped = fn.drop_duplicate_rows(sample_df)
        assert n_dropped == 0

    def test_invalid_subset_columns_ignored(self, sample_df):
        df, n_dropped = fn.drop_duplicate_rows(sample_df, subset=["nonexistent_col"])
        assert n_dropped == 0


# ── handle_missing_values ─────────────────────────────────────────────────────

class TestHandleMissing:

    def test_constant_fill(self):
        df = pd.DataFrame({"region": ["North", None, "South", None]})
        df, fills, rows_dropped = fn.handle_missing_values(df, {"region": "Unknown"})
        assert df["region"].isnull().sum() == 0
        assert fills["region"] == "Unknown"
        assert rows_dropped == 0

    def test_mean_fill(self):
        df = pd.DataFrame({"revenue": [100.0, 200.0, None, None]})
        df, fills, _ = fn.handle_missing_values(df, {"revenue": "mean"})
        assert df["revenue"].isnull().sum() == 0
        assert fills["revenue"] == pytest.approx(150.0)

    def test_median_fill(self):
        df = pd.DataFrame({"revenue": [100.0, 200.0, 300.0, None]})
        df, fills, _ = fn.handle_missing_values(df, {"revenue": "median"})
        assert df["revenue"].isnull().sum() == 0
        assert fills["revenue"] == pytest.approx(200.0)

    def test_mode_fill(self):
        df = pd.DataFrame({"category": ["A", "A", "B", None]})
        df, fills, _ = fn.handle_missing_values(df, {"category": "mode"})
        assert df["category"].isnull().sum() == 0
        assert fills["category"] == "A"

    def test_drop_strategy(self):
        df = pd.DataFrame({"order_id": ["A", None, "C"], "revenue": [10, 20, 30]})
        df, _, rows_dropped = fn.handle_missing_values(df, {"order_id": "drop"})
        assert rows_dropped == 1
        assert len(df) == 2

    def test_missing_column_skipped(self, sample_df):
        df, fills, _ = fn.handle_missing_values(sample_df, {"nonexistent": "mean"})
        assert "nonexistent" not in fills


# ── convert_dtypes ────────────────────────────────────────────────────────────

class TestConvertDtypes:

    def test_datetime_conversion(self):
        df = pd.DataFrame({"date": ["2024-01-15", "2024-02-10", "2024-03-22"]})
        df, applied = fn.convert_dtypes(df, {"date": "datetime"})
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert "date" in applied

    def test_numeric_conversion(self):
        df = pd.DataFrame({"quantity": ["2", "five", "3"]})
        df, applied = fn.convert_dtypes(df, {"quantity": "numeric"})
        assert pd.api.types.is_numeric_dtype(df["quantity"])
        assert pd.isna(df["quantity"].iloc[1])  # "five" → NaN

    def test_int_conversion(self):
        df = pd.DataFrame({"quantity": [2.0, 5.0, 1.0]})
        df, applied = fn.convert_dtypes(df, {"quantity": "int"})
        assert str(df["quantity"].dtype) == "Int64"

    def test_float_conversion(self):
        df = pd.DataFrame({"revenue": ["1299.00", "89.99"]})
        df, applied = fn.convert_dtypes(df, {"revenue": "float"})
        assert df["revenue"].dtype == "float64"

    def test_missing_column_skipped(self, sample_df):
        df, applied = fn.convert_dtypes(sample_df, {"nonexistent": "datetime"})
        assert "nonexistent" not in applied

    def test_unknown_dtype_skipped(self, sample_df):
        df, applied = fn.convert_dtypes(sample_df, {"revenue": "hexadecimal"})
        assert "revenue" not in applied


# ── drop_columns ──────────────────────────────────────────────────────────────

class TestDropColumns:

    def test_drops_existing_columns(self, sample_df):
        df, dropped = fn.drop_columns(sample_df, ["region", "sales_rep"])
        assert "region" not in df.columns
        assert "sales_rep" not in df.columns
        assert dropped == ["region", "sales_rep"]

    def test_ignores_missing_columns(self, sample_df):
        df, dropped = fn.drop_columns(sample_df, ["nonexistent"])
        assert dropped == []
        assert len(df.columns) == len(sample_df.columns)

    def test_does_not_mutate_input(self, sample_df):
        original_cols = list(sample_df.columns)
        fn.drop_columns(sample_df, ["region"])
        assert list(sample_df.columns) == original_cols


# ── DataCleaner (integration) ─────────────────────────────────────────────────

class TestDataCleaner:

    def test_report_unavailable_before_clean(self):
        cleaner = DataCleaner()
        with pytest.raises(RuntimeError):
            _ = cleaner.report

    def test_returns_dataframe(self, sample_df):
        cleaner = DataCleaner()
        result = cleaner.clean(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_report_populated_after_clean(self, sample_df):
        cleaner = DataCleaner()
        cleaner.clean(sample_df)
        assert cleaner.report is not None
        assert cleaner.report.original_shape == sample_df.shape

    def test_does_not_mutate_input(self, dirty_df):
        original_shape = dirty_df.shape
        cleaner = DataCleaner()
        cleaner.clean(dirty_df)
        assert dirty_df.shape == original_shape

    def test_duplicate_removal_integration(self):
        df = pd.DataFrame({
            "order_id": ["A", "A", "B"],
            "revenue":  [100.0, 100.0, 200.0],
        })
        config = CleaningConfig(
            normalize_columns=False,
            drop_duplicates=True,
            duplicate_subset=["order_id"],
        )
        cleaner = DataCleaner(config)
        result = cleaner.clean(df)
        assert len(result) == 2
        assert cleaner.report.rows_dropped_duplicates == 1

    def test_full_pipeline_dirty_data(self, dirty_df):
        config = CleaningConfig(
            normalize_columns=True,
            drop_duplicates=True,
            duplicate_subset=["order_id"],
            fill_missing={"region": "Unknown", "revenue": "median"},
            type_conversions={"date": "datetime", "quantity": "numeric"},
        )
        cleaner = DataCleaner(config)
        clean = cleaner.clean(dirty_df)

        assert clean["region"].isnull().sum() == 0
        assert cleaner.report.rows_dropped_duplicates >= 1
        assert cleaner.report.original_shape[0] > cleaner.report.final_shape[0]

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["order_id", "revenue"])
        cleaner = DataCleaner()
        result = cleaner.clean(df)
        assert result.empty
        assert cleaner.report.final_shape == (0, 2)

    def test_from_json_factory(self, tmp_path):
        config = CleaningConfig(fill_missing={"region": "Unknown"})
        json_path = str(tmp_path / "cfg.json")
        config.to_json(json_path)

        cleaner = DataCleaner.from_json(json_path)
        assert cleaner.config.fill_missing == {"region": "Unknown"}
