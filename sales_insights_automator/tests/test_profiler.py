"""Unit tests for the Data Profiling layer."""

import json
import pytest
import pandas as pd
import numpy as np

from profiling.profiler import DataProfiler, DataProfile, ColumnProfile


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A tidy 5-row DataFrame with no quality issues."""
    return pd.DataFrame({
        "order_id":  ["A1", "A2", "A3", "A4", "A5"],
        "region":    ["North", "South", "East", "West", "North"],
        "revenue":   [100.0, 200.0, 150.0, 300.0, 250.0],
        "quantity":  [1, 2, 1, 3, 2],
    })


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """DataFrame with duplicates, nulls, outliers, and a constant column."""
    return pd.DataFrame({
        "id":        [1, 2, 2, 3, 4, 5],           # rows at index 1 & 2 are exact dupes
        "name":      ["Alice", "Bob", "Bob", "Carol", None, "Dave"],  # index 4 is null
        "revenue":   [100.0, 200.0, 200.0, None, 150.0, 9999999.0],  # null + outlier
        "status":    ["open"] * 6,                  # constant column
    })


@pytest.fixture
def profiler() -> DataProfiler:
    return DataProfiler()


# ── DataProfiler.profile() ────────────────────────────────────────────────────

class TestProfileShape:
    def test_total_rows(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        assert p.total_rows == 5

    def test_total_columns(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        assert p.total_columns == 4

    def test_returns_dataprofile(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        assert isinstance(p, DataProfile)

    def test_column_count_matches(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        assert len(p.columns) == 4


# ── Duplicate detection ───────────────────────────────────────────────────────

class TestDuplicates:
    def test_no_duplicates_in_clean_data(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        assert p.duplicate_rows == 0
        assert p.duplicate_pct == 0.0

    def test_detects_duplicates(self, profiler, dirty_df):
        p = profiler.profile(dirty_df)
        assert p.duplicate_rows == 1

    def test_duplicate_pct_calculated(self, profiler, dirty_df):
        p = profiler.profile(dirty_df)
        # 1 out of 6 rows = 16.67%
        assert p.duplicate_pct == pytest.approx(16.67, abs=0.1)


# ── Null detection ────────────────────────────────────────────────────────────

class TestNulls:
    def test_no_nulls_in_clean_data(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        assert p.total_null_cells == 0
        assert p.columns_with_nulls == []

    def test_detects_null_columns(self, profiler, dirty_df):
        p = profiler.profile(dirty_df)
        # 'name' has 2 nulls, 'revenue' has 1 null
        assert "name" in p.columns_with_nulls
        assert "revenue" in p.columns_with_nulls

    def test_null_count_per_column(self, profiler, dirty_df):
        p = profiler.profile(dirty_df)
        # dirty_df: name has 1 null (index 4), revenue has 1 null (index 3)
        name_col = p.get_column("name")
        assert name_col.null_count == 1

    def test_null_pct_per_column(self, profiler, dirty_df):
        p = profiler.profile(dirty_df)
        # 1 null out of 6 rows = 16.67%
        name_col = p.get_column("name")
        assert name_col.null_pct == pytest.approx(16.67, abs=0.1)

    def test_null_density_calculated(self, profiler, dirty_df):
        p = profiler.profile(dirty_df)
        # 2 nulls across 24 cells (6×4) = 8.33%
        assert p.null_density_pct == pytest.approx(8.33, abs=0.1)


# ── Constant column detection ─────────────────────────────────────────────────

class TestConstantColumns:
    def test_detects_constant_column(self, profiler, dirty_df):
        p = profiler.profile(dirty_df)
        assert "status" in p.constant_columns

    def test_constant_flag_on_column_profile(self, profiler, dirty_df):
        p = profiler.profile(dirty_df)
        col = p.get_column("status")
        assert col.is_constant is True

    def test_non_constant_not_flagged(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        assert p.constant_columns == []


# ── Numeric statistics ────────────────────────────────────────────────────────

class TestNumericStats:
    def test_numeric_kind_inferred(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        rev = p.get_column("revenue")
        assert rev.inferred_kind == "numeric"

    def test_mean_calculated(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        rev = p.get_column("revenue")
        assert rev.mean == pytest.approx(200.0, abs=0.01)

    def test_min_max(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        rev = p.get_column("revenue")
        assert rev.min == 100.0
        assert rev.max == 300.0

    def test_no_outliers_in_clean_data(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        rev = p.get_column("revenue")
        assert rev.outlier_count == 0

    def test_detects_outlier(self, profiler, dirty_df):
        p = profiler.profile(dirty_df)
        rev = p.get_column("revenue")
        # 9999999 should be detected as an outlier
        assert rev is not None
        assert rev.outlier_count is not None and rev.outlier_count >= 1

    def test_outlier_column_in_flag_list(self, profiler, dirty_df):
        p = profiler.profile(dirty_df)
        assert "revenue" in p.outlier_columns

    def test_skewness_returned(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        rev = p.get_column("revenue")
        assert rev.skewness is not None

    def test_zero_count(self):
        df = pd.DataFrame({"x": [0, 0, 1, 2, 3]})
        p  = DataProfiler().profile(df)
        col = p.get_column("x")
        assert col.zero_count == 2


# ── Categorical statistics ────────────────────────────────────────────────────

class TestCategoricalStats:
    def test_categorical_kind_inferred(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        region = p.get_column("region")
        assert region.inferred_kind == "categorical"

    def test_top_value_populated(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        region = p.get_column("region")
        assert region.top_value == "North"
        assert region.top_value_count == 2

    def test_value_counts_populated(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        region = p.get_column("region")
        assert isinstance(region.value_counts, dict)
        assert len(region.value_counts) <= 5


# ── Likely-ID detection ───────────────────────────────────────────────────────

class TestLikelyId:
    def test_high_cardinality_flagged(self):
        df = pd.DataFrame({"uid": [f"U{i}" for i in range(100)], "val": [1] * 100})
        p  = DataProfiler().profile(df)
        assert "uid" in p.likely_id_columns

    def test_low_cardinality_not_flagged(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        assert "region" not in p.likely_id_columns


# ── Quality score ─────────────────────────────────────────────────────────────

class TestQualityScore:
    def test_clean_data_has_high_score(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        assert p.quality_score >= 90

    def test_dirty_data_has_lower_score(self, profiler, dirty_df):
        p_clean = DataProfiler().profile(
            pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        )
        p_dirty = DataProfiler().profile(dirty_df)
        assert p_dirty.quality_score < p_clean.quality_score


# ── Serialisation ─────────────────────────────────────────────────────────────

class TestSerialisation:
    def test_to_dict_returns_dict(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        d = p.to_dict()
        assert isinstance(d, dict)
        assert "total_rows" in d
        assert "columns" in d

    def test_to_json_returns_valid_json(self, profiler, clean_df):
        p      = profiler.profile(clean_df)
        j      = p.to_json()
        parsed = json.loads(j)
        assert parsed["total_rows"] == 5

    def test_to_json_saves_file(self, profiler, clean_df, tmp_path):
        p    = profiler.profile(clean_df)
        path = str(tmp_path / "profile.json")
        p.to_json(path)
        with open(path) as fh:
            parsed = json.load(fh)
        assert parsed["total_rows"] == 5


# ── DataProfile helper properties ─────────────────────────────────────────────

class TestDataProfileHelpers:
    def test_get_column_returns_none_for_missing(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        assert p.get_column("nonexistent") is None

    def test_numeric_columns_property(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        names = [c.name for c in p.numeric_columns]
        assert "revenue" in names
        assert "quantity" in names
        assert "region" not in names

    def test_categorical_columns_property(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        names = [c.name for c in p.categorical_columns]
        assert "region" in names

    def test_repr(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        r = repr(p)
        assert "5" in r
        assert "quality=" in r


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_null_column(self):
        df = pd.DataFrame({"x": [None, None, None], "y": [1, 2, 3]})
        p  = DataProfiler().profile(df)
        col = p.get_column("x")
        assert col.null_count == 3

    def test_single_row(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        p  = DataProfiler().profile(df)
        assert p.total_rows == 1

    def test_single_column(self):
        df = pd.DataFrame({"revenue": [10.0, 20.0, 30.0]})
        p  = DataProfiler().profile(df)
        assert p.total_columns == 1

    def test_empty_dataframe_shape(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        p  = DataProfiler().profile(df)
        assert p.total_rows == 0
        assert p.duplicate_rows == 0

    def test_memory_usage_positive(self, profiler, clean_df):
        p = profiler.profile(clean_df)
        assert p.memory_usage_mb >= 0
