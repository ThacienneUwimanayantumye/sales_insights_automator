"""Unit tests for SchemaConfig and SchemaWizard."""

import json
import pytest
import pandas as pd

from config.schema import SchemaConfig, ALL_ROLES, REQUIRED_ROLES
from profiling.schema_wizard import SchemaWizard


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def standard_df() -> pd.DataFrame:
    """DataFrame that already uses standard column names."""
    return pd.DataFrame({
        "order_id":    ["O1", "O2", "O3"],
        "date":        ["2024-01-01", "2024-01-02", "2024-01-03"],
        "product":     ["Laptop", "Phone", "Tablet"],
        "category":    ["Electronics", "Electronics", "Electronics"],
        "region":      ["North", "South", "East"],
        "sales_rep":   ["Alice", "Bob", "Alice"],
        "quantity":    [1, 2, 1],
        "unit_price":  [1000.0, 500.0, 700.0],
        "discount_pct":[0.05, 0.10, 0.00],
        "revenue":     [950.0, 900.0, 700.0],
    })


@pytest.fixture
def renamed_df() -> pd.DataFrame:
    """Same data but with non-standard column names — a real-world scenario."""
    return pd.DataFrame({
        "transaction_id": ["O1", "O2", "O3"],
        "sale_date":      ["2024-01-01", "2024-01-02", "2024-01-03"],
        "item_name":      ["Laptop", "Phone", "Tablet"],
        "dept":           ["Electronics", "Electronics", "Electronics"],
        "territory":      ["North", "South", "East"],
        "agent":          ["Alice", "Bob", "Alice"],
        "qty":            [1, 2, 1],
        "price":          [1000.0, 500.0, 700.0],
        "disc":           [0.05, 0.10, 0.00],
        "total_sales":    [950.0, 900.0, 700.0],
    })


@pytest.fixture
def custom_schema() -> SchemaConfig:
    """SchemaConfig matching the renamed_df fixture."""
    return SchemaConfig(
        order_id   = "transaction_id",
        date       = "sale_date",
        product    = "item_name",
        category   = "dept",
        region     = "territory",
        sales_rep  = "agent",
        quantity   = "qty",
        unit_price = "price",
        discount   = "disc",
        revenue    = "total_sales",
    )


# ── SchemaConfig: rename_to_standard ─────────────────────────────────────────

class TestRenameToStandard:
    def test_no_rename_needed_for_standard_names(self, standard_df):
        schema = SchemaConfig()
        result = schema.rename_to_standard(standard_df)
        assert list(result.columns) == list(standard_df.columns)

    def test_renames_non_standard_columns(self, renamed_df, custom_schema):
        result = custom_schema.rename_to_standard(renamed_df)
        assert "revenue"    in result.columns
        assert "date"       in result.columns
        assert "order_id"   in result.columns
        assert "total_sales" not in result.columns
        assert "sale_date"   not in result.columns

    def test_does_not_mutate_original(self, renamed_df, custom_schema):
        original_cols = list(renamed_df.columns)
        custom_schema.rename_to_standard(renamed_df)
        assert list(renamed_df.columns) == original_cols

    def test_data_values_preserved(self, renamed_df, custom_schema):
        result = custom_schema.rename_to_standard(renamed_df)
        assert list(result["revenue"]) == list(renamed_df["total_sales"])
        assert list(result["date"])    == list(renamed_df["sale_date"])

    def test_skips_none_roles(self, renamed_df):
        schema = SchemaConfig(revenue="total_sales", category=None)
        result = schema.rename_to_standard(renamed_df)
        assert "revenue" in result.columns
        # category was None so its original column should remain unchanged
        assert "dept" in result.columns

    def test_skips_missing_columns(self, renamed_df):
        schema = SchemaConfig(revenue="nonexistent_column")
        result = schema.rename_to_standard(renamed_df)
        # No crash — just skips the column that doesn't exist
        assert "nonexistent_column" not in result.columns


# ── SchemaConfig: validate ────────────────────────────────────────────────────

class TestValidate:
    def test_valid_standard_config(self, standard_df):
        schema = SchemaConfig()
        errors = schema.validate(standard_df)
        assert errors == []

    def test_valid_custom_config(self, renamed_df, custom_schema):
        errors = custom_schema.validate(renamed_df)
        assert errors == []

    def test_missing_required_column(self, standard_df):
        schema = SchemaConfig(revenue="nonexistent")
        errors = schema.validate(standard_df)
        assert any("revenue" in e for e in errors)

    def test_none_required_role(self, standard_df):
        schema = SchemaConfig(revenue=None)
        errors = schema.validate(standard_df)
        assert any("revenue" in e for e in errors)

    def test_optional_roles_not_validated(self, standard_df):
        schema = SchemaConfig(category=None, sales_rep=None)
        errors = schema.validate(standard_df)
        assert errors == []


# ── SchemaConfig: serialisation ───────────────────────────────────────────────

class TestSerialisation:
    def test_to_dict(self):
        schema = SchemaConfig(revenue="total_sales")
        d = schema.to_dict()
        assert d["revenue"] == "total_sales"
        assert "order_id" in d

    def test_from_dict(self):
        schema = SchemaConfig.from_dict({"revenue": "amt", "date": "txn_date"})
        assert schema.revenue == "amt"
        assert schema.date    == "txn_date"
        # Unspecified roles get defaults
        assert schema.order_id == "order_id"

    def test_from_dict_ignores_unknown_keys(self):
        schema = SchemaConfig.from_dict({"revenue": "sales", "unknown_key": "x"})
        assert schema.revenue == "sales"

    def test_round_trip_json(self, tmp_path):
        original = SchemaConfig(revenue="total_sales", date="sale_date")
        path = str(tmp_path / "schema.json")
        original.to_json(path)
        loaded = SchemaConfig.from_json(path)
        assert loaded.revenue == "total_sales"
        assert loaded.date    == "sale_date"

    def test_json_file_is_valid_json(self, tmp_path):
        schema = SchemaConfig(revenue="sales")
        path = str(tmp_path / "schema.json")
        schema.to_json(path)
        with open(path) as fh:
            data = json.load(fh)
        assert data["revenue"] == "sales"

    def test_creates_parent_directory(self, tmp_path):
        schema = SchemaConfig()
        path = str(tmp_path / "sub" / "dir" / "schema.json")
        schema.to_json(path)
        loaded = SchemaConfig.from_json(path)
        assert loaded.revenue == "revenue"


# ── SchemaConfig: helpers ─────────────────────────────────────────────────────

class TestHelpers:
    def test_mapped_roles_excludes_none(self):
        schema = SchemaConfig(category=None, sales_rep=None)
        mapped = schema.mapped_roles()
        assert "category"  not in mapped
        assert "sales_rep" not in mapped
        assert "revenue"   in mapped

    def test_repr(self):
        schema = SchemaConfig()
        r = repr(schema)
        assert "SchemaConfig" in r
        assert "mapped" in r

    def test_summary_contains_all_roles(self):
        schema = SchemaConfig()
        s = schema.summary()
        for role in ALL_ROLES:
            assert role in s


# ── SchemaWizard: auto-detection ──────────────────────────────────────────────

class TestSchemaWizardDetect:
    def test_detects_standard_column_names(self, standard_df):
        wizard = SchemaWizard()
        schema = wizard.detect(standard_df)
        assert schema.revenue   == "revenue"
        assert schema.order_id  == "order_id"

    def test_detects_renamed_columns(self, renamed_df):
        wizard = SchemaWizard()
        schema = wizard.detect(renamed_df)
        # Should detect total_sales as revenue
        assert schema.revenue == "total_sales"

    def test_detects_date_column(self, renamed_df):
        wizard = SchemaWizard()
        schema = wizard.detect(renamed_df)
        assert schema.date == "sale_date"

    def test_detects_order_id_column(self, renamed_df):
        wizard = SchemaWizard()
        schema = wizard.detect(renamed_df)
        assert schema.order_id == "transaction_id"

    def test_no_column_assigned_twice(self, standard_df):
        wizard = SchemaWizard()
        schema = wizard.detect(standard_df)
        assigned = [v for v in schema.mapped_roles().values() if v]
        assert len(assigned) == len(set(assigned)), "A column was assigned to multiple roles"

    def test_returns_schema_config_instance(self, standard_df):
        schema = SchemaWizard().detect(standard_df)
        assert isinstance(schema, SchemaConfig)

    def test_unrecognisable_columns_get_none(self):
        df = pd.DataFrame({"aaa": [1], "bbb": ["x"], "ccc": [0.5]})
        schema = SchemaWizard(min_confidence=5.0).detect(df)
        # With high threshold and unrecognisable names, roles should be None
        none_roles = [r for r in ALL_ROLES if getattr(schema, r) is None]
        assert len(none_roles) > 0


# ── SchemaWizard: silent run ──────────────────────────────────────────────────

class TestSchemaWizardSilentRun:
    def test_silent_run_returns_schema(self, standard_df):
        schema = SchemaWizard().run(standard_df, silent=True)
        assert isinstance(schema, SchemaConfig)

    def test_silent_run_saves_json(self, standard_df, tmp_path):
        path = str(tmp_path / "schema.json")
        SchemaWizard().run(standard_df, save_path=path, silent=True)
        loaded = SchemaConfig.from_json(path)
        assert isinstance(loaded, SchemaConfig)

    def test_silent_run_does_not_prompt(self, standard_df, monkeypatch):
        """Ensure no input() calls are made in silent mode."""
        def fail_input(prompt=""):
            raise AssertionError("input() was called in silent mode")
        monkeypatch.setattr("builtins.input", fail_input)
        SchemaWizard().run(standard_df, silent=True)   # must not raise


# ── Integration: SalesAnalyzer with SchemaConfig ──────────────────────────────

class TestSalesAnalyzerIntegration:
    """Verify that the analysis pipeline works end-to-end with a custom schema."""

    def _make_renamed_df(self) -> pd.DataFrame:
        """500-row dataset with non-standard column names."""
        import numpy as np
        rng = np.random.default_rng(42)
        n = 100
        return pd.DataFrame({
            "txn_id":      [f"T{i:04d}" for i in range(n)],
            "txn_date":    pd.date_range("2024-01-01", periods=n, freq="D"),
            "item":        rng.choice(["Laptop", "Phone", "Tablet"], n),
            "dept":        rng.choice(["Electronics", "Accessories"], n),
            "area":        rng.choice(["North", "South", "East"], n),
            "rep_name":    rng.choice(["Alice", "Bob"], n),
            "qty":         rng.integers(1, 5, n),
            "price":       rng.choice([500.0, 1000.0, 700.0], n),
            "disc":        rng.choice([0.0, 0.05, 0.10], n),
            "total_sales": rng.uniform(400, 1200, n).round(2),
        })

    def test_analyzer_with_schema_succeeds(self):
        from analysis.analyzer import SalesAnalyzer
        df = self._make_renamed_df()
        schema = SchemaConfig(
            order_id   = "txn_id",
            date       = "txn_date",
            product    = "item",
            category   = "dept",
            region     = "area",
            sales_rep  = "rep_name",
            quantity   = "qty",
            unit_price = "price",
            discount   = "disc",
            revenue    = "total_sales",
        )
        analyzer = SalesAnalyzer(schema=schema)
        result = analyzer.analyze(df)
        assert result.summary_stats["total_revenue"] > 0
        assert result.row_count == 100

    def test_analyzer_raises_on_invalid_schema(self):
        from analysis.analyzer import SalesAnalyzer
        df = self._make_renamed_df()
        bad_schema = SchemaConfig(revenue="nonexistent_col")
        analyzer = SalesAnalyzer(schema=bad_schema)
        with pytest.raises(ValueError, match="Schema validation failed"):
            analyzer.analyze(df)
