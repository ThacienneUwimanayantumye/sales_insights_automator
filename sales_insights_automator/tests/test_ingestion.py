"""
Unit tests for the ingestion layer.

Run with:
    pytest tests/test_ingestion.py -v

Tests use the generated sample data.  Run scripts/create_sample_data.py first
if the sample files do not yet exist.
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.base import DataSource, DataSourceError
from ingestion.csv_source import CSVSource
from ingestion.sqlite_source import SQLiteSource
from ingestion.google_drive_source import GoogleDriveSource


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_csv(tmp_path) -> str:
    """Write a tiny CSV to a temp directory and return its path."""
    data = (
        "order_id,date,product,category,region,sales_rep,quantity,unit_price,discount_pct,revenue\n"
        "ORD-00001,2024-01-15,Laptop Pro 15,Computers,North,Alice Martin,2,1299.00,0.05,2468.10\n"
        "ORD-00002,2024-02-10,USB-C Hub,Accessories,South,Bob Chen,5,49.99,0.00,249.95\n"
        "ORD-00003,2024-03-22,Monitor 27in,Displays,West,Carla Diaz,1,449.00,0.10,404.10\n"
    )
    csv_file = tmp_path / "test_sales.csv"
    csv_file.write_text(data)
    return str(csv_file)


@pytest.fixture()
def sample_sqlite(tmp_path) -> str:
    """Create a tiny SQLite database and return its path."""
    db_path = str(tmp_path / "test_sales.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE sales (
            order_id    TEXT,
            date        TEXT,
            product     TEXT,
            category    TEXT,
            region      TEXT,
            sales_rep   TEXT,
            quantity    INTEGER,
            unit_price  REAL,
            discount_pct REAL,
            revenue     REAL
        )
        """
    )
    conn.execute(
        "INSERT INTO sales VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("ORD-00001", "2024-01-15", "Laptop Pro 15", "Computers", "North",
         "Alice Martin", 2, 1299.00, 0.05, 2468.10),
    )
    conn.execute(
        "INSERT INTO sales VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("ORD-00002", "2024-02-10", "USB-C Hub", "Accessories", "South",
         "Bob Chen", 5, 49.99, 0.00, 249.95),
    )
    conn.commit()
    conn.close()
    return db_path


# ── DataSource base class ─────────────────────────────────────────────────────

class TestDataSourceInterface:
    """Ensure the abstract interface cannot be instantiated directly."""

    def test_cannot_instantiate_base_class(self):
        with pytest.raises(TypeError):
            DataSource()  # type: ignore[abstract]


# ── CSVSource ─────────────────────────────────────────────────────────────────

class TestCSVSource:

    def test_validate_existing_file(self, sample_csv):
        source = CSVSource(sample_csv)
        assert source.validate() is True

    def test_validate_missing_file(self, tmp_path):
        source = CSVSource(str(tmp_path / "nonexistent.csv"))
        assert source.validate() is False

    def test_load_returns_dataframe(self, sample_csv):
        source = CSVSource(sample_csv)
        df = source.load()
        assert isinstance(df, pd.DataFrame)

    def test_load_correct_row_count(self, sample_csv):
        source = CSVSource(sample_csv)
        df = source.load()
        assert len(df) == 3

    def test_load_correct_columns(self, sample_csv):
        expected_columns = {
            "order_id", "date", "product", "category", "region",
            "sales_rep", "quantity", "unit_price", "discount_pct", "revenue",
        }
        source = CSVSource(sample_csv)
        df = source.load()
        assert expected_columns.issubset(set(df.columns))

    def test_source_file_column_added(self, sample_csv):
        source = CSVSource(sample_csv)
        df = source.load()
        assert "_source_file" in df.columns

    def test_load_validated_raises_on_missing_file(self, tmp_path):
        source = CSVSource(str(tmp_path / "missing.csv"))
        with pytest.raises(DataSourceError):
            source.load_validated()

    def test_describe_contains_filepath(self, sample_csv):
        source = CSVSource(sample_csv)
        assert sample_csv in source.describe()


# ── SQLiteSource ──────────────────────────────────────────────────────────────

class TestSQLiteSource:

    def test_validate_existing_db(self, sample_sqlite):
        source = SQLiteSource(db_path=sample_sqlite, table="sales")
        assert source.validate() is True

    def test_validate_missing_db(self, tmp_path):
        source = SQLiteSource(db_path=str(tmp_path / "ghost.db"), table="sales")
        assert source.validate() is False

    def test_load_with_table_name(self, sample_sqlite):
        source = SQLiteSource(db_path=sample_sqlite, table="sales")
        df = source.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_load_with_raw_query(self, sample_sqlite):
        source = SQLiteSource(
            db_path=sample_sqlite,
            query="SELECT * FROM sales WHERE region = 'North'",
        )
        df = source.load()
        assert len(df) == 1
        assert df.iloc[0]["region"] == "North"

    def test_mutually_exclusive_params(self):
        with pytest.raises(ValueError):
            SQLiteSource(db_path="x.db", query="SELECT 1", table="sales")

    def test_both_params_missing(self):
        with pytest.raises(ValueError):
            SQLiteSource(db_path="x.db")

    def test_list_tables(self, sample_sqlite):
        source = SQLiteSource(db_path=sample_sqlite, table="sales")
        tables = source.list_tables()
        assert "sales" in tables

    def test_load_validated_raises_on_missing_db(self, tmp_path):
        source = SQLiteSource(db_path=str(tmp_path / "ghost.db"), table="sales")
        with pytest.raises(DataSourceError):
            source.load_validated()


# ── GoogleDriveSource (stub) ──────────────────────────────────────────────────

class TestGoogleDriveSourceStub:

    def test_load_raises_not_implemented(self):
        source = GoogleDriveSource(file_id="fake_id")
        with pytest.raises(NotImplementedError):
            source.load()

    def test_validate_raises_not_implemented(self):
        source = GoogleDriveSource(file_id="fake_id")
        with pytest.raises(NotImplementedError):
            source.validate()

    def test_describe_mentions_stub(self):
        source = GoogleDriveSource(file_id="fake_id")
        assert "STUB" in source.describe()
