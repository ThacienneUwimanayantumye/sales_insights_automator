"""
Ingestion layer demo script.

Demonstrates loading data from CSV and SQLite using the connector classes.
Run from the project root:

    python scripts/demo_ingestion.py

Expected output: row counts, schema, and a sample of the loaded data.
"""

import sys
from pathlib import Path

# Allow running from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SAMPLE_CSV_PATH, SAMPLE_SQLITE_PATH
from ingestion.csv_source import CSVSource
from ingestion.sqlite_source import SQLiteSource
from ingestion.google_drive_source import GoogleDriveSource


# ── Helpers ───────────────────────────────────────────────────────────────────

DIVIDER = "─" * 60


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def print_summary(df, label: str) -> None:
    print(f"\n[{label}] Shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[{label}] Columns     : {list(df.columns)}")
    print(f"[{label}] Date range  : {df['date'].min()} → {df['date'].max()}")
    print(f"[{label}] Total rev   : ${df['revenue'].sum():,.2f}")
    print(f"\n[{label}] First 3 rows:\n{df.head(3).to_string(index=False)}")


# ── Demo 1 — CSV connector ────────────────────────────────────────────────────

def demo_csv() -> None:
    section("DEMO 1 — CSVSource")

    source = CSVSource(
        filepath=SAMPLE_CSV_PATH,
        parse_dates=["date"],
    )

    print(f"Connector  : {source.describe()}")
    print(f"Validation : {source.validate()}")

    df = source.load_validated()
    print_summary(df, "CSV")


# ── Demo 2 — SQLite connector (full table) ────────────────────────────────────

def demo_sqlite_table() -> None:
    section("DEMO 2 — SQLiteSource (full table)")

    source = SQLiteSource(db_path=SAMPLE_SQLITE_PATH, table="sales")

    print(f"Connector  : {source.describe()}")
    print(f"Tables     : {source.list_tables()}")
    print(f"Validation : {source.validate()}")

    df = source.load_validated()
    print_summary(df, "SQLite/table")


# ── Demo 3 — SQLite connector (custom query) ─────────────────────────────────

def demo_sqlite_query() -> None:
    section("DEMO 3 — SQLiteSource (custom query — West region only)")

    source = SQLiteSource(
        db_path=SAMPLE_SQLITE_PATH,
        query="SELECT * FROM sales WHERE region = 'West' ORDER BY date",
    )

    print(f"Connector  : {source.describe()}")
    df = source.load_validated()
    print_summary(df, "SQLite/query")

    # Quick aggregation to show the data is sensible
    by_product = (
        df.groupby("product")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    by_product.columns = ["product", "total_revenue"]
    print(f"\n[SQLite/query] Revenue by product (West):\n{by_product.to_string(index=False)}")


# ── Demo 4 — SQLite monthly_revenue view ─────────────────────────────────────

def demo_sqlite_view() -> None:
    section("DEMO 4 — SQLiteSource (monthly_revenue view)")

    source = SQLiteSource(
        db_path=SAMPLE_SQLITE_PATH,
        query="SELECT * FROM monthly_revenue LIMIT 12",
    )

    df = source.load_validated()
    print(f"\n[SQLite/view] Monthly revenue sample:\n{df.to_string(index=False)}")


# ── Demo 5 — GoogleDriveSource stub ──────────────────────────────────────────

def demo_google_drive_stub() -> None:
    section("DEMO 5 — GoogleDriveSource (stub — expected NotImplementedError)")

    source = GoogleDriveSource(file_id="some_fake_file_id_here")
    print(f"Connector  : {source.describe()}")

    try:
        source.load_validated()
    except NotImplementedError as exc:
        print(f"[GoogleDriveSource] Caught expected NotImplementedError: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nSales Insights Automator — Ingestion Layer Demo")
    print("=" * 60)

    demo_csv()
    demo_sqlite_table()
    demo_sqlite_query()
    demo_sqlite_view()
    demo_google_drive_stub()

    print(f"\n{'=' * 60}")
    print("All ingestion demos complete.")
