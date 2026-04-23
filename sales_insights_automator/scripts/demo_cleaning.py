"""
Cleaning layer demo script.

Demonstrates four scenarios:
  1. Default DataCleaner (zero config — sensible defaults)
  2. Programmatic config
  3. Config loaded from a JSON file  ← the impressive bit
  4. Injecting intentional data quality issues, then cleaning them

Run from the project root:
    python scripts/demo_cleaning.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.csv_source import CSVSource
from cleaning.cleaner import DataCleaner
from cleaning.config import CleaningConfig
from config.settings import SAMPLE_CSV_PATH

DIVIDER = "─" * 62


def section(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


# ── Demo 1 — Default cleaner (no config) ──────────────────────────────────────

def demo_default_cleaner() -> None:
    section("DEMO 1 — Default DataCleaner (zero config)")

    raw_df = CSVSource(SAMPLE_CSV_PATH).load()
    cleaner = DataCleaner()
    clean_df = cleaner.clean(raw_df)
    print(cleaner.report.summary())


# ── Demo 2 — Programmatic config ─────────────────────────────────────────────

def demo_programmatic_config() -> None:
    section("DEMO 2 — Programmatic CleaningConfig")

    raw_df = CSVSource(SAMPLE_CSV_PATH).load()

    config = CleaningConfig(
        normalize_columns=True,
        drop_duplicates=True,
        duplicate_subset=["order_id"],
        type_conversions={
            "date": "datetime",
            "quantity": "int",
            "revenue": "float",
        },
    )

    cleaner = DataCleaner(config)
    clean_df = cleaner.clean(raw_df)

    print(f"\nDate column dtype  : {clean_df['date'].dtype}")
    print(f"Quantity dtype     : {clean_df['quantity'].dtype}")
    print(cleaner.report.summary())


# ── Demo 3 — JSON-driven config ───────────────────────────────────────────────

def demo_json_config() -> None:
    section("DEMO 3 — Config loaded from JSON file")

    config_path = str(PROJECT_ROOT / "config" / "default_cleaning.json")
    print(f"Loading config from: {config_path}")

    raw_df = CSVSource(SAMPLE_CSV_PATH).load()
    cleaner = DataCleaner.from_json(config_path)

    print(f"Active config      : {cleaner.config}")
    clean_df = cleaner.clean(raw_df)
    print(cleaner.report.summary())

    # Show that we can serialise the report for logging / storage
    report_json = cleaner.report.to_json()
    print(f"\nReport as JSON (first 300 chars):\n{report_json[:300]}...")


# ── Demo 4 — Dirty data scenario ─────────────────────────────────────────────

def demo_dirty_data() -> None:
    section("DEMO 4 — Intentionally dirty data")

    # Build a DataFrame with realistic data quality issues
    dirty_df = pd.DataFrame({
        "Order ID":    ["ORD-001", "ORD-002", "ORD-002", "ORD-003", "ORD-004", "ORD-005"],
        "Date":        ["2024-01-15", "2024-02-10", "2024-02-10", None, "BAD DATE", "2024-04-01"],
        "Product":     ["Laptop", "Keyboard", "Keyboard", "Monitor", None, "Headphones"],
        "Region":      ["North", None, None, "South", "West", None],
        "Sales Rep.":  ["Alice", "Bob", "Bob", None, "Carla", "David"],
        "Quantity":    [2, "five", 5, 1, 3, 4],   # "five" — bad type
        "Unit Price":  [1299.0, 89.99, 89.99, 449.0, 129.0, 349.0],
        "Revenue":     [2598.0, 449.95, 449.95, None, 387.0, 1396.0],
    })

    print(f"\nDirty DataFrame ({len(dirty_df)} rows):")
    print(dirty_df.to_string(index=False))
    print(f"\nNull counts:\n{dirty_df.isnull().sum().to_string()}")

    config = CleaningConfig(
        normalize_columns=True,
        drop_duplicates=True,
        duplicate_subset=["order_id"],
        fill_missing={
            "region":    "Unknown",
            "sales_rep": "Unassigned",
            "revenue":   "median",
        },
        type_conversions={
            "date":       "datetime",
            "quantity":   "numeric",
            "unit_price": "float",
            "revenue":    "float",
        },
    )

    cleaner = DataCleaner(config)
    clean_df = cleaner.clean(dirty_df)

    print(f"\nClean DataFrame ({len(clean_df)} rows):")
    print(clean_df.to_string(index=False))
    print(cleaner.report.summary())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nSales Insights Automator — Cleaning Layer Demo")
    print("=" * 62)

    demo_default_cleaner()
    demo_programmatic_config()
    demo_json_config()
    demo_dirty_data()

    print(f"\n{'=' * 62}")
    print("All cleaning demos complete.")
