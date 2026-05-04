"""
Data profiling demo — shows the full quality report on raw ingested data.

Run from the project root:
    python scripts/demo_profiler.py

The script runs two scenarios:
  1. Clean sample data    — shows what a healthy profile looks like
  2. Dirty injected data  — shows how the profiler catches real problems
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.csv_source import CSVSource
from profiling.profiler import DataProfiler
from config.settings import SAMPLE_CSV_PATH

DIVIDER = "=" * 65


def section(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


# ── Scenario 1 — clean sample data ───────────────────────────────────────────

def demo_clean_data() -> None:
    section("SCENARIO 1 — Clean sample data (500 rows)")

    raw_df = CSVSource(SAMPLE_CSV_PATH).load()
    profiler = DataProfiler()
    profile  = profiler.profile(raw_df)
    profile.print_report()

    # Save to JSON for reference
    out_path = str(PROJECT_ROOT / "data" / "raw" / "data_profile_clean.json")
    profile.to_json(out_path)
    print(f"\n  Profile saved → {out_path}")


# ── Scenario 2 — dirty injected data ─────────────────────────────────────────

def demo_dirty_data() -> None:
    section("SCENARIO 2 — Dirty data with injected quality issues")

    # Build a DataFrame that deliberately has all common quality problems
    dirty_df = pd.DataFrame({
        "order_id": [
            "ORD-001", "ORD-002", "ORD-002",   # ← duplicate
            "ORD-003", "ORD-004", "ORD-005",
            "ORD-006", "ORD-007", "ORD-008", "ORD-009",
        ],
        "date": [
            "2024-01-15", "2024-02-10", "2024-02-10",
            None,          "BAD-DATE",   "2024-04-01",   # ← nulls + bad type
            "2024-05-20",  "2024-06-30", "2024-07-14",   "2024-08-22",
        ],
        "product": [
            "Laptop", "Keyboard", "Keyboard",
            "Monitor", None, "Headphones",                # ← null
            "Laptop", "Monitor", "Keyboard", "Headphones",
        ],
        "region": [
            "North", None, None,                          # ← nulls (33%)
            "South", "West", None,
            "East",  "North", "South", "West",
        ],
        "sales_rep": [
            "Alice", "Bob", "Bob",
            None, "Carla", "Alice",                       # ← null
            "Bob", "Carla", "Alice", "Bob",
        ],
        "quantity": [
            2, "five", "five",                            # ← non-numeric
            1, 3, 4,
            2, 5, 1, 3,
        ],
        "unit_price": [
            1299.0, 89.99, 89.99,
            449.0, 129.0, 349.0,
            1299.0, 449.0, 89.99, 349.0,
        ],
        "revenue": [
            2598.0, 449.95, 449.95,
            None, 387.0, 1396.0,                          # ← null
            2598.0, 2245.0, 89.99,
            99999999.99,                                  # ← extreme outlier
        ],
        "status": [
            "confirmed", "confirmed", "confirmed",
            "confirmed", "confirmed", "confirmed",
            "confirmed", "confirmed", "confirmed",
            "confirmed",                                  # ← constant column
        ],
    })

    print(f"\n  Dirty DataFrame shape: {dirty_df.shape}")
    print(f"\n{dirty_df.to_string(index=False)}\n")

    profiler = DataProfiler()
    profile  = profiler.profile(dirty_df)
    profile.print_report()

    out_path = str(PROJECT_ROOT / "data" / "raw" / "data_profile_dirty.json")
    profile.to_json(out_path)
    print(f"\n  Profile saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nSales Insights Automator — Data Profiling Demo")
    print(DIVIDER)
    demo_clean_data()
    demo_dirty_data()
    print(f"\n{DIVIDER}")
    print("Profiling complete.")
    print("Next step: pass the raw DataFrame to DataCleaner with rules")
    print("informed by this profile.")
