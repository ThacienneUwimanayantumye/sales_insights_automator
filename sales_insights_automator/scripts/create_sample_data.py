"""
One-time script to generate sample CSV and SQLite data for local development.

Run from the project root:
    python scripts/create_sample_data.py
"""

import os
import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Allow running from any directory by adding project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SAMPLE_CSV_PATH, SAMPLE_SQLITE_PATH

# ── Seed for reproducibility ─────────────────────────────────────────────────
random.seed(42)

# ── Domain data ───────────────────────────────────────────────────────────────
PRODUCTS = [
    "Laptop Pro 15",
    "Wireless Keyboard",
    "USB-C Hub",
    "Monitor 27in",
    "Ergonomic Chair",
    "Standing Desk",
    "Webcam HD",
    "Noise-Cancelling Headphones",
    "SSD 1TB",
    "Mechanical Keyboard",
]

REGIONS = ["North", "South", "East", "West", "Central"]

SALES_REPS = [
    "Alice Martin",
    "Bob Chen",
    "Carla Diaz",
    "David Kim",
    "Eva Rossi",
    "Frank Okafor",
]

CATEGORIES = {
    "Laptop Pro 15": "Computers",
    "Wireless Keyboard": "Peripherals",
    "USB-C Hub": "Accessories",
    "Monitor 27in": "Displays",
    "Ergonomic Chair": "Furniture",
    "Standing Desk": "Furniture",
    "Webcam HD": "Peripherals",
    "Noise-Cancelling Headphones": "Audio",
    "SSD 1TB": "Storage",
    "Mechanical Keyboard": "Peripherals",
}

UNIT_PRICES = {
    "Laptop Pro 15": 1299.00,
    "Wireless Keyboard": 89.99,
    "USB-C Hub": 49.99,
    "Monitor 27in": 449.00,
    "Ergonomic Chair": 599.00,
    "Standing Desk": 799.00,
    "Webcam HD": 129.00,
    "Noise-Cancelling Headphones": 349.00,
    "SSD 1TB": 109.99,
    "Mechanical Keyboard": 159.00,
}


def generate_sales_records(n: int = 500) -> pd.DataFrame:
    """Generate ``n`` synthetic sales transactions."""
    start_date = date(2024, 1, 1)
    records = []

    for i in range(1, n + 1):
        product = random.choice(PRODUCTS)
        quantity = random.randint(1, 10)
        unit_price = UNIT_PRICES[product]
        # Apply a small random discount (0–15 %)
        discount_pct = round(random.uniform(0, 0.15), 2)
        revenue = round(unit_price * quantity * (1 - discount_pct), 2)
        sale_date = start_date + timedelta(days=random.randint(0, 364))

        records.append(
            {
                "order_id": f"ORD-{i:05d}",
                "date": sale_date.isoformat(),
                "product": product,
                "category": CATEGORIES[product],
                "region": random.choice(REGIONS),
                "sales_rep": random.choice(SALES_REPS),
                "quantity": quantity,
                "unit_price": unit_price,
                "discount_pct": discount_pct,
                "revenue": revenue,
            }
        )

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[create_sample_data] CSV saved → {path}  ({len(df):,} rows)")


def save_sqlite(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)

    # Main transactions table
    df.to_sql("sales", conn, if_exists="replace", index=False)

    # Pre-aggregated monthly summary view — useful for quick dashboard queries
    conn.execute(
        """
        CREATE VIEW IF NOT EXISTS monthly_revenue AS
        SELECT
            strftime('%Y-%m', date)  AS month,
            region,
            category,
            SUM(revenue)             AS total_revenue,
            SUM(quantity)            AS total_units,
            COUNT(order_id)          AS order_count
        FROM sales
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3;
        """
    )
    conn.commit()
    conn.close()
    print(f"[create_sample_data] SQLite saved → {path}  ({len(df):,} rows, view: monthly_revenue)")


if __name__ == "__main__":
    df = generate_sales_records(n=500)
    save_csv(df, SAMPLE_CSV_PATH)
    save_sqlite(df, SAMPLE_SQLITE_PATH)
    print("\nSample data ready. Run demo_ingestion.py to test the connectors.")
