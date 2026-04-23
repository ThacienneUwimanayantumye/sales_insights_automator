"""
Analysis layer demo script.

Shows the full Stage 1 → 2 → 3 pipeline in action:
  CSV ingestion → cleaning → analysis → structured report

Run from the project root:
    python scripts/demo_analysis.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.csv_source import CSVSource
from cleaning.cleaner import DataCleaner
from cleaning.config import CleaningConfig
from analysis.analyzer import SalesAnalyzer
from analysis import metrics as m
from analysis import trends as t
from config.settings import SAMPLE_CSV_PATH

DIVIDER = "─" * 64


def section(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


# ── Stage 1: Ingest ───────────────────────────────────────────────────────────

def ingest() -> object:
    section("STAGE 1 — Ingestion")
    source = CSVSource(SAMPLE_CSV_PATH)
    df = source.load_validated()
    print(f"Raw shape: {df.shape}")
    return df


# ── Stage 2: Clean ────────────────────────────────────────────────────────────

def clean(raw_df) -> object:
    section("STAGE 2 — Cleaning")
    config = CleaningConfig(
        normalize_columns=True,
        drop_duplicates=True,
        duplicate_subset=["order_id"],
        fill_missing={"region": "Unknown", "revenue": "median"},
        type_conversions={
            "date": "datetime",
            "quantity": "int",
            "revenue": "float",
            "unit_price": "float",
            "discount_pct": "float",
        },
    )
    cleaner = DataCleaner(config)
    clean_df = cleaner.clean(raw_df)
    print(cleaner.report.summary())
    return clean_df


# ── Stage 3: Analyse ──────────────────────────────────────────────────────────

def analyse(clean_df) -> object:
    section("STAGE 3 — Analysis")
    analyzer = SalesAnalyzer(top_n=5, rolling_window=3)
    result = analyzer.analyze(clean_df)
    return result


# ── Demo output sections ──────────────────────────────────────────────────────

def print_kpis(result) -> None:
    section("KEY PERFORMANCE INDICATORS")
    stats = result.summary_stats
    print(f"  Total Revenue        : ${stats['total_revenue']:>12,.2f}")
    print(f"  Total Orders         : {stats['total_orders']:>12,}")
    print(f"  Total Units Sold     : {stats['total_units_sold']:>12,}")
    print(f"  Average Order Value  : ${stats['average_order_value']:>12,.2f}")
    print(f"  Avg Discount         : {stats['average_discount_pct']:>11.1f}%")
    print(f"  Median Order Value   : ${stats['median_order_value']:>12,.2f}")


def print_revenue_breakdowns(result) -> None:
    section("REVENUE BY REGION")
    print(result.revenue_by_region.to_string(index=False))

    section("REVENUE BY CATEGORY")
    print(result.revenue_by_category.to_string(index=False))

    section("TOP 5 PRODUCTS")
    print(result.revenue_by_product.head(5).to_string(index=False))

    section("SALES REP PERFORMANCE")
    print(result.revenue_by_sales_rep.to_string(index=False))


def print_trends(result) -> None:
    section("MONTHLY REVENUE TREND")
    trend_cols = ["month", "total_revenue", "mom_growth_pct", "rolling_3m_avg_revenue"]
    available  = [c for c in trend_cols if c in result.monthly_trend.columns]
    print(result.monthly_trend[available].to_string(index=False))

    section("TREND SUMMARY")
    for k, v in result.trend_summary.items():
        label = k.replace("_", " ").title()
        val   = f"${v:,.2f}" if "revenue" in k else f"{v}%" if "pct" in k or "growth" in k else str(v)
        print(f"  {label:<35}: {val}")

    section("BEST PERFORMING MONTHS")
    print(result.best_months[["month", "total_revenue", "order_count"]].to_string(index=False))

    section("WORST PERFORMING MONTHS")
    print(result.worst_months[["month", "total_revenue", "order_count"]].to_string(index=False))

    section("REVENUE BY DAY OF WEEK")
    print(result.revenue_by_weekday.to_string(index=False))


def print_discount_analysis(result) -> None:
    section("DISCOUNT ANALYSIS")
    ds = result.discount_stats
    print(f"  Avg Discount         : {ds['avg_discount_pct']}%")
    print(f"  Max Discount Given   : {ds['max_discount_pct']}%")
    print(f"  Orders With Discount : {ds['orders_with_discount']:,}")
    print(f"  Discount Rate        : {ds['discount_rate']:.1%}")
    print(f"  Revenue Lost         : ${ds['revenue_lost_to_discount']:,.2f}")


def print_ai_prompt_preview(result) -> None:
    section("AI PROMPT CONTEXT PREVIEW  (first 1,200 chars)")
    summary = result.text_summary()
    print(summary[:1200])
    print(f"\n... ({len(summary):,} total chars — will be passed to OpenAI in Stage 4)")


def save_json_result(result) -> None:
    section("SERIALISATION — saving result to data/raw/analysis_result.json")
    output_path = str(PROJECT_ROOT / "data" / "raw" / "analysis_result.json")
    result.to_json(output_path)
    print(f"  Saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nSales Insights Automator — Full Pipeline Demo (Stages 1–3)")
    print("=" * 64)

    raw_df   = ingest()
    clean_df = clean(raw_df)
    result   = analyse(clean_df)

    print_kpis(result)
    print_revenue_breakdowns(result)
    print_trends(result)
    print_discount_analysis(result)
    print_ai_prompt_preview(result)
    save_json_result(result)

    print(f"\n{'=' * 64}")
    print("Pipeline complete.  Ready for Stage 4 — AI insight generation.")
