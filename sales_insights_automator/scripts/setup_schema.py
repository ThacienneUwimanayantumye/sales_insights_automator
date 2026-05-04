"""
Schema Setup Wizard — run this once when you connect a new dataset.

The wizard:
  1. Loads your data file (CSV or SQLite)
  2. Runs the DataProfiler so you can see exactly what each column looks like
  3. Auto-detects which column plays which semantic role (revenue, date, etc.)
  4. Walks you through each mapping so you can confirm or correct it
  5. Saves the result as a JSON file you can reuse in every subsequent run

Run from the project root
--------------------------
    # Interactive (recommended for new datasets):
    python scripts/setup_schema.py --data data/samples/sample_sales.csv

    # Save the config to a custom location:
    python scripts/setup_schema.py --data data/samples/sample_sales.csv \\
                                   --output config/my_schema.json

    # Auto-detect only, no prompts (for pipelines / CI):
    python scripts/setup_schema.py --data my_data.csv --auto

    # SQLite source:
    python scripts/setup_schema.py --data data/samples/sample_sales.db \\
                                   --table sales

After running
-------------
Use the saved config in your analysis pipeline:

    from config.schema import SchemaConfig
    from analysis.analyzer import SalesAnalyzer

    schema   = SchemaConfig.from_json("config/my_schema.json")
    analyzer = SalesAnalyzer(schema=schema)
    result   = analyzer.analyze(df)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.csv_source    import CSVSource
from ingestion.sqlite_source import SQLiteSource
from profiling.profiler      import DataProfiler
from profiling.schema_wizard import SchemaWizard


def load_data(data_path: str, table: str | None):
    """Load a DataFrame from the given path (CSV or SQLite)."""
    path = Path(data_path)
    if not path.exists():
        print(f"\n  Error: file not found: {data_path}")
        sys.exit(1)

    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        return CSVSource(str(path)).load()
    elif suffix in (".db", ".sqlite", ".sqlite3"):
        source = SQLiteSource(str(path), table_name=table) if table else SQLiteSource(str(path))
        return source.load()
    else:
        print(f"\n  Unsupported file type: {suffix}")
        print("  Supported: .csv, .tsv, .db, .sqlite")
        sys.exit(1)


def default_output_path(data_path: str) -> str:
    stem = Path(data_path).stem
    return str(PROJECT_ROOT / "config" / f"{stem}_schema.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive schema mapping wizard for the Sales Insights Automator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="PATH",
        help="Path to your data file (CSV or SQLite).",
    )
    parser.add_argument(
        "--table",
        default=None,
        metavar="TABLE",
        help="SQLite table name (required if --data is a .db file).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Where to save the schema JSON. "
             "Defaults to config/<filename>_schema.json.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect only, skip interactive prompts. "
             "Useful for automated pipelines.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=True,
        help="Show a data quality profile before mapping (default: on).",
    )
    parser.add_argument(
        "--no-profile",
        dest="profile",
        action="store_false",
        help="Skip the data profiling step.",
    )
    args = parser.parse_args()

    output_path = args.output or default_output_path(args.data)
    W = 65

    print(f"\n{'═'*W}")
    print("  Sales Insights Automator — Schema Setup")
    print(f"{'═'*W}")
    print(f"\n  Data file : {args.data}")
    print(f"  Output    : {output_path}")
    print(f"  Mode      : {'auto-detect (no prompts)' if args.auto else 'interactive'}")

    # ── Load data ─────────────────────────────────────────────────────
    print(f"\n  Loading data...")
    df = load_data(args.data, args.table)
    print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")

    # ── Profile (optional) ────────────────────────────────────────────
    if args.profile:
        print(f"\n{'─'*W}")
        print("  Running data profile — this helps you understand each column")
        print(f"{'─'*W}\n")
        profiler = DataProfiler()
        profile  = profiler.profile(df)

        # Print a compact version (just the column types table)
        print(f"\n  {'#':>3}  {'Column':<28}  {'Kind':<12}  "
              f"{'Nulls':>6}  {'Unique':>7}  Sample values")
        print(f"  {'─'*3}  {'─'*28}  {'─'*12}  "
              f"{'─'*6}  {'─'*7}  {'─'*20}")
        for i, cp in enumerate(profile.columns, 1):
            sample  = ", ".join(cp.sample_values[:2])
            if len(sample) > 25:
                sample = sample[:22] + "..."
            nulls   = f"{cp.null_count}" if cp.null_count else "✓"
            print(
                f"  {i:>3}  {cp.name:<28}  {cp.inferred_kind:<12}  "
                f"{nulls:>6}  {cp.unique_count:>7,}  {sample}"
            )

    # ── Run wizard ────────────────────────────────────────────────────
    wizard = SchemaWizard()
    schema = wizard.run(df, save_path=output_path, silent=args.auto)

    # ── Validate ──────────────────────────────────────────────────────
    errors = schema.validate(df)
    if errors:
        print(f"\n  ⚠  Validation warnings:")
        for e in errors:
            print(f"     - {e}")
        print("\n  You can re-run this wizard at any time to fix the mapping.")
    else:
        print("\n  ✓ Schema is valid — all required roles are mapped.")

    # ── Usage hint ────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  NEXT STEPS\n")
    print(f"  from config.schema import SchemaConfig")
    print(f"  from analysis.analyzer import SalesAnalyzer\n")
    print(f"  schema   = SchemaConfig.from_json('{output_path}')")
    print(f"  analyzer = SalesAnalyzer(schema=schema)")
    print(f"  result   = analyzer.analyze(your_cleaned_df)")
    print(f"\n{'═'*W}\n")


if __name__ == "__main__":
    main()
