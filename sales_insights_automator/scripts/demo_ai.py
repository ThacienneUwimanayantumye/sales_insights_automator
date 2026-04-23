"""
AI insight generation demo — full pipeline: Stages 1 → 2 → 3 → 4.

This script always works, regardless of whether you have an OpenAI API key:
  - With a key set in .env  → real API call, real insights
  - Without a key           → dry run shows prompt preview + placeholder

Run from the project root:
    python scripts/demo_ai.py                     # auto-detects key
    python scripts/demo_ai.py --dry-run           # force dry run
    python scripts/demo_ai.py --template exec     # executive summary
    python scripts/demo_ai.py --template recs     # recommendations
    python scripts/demo_ai.py --template anomalies
    python scripts/demo_ai.py --template full     # default
    python scripts/demo_ai.py --all-templates     # run all 4 templates
    python scripts/demo_ai.py --save              # save report to file
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before importing settings
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

from ingestion.csv_source import CSVSource
from cleaning.cleaner import DataCleaner
from cleaning.config import CleaningConfig
from analysis.analyzer import SalesAnalyzer
from ai.insight_generator import InsightGenerator
from ai.prompt_builder import PromptBuilder
from ai.llm_client import LLMClient
from config.settings import OPENAI_API_KEY, SAMPLE_CSV_PATH

DIVIDER = "─" * 64
TEMPLATE_ALIASES = {
    "full":      "full_report",
    "exec":      "executive_summary",
    "recs":      "recommendations",
    "anomalies": "anomalies",
}


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def run_pipeline():
    """Run stages 1–3 and return a clean AnalysisResult."""
    print("\n[Pipeline] Stage 1 — Ingesting data...")
    raw_df = CSVSource(SAMPLE_CSV_PATH).load_validated()

    print("[Pipeline] Stage 2 — Cleaning data...")
    config = CleaningConfig(
        normalize_columns=True,
        drop_duplicates=True,
        duplicate_subset=["order_id"],
        fill_missing={"region": "Unknown", "revenue": "median"},
        type_conversions={
            "date": "datetime", "quantity": "int",
            "revenue": "float", "unit_price": "float", "discount_pct": "float",
        },
    )
    clean_df = DataCleaner(config).clean(raw_df)

    print("[Pipeline] Stage 3 — Analysing data...")
    result = SalesAnalyzer().analyze(clean_df)

    print(
        f"[Pipeline] Ready — {result.row_count:,} orders, "
        f"period {result.date_range['from']} → {result.date_range['to']}, "
        f"total revenue ${result.summary_stats['total_revenue']:,.2f}\n"
    )
    return result


# ── Demo modes ────────────────────────────────────────────────────────────────

def demo_prompt_preview(result) -> None:
    """Show what the prompt looks like before any API call."""
    print(f"\n{DIVIDER}")
    print("  PROMPT PREVIEW")
    print(DIVIDER)

    builder = PromptBuilder()
    for name, label in [
        ("full_report",        "Full Report"),
        ("executive_summary",  "Executive Summary"),
        ("recommendations",    "Action Recommendations"),
        ("anomalies",          "Anomaly Analysis"),
    ]:
        payload = builder.build(result, template=name)
        print(
            f"  [{label}]  "
            f"~{payload.estimated_tokens:,} tokens  |  "
            f"{len(payload.user):,} user chars"
        )

    # Show the first 800 chars of the full_report user prompt
    payload = builder.build(result, template="full_report")
    print(f"\n  System prompt ({len(payload.system)} chars):\n")
    print("\n".join(f"    {line}" for line in payload.system.strip().splitlines()[:6]))
    print(f"\n  User prompt (first 800 chars):\n")
    print(f"    {payload.user[:800].replace(chr(10), chr(10) + '    ')}...")


def demo_single_template(result, template: str, dry_run: bool, save: bool) -> None:
    """Generate insights for a single template."""
    label = PromptBuilder.available_templates().get(template, template)

    print(f"\n{DIVIDER}")
    print(f"  STAGE 4 — AI INSIGHTS  [{label}]")
    print(DIVIDER)

    generator = InsightGenerator(dry_run=dry_run)
    report    = generator.generate(result, template=template)
    report.print_cli()

    if save:
        out_path = str(PROJECT_ROOT / "data" / "raw" / f"insights_{template}.md")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(report.to_markdown())
        json_path = str(PROJECT_ROOT / "data" / "raw" / f"insights_{template}.json")
        report.to_json(json_path)
        print(f"\n  Saved → {out_path}")
        print(f"  Saved → {json_path}")


def demo_all_templates(result, dry_run: bool, save: bool) -> None:
    """Generate insights for all 4 templates back-to-back."""
    for template in PromptBuilder.available_templates():
        demo_single_template(result, template, dry_run=dry_run, save=save)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sales Insights Automator — AI demo (Stages 1–4)"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Skip the OpenAI API call and show a prompt preview instead",
    )
    p.add_argument(
        "--template", default="full",
        choices=list(TEMPLATE_ALIASES.keys()),
        help="Prompt template to use (default: full)",
    )
    p.add_argument(
        "--all-templates", action="store_true",
        help="Run all 4 prompt templates in sequence",
    )
    p.add_argument(
        "--save", action="store_true",
        help="Save the generated report to data/raw/",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Auto-detect dry-run if no API key is present
    has_key = bool(OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."))
    dry_run = args.dry_run or not has_key

    if not has_key and not args.dry_run:
        print(
            "\n[demo_ai] No OPENAI_API_KEY found in environment.\n"
            "  → Running in dry-run mode. Set OPENAI_API_KEY in .env for real insights.\n"
        )

    print("Sales Insights Automator — AI Insight Generation Demo (Stages 1–4)")
    print("=" * 64)

    # Stages 1–3
    result = run_pipeline()

    # Prompt preview (always shown)
    demo_prompt_preview(result)

    # Stage 4 — AI generation
    template = TEMPLATE_ALIASES[args.template]

    if args.all_templates:
        demo_all_templates(result, dry_run=dry_run, save=args.save)
    else:
        demo_single_template(result, template, dry_run=dry_run, save=args.save)

    print(f"\n{'=' * 64}")
    if dry_run:
        print("Dry-run complete. Add OPENAI_API_KEY to .env for real AI insights.")
    else:
        print("Pipeline complete — Stages 1 → 2 → 3 → 4 done.")
