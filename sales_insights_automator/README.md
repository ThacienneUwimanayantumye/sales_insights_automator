# Sales Insights Automator

An AI-powered internal tool that ingests sales data from multiple sources,
cleans it, runs descriptive analytics, and generates natural-language business
insights using the OpenAI API.

> **Portfolio project** — designed to demonstrate production-level thinking for
> data analyst / data engineer roles: clean architecture, modularity, and
> explainability over cleverness.

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│                             Pipeline Stages                               │
│                                                                           │
│  1. Ingestion → 1b. Profiling → 2. Cleaning → 3. Analysis → 4. AI       │
│                 (data quality          ↑ informed by profile              │
│                  report)                                                  │
│                                                                           │
│  Output:  CLI (demo scripts)  │  Streamlit UI (app/)                     │
└───────────────────────────────────────────────────────────────────────────┘
```

Each stage is a self-contained Python module.  No stage knows about the
internals of another — they communicate only through DataFrames.

---

## Project Structure

```
sales_insights_automator/
├── config/
│   ├── settings.py          # Centralised config + env vars
│   └── google_credentials.json  # (gitignored) Drive service account
│
├── ingestion/               # Stage 1 — pluggable data source connectors
│   ├── base.py              # DataSource abstract base class
│   ├── csv_source.py        # CSVSource
│   ├── sqlite_source.py     # SQLiteSource
│   ├── kaggle_source.py     # KaggleSource
│   └── google_drive_source.py  # GoogleDriveSource (stub)
│
├── profiling/               # Stage 1b — data quality report before cleaning
│   └── profiler.py          # DataProfiler → DataProfile (+ ColumnProfile)
│
├── data/
│   ├── samples/             # Generated sample data (CSV + SQLite)
│   └── raw/                 # Downloaded / ingested files land here
│
├── scripts/
│   ├── create_sample_data.py   # Generates sample_sales.csv + .db
│   └── demo_ingestion.py       # End-to-end ingestion demo
│
├── tests/
│   └── test_ingestion.py    # pytest unit tests for the ingestion layer
│
├── .env.example             # Template for environment variables
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (needed for Stage 4)
```

### 3. Generate sample data

```bash
python scripts/create_sample_data.py
```

### 4. Run the ingestion demo

```bash
python scripts/demo_ingestion.py
```

### 5. Launch the Streamlit UI

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. Run the test suite

```bash
pytest tests/ -v
```

---

## Streamlit UI

The web interface provides a complete guided workflow across 4 pages:

| Page | What it does |
|---|---|
| **📂 Upload & Profile** | Upload a CSV, see a full data quality report with quality score, null analysis, outlier flags, and cleaning recommendations |
| **🔧 Schema Setup** | Auto-detects which column is revenue, date, region, etc. Confirm with dropdowns — no JSON files needed |
| **📈 Dashboard** | Interactive Plotly charts: monthly trend, revenue by region/product/category/rep, discount gauge, day-of-week pattern, regional multi-line trend |
| **🤖 AI Insights** | Generate AI-written business reports with privacy controls. Choose from Full Report, Executive Summary, Recommendations, or Anomaly Detection |

```bash
streamlit run app/main.py
```

---

## Schema Wizard — Bring Your Own Dataset

Every company names columns differently.  One dataset calls it `"revenue"`,
another `"total_sales"`, a third `"amt"`.  The Schema Wizard solves this
**without requiring any manual JSON editing**.

### How it works

```
1. Load your file         → detect columns automatically
2. Auto-detect roles      → score every column against 10 semantic roles
3. Show you the mapping   → one column at a time, with samples
4. You confirm or correct → just press Enter for obvious matches
5. Save the result        → reusable JSON file for all future runs
```

### Run the wizard

```bash
# Interactive (recommended for any new dataset):
python scripts/setup_schema.py --data path/to/your_file.csv

# Auto-detect only, no prompts (for automated pipelines):
python scripts/setup_schema.py --data path/to/your_file.csv --auto

# SQLite source:
python scripts/setup_schema.py --data sales.db --table transactions
```

### What you see

```
  #   Column                        Type          Nulls   Unique  Sample values
  ─── ──────────────────────────── ────────────  ─────── ─────── ─────────────
    1  transaction_id               str                ✓     500  TXN-0001, ...
    2  sale_date                    str                ✓     265  2024-01-01, ...
    3  total_sales                  float64            ✓     428  950.0, 900.0, ...
    ...

  Role  1/10: order_id  [REQUIRED]
  Unique identifier per transaction
  Suggested: 'transaction_id'  (str)  samples: TXN-0001, TXN-0002

  [Enter]=accept  [s]=skip  [?]=show all columns  [1–11]=pick manually
  >                             ← just press Enter
```

### Use the saved schema in analysis

```python
from config.schema import SchemaConfig
from analysis.analyzer import SalesAnalyzer

schema   = SchemaConfig.from_json("config/your_file_schema.json")
analyzer = SalesAnalyzer(schema=schema)
result   = analyzer.analyze(your_df)
```

### Supported semantic roles

| Role | Required | Meaning |
|---|---|---|
| `order_id` | ✅ | Unique transaction identifier |
| `date` | ✅ | Transaction date |
| `revenue` | ✅ | Monetary value per transaction |
| `product` | optional | Product name or SKU |
| `category` | optional | Product group or department |
| `region` | optional | Sales territory or area |
| `sales_rep` | optional | Salesperson name or ID |
| `quantity` | optional | Units sold per transaction |
| `unit_price` | optional | Price per unit |
| `discount` | optional | Discount applied (0–1 fraction) |

Missing optional roles are gracefully skipped in analysis.

---

## Profiling Layer — Data Quality Report

Run `DataProfiler` immediately after loading raw data and **before** cleaning.
It answers the questions every data analyst asks on first contact with a dataset.

```python
from ingestion.csv_source import CSVSource
from profiling.profiler import DataProfiler

raw_df  = CSVSource("data/samples/sample_sales.csv").load()
profile = DataProfiler().profile(raw_df)
profile.print_report()          # rich terminal output
profile.to_json("profile.json") # save for later
```

### What the report covers

| Section | Detail |
|---|---|
| **Overview** | Row/column count, memory usage, column kinds, data quality score (0–100) |
| **Duplicates** | Exact duplicate row count and percentage |
| **Missing values** | Per-column null count and %, total null density, severity classification |
| **Column types** | pandas dtype + inferred kind (numeric / categorical / datetime / boolean) |
| **Cardinality** | Unique-value % — flags likely ID columns (≥95%) and constant columns |
| **Numeric stats** | Min, max, mean, median, std, Q1/Q3/IQR, skewness, zero count |
| **Outlier detection** | Tukey IQR fence — count and % of rows outside [Q1−1.5×IQR, Q3+1.5×IQR] |
| **Categorical stats** | Top-5 value counts with mini bar charts |
| **Quality flags** | Constant columns, high-null columns (>20%), outlier columns, likely IDs |
| **Recommendations** | Prioritised, actionable cleaning steps derived automatically from the flags |

### Quick-start

```bash
python scripts/demo_profiler.py   # runs on clean + deliberately dirty data
```

---

## Ingestion Layer — Connectors

| Connector | Status | Notes |
|---|---|---|
| `CSVSource` | ✅ Production-ready | Supports single file or glob pattern |
| `SQLiteSource` | ✅ Production-ready | Table name or raw SQL query |
| `KaggleSource` | ✅ Functional | Requires `~/.kaggle/kaggle.json` |
| `GoogleDriveSource` | 🚧 Stub | Interface defined, implementation pending |

### Adding a new connector

1. Create `ingestion/my_source.py`
2. Inherit from `DataSource`
3. Implement `load()` and `validate()`
4. Export from `ingestion/__init__.py`

That's it — nothing else in the pipeline needs to change.

---

## Cleaning Layer

The cleaning layer is config-driven — all rules live in `config/default_cleaning.json`.
Non-engineers can change cleaning behaviour by editing that file, no Python required.

| Module | Purpose |
|---|---|
| `cleaning/config.py` | `CleaningConfig` dataclass — load rules from code, dict, JSON, or YAML |
| `cleaning/functions.py` | Pure, single-responsibility functions (normalise, dedup, fill, convert, drop) |
| `cleaning/report.py` | `CleaningReport` — full audit trail: rows removed, fills applied, type changes |
| `cleaning/cleaner.py` | `DataCleaner` — orchestrates all steps, exposes `.report` after each run |

### Supported null-fill strategies

| Strategy | Behaviour |
|---|---|
| `"mean"` | Fill with column mean (numeric only) |
| `"median"` | Fill with column median (numeric only) |
| `"mode"` | Fill with most frequent value |
| `"drop"` | Drop rows where this column is null |
| any other value | Use as a literal constant (e.g. `"Unknown"`, `0`) |

### Supported type conversions

`"datetime"`, `"numeric"`, `"int"` (nullable), `"float"`, `"str"`

---

## Analysis Layer

The analysis layer sits between the cleaning layer and the AI layer.  It receives a cleaned DataFrame and produces a fully populated `AnalysisResult` — a single object that holds every computed metric and is the direct input to the OpenAI prompt in Stage 4.

| Module | Purpose |
|---|---|
| `analysis/metrics.py` | Pure KPI functions: summary stats, revenue by dimension, discount analysis, cross-tab |
| `analysis/trends.py` | Time-series: monthly revenue, MoM growth, rolling average, best/worst periods, day-of-week |
| `analysis/insight_builder.py` | `AnalysisResult` dataclass — holds all data, exposes `text_summary()` for the AI prompt |
| `analysis/analyzer.py` | `SalesAnalyzer` — orchestrates metrics + trends, returns `AnalysisResult` |

### Key design: `text_summary()` as the Stage 4 bridge

`AnalysisResult.text_summary()` produces a compact, structured plain-text block of all findings. This is passed verbatim as context to the OpenAI API in Stage 4 — keeping the AI layer completely decoupled from pandas DataFrames.

---

## AI Insight Generation Layer

The AI layer takes an `AnalysisResult` from Stage 3 and produces human-readable business insights via the OpenAI Chat API.

| Module | Purpose |
|---|---|
| `ai/prompt_builder.py` | `PromptBuilder` — constructs system + user prompts from `AnalysisResult`. Supports 4 templates. |
| `ai/llm_client.py` | `LLMClient` — thin OpenAI wrapper with retry, token tracking, and lazy auth |
| `ai/insight_generator.py` | `InsightGenerator` — orchestrates builder + client, returns `InsightReport` |

### Prompt templates

| Template | Use case |
|---|---|
| `full_report` | Comprehensive analysis (default) |
| `executive_summary` | 3–4 paragraph narrative for C-suite, no bullet points |
| `recommendations` | 5 specific, prioritised action items with expected impact |
| `anomalies` | Focus on unusual patterns, outliers, and investigative leads |

### Dry-run mode

No API key? No problem. The demo auto-detects and falls back gracefully:

```bash
python scripts/demo_ai.py              # auto-detects key; dry-runs if missing
python scripts/demo_ai.py --dry-run    # force dry run
python scripts/demo_ai.py --template recs --save  # save report to data/raw/
```

### InsightReport output

`InsightReport` exposes three formats:
- `.print_cli()` — formatted terminal output
- `.to_markdown()` — Markdown document (ready for Streamlit)
- `.to_json(path)` — JSON for storage / audit trail

---

## Privacy Layer

The privacy layer sits between the analysis layer and the AI layer. It ensures **no sensitive data ever leaves the machine unprotected**.

| Module | Purpose |
|---|---|
| `privacy/config.py` | `PrivacyConfig` — declarative rules, loadable from `config/privacy_config.json` |
| `privacy/anonymizer.py` | `DataAnonymizer` — masks names, rounds revenues, strips dates before AI prompt |
| `privacy/audit_log.py` | `AuditLogger` — JSONL audit trail of every API call (metadata only, never content) |

### What happens before each API call

```
AnalysisResult (real data, stays local)
       ↓  DataAnonymizer
Safe copy (rep names → "Sales Rep A", revenues rounded to nearest $1,000)
       ↓  PromptBuilder (adds CONFIDENTIALITY NOTICE to system prompt)
OpenAI API  ← only ever sees anonymised data
       ↓  AuditLogger
data/audit/api_calls.jsonl  (metadata only — never prompt content)
```

### Default settings (`config/privacy_config.json`)

| Rule | Default | Effect |
|---|---|---|
| `mask_rep_names` | `true` | "Alice Martin" → "Sales Rep A" |
| `mask_product_names` | `false` | Product names sent as-is |
| `mask_region_names` | `false` | Region names sent as-is |
| `round_revenue_to` | `1000` | $956,745 → $957,000 |
| `request_no_training` | `true` | Adds confidentiality notice to system prompt |
| `strip_exact_dates` | `false` | Exact period is included |
| `enable_audit_log` | `true` | Every call logged to `data/audit/api_calls.jsonl` |

---

## Roadmap

- [x] **Stage 1** — Ingestion layer (CSV, SQLite, Kaggle, Drive stub)
- [x] **Stage 2** — Data cleaning layer (null handling, type coercion, deduplication, config-driven rules)
- [x] **Stage 3** — Analysis layer (KPIs, trends, MoM growth, discount analysis, AnalysisResult)
- [x] **Stage 4** — AI insight generation (OpenAI API, 4 prompt templates, dry-run mode)
- [ ] **Stage 5** — Streamlit dashboard UI

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Data processing | pandas, numpy |
| AI insights | OpenAI API (`gpt-4o-mini`) |
| UI | Streamlit |
| Local DB | SQLite (stdlib) |
| External data | Kaggle API |
| Testing | pytest |
