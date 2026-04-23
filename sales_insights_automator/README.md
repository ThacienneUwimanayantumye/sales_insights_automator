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
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Stages                          │
│                                                                 │
│  1. Ingestion   →  2. Cleaning  →  3. Analysis  →  4. AI       │
│  (this stage)                                        Insights   │
│                                                                 │
│  Output:  CLI (MVP)  │  Streamlit UI (later)                    │
└─────────────────────────────────────────────────────────────────┘
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

### 5. Run the test suite

```bash
pytest tests/ -v
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
