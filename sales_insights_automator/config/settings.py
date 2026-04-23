"""
Central configuration for the Sales Insights Automator.

All tuneable values (paths, API keys, model names) live here so that
the rest of the codebase stays free of hard-coded strings.

Environment variables override the defaults when set — this makes the
project Docker-friendly and CI-friendly without any code changes.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
RAW_DIR = DATA_DIR / "raw"

SAMPLE_CSV_PATH = str(SAMPLES_DIR / "sample_sales.csv")
SAMPLE_SQLITE_PATH = str(SAMPLES_DIR / "sample_sales.db")

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))

# ── Kaggle ────────────────────────────────────────────────────────────────────
KAGGLE_DOWNLOAD_DIR: str = str(RAW_DIR / "kaggle")

# ── Google Drive ──────────────────────────────────────────────────────────────
GDRIVE_CREDENTIALS_PATH: str = os.getenv(
    "GDRIVE_CREDENTIALS_PATH", str(ROOT_DIR / "config" / "google_credentials.json")
)
GDRIVE_DOWNLOAD_DIR: str = str(RAW_DIR / "gdrive")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
