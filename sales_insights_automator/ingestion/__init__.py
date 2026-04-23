"""
Ingestion layer for the Sales Insights Automator.

Exposes all data source connectors from a single import point.
"""

from ingestion.base import DataSource
from ingestion.csv_source import CSVSource
from ingestion.sqlite_source import SQLiteSource
from ingestion.kaggle_source import KaggleSource
from ingestion.google_drive_source import GoogleDriveSource

__all__ = [
    "DataSource",
    "CSVSource",
    "SQLiteSource",
    "KaggleSource",
    "GoogleDriveSource",
]
