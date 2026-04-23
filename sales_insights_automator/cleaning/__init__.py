"""
Cleaning layer for the Sales Insights Automator.

Public interface — import everything you need from here.
"""

from cleaning.config import CleaningConfig
from cleaning.cleaner import DataCleaner
from cleaning.report import CleaningReport

__all__ = ["DataCleaner", "CleaningConfig", "CleaningReport"]
