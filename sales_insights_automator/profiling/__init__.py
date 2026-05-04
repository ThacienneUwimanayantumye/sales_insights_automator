"""
Data profiling layer for the Sales Insights Automator.
Run immediately after ingestion to understand data quality before cleaning.
"""

from profiling.profiler import DataProfiler, DataProfile, ColumnProfile

__all__ = ["DataProfiler", "DataProfile", "ColumnProfile"]
