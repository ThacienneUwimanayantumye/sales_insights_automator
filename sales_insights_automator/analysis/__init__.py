"""
Analysis layer for the Sales Insights Automator.

Public interface — import the two classes you need from here.
"""

from analysis.analyzer import SalesAnalyzer
from analysis.insight_builder import AnalysisResult

__all__ = ["SalesAnalyzer", "AnalysisResult"]
