"""
DataAnonymizer — masks sensitive fields in an AnalysisResult before the
data is passed to the AI prompt builder.

Design
------
The anonymizer works on a *copy* of the AnalysisResult — the original
object is never modified.  This means:

  - The Streamlit UI (Stage 5) can display real names and precise numbers
  - Only the text sent to the OpenAI API is anonymised
  - The mapping (real name → masked label) is stored so the AI's response
    can be de-anonymised before displaying to the user if needed

Anonymisation strategy
----------------------
Names (reps, products, regions) are replaced with stable, alphabetic
labels derived from the order they first appear:
    Alice Martin  → Sales Rep A
    Bob Chen      → Sales Rep B
    ...

This preserves relative comparisons ("Rep A outperformed Rep B") while
making individual identity impossible to infer.

Revenue rounding reduces the precision of financial figures:
    $956,745.51  → $957,000  (round_revenue_to=1000)

This makes it harder to reverse-engineer exact figures from the AI output.
"""

import copy
import re
import string
from typing import Dict, Optional

import pandas as pd

from analysis.insight_builder import AnalysisResult
from privacy.config import PrivacyConfig


class DataAnonymizer:
    """Applies privacy masks to an AnalysisResult before AI prompt generation.

    Parameters
    ----------
    config : PrivacyConfig
        Which masks to apply.

    Attributes
    ----------
    mappings : dict
        After calling ``anonymize()``, this holds the full substitution
        mapping: ``{"Alice Martin": "Sales Rep A", ...}``.
        Useful for de-anonymising the AI response if needed.

    Examples
    --------
    >>> config = PrivacyConfig.maximum()
    >>> anon = DataAnonymizer(config)
    >>> safe_result = anon.anonymize(result)
    >>> print(anon.mappings)
    {"Alice Martin": "Sales Rep A", "Laptop Pro 15": "Product A", ...}
    """

    def __init__(self, config: PrivacyConfig) -> None:
        self.config   = config
        self.mappings: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def anonymize(self, result: AnalysisResult) -> AnalysisResult:
        """Return a privacy-safe copy of the AnalysisResult.

        The returned object has the same structure as the original but with
        sensitive fields replaced according to ``self.config``.
        The original object is never mutated.

        Parameters
        ----------
        result : AnalysisResult

        Returns
        -------
        AnalysisResult
            A deep copy with masked fields.
        """
        self.mappings = {}
        safe = copy.deepcopy(result)

        if self.config.mask_rep_names:
            safe.revenue_by_sales_rep = self._mask_column(
                safe.revenue_by_sales_rep, "sales_rep", "Sales Rep"
            )

        if self.config.mask_product_names:
            safe.revenue_by_product = self._mask_column(
                safe.revenue_by_product, "product", "Product"
            )

        if self.config.mask_region_names:
            safe.revenue_by_region = self._mask_column(
                safe.revenue_by_region, "region", "Region"
            )

        if self.config.round_revenue_to > 0:
            safe = self._round_revenues(safe, self.config.round_revenue_to)

        if self.config.strip_exact_dates:
            safe = self._strip_dates(safe)

        return safe

    def anonymize_text(self, text: str) -> str:
        """Apply all accumulated mappings to a plain-text string.

        Call this *after* ``anonymize()`` so the mappings dict is populated.
        Used to sanitise any remaining free-text before it is sent to the API.

        Parameters
        ----------
        text : str
            Any text that may contain real names or values.

        Returns
        -------
        str
            Text with all known real values replaced by their masked labels.
        """
        result = text
        # Sort by length descending so "Alice Martin" is replaced before "Alice"
        for real, masked in sorted(self.mappings.items(), key=lambda x: -len(x[0])):
            result = result.replace(real, masked)
        return result

    @property
    def masked_fields(self) -> list:
        """List of field categories that were masked in the last run."""
        fields = []
        if self.config.mask_rep_names:     fields.append("sales_rep")
        if self.config.mask_product_names: fields.append("product")
        if self.config.mask_region_names:  fields.append("region")
        if self.config.round_revenue_to:   fields.append("revenue_rounded")
        if self.config.strip_exact_dates:  fields.append("dates")
        return fields

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _mask_column(
        self,
        df: pd.DataFrame,
        column: str,
        label_prefix: str,
    ) -> pd.DataFrame:
        """Replace all values in ``column`` with ``label_prefix A/B/C/...``."""
        if df is None or df.empty or column not in df.columns:
            return df

        df = df.copy()
        unique_values = df[column].dropna().unique().tolist()

        for idx, real_value in enumerate(unique_values):
            label = f"{label_prefix} {self._index_to_letter(idx)}"
            self.mappings[str(real_value)] = label

        df[column] = df[column].map(
            lambda v: self.mappings.get(str(v), str(v))
        )
        return df

    @staticmethod
    def _index_to_letter(idx: int) -> str:
        """Convert 0→A, 1→B, ..., 25→Z, 26→AA, 27→AB, ..."""
        letters = string.ascii_uppercase
        if idx < 26:
            return letters[idx]
        return letters[idx // 26 - 1] + letters[idx % 26]

    @staticmethod
    def _round_revenues(result: AnalysisResult, precision: int) -> AnalysisResult:
        """Round all revenue columns in every DataFrame to the nearest ``precision``."""
        revenue_col = "total_revenue"

        for attr in [
            "revenue_by_region", "revenue_by_product",
            "revenue_by_category", "revenue_by_sales_rep",
        ]:
            df = getattr(result, attr)
            if df is None or df.empty:
                continue
            if revenue_col in df.columns:
                df = df.copy()
                df[revenue_col] = (df[revenue_col] / precision).round() * precision
                setattr(result, attr, df)

        # Also round the summary stats
        stats = result.summary_stats.copy()
        for key in ["total_revenue", "average_order_value", "median_order_value",
                    "min_order_value", "max_order_value"]:
            if key in stats:
                stats[key] = round(stats[key] / precision) * precision
        result.summary_stats = stats

        return result

    @staticmethod
    def _strip_dates(result: AnalysisResult) -> AnalysisResult:
        """Replace the exact date range with a generic label."""
        result.date_range = {"from": "Period Start", "to": "Period End"}

        if result.monthly_trend is not None and not result.monthly_trend.empty and "month" in result.monthly_trend.columns:
            trend = result.monthly_trend.copy()
            trend["month"] = [
                f"Month {i+1}" for i in range(len(trend))
            ]
            result.monthly_trend = trend

        return result
