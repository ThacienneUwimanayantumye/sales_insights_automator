"""
AnalysisResult — the structured output of the analysis layer.

This dataclass acts as the contract between Stage 3 (analysis) and
Stage 4 (AI insight generation).  It holds every computed metric and
trend in one place and exposes:

  ``text_summary()``
      Formats all findings as a clean, structured text block.
      This is intentionally designed to be the prompt context passed
      to the OpenAI API in Stage 4 — compact enough to fit in a prompt,
      rich enough to support meaningful insights.

  ``to_dict()`` / ``to_json()``
      Serialise the result for storage, logging, or the Streamlit UI.

Design note
-----------
DataFrames are stored as-is inside the result for programmatic access.
The ``text_summary()`` method converts them to a compact text format for
the AI.  This keeps the two consumers (code and AI) cleanly separated.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class AnalysisResult:
    """All computed metrics and trends from a single analysis run.

    This object is produced by ``SalesAnalyzer.analyze()`` and consumed by:
      - The CLI output layer (pretty-prints ``text_summary()``)
      - The AI layer (passes ``text_summary()`` as prompt context)
      - The Streamlit UI (reads the DataFrame fields directly)

    Attributes
    ----------
    summary_stats : dict
        High-level KPIs (total revenue, AOV, units sold, etc.)
    revenue_by_region : pd.DataFrame
    revenue_by_product : pd.DataFrame
    revenue_by_category : pd.DataFrame
    revenue_by_sales_rep : pd.DataFrame
    discount_stats : dict
        Discount penetration and revenue impact.
    monthly_trend : pd.DataFrame
        Monthly revenue with MoM growth and rolling average.
    trend_summary : dict
        Best/worst months, overall growth rate, etc.
    best_months : pd.DataFrame
        Top-N revenue months.
    worst_months : pd.DataFrame
        Bottom-N revenue months.
    revenue_by_weekday : pd.DataFrame
    regional_trend : pd.DataFrame
        Wide-format monthly revenue split by region.
    date_range : dict
        ``{"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}``
    row_count : int
        Number of rows in the cleaned dataset that was analysed.
    generated_at : datetime
        Timestamp of the analysis run.
    """

    summary_stats:        Dict[str, Any]   = field(default_factory=dict)
    revenue_by_region:    pd.DataFrame     = field(default_factory=pd.DataFrame)
    revenue_by_product:   pd.DataFrame     = field(default_factory=pd.DataFrame)
    revenue_by_category:  pd.DataFrame     = field(default_factory=pd.DataFrame)
    revenue_by_sales_rep: pd.DataFrame     = field(default_factory=pd.DataFrame)
    discount_stats:       Dict[str, Any]   = field(default_factory=dict)
    monthly_trend:        pd.DataFrame     = field(default_factory=pd.DataFrame)
    trend_summary:        Dict[str, Any]   = field(default_factory=dict)
    best_months:          pd.DataFrame     = field(default_factory=pd.DataFrame)
    worst_months:         pd.DataFrame     = field(default_factory=pd.DataFrame)
    revenue_by_weekday:   pd.DataFrame     = field(default_factory=pd.DataFrame)
    regional_trend:       pd.DataFrame     = field(default_factory=pd.DataFrame)
    date_range:           Dict[str, str]   = field(default_factory=dict)
    row_count:            int              = 0
    generated_at:         datetime         = field(default_factory=datetime.now)

    # ------------------------------------------------------------------ #
    # AI-ready text summary                                               #
    # ------------------------------------------------------------------ #

    def text_summary(self, max_rows: int = 5) -> str:
        """Format all analysis findings as structured plain text.

        This output is designed to be dropped directly into an OpenAI
        prompt as context.  It is compact (avoids large tables), factual,
        and follows a consistent structure so the AI can reliably parse it.

        Parameters
        ----------
        max_rows : int
            Maximum rows to include for each DataFrame section.
            Keeps the prompt within token budget.

        Returns
        -------
        str
        """
        lines = [
            "=== SALES ANALYSIS REPORT ===",
            f"Generated : {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
            f"Dataset   : {self.row_count:,} orders  |  "
            f"Period: {self.date_range.get('from', 'N/A')} → {self.date_range.get('to', 'N/A')}",
            "",
        ]

        # ── Summary KPIs ──────────────────────────────────────────────
        lines.append("--- KEY PERFORMANCE INDICATORS ---")
        for k, v in self.summary_stats.items():
            label = k.replace("_", " ").title()
            if "revenue" in k or "value" in k:
                lines.append(f"  {label:<30}: ${v:,.2f}")
            elif "pct" in k or "rate" in k:
                lines.append(f"  {label:<30}: {v}%")
            else:
                lines.append(f"  {label:<30}: {v:,}" if isinstance(v, int) else f"  {label:<30}: {v}")
        lines.append("")

        # ── Revenue by region ─────────────────────────────────────────
        lines.append("--- REVENUE BY REGION ---")
        lines.append(self._df_to_text(self.revenue_by_region, max_rows))
        lines.append("")

        # ── Revenue by product ────────────────────────────────────────
        lines.append("--- TOP PRODUCTS BY REVENUE ---")
        lines.append(self._df_to_text(self.revenue_by_product, max_rows))
        lines.append("")

        # ── Revenue by category ───────────────────────────────────────
        lines.append("--- REVENUE BY CATEGORY ---")
        lines.append(self._df_to_text(self.revenue_by_category, max_rows))
        lines.append("")

        # ── Sales rep performance ─────────────────────────────────────
        lines.append("--- SALES REP PERFORMANCE ---")
        lines.append(self._df_to_text(self.revenue_by_sales_rep, max_rows))
        lines.append("")

        # ── Discount analysis ─────────────────────────────────────────
        lines.append("--- DISCOUNT ANALYSIS ---")
        for k, v in self.discount_stats.items():
            label = k.replace("_", " ").title()
            if "revenue" in k:
                lines.append(f"  {label:<35}: ${v:,.2f}")
            elif "pct" in k:
                lines.append(f"  {label:<35}: {v}%")
            elif "rate" in k:
                lines.append(f"  {label:<35}: {v:.1%}")
            else:
                lines.append(f"  {label:<35}: {v}")
        lines.append("")

        # ── Monthly trends ────────────────────────────────────────────
        lines.append("--- MONTHLY REVENUE TREND ---")
        lines.append(self._df_to_text(self.monthly_trend, max_rows))
        lines.append("")

        # ── Trend summary ─────────────────────────────────────────────
        lines.append("--- TREND SUMMARY ---")
        for k, v in self.trend_summary.items():
            label = k.replace("_", " ").title()
            if "revenue" in k:
                lines.append(f"  {label:<35}: ${v:,.2f}")
            elif "pct" in k or "growth" in k:
                lines.append(f"  {label:<35}: {v}%")
            else:
                lines.append(f"  {label:<35}: {v}")
        lines.append("")

        # ── Best / worst months ───────────────────────────────────────
        lines.append("--- BEST PERFORMING MONTHS ---")
        lines.append(self._df_to_text(self.best_months, max_rows))
        lines.append("")

        lines.append("--- WORST PERFORMING MONTHS ---")
        lines.append(self._df_to_text(self.worst_months, max_rows))
        lines.append("")

        # ── Day-of-week pattern ───────────────────────────────────────
        lines.append("--- REVENUE BY DAY OF WEEK ---")
        lines.append(self._df_to_text(self.revenue_by_weekday, 7))

        lines.append("")
        lines.append("=== END OF REPORT ===")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Convert the result to a JSON-serialisable dictionary.

        DataFrames are converted to lists-of-dicts (record orientation).
        """
        def df_to_records(df: pd.DataFrame) -> list:
            return df.where(pd.notnull(df), None).to_dict(orient="records")

        return {
            "generated_at":       self.generated_at.isoformat(),
            "date_range":         self.date_range,
            "row_count":          self.row_count,
            "summary_stats":      self.summary_stats,
            "discount_stats":     self.discount_stats,
            "trend_summary":      self.trend_summary,
            "revenue_by_region":  df_to_records(self.revenue_by_region),
            "revenue_by_product": df_to_records(self.revenue_by_product),
            "revenue_by_category":df_to_records(self.revenue_by_category),
            "revenue_by_sales_rep":df_to_records(self.revenue_by_sales_rep),
            "monthly_trend":      df_to_records(self.monthly_trend),
            "best_months":        df_to_records(self.best_months),
            "worst_months":       df_to_records(self.worst_months),
            "revenue_by_weekday": df_to_records(self.revenue_by_weekday),
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialise to a JSON string and optionally write to a file."""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json_str)
        return json_str

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _df_to_text(df: pd.DataFrame, max_rows: int) -> str:
        """Render a DataFrame as a compact, fixed-width text table."""
        if df.empty:
            return "  (no data)"
        display = df.head(max_rows)
        rows = display.to_string(index=False).split("\n")
        return "\n".join(f"  {r}" for r in rows)

    def __repr__(self) -> str:
        return (
            f"AnalysisResult("
            f"rows={self.row_count}, "
            f"period={self.date_range.get('from')}→{self.date_range.get('to')}, "
            f"generated={self.generated_at.strftime('%Y-%m-%d %H:%M')})"
        )
