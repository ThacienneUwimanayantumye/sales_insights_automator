"""
SalesAnalyzer — the public interface of the analysis layer.

Orchestrates all metric and trend computations in a single ``analyze()``
call and returns a fully populated ``AnalysisResult``.

Pipeline position
-----------------
  Ingestion → Cleaning → **Analysis** → AI insights → Output

The SalesAnalyzer sits between the cleaning layer and the AI layer.
It expects a *cleaned* DataFrame (datetime date column, correct dtypes)
and produces an ``AnalysisResult`` that the AI layer will summarise.

Usage
-----
    from analysis.analyzer import SalesAnalyzer

    analyzer = SalesAnalyzer()
    result   = analyzer.analyze(clean_df)

    # For the CLI
    print(result.text_summary())

    # For the AI layer (Stage 4)
    prompt_context = result.text_summary()

    # For the Streamlit UI (Stage 5)
    st.dataframe(result.revenue_by_region)
"""

import pandas as pd

from analysis import metrics as m
from analysis import trends  as t
from analysis.insight_builder import AnalysisResult
from config.schema import SchemaConfig


class SalesAnalyzer:
    """Runs the full analysis pipeline on a cleaned sales DataFrame.

    Parameters
    ----------
    top_n : int
        Number of top performers to surface in each dimension.
        Defaults to 5.
    rolling_window : int
        Window size (months) for the rolling revenue average.
        Defaults to 3.
    best_worst_n : int
        Number of best/worst months to identify.
        Defaults to 3.
    schema : SchemaConfig, optional
        Maps your dataset's actual column names to the standard roles the
        analysis layer expects.  When provided, columns are transparently
        renamed at the start of ``analyze()`` — nothing else changes.

        If your dataset already uses the standard column names
        (order_id, date, revenue, …) you don't need to provide a schema.

        Obtain a schema using the interactive wizard::

            from profiling.schema_wizard import SchemaWizard
            schema = SchemaWizard().run(raw_df, save_path="config/schema.json")

        Or load a previously saved one::

            schema = SchemaConfig.from_json("config/schema.json")

    Attributes
    ----------
    result : AnalysisResult
        Available after calling ``analyze()``.

    Examples
    --------
    >>> analyzer = SalesAnalyzer()
    >>> result = analyzer.analyze(clean_df)
    >>> print(result.text_summary())
    """

    def __init__(
        self,
        top_n: int = 5,
        rolling_window: int = 3,
        best_worst_n: int = 3,
        schema: SchemaConfig | None = None,
    ) -> None:
        self.top_n          = top_n
        self.rolling_window = rolling_window
        self.best_worst_n   = best_worst_n
        self.schema         = schema
        self._result: AnalysisResult | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        """Run the complete analysis pipeline.

        Steps (in order):
          1. Validate input is non-empty
          2. Compute summary statistics
          3. Compute revenue breakdowns by region, product, category, rep
          4. Discount analysis
          5. Monthly trend + MoM growth + rolling average
          6. Trend summary (best/worst months, overall growth)
          7. Day-of-week revenue pattern
          8. Regional monthly trend (multi-series)
          9. Assemble into AnalysisResult

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned DataFrame from the cleaning layer.  Must have a
            datetime ``date`` column.

        Returns
        -------
        AnalysisResult
            Fully populated result object.

        Raises
        ------
        ValueError
            If the DataFrame is empty.
        """
        if df.empty:
            raise ValueError("Cannot analyze an empty DataFrame.")

        # ── Apply schema mapping ──────────────────────────────────────
        # Rename dataset-specific column names to the standard names the
        # analysis functions expect.  The original DataFrame is unchanged.
        if self.schema is not None:
            errors = self.schema.validate(df)
            if errors:
                raise ValueError(
                    "Schema validation failed:\n" +
                    "\n".join(f"  - {e}" for e in errors)
                )
            df = self.schema.rename_to_standard(df)
            print(f"[SalesAnalyzer] Schema mapping applied — "
                  f"{len(self.schema.mapped_roles())} roles mapped.")

        print(f"[SalesAnalyzer] Starting analysis on {len(df):,} rows...")

        # ── Date range metadata ───────────────────────────────────────
        date_col = df["date"] if "date" in df.columns else df.iloc[:, 0]
        if pd.api.types.is_datetime64_any_dtype(date_col):
            date_range = {
                "from": str(date_col.min().date()),
                "to":   str(date_col.max().date()),
            }
        else:
            date_range = {"from": "unknown", "to": "unknown"}

        has_col = lambda col: col in df.columns   # noqa: E731

        # ── Step 1: Summary KPIs ──────────────────────────────────────
        print("[SalesAnalyzer] Computing summary statistics...")
        summary_stats = m.compute_summary_stats(df)

        # ── Step 2: Revenue breakdowns (optional columns) ─────────────
        print("[SalesAnalyzer] Computing revenue breakdowns...")
        rev_by_region   = m.revenue_by_region(df)   if has_col(m.COL_REGION)    else None
        rev_by_product  = m.revenue_by_product(df)  if has_col(m.COL_PRODUCT)   else None
        rev_by_category = m.revenue_by_category(df) if has_col(m.COL_CATEGORY)  else None
        rev_by_rep      = m.sales_rep_performance(df) if has_col(m.COL_SALES_REP) else None

        # ── Step 3: Discount analysis (optional) ──────────────────────
        # discount_analysis() already returns None if discount_pct is absent
        print("[SalesAnalyzer] Computing discount analysis...")
        discount_stats = m.discount_analysis(df)

        # ── Step 4: Monthly trend ─────────────────────────────────────
        print("[SalesAnalyzer] Computing monthly trends...")
        monthly = t.monthly_revenue(df)
        monthly = t.compute_growth_rates(monthly)
        monthly = t.rolling_revenue(monthly, window=self.rolling_window)
        trend_summ = t.trend_summary(monthly)
        best_months, worst_months = t.best_and_worst_periods(monthly, n=self.best_worst_n)

        # ── Step 5: Day-of-week pattern ───────────────────────────────
        print("[SalesAnalyzer] Computing day-of-week pattern...")
        dow = t.revenue_by_day_of_week(df)

        # ── Step 6: Regional monthly trend (optional) ─────────────────
        print("[SalesAnalyzer] Computing regional monthly trend...")
        regional_trend = (
            t.monthly_revenue_by_region(df) if has_col(m.COL_REGION) else None
        )

        # ── Assemble ──────────────────────────────────────────────────
        result = AnalysisResult(
            summary_stats        = summary_stats,
            revenue_by_region    = rev_by_region,
            revenue_by_product   = rev_by_product,
            revenue_by_category  = rev_by_category,
            revenue_by_sales_rep = rev_by_rep,
            discount_stats       = discount_stats,
            monthly_trend        = monthly,
            trend_summary        = trend_summ,
            best_months          = best_months,
            worst_months         = worst_months,
            revenue_by_weekday   = dow,
            regional_trend       = regional_trend,
            date_range           = date_range,
            row_count            = len(df),
        )

        self._result = result
        print(
            f"[SalesAnalyzer] Analysis complete — "
            f"period: {date_range['from']} → {date_range['to']}, "
            f"total revenue: ${summary_stats['total_revenue']:,.2f}"
        )
        return result

    @property
    def result(self) -> AnalysisResult:
        """The result from the last ``analyze()`` call.

        Raises
        ------
        RuntimeError
            If ``analyze()`` has not been called yet.
        """
        if self._result is None:
            raise RuntimeError("No result available. Call analyze(df) first.")
        return self._result

    def __repr__(self) -> str:
        schema_str = repr(self.schema) if self.schema else "default column names"
        return (
            f"SalesAnalyzer("
            f"top_n={self.top_n}, "
            f"rolling_window={self.rolling_window}, "
            f"schema={schema_str})"
        )
