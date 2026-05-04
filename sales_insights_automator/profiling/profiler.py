"""
DataProfiler — produces a detailed quality report on a raw DataFrame.

Run this immediately after ingestion, before cleaning.  It answers the
questions a data analyst asks on first contact with a new dataset:

  - How many rows and columns?
  - What type is each variable?
  - Which columns have missing values, and how many?
  - Are there duplicate records?
  - What does the distribution of each numeric column look like?
  - Are there outliers?
  - What are the most frequent values in categorical columns?
  - Are any columns constant (zero variance)?
  - Are any columns nearly unique (high cardinality)?

The result is a DataProfile object that can be printed to the terminal,
serialised to JSON, or passed to the AI layer for automated commentary.

Pipeline position
-----------------
  Ingestion → **Profiling** → Cleaning → Analysis → AI → Output

Usage
-----
    profiler = DataProfiler()
    profile  = profiler.profile(raw_df)
    profile.print_report()
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


# ── Column-level profile ──────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    """Quality and statistical summary for a single column.

    Attributes
    ----------
    name : str
    dtype : str
        Pandas dtype string (e.g. ``"object"``, ``"int64"``, ``"float64"``,
        ``"datetime64[ns]"``).
    inferred_kind : str
        High-level kind: ``"numeric"``, ``"categorical"``, ``"datetime"``,
        ``"boolean"``, ``"unknown"``.
    null_count : int
    null_pct : float
        Percentage of rows that are null (0–100).
    unique_count : int
        Number of distinct non-null values.
    cardinality_pct : float
        ``unique_count / total_rows * 100``.  High values (>90%) suggest
        an ID or free-text field.
    is_constant : bool
        True if all non-null values are identical.
    is_likely_id : bool
        True if cardinality is ≥ 95% — likely a primary-key style column.
    sample_values : list
        Up to 5 representative non-null values.

    Numeric-only (None for non-numeric columns)
    -------------------------------------------
    min, max, mean, median, std : float or None
    q1, q3 : float or None
        25th and 75th percentiles.
    iqr : float or None
        Interquartile range (Q3 − Q1).
    outlier_count : int or None
        Rows outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR] (Tukey fences).
    outlier_pct : float or None
    skewness : float or None
        Measure of distribution asymmetry.  |skew| > 1 = highly skewed.
    zero_count : int or None
        Number of zero values.

    Categorical-only (None for numeric/datetime)
    --------------------------------------------
    top_value : str or None
        Most frequent value.
    top_value_count : int or None
    top_value_pct : float or None
        Frequency of the most common value as a percentage.
    value_counts : dict or None
        Top-5 value → count mapping.
    """

    # ── Core ──────────────────────────────────────────────────────────
    name:            str
    dtype:           str
    inferred_kind:   str
    null_count:      int
    null_pct:        float
    unique_count:    int
    cardinality_pct: float
    is_constant:     bool
    is_likely_id:    bool
    sample_values:   List[Any]

    # ── Numeric ───────────────────────────────────────────────────────
    min:           Optional[float] = None
    max:           Optional[float] = None
    mean:          Optional[float] = None
    median:        Optional[float] = None
    std:           Optional[float] = None
    q1:            Optional[float] = None
    q3:            Optional[float] = None
    iqr:           Optional[float] = None
    outlier_count: Optional[int]   = None
    outlier_pct:   Optional[float] = None
    skewness:      Optional[float] = None
    zero_count:    Optional[int]   = None

    # ── Categorical ───────────────────────────────────────────────────
    top_value:       Optional[str]   = None
    top_value_count: Optional[int]   = None
    top_value_pct:   Optional[float] = None
    value_counts:    Optional[dict]  = None

    @property
    def has_nulls(self) -> bool:
        return self.null_count > 0

    @property
    def has_outliers(self) -> bool:
        return bool(self.outlier_count and self.outlier_count > 0)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ── Dataset-level profile ─────────────────────────────────────────────────────

@dataclass
class DataProfile:
    """Complete quality report for a DataFrame.

    This is the object returned by ``DataProfiler.profile()``.
    Call ``print_report()`` to display it, or ``to_json()`` to save it.
    """

    # ── Shape ─────────────────────────────────────────────────────────
    total_rows:     int
    total_columns:  int
    memory_usage_mb: float

    # ── Duplicates ────────────────────────────────────────────────────
    duplicate_rows:  int
    duplicate_pct:   float

    # ── Nulls ─────────────────────────────────────────────────────────
    total_null_cells:  int
    null_density_pct:  float   # (total nulls) / (rows × cols) × 100
    columns_with_nulls: List[str]

    # ── Column profiles ───────────────────────────────────────────────
    columns: List[ColumnProfile]

    # ── Data quality flags ────────────────────────────────────────────
    constant_columns:        List[str]   # zero variance
    likely_id_columns:       List[str]   # cardinality ≥ 95%
    high_null_columns:       List[str]   # > 20% nulls
    outlier_columns:         List[str]   # numeric cols with outliers
    high_cardinality_columns:List[str]   # cardinality 50–95% (watchlist)

    # ── Metadata ──────────────────────────────────────────────────────
    profiled_at: datetime = field(default_factory=datetime.now)

    # ------------------------------------------------------------------ #
    # Convenience lookups                                                 #
    # ------------------------------------------------------------------ #

    def get_column(self, name: str) -> Optional[ColumnProfile]:
        """Return the ColumnProfile for a given column name."""
        return next((c for c in self.columns if c.name == name), None)

    @property
    def numeric_columns(self) -> List[ColumnProfile]:
        return [c for c in self.columns if c.inferred_kind == "numeric"]

    @property
    def categorical_columns(self) -> List[ColumnProfile]:
        return [c for c in self.columns if c.inferred_kind == "categorical"]

    @property
    def datetime_columns(self) -> List[ColumnProfile]:
        return [c for c in self.columns if c.inferred_kind == "datetime"]

    @property
    def quality_score(self) -> float:
        """Simple 0–100 data quality score.

        Penalises: nulls, duplicates, constant columns, high-null columns.
        A score above 80 is generally considered acceptable.
        """
        score = 100.0
        score -= min(self.null_density_pct * 2, 30)        # up to -30 for nulls
        score -= min(self.duplicate_pct * 2, 20)           # up to -20 for dupes
        score -= len(self.constant_columns) * 5            # -5 per constant col
        score -= len(self.high_null_columns) * 5           # -5 per high-null col
        return round(max(score, 0.0), 1)

    # ------------------------------------------------------------------ #
    # Terminal report                                                     #
    # ------------------------------------------------------------------ #

    def print_report(self) -> None:
        """Print the full profiling report to the terminal."""
        print(self._render())

    def _render(self) -> str:
        lines = []
        W = 65

        def rule(char="─"): lines.append(char * W)
        def header(t):
            rule("═")
            lines.append(f"  {t}")
            rule("═")
        def section(t):
            rule()
            lines.append(f"  {t}")
            rule()
        def row(label, value, indent=2):
            pad = " " * indent
            lines.append(f"{pad}{label:<35}: {value}")

        header("DATA PROFILE REPORT")
        lines.append(f"  Generated : {self.profiled_at.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # ── Overview ─────────────────────────────────────────────────
        section("1. DATASET OVERVIEW")
        row("Total rows",             f"{self.total_rows:,}")
        row("Total columns",          f"{self.total_columns}")
        row("Memory usage",           f"{self.memory_usage_mb:.2f} MB")
        row("Data quality score",     f"{self.quality_score}/100")
        lines.append("")
        row("Numeric columns",        f"{len(self.numeric_columns)}")
        row("Categorical columns",    f"{len(self.categorical_columns)}")
        row("Datetime columns",       f"{len(self.datetime_columns)}")

        # ── Duplicates ────────────────────────────────────────────────
        section("2. DUPLICATE RECORDS")
        if self.duplicate_rows == 0:
            lines.append("  ✓ No duplicate rows found.")
        else:
            row("Duplicate rows",     f"{self.duplicate_rows:,}  ({self.duplicate_pct:.1f}% of data)", 2)
            lines.append("  ⚠  Recommendation: Review and remove duplicates before analysis.")

        # ── Missing values ────────────────────────────────────────────
        section("3. MISSING VALUES")
        if not self.columns_with_nulls:
            lines.append("  ✓ No missing values found in any column.")
        else:
            row("Total null cells",   f"{self.total_null_cells:,}")
            row("Null density",       f"{self.null_density_pct:.2f}% of all cells")
            row("Columns affected",   f"{len(self.columns_with_nulls)}/{self.total_columns}")
            lines.append("")
            lines.append(f"  {'Column':<25} {'Nulls':>7} {'%':>7}  {'Severity'}")
            lines.append(f"  {'─'*25} {'─'*7} {'─'*7}  {'─'*10}")
            for col_name in self.columns_with_nulls:
                cp = self.get_column(col_name)
                if cp:
                    severity = "HIGH ⚠" if cp.null_pct > 20 else ("MED" if cp.null_pct > 5 else "LOW")
                    lines.append(f"  {cp.name:<25} {cp.null_count:>7,} {cp.null_pct:>6.1f}%  {severity}")

        # ── Column details ────────────────────────────────────────────
        section("4. COLUMN TYPES & CARDINALITY")
        lines.append(f"  {'Column':<25} {'Type':<12} {'Kind':<12} {'Unique':>7}  {'Cardinality'}")
        lines.append(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*7}  {'─'*12}")
        for cp in self.columns:
            flag = " ← ID?"  if cp.is_likely_id else (" ← constant!" if cp.is_constant else "")
            lines.append(
                f"  {cp.name:<25} {cp.dtype:<12} {cp.inferred_kind:<12} "
                f"{cp.unique_count:>7,}  {cp.cardinality_pct:>5.1f}%{flag}"
            )

        # ── Numeric stats ─────────────────────────────────────────────
        if self.numeric_columns:
            section("5. NUMERIC COLUMN STATISTICS")
            for cp in self.numeric_columns:
                lines.append(f"\n  [{cp.name}]")
                row("Range",    f"{cp.min:,.2f} → {cp.max:,.2f}")
                row("Mean",     f"{cp.mean:,.2f}")
                row("Median",   f"{cp.median:,.2f}")
                row("Std dev",  f"{cp.std:,.2f}")
                row("Skewness", f"{cp.skewness:,.2f}  {'← highly skewed ⚠' if abs(cp.skewness) > 1 else ''}")
                if cp.zero_count:
                    row("Zero values", f"{cp.zero_count:,}  ({cp.zero_count/self.total_rows*100:.1f}%)")
                if cp.outlier_count:
                    row("Outliers (IQR)", f"{cp.outlier_count:,}  ({cp.outlier_pct:.1f}%)  ⚠")
                else:
                    row("Outliers (IQR)", "None detected  ✓")

        # ── Categorical stats ─────────────────────────────────────────
        if self.categorical_columns:
            section("6. CATEGORICAL COLUMN STATISTICS")
            for cp in self.categorical_columns:
                lines.append(f"\n  [{cp.name}]  ({cp.unique_count} unique values)")
                if cp.value_counts:
                    for val, count in list(cp.value_counts.items())[:5]:
                        pct = count / self.total_rows * 100
                        bar = "█" * int(pct / 5)
                        lines.append(f"    {str(val):<25} {count:>6,}  ({pct:>5.1f}%)  {bar}")
                if cp.is_constant:
                    lines.append(f"    ⚠  Constant column — all values are '{cp.top_value}'")

        # ── Data quality flags ────────────────────────────────────────
        section("7. DATA QUALITY FLAGS")
        flags_found = False
        if self.constant_columns:
            lines.append(f"  ⚠  Constant columns (consider dropping): {self.constant_columns}")
            flags_found = True
        if self.high_null_columns:
            lines.append(f"  ⚠  High-null columns (>20% missing): {self.high_null_columns}")
            flags_found = True
        if self.outlier_columns:
            lines.append(f"  ⚠  Columns with numeric outliers: {self.outlier_columns}")
            flags_found = True
        if self.likely_id_columns:
            lines.append(f"  ℹ  Likely ID/key columns: {self.likely_id_columns}")
            flags_found = True
        if not flags_found:
            lines.append("  ✓ No critical data quality issues found.")

        # ── Recommendations ───────────────────────────────────────────
        section("8. CLEANING RECOMMENDATIONS")
        recs = self._generate_recommendations()
        if recs:
            for i, rec in enumerate(recs, 1):
                lines.append(f"  {i}. {rec}")
        else:
            lines.append("  ✓ Data appears ready for analysis.")

        rule("═")
        return "\n".join(lines)

    def _generate_recommendations(self) -> List[str]:
        recs = []
        if self.duplicate_rows > 0:
            recs.append(
                f"Remove {self.duplicate_rows:,} duplicate rows "
                f"({self.duplicate_pct:.1f}% of data)."
            )
        for col_name in self.high_null_columns:
            cp = self.get_column(col_name)
            if cp:
                strategy = "median/mean" if cp.inferred_kind == "numeric" else "'Unknown' / mode"
                recs.append(
                    f"Column '{col_name}' has {cp.null_pct:.1f}% nulls — "
                    f"fill with {strategy} or drop rows."
                )
        for col_name in self.columns_with_nulls:
            if col_name not in self.high_null_columns:
                cp = self.get_column(col_name)
                if cp:
                    recs.append(
                        f"Column '{col_name}' has {cp.null_count} nulls ({cp.null_pct:.1f}%) — "
                        f"consider filling."
                    )
        for col_name in self.outlier_columns:
            cp = self.get_column(col_name)
            if cp:
                recs.append(
                    f"Column '{col_name}' has {cp.outlier_count} outliers "
                    f"({cp.outlier_pct:.1f}%) — investigate before analysis."
                )
        for col_name in self.constant_columns:
            recs.append(f"Column '{col_name}' is constant — drop it (adds no information).")
        return recs

    # ------------------------------------------------------------------ #
    # Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "profiled_at":            self.profiled_at.isoformat(),
            "total_rows":             self.total_rows,
            "total_columns":          self.total_columns,
            "memory_usage_mb":        self.memory_usage_mb,
            "quality_score":          self.quality_score,
            "duplicate_rows":         self.duplicate_rows,
            "duplicate_pct":          self.duplicate_pct,
            "total_null_cells":       self.total_null_cells,
            "null_density_pct":       self.null_density_pct,
            "columns_with_nulls":     self.columns_with_nulls,
            "constant_columns":       self.constant_columns,
            "likely_id_columns":      self.likely_id_columns,
            "high_null_columns":      self.high_null_columns,
            "outlier_columns":        self.outlier_columns,
            "columns":                [c.to_dict() for c in self.columns],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json_str)
        return json_str

    def __repr__(self) -> str:
        return (
            f"DataProfile({self.total_rows:,} rows × {self.total_columns} cols, "
            f"quality={self.quality_score}/100, "
            f"nulls={self.total_null_cells}, "
            f"dupes={self.duplicate_rows})"
        )


# ── DataProfiler ──────────────────────────────────────────────────────────────

class DataProfiler:
    """Analyses a raw DataFrame and returns a DataProfile report.

    Parameters
    ----------
    high_null_threshold : float
        Columns with more than this percentage of nulls are flagged as
        high-null.  Defaults to 20.0 (%).
    high_cardinality_threshold : float
        Columns with cardinality above this % (but below likely-ID threshold)
        are placed on a watchlist.  Defaults to 50.0 (%).
    likely_id_threshold : float
        Columns with cardinality above this % are flagged as likely ID columns.
        Defaults to 95.0 (%).
    outlier_iqr_multiplier : float
        Tukey fence multiplier for outlier detection.  Standard value is 1.5.

    Examples
    --------
    >>> profiler = DataProfiler()
    >>> profile  = profiler.profile(raw_df)
    >>> profile.print_report()
    """

    def __init__(
        self,
        high_null_threshold:       float = 20.0,
        high_cardinality_threshold:float = 50.0,
        likely_id_threshold:       float = 95.0,
        outlier_iqr_multiplier:    float = 1.5,
    ) -> None:
        self.high_null_threshold        = high_null_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.likely_id_threshold        = likely_id_threshold
        self.outlier_iqr_multiplier     = outlier_iqr_multiplier

    def profile(self, df: pd.DataFrame) -> DataProfile:
        """Run the full profiling analysis and return a DataProfile.

        Parameters
        ----------
        df : pd.DataFrame
            Raw ingested DataFrame — before any cleaning.

        Returns
        -------
        DataProfile
        """
        n_rows, n_cols = df.shape
        print(f"[DataProfiler] Profiling {n_rows:,} rows × {n_cols} columns...")

        col_profiles = [self._profile_column(df, col, n_rows) for col in df.columns]

        total_nulls   = int(df.isnull().sum().sum())
        null_density  = round(total_nulls / (n_rows * n_cols) * 100, 4) if n_rows * n_cols else 0
        dup_rows      = int(df.duplicated().sum())
        mem_mb        = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3)

        cols_with_nulls    = [c.name for c in col_profiles if c.has_nulls]
        constant_cols      = [c.name for c in col_profiles if c.is_constant]
        likely_id_cols     = [c.name for c in col_profiles if c.is_likely_id]
        high_null_cols     = [c.name for c in col_profiles if c.null_pct > self.high_null_threshold]
        outlier_cols       = [c.name for c in col_profiles if c.has_outliers]
        high_card_cols     = [
            c.name for c in col_profiles
            if self.high_cardinality_threshold <= c.cardinality_pct < self.likely_id_threshold
        ]

        profile = DataProfile(
            total_rows              = n_rows,
            total_columns           = n_cols,
            memory_usage_mb         = mem_mb,
            duplicate_rows          = dup_rows,
            duplicate_pct           = round(dup_rows / n_rows * 100, 2) if n_rows else 0,
            total_null_cells        = total_nulls,
            null_density_pct        = null_density,
            columns_with_nulls      = cols_with_nulls,
            columns                 = col_profiles,
            constant_columns        = constant_cols,
            likely_id_columns       = likely_id_cols,
            high_null_columns       = high_null_cols,
            outlier_columns         = outlier_cols,
            high_cardinality_columns= high_card_cols,
        )

        print(f"[DataProfiler] Done — quality score: {profile.quality_score}/100")
        return profile

    # ------------------------------------------------------------------ #
    # Column profiling                                                    #
    # ------------------------------------------------------------------ #

    def _profile_column(
        self,
        df: pd.DataFrame,
        col: str,
        n_rows: int,
    ) -> ColumnProfile:
        series       = df[col]
        null_count   = int(series.isnull().sum())
        null_pct     = round(null_count / n_rows * 100, 2) if n_rows else 0
        non_null     = series.dropna()
        unique_count = int(non_null.nunique())
        cardinality  = round(unique_count / n_rows * 100, 2) if n_rows else 0
        is_constant  = unique_count == 1
        is_likely_id = cardinality >= self.likely_id_threshold

        kind = self._infer_kind(series)
        sample_values = [
            str(v) for v in non_null.unique()[:5].tolist()
        ]

        base = dict(
            name            = col,
            dtype           = str(series.dtype),
            inferred_kind   = kind,
            null_count      = null_count,
            null_pct        = null_pct,
            unique_count    = unique_count,
            cardinality_pct = cardinality,
            is_constant     = is_constant,
            is_likely_id    = is_likely_id,
            sample_values   = sample_values,
        )

        if kind == "numeric":
            return ColumnProfile(**base, **self._numeric_stats(non_null, n_rows))
        elif kind == "categorical":
            return ColumnProfile(**base, **self._categorical_stats(non_null, n_rows))
        else:
            return ColumnProfile(**base)

    def _numeric_stats(self, series: pd.Series, n_rows: int) -> dict:
        s   = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return {}
        q1  = float(s.quantile(0.25))
        q3  = float(s.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - self.outlier_iqr_multiplier * iqr
        upper_fence = q3 + self.outlier_iqr_multiplier * iqr
        outliers = int(((s < lower_fence) | (s > upper_fence)).sum())

        return {
            "min":           round(float(s.min()), 4),
            "max":           round(float(s.max()), 4),
            "mean":          round(float(s.mean()), 4),
            "median":        round(float(s.median()), 4),
            "std":           round(float(s.std()), 4),
            "q1":            round(q1, 4),
            "q3":            round(q3, 4),
            "iqr":           round(iqr, 4),
            "outlier_count": outliers,
            "outlier_pct":   round(outliers / n_rows * 100, 2) if n_rows else 0,
            "skewness":      round(float(s.skew()), 4),
            "zero_count":    int((s == 0).sum()),
        }

    @staticmethod
    def _categorical_stats(series: pd.Series, n_rows: int) -> dict:
        vc = series.value_counts()
        if vc.empty:
            return {}
        top_val   = vc.index[0]
        top_count = int(vc.iloc[0])
        return {
            "top_value":       str(top_val),
            "top_value_count": top_count,
            "top_value_pct":   round(top_count / n_rows * 100, 2) if n_rows else 0,
            "value_counts":    {str(k): int(v) for k, v in vc.head(5).items()},
        }

    @staticmethod
    def _infer_kind(series: pd.Series) -> str:
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        # Try to detect dates stored as strings
        if series.dtype == object:
            sample = series.dropna().head(20).astype(str)
            try:
                pd.to_datetime(sample, infer_datetime_format=True)
                return "datetime"
            except Exception:
                pass
        return "categorical"
