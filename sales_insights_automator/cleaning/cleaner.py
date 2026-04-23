"""
DataCleaner — the public interface of the cleaning layer.

This class orchestrates all cleaning steps in a fixed, deliberate order:

  1. Drop unwanted columns      (before anything else to reduce noise)
  2. Normalise column names     (so subsequent steps use stable names)
  3. Remove duplicate rows
  4. Handle missing values      (fill or drop per column rules)
  5. Convert dtypes             (after nulls are handled to avoid coercion errors)

Each step delegates to a pure function in ``cleaning.functions``.
The result of every step is recorded in a ``CleaningReport`` that is
available after the call via ``cleaner.report``.

Usage
-----
    from cleaning.cleaner import DataCleaner
    from cleaning.config import CleaningConfig

    config = CleaningConfig.from_json("config/default_cleaning.json")
    cleaner = DataCleaner(config)

    clean_df = cleaner.clean(raw_df)
    print(cleaner.report.summary())
"""

import pandas as pd

from cleaning.config import CleaningConfig
from cleaning.report import CleaningReport
from cleaning import functions as fn


class DataCleaner:
    """Applies a configurable sequence of cleaning steps to a DataFrame.

    Parameters
    ----------
    config : CleaningConfig, optional
        Rules that drive the cleaning process.  If omitted, sensible
        defaults are used (normalise columns, drop duplicates, no fills,
        no type conversions).

    Attributes
    ----------
    config : CleaningConfig
        The active cleaning configuration.
    report : CleaningReport
        Available after calling ``clean()``.  Raises ``RuntimeError``
        if accessed before a cleaning run has been performed.

    Examples
    --------
    >>> cleaner = DataCleaner()
    >>> clean_df = cleaner.clean(raw_df)
    >>> print(cleaner.report.summary())

    >>> # With config-driven rules
    >>> config = CleaningConfig(
    ...     fill_missing={"region": "Unknown"},
    ...     type_conversions={"date": "datetime", "quantity": "int"},
    ... )
    >>> cleaner = DataCleaner(config)
    >>> clean_df = cleaner.clean(raw_df)
    """

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()
        self._report: CleaningReport | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline and return the cleaned DataFrame.

        The input DataFrame is never modified (all operations work on copies).

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data from the ingestion layer.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.

        Side-effects
        ------------
        Populates ``self.report`` with a full audit trail.
        """
        if df.empty:
            self._report = CleaningReport(
                original_shape=(0, len(df.columns)),
                final_shape=(0, len(df.columns)),
            )
            return df.copy()

        report = CleaningReport()
        report.original_shape = df.shape
        report.null_counts_before = fn.null_counts(df)

        # ── Step 1: Drop unwanted columns ─────────────────────────────
        if self.config.drop_columns:
            df, actually_dropped = fn.drop_columns(df, self.config.drop_columns)
            report.columns_dropped = actually_dropped
            if actually_dropped:
                print(f"[DataCleaner] Dropped columns: {actually_dropped}")

        # ── Step 2: Normalise column names ────────────────────────────
        if self.config.normalize_columns:
            df, renamed = fn.normalize_column_names(df)
            report.columns_renamed = renamed
            if renamed:
                print(f"[DataCleaner] Renamed {len(renamed)} column(s)")

        # ── Step 3: Remove duplicates ─────────────────────────────────
        if self.config.drop_duplicates:
            df, n_dupes = fn.drop_duplicate_rows(df, self.config.duplicate_subset)
            report.rows_dropped_duplicates = n_dupes
            if n_dupes:
                print(f"[DataCleaner] Removed {n_dupes:,} duplicate row(s)")

        # ── Step 4: Handle missing values ─────────────────────────────
        if self.config.fill_missing:
            df, fills_applied, rows_dropped_nulls = fn.handle_missing_values(
                df, self.config.fill_missing
            )
            report.null_fills = fills_applied
            report.rows_dropped_nulls = rows_dropped_nulls
            if fills_applied:
                print(f"[DataCleaner] Filled nulls in columns: {list(fills_applied.keys())}")
            if rows_dropped_nulls:
                print(f"[DataCleaner] Dropped {rows_dropped_nulls:,} row(s) with null values")

        # ── Step 5: Convert dtypes ────────────────────────────────────
        if self.config.type_conversions:
            df, conversions_applied = fn.convert_dtypes(df, self.config.type_conversions)
            report.type_conversions = conversions_applied
            if conversions_applied:
                print(f"[DataCleaner] Applied type conversions: {conversions_applied}")

        # ── Finalise report ───────────────────────────────────────────
        report.final_shape = df.shape
        report.null_counts_after = fn.null_counts(df)
        self._report = report

        print(
            f"[DataCleaner] Done — {report.original_shape[0]:,} rows in, "
            f"{report.final_shape[0]:,} rows out "
            f"({report.rows_removed:,} removed, {report.retention_rate:.1%} retained)"
        )

        return df

    @property
    def report(self) -> CleaningReport:
        """Audit trail of the last cleaning run.

        Raises
        ------
        RuntimeError
            If ``clean()`` has not been called yet.
        """
        if self._report is None:
            raise RuntimeError(
                "No cleaning report available. Call clean(df) first."
            )
        return self._report

    # ------------------------------------------------------------------ #
    # Convenience factory                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_json(cls, config_path: str) -> "DataCleaner":
        """Create a DataCleaner pre-loaded with rules from a JSON config file.

        Parameters
        ----------
        config_path : str
            Path to a JSON file compatible with ``CleaningConfig.from_json``.

        Examples
        --------
        >>> cleaner = DataCleaner.from_json("config/default_cleaning.json")
        >>> clean_df = cleaner.clean(raw_df)
        """
        config = CleaningConfig.from_json(config_path)
        return cls(config)

    def __repr__(self) -> str:
        return f"DataCleaner(config={self.config!r})"
