"""
CSV data source connector.

Reads one or more CSV files from the local filesystem.  Supports a single
file path or a glob pattern so callers can load an entire directory of monthly
export files in one call (e.g. ``data/raw/sales_*.csv``).
"""

import glob
import os
from typing import Optional

import pandas as pd

from ingestion.base import DataSource, DataSourceError


class CSVSource(DataSource):
    """Loads sales data from a local CSV file or a glob pattern.

    Parameters
    ----------
    filepath : str
        Path to a single CSV file or a glob pattern such as
        ``data/raw/sales_*.csv``.
    delimiter : str, optional
        Column delimiter.  Defaults to ``","`` (standard CSV).
    encoding : str, optional
        File encoding.  Defaults to ``"utf-8"``.
    parse_dates : list[str], optional
        Column names that should be parsed as datetime objects.

    Examples
    --------
    >>> source = CSVSource("data/samples/sample_sales.csv", parse_dates=["date"])
    >>> df = source.load_validated()
    >>> print(df.head())
    """

    def __init__(
        self,
        filepath: str,
        delimiter: str = ",",
        encoding: str = "utf-8",
        parse_dates: Optional[list] = None,
    ) -> None:
        self.filepath = filepath
        self.delimiter = delimiter
        self.encoding = encoding
        self.parse_dates = parse_dates or []

    # ------------------------------------------------------------------ #

    def validate(self) -> bool:
        """Return True if at least one file matches ``self.filepath``."""
        matched = glob.glob(self.filepath)
        if not matched:
            print(f"[CSVSource] No files found matching: {self.filepath}")
            return False
        return True

    def load(self) -> pd.DataFrame:
        """Read CSV file(s) and return a single concatenated DataFrame.

        When a glob pattern matches multiple files, all files are read and
        concatenated row-wise.  A ``_source_file`` column is added so
        downstream code can trace each row back to its origin file.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        DataSourceError
            If a matched file cannot be parsed.
        """
        matched_files = glob.glob(self.filepath)

        if not matched_files:
            raise DataSourceError(f"No CSV files found matching: {self.filepath}")

        frames = []
        for path in sorted(matched_files):
            try:
                df = pd.read_csv(
                    path,
                    delimiter=self.delimiter,
                    encoding=self.encoding,
                    parse_dates=self.parse_dates if self.parse_dates else False,
                )
                df["_source_file"] = os.path.basename(path)
                frames.append(df)
                print(f"[CSVSource] Loaded {len(df):,} rows from '{path}'")
            except Exception as exc:
                raise DataSourceError(f"Failed to read '{path}': {exc}") from exc

        combined = pd.concat(frames, ignore_index=True)
        print(f"[CSVSource] Total rows loaded: {len(combined):,}")
        return combined

    def describe(self) -> str:
        return f"CSVSource(filepath='{self.filepath}')"
