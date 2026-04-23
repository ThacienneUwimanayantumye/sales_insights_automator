"""
Kaggle data source connector.

Downloads a dataset from Kaggle using the official Kaggle API client and
returns the requested file as a DataFrame.

Prerequisites
-------------
1. Install the client:  ``pip install kaggle``
2. Place your API token at ``~/.kaggle/kaggle.json``
   (download from https://www.kaggle.com/settings → "Create New Token").
3. Set file permissions:  ``chmod 600 ~/.kaggle/kaggle.json``

The connector caches the downloaded file in ``download_dir`` so subsequent
calls do not re-download unless ``force_download=True``.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from ingestion.base import DataSource, DataSourceError


class KaggleSource(DataSource):
    """Downloads a Kaggle dataset file and loads it as a DataFrame.

    Parameters
    ----------
    dataset : str
        Kaggle dataset identifier in ``owner/dataset-name`` format,
        e.g. ``"abhishek/top-5000-youtube-channels"``.
    filename : str
        The specific file inside the dataset to load, e.g. ``"sales.csv"``.
    download_dir : str, optional
        Local directory where the file will be saved.
        Defaults to ``data/raw/kaggle/``.
    force_download : bool, optional
        If True, re-download even if the file already exists locally.
        Defaults to False.
    parse_dates : list[str], optional
        Column names to parse as datetime.

    Examples
    --------
    >>> source = KaggleSource(
    ...     dataset="carrie1/ecommerce-data",
    ...     filename="data.csv",
    ...     parse_dates=["InvoiceDate"],
    ... )
    >>> df = source.load_validated()
    """

    def __init__(
        self,
        dataset: str,
        filename: str,
        download_dir: str = "data/raw/kaggle",
        force_download: bool = False,
        parse_dates: Optional[list] = None,
    ) -> None:
        self.dataset = dataset
        self.filename = filename
        self.download_dir = download_dir
        self.force_download = force_download
        self.parse_dates = parse_dates or []

    # ------------------------------------------------------------------ #

    @property
    def _local_path(self) -> Path:
        """Full local path where the downloaded file will be stored."""
        return Path(self.download_dir) / self.filename

    def validate(self) -> bool:
        """Return True if the Kaggle client is importable and credentials exist."""
        try:
            import kaggle  # noqa: F401  (import check only)
        except ImportError:
            print(
                "[KaggleSource] 'kaggle' package not installed. "
                "Run: pip install kaggle"
            )
            return False

        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            print(
                f"[KaggleSource] Kaggle credentials not found at {kaggle_json}. "
                "Download your API token from https://www.kaggle.com/settings."
            )
            return False

        return True

    def load(self) -> pd.DataFrame:
        """Download the dataset (if needed) and return it as a DataFrame.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        DataSourceError
            If the download fails or the file cannot be parsed.
        """
        self._download_if_needed()

        try:
            df = pd.read_csv(
                self._local_path,
                parse_dates=self.parse_dates if self.parse_dates else False,
                encoding="utf-8",
                encoding_errors="replace",  # Kaggle exports sometimes have mixed encodings
            )
        except Exception as exc:
            raise DataSourceError(
                f"Failed to read downloaded file '{self._local_path}': {exc}"
            ) from exc

        print(
            f"[KaggleSource] Loaded {len(df):,} rows from "
            f"Kaggle dataset '{self.dataset}' / '{self.filename}'"
        )
        return df

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _download_if_needed(self) -> None:
        """Download the dataset file if it is not already cached locally."""
        if self._local_path.exists() and not self.force_download:
            print(
                f"[KaggleSource] Using cached file: {self._local_path} "
                "(pass force_download=True to re-download)"
            )
            return

        os.makedirs(self.download_dir, exist_ok=True)

        try:
            import kaggle

            print(
                f"[KaggleSource] Downloading '{self.filename}' "
                f"from Kaggle dataset '{self.dataset}'..."
            )
            kaggle.api.authenticate()
            kaggle.api.dataset_download_file(
                self.dataset,
                file_name=self.filename,
                path=self.download_dir,
                force=self.force_download,
                quiet=False,
            )

            # Kaggle sometimes appends .zip — unzip if necessary
            zipped = self._local_path.with_suffix(self._local_path.suffix + ".zip")
            if zipped.exists():
                import zipfile

                with zipfile.ZipFile(zipped, "r") as zf:
                    zf.extractall(self.download_dir)
                zipped.unlink()
                print(f"[KaggleSource] Unzipped to {self.download_dir}")

        except Exception as exc:
            raise DataSourceError(
                f"Kaggle download failed for '{self.dataset}/{self.filename}': {exc}"
            ) from exc

    def describe(self) -> str:
        return f"KaggleSource(dataset='{self.dataset}', file='{self.filename}')"
