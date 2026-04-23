"""
Google Drive data source connector (stub).

This connector is intentionally left as a scaffold.  A production
implementation would use the Google Drive API v3 via ``google-api-python-client``
and ``google-auth-oauthlib``.

Implementation roadmap
----------------------
Phase 1 (this file): Define the interface — constructor, validate, load.
Phase 2: Authenticate with a service account JSON key or OAuth2 flow.
Phase 3: Download the file by ``file_id`` and read into a DataFrame.
Phase 4: Add caching so repeat calls don't re-download unchanged files.

To implement Phase 2+, install:
    pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
"""

import pandas as pd

from ingestion.base import DataSource, DataSourceError


class GoogleDriveSource(DataSource):
    """Loads a file from Google Drive by its file ID.

    .. note::
        This is a **stub**.  ``load()`` and ``validate()`` raise
        ``NotImplementedError`` until the Google Drive API integration is
        implemented.  The constructor is fully defined so that the rest of
        the pipeline can reference this class without breaking.

    Parameters
    ----------
    file_id : str
        The Google Drive file ID (the long alphanumeric string in the
        shareable link after ``/d/``).
    credentials_path : str, optional
        Path to a ``service_account.json`` or OAuth2 ``credentials.json``
        file.  Defaults to ``config/google_credentials.json``.
    download_dir : str, optional
        Local directory to cache the downloaded file.
        Defaults to ``data/raw/gdrive/``.
    mime_type : str, optional
        Expected MIME type of the file.  Used to select the correct export
        format when the file is a Google Sheets document.
        Defaults to ``"text/csv"``.

    Examples
    --------
    >>> # This will raise NotImplementedError until fully implemented.
    >>> source = GoogleDriveSource(file_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms")
    >>> df = source.load_validated()
    """

    def __init__(
        self,
        file_id: str,
        credentials_path: str = "config/google_credentials.json",
        download_dir: str = "data/raw/gdrive",
        mime_type: str = "text/csv",
    ) -> None:
        self.file_id = file_id
        self.credentials_path = credentials_path
        self.download_dir = download_dir
        self.mime_type = mime_type

    # ------------------------------------------------------------------ #

    def validate(self) -> bool:
        """Check whether the Google Drive file is accessible.

        .. note::
            Not yet implemented.  Will verify credentials and that the
            file exists and is readable.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "GoogleDriveSource is not yet implemented. "
            "See the module docstring for the implementation roadmap."
        )

    def load(self) -> pd.DataFrame:
        """Download the Drive file and return it as a DataFrame.

        .. note::
            Not yet implemented.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "GoogleDriveSource is not yet implemented. "
            "See the module docstring for the implementation roadmap."
        )

    def describe(self) -> str:
        return f"GoogleDriveSource(file_id='{self.file_id}') [STUB — not implemented]"
