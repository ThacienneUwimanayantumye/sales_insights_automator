"""
Google Drive data source connector.

Uses Drive API v3 with read-only scope. Supports:

- **Service account** JSON (``type: service_account``). Share the Drive file
  with the service account email.
- **OAuth 2.0** desktop client JSON (``installed`` or ``web``) plus a user
  token file produced by ``scripts/setup_gdrive_oauth.py``.

Loads **CSV/TSV** uploaded files and **Google Sheets** (exported as CSV via the
API). Other binary formats are rejected with a clear error.

Dependencies::

    pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ingestion.base import DataSource, DataSourceError

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Native Google Sheets — export as CSV
MIME_GOOGLE_SHEET = "application/vnd.google-apps.spreadsheet"
# Typical uploaded tabular files on Drive
MIME_CSV_ALIASES = frozenset(
    {
        "text/csv",
        "text/tab-separated-values",
        "text/plain",
        "application/vnd.ms-excel",
    }
)


def _ensure_google_installed() -> None:
    try:
        import google.auth  # noqa: F401
        from googleapiclient.discovery import build  # noqa: F401
    except ImportError as exc:
        raise DataSourceError(
            "Google API libraries are not installed. Run: "
            "pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"
        ) from exc


def _load_credentials(credentials_path: Path, token_path: Optional[Path]) -> Any:
    if not credentials_path.is_file():
        raise DataSourceError(f"Credentials file not found: {credentials_path}")

    try:
        raw = json.loads(credentials_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataSourceError(f"Invalid JSON in credentials file: {exc}") from exc

    cred_type = raw.get("type")
    if cred_type == "service_account":
        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_file(
            str(credentials_path), scopes=SCOPES
        )

    if "installed" in raw or "web" in raw:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        if token_path is None or not token_path.is_file():
            loc = token_path if token_path else "(no path)"
            raise DataSourceError(
                "OAuth client JSON detected but no token file found at "
                f"{loc}. Run: python scripts/setup_gdrive_oauth.py"
            )
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.valid:
            return creds
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
            return creds
        raise DataSourceError(
            "OAuth token is missing or invalid. Re-run scripts/setup_gdrive_oauth.py"
        )

    raise DataSourceError(
        "Unrecognized credentials JSON: expected a service account key "
        "(type: service_account) or OAuth client secrets (installed / web)."
    )


def _build_drive_service(
    credentials_path: str, token_path: Optional[str] = None
) -> Any:
    _ensure_google_installed()
    from googleapiclient.discovery import build

    cpath = Path(credentials_path).expanduser()
    tpath = Path(token_path).expanduser() if token_path else None
    creds = _load_credentials(cpath, tpath)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


class GoogleDriveSource(DataSource):
    """Load a CSV/TSV or Google Sheet from Drive by file ID."""

    def __init__(
        self,
        file_id: str,
        credentials_path: Optional[str] = None,
        token_path: Optional[str] = None,
        download_dir: Optional[str] = None,
        sheet_export_mime: str = "text/csv",
    ) -> None:
        from config import settings

        self.file_id = (file_id or "").strip()
        self.credentials_path = credentials_path or settings.GDRIVE_CREDENTIALS_PATH
        self.token_path = token_path.strip() if token_path and token_path.strip() else None
        self.download_dir = download_dir or settings.GDRIVE_DOWNLOAD_DIR
        self.sheet_export_mime = sheet_export_mime

        self._token_file = (
            Path(self.token_path).expanduser()
            if self.token_path
            else Path(settings.GDRIVE_TOKEN_PATH).expanduser()
        )

    def _service(self) -> Any:
        return _build_drive_service(
            self.credentials_path,
            str(self._token_file),
        )

    def _fetch_metadata(self, service: Any) -> dict[str, Any]:
        from googleapiclient.errors import HttpError

        try:
            return (
                service.files()
                .get(fileId=self.file_id, fields="id,name,mimeType")
                .execute()
            )
        except HttpError as exc:
            status = getattr(exc.resp, "status", None)
            if status == 404:
                raise DataSourceError(
                    f"Drive file not found (404). Check file ID and sharing: {self.file_id}"
                ) from exc
            raise DataSourceError(f"Drive API error ({status}): {exc}") from exc

    def _download_raw_bytes(self, service: Any, meta: dict[str, Any]) -> bytes:
        from googleapiclient.http import MediaIoBaseDownload

        mime = meta.get("mimeType") or ""
        fh = io.BytesIO()

        if mime == MIME_GOOGLE_SHEET:
            request = service.files().export_media(
                fileId=self.file_id, mimeType=self.sheet_export_mime
            )
        else:
            request = service.files().get_media(fileId=self.file_id)

        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return fh.getvalue()

    def _infer_sep(self, name: str, mime: str) -> str:
        lower = name.lower()
        if lower.endswith(".tsv") or mime == "text/tab-separated-values":
            return "\t"
        return ","

    def _ensure_readable_tabular(self, meta: dict[str, Any]) -> None:
        mime = meta.get("mimeType") or ""
        name = (meta.get("name") or "").lower()

        if mime == MIME_GOOGLE_SHEET:
            return
        if mime in MIME_CSV_ALIASES:
            return
        if name.endswith(".csv") or name.endswith(".tsv"):
            return

        raise DataSourceError(
            f"This connector only loads CSV/TSV files or Google Sheets. "
            f"Got mimeType={mime!r}, name={meta.get('name')!r}."
        )

    def _read_dataframe(self, raw: bytes, meta: dict[str, Any]) -> pd.DataFrame:
        name = meta.get("name") or f"gdrive_{self.file_id}.csv"
        mime = meta.get("mimeType") or ""
        sep = self._infer_sep(name, mime)
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                sep=sep,
                encoding="utf-8-sig",
            )
        except Exception as exc:
            raise DataSourceError(f"Could not parse tabular data from Drive: {exc}") from exc

        df["_source_file"] = name
        print(f"[GoogleDriveSource] Loaded {len(df):,} rows from '{name}' ({self.file_id})")
        return df

    def validate(self) -> bool:
        if not self.file_id:
            print("[GoogleDriveSource] Missing file_id.")
            return False
        try:
            service = self._service()
            meta = self._fetch_metadata(service)
            self._ensure_readable_tabular(meta)
            return True
        except DataSourceError as exc:
            print(f"[GoogleDriveSource] {exc}")
            return False
        except Exception as exc:
            print(f"[GoogleDriveSource] Validation failed: {exc}")
            return False

    def load(self) -> pd.DataFrame:
        service = self._service()
        meta = self._fetch_metadata(service)
        self._ensure_readable_tabular(meta)
        raw = self._download_raw_bytes(service, meta)
        if not raw:
            raise DataSourceError("Downloaded file is empty.")
        return self._read_dataframe(raw, meta)

    def describe(self) -> str:
        return (
            f"GoogleDriveSource(file_id={self.file_id!r}, "
            f"credentials_path={self.credentials_path!r})"
        )
