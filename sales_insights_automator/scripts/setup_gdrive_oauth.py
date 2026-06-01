"""
One-time OAuth setup for Google Drive (user consent, read-only).

Place a Google Cloud **OAuth 2.0 Client ID** JSON (Desktop app) at the path
defined by GDRIVE_CREDENTIALS_PATH (default: config/google_credentials.json).

Run from the project root::

    python scripts/setup_gdrive_oauth.py

A browser opens for consent; the refresh token is saved to GDRIVE_TOKEN_PATH
(default: config/gdrive_token.json). Use that token with GoogleDriveSource —
service accounts do not need this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from google_auth_oauthlib.flow import InstalledAppFlow

from config.settings import GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def main() -> None:
    secrets = Path(GDRIVE_CREDENTIALS_PATH).expanduser()
    if not secrets.is_file():
        raise SystemExit(
            f"Missing OAuth client secrets file: {secrets}\n"
            "Download JSON from Google Cloud Console → APIs & Services → "
            "Credentials → OAuth 2.0 Client ID (Desktop app)."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
    creds = flow.run_local_server(port=0)

    out = Path(GDRIVE_TOKEN_PATH).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(creds.to_json(), encoding="utf-8")
    print(f"Saved OAuth token to {out}")


if __name__ == "__main__":
    main()
