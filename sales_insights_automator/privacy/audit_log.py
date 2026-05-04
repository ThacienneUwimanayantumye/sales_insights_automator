"""
AuditLogger — records metadata of every AI API call.

What is logged
--------------
Every log entry records *metadata only* — never the prompt content,
never the generated insights, never the raw data.

Each entry contains:
  - Timestamp
  - Template used
  - Model called
  - Token counts (prompt + completion)
  - Data shape (row count, date range)
  - Which fields were masked by the anonymizer
  - Privacy config summary
  - Latency

What is NOT logged
------------------
  - The prompt text
  - The AI response text
  - Any actual sales figures or names

This design means the audit log can be shared with a compliance team
without exposing any business data.

Log format
----------
One JSON object per line (JSONL) written to ``data/audit/api_calls.jsonl``.
JSONL is append-only, human-readable, and trivially parseable by pandas
for compliance reporting:

    pd.read_json("data/audit/api_calls.jsonl", lines=True)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_LOG_PATH = "data/audit/api_calls.jsonl"


class AuditLogger:
    """Appends structured metadata records to a JSONL audit log file.

    Parameters
    ----------
    log_path : str
        Path to the JSONL log file.  Created automatically if it does
        not exist.  Defaults to ``data/audit/api_calls.jsonl``.

    Examples
    --------
    >>> logger = AuditLogger()
    >>> logger.log_api_call(
    ...     template      = "full_report",
    ...     model         = "gpt-4o-mini",
    ...     prompt_tokens = 1645,
    ...     completion_tokens = 412,
    ...     row_count     = 500,
    ...     date_range    = {"from": "2024-01-01", "to": "2024-12-30"},
    ...     masked_fields = ["sales_rep", "revenue_rounded"],
    ...     latency_ms    = 1843.0,
    ...     privacy_summary = "rep names masked | revenue rounded to 1000",
    ... )
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH) -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def log_api_call(
        self,
        template:          str,
        model:             str,
        prompt_tokens:     int,
        completion_tokens: int,
        row_count:         int,
        date_range:        Dict[str, str],
        masked_fields:     List[str],
        latency_ms:        float,
        privacy_summary:   str,
        was_dry_run:       bool = False,
    ) -> None:
        """Append one metadata record to the audit log.

        This method never fails — if the log cannot be written (e.g. disk
        full, permission error), it prints a warning and continues.  Audit
        failures must never break the main pipeline.
        """
        entry = {
            "timestamp":         datetime.now().isoformat(),
            "event":             "dry_run" if was_dry_run else "api_call",
            "template":          template,
            "model":             model,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
            "latency_ms":        round(latency_ms, 1),
            "data_row_count":    row_count,
            "data_period":       date_range,
            "fields_masked":     masked_fields,
            "privacy_summary":   privacy_summary,
            # Explicitly document what was NOT sent
            "content_logged":    False,
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            print(f"[AuditLogger] ⚠ Could not write audit log: {exc}")

    def read_log(self) -> list:
        """Read all audit log entries as a list of dicts.

        Returns an empty list if the log file does not exist yet.
        """
        if not os.path.isfile(self.log_path):
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    def print_summary(self) -> None:
        """Print a human-readable summary of all logged API calls."""
        entries = self.read_log()
        if not entries:
            print("[AuditLogger] No API calls logged yet.")
            return

        total_tokens = sum(e.get("total_tokens", 0) for e in entries)
        real_calls   = [e for e in entries if e.get("event") == "api_call"]
        dry_runs     = [e for e in entries if e.get("event") == "dry_run"]

        print(f"\n{'─' * 55}")
        print(f"  AUDIT LOG SUMMARY  —  {self.log_path}")
        print(f"{'─' * 55}")
        print(f"  Total entries      : {len(entries)}")
        print(f"  Real API calls     : {len(real_calls)}")
        print(f"  Dry runs           : {len(dry_runs)}")
        print(f"  Total tokens used  : {total_tokens:,}")
        if entries:
            print(f"  First call         : {entries[0]['timestamp'][:19]}")
            print(f"  Last call          : {entries[-1]['timestamp'][:19]}")
        print(f"  Content logged     : Never (metadata only)")
        print(f"{'─' * 55}")
