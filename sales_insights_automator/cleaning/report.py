"""
CleaningReport — an audit trail of every change the DataCleaner made.

After calling DataCleaner.clean(), the report is available via the
``cleaner.report`` property.  It tells you exactly:

  - How many rows / columns were in the original vs final DataFrame
  - How many rows were removed (duplicates, null-drops)
  - Which columns were renamed
  - Which nulls were filled (and with what values)
  - Which type conversions were applied

This is invaluable in production pipelines where data quality must be
traceable, and it's a great talking point in interviews.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CleaningReport:
    """Immutable record of all changes applied during a cleaning run.

    DataCleaner builds and populates this object while cleaning; it is
    then exposed read-only via ``cleaner.report``.

    Attributes
    ----------
    original_shape : tuple[int, int]
        (rows, columns) before cleaning.
    final_shape : tuple[int, int]
        (rows, columns) after cleaning.
    columns_dropped : list[str]
        Columns removed by the ``drop_columns`` rule.
    columns_renamed : dict[str, str]
        Mapping of original column name → normalised name.
    rows_dropped_duplicates : int
        Number of duplicate rows removed.
    rows_dropped_nulls : int
        Number of rows removed because a required column was null.
    null_fills : dict[str, Any]
        Maps column name → the actual value used to fill nulls
        (e.g. ``{"revenue": 450.3}`` for a median fill, or
         ``{"region": "Unknown"}`` for a constant fill).
    null_counts_before : dict[str, int]
        Null count per column before cleaning.
    null_counts_after : dict[str, int]
        Null count per column after cleaning.
    type_conversions : dict[str, str]
        Maps column name → dtype string that was applied.
    """

    original_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)

    columns_dropped: List[str] = field(default_factory=list)
    columns_renamed: Dict[str, str] = field(default_factory=dict)

    rows_dropped_duplicates: int = 0
    rows_dropped_nulls: int = 0

    null_fills: Dict[str, Any] = field(default_factory=dict)
    null_counts_before: Dict[str, int] = field(default_factory=dict)
    null_counts_after: Dict[str, int] = field(default_factory=dict)

    type_conversions: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Computed properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def rows_removed(self) -> int:
        """Total rows removed across all cleaning steps."""
        return self.original_shape[0] - self.final_shape[0]

    @property
    def retention_rate(self) -> float:
        """Fraction of original rows that survived cleaning (0.0–1.0)."""
        if self.original_shape[0] == 0:
            return 1.0
        return self.final_shape[0] / self.original_shape[0]

    @property
    def total_nulls_filled(self) -> int:
        """Total null cells that were filled (not dropped) across all columns."""
        filled = 0
        for col, before in self.null_counts_before.items():
            after = self.null_counts_after.get(col, 0)
            if col in self.null_fills:
                filled += max(0, before - after)
        return filled

    # ------------------------------------------------------------------ #
    # Human-readable output                                               #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Return a formatted multi-line summary of the cleaning run."""
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║              DataCleaner — Cleaning Report               ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"  Original shape     : {self.original_shape[0]:,} rows × {self.original_shape[1]} cols",
            f"  Final shape        : {self.final_shape[0]:,} rows × {self.final_shape[1]} cols",
            f"  Rows removed       : {self.rows_removed:,}  "
            f"(retention: {self.retention_rate:.1%})",
            "",
        ]

        if self.columns_dropped:
            lines.append(f"  Columns dropped    : {self.columns_dropped}")

        if self.columns_renamed:
            renamed = [f"{old} → {new}" for old, new in self.columns_renamed.items()]
            lines.append(f"  Columns renamed    : {len(renamed)}")
            for r in renamed:
                lines.append(f"    • {r}")

        if self.rows_dropped_duplicates:
            lines.append(f"  Duplicate rows     : {self.rows_dropped_duplicates:,} removed")

        if self.rows_dropped_nulls:
            lines.append(f"  Null-drop rows     : {self.rows_dropped_nulls:,} removed")

        if self.null_fills:
            lines.append(f"  Null fills         : {self.total_nulls_filled:,} cells filled")
            for col, value in self.null_fills.items():
                before = self.null_counts_before.get(col, 0)
                after = self.null_counts_after.get(col, 0)
                lines.append(f"    • {col}: {before} → {after} nulls  (filled with {value!r})")

        if self.type_conversions:
            lines.append(f"  Type conversions   : {len(self.type_conversions)}")
            for col, dtype in self.type_conversions.items():
                lines.append(f"    • {col} → {dtype}")

        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Serialise the report to a JSON-safe dictionary."""
        return {
            "original_shape": list(self.original_shape),
            "final_shape": list(self.final_shape),
            "rows_removed": self.rows_removed,
            "retention_rate": round(self.retention_rate, 4),
            "columns_dropped": self.columns_dropped,
            "columns_renamed": self.columns_renamed,
            "rows_dropped_duplicates": self.rows_dropped_duplicates,
            "rows_dropped_nulls": self.rows_dropped_nulls,
            "null_fills": {k: str(v) for k, v in self.null_fills.items()},
            "total_nulls_filled": self.total_nulls_filled,
            "type_conversions": self.type_conversions,
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """Return (and optionally save) the report as a JSON string."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json_str)
        return json_str

    def __repr__(self) -> str:
        return (
            f"CleaningReport("
            f"rows={self.original_shape[0]}→{self.final_shape[0]}, "
            f"removed={self.rows_removed}, "
            f"retention={self.retention_rate:.1%})"
        )
