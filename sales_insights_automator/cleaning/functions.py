"""
Pure, single-responsibility cleaning functions.

Each function in this module:
  - Takes a DataFrame (and optionally some parameters)
  - Returns a *new* DataFrame  (never mutates the input)
  - Returns auxiliary metadata so the caller can build an audit trail

Keeping these as pure module-level functions (not methods) makes them:
  - Trivially unit-testable in isolation
  - Reusable outside of the DataCleaner class
  - Easy to explain — one function, one job
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from cleaning.config import NULL_STRATEGIES, STATISTICAL_STRATEGIES


# ── 1. Column normalisation ───────────────────────────────────────────────────

def normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Normalise all column names to snake_case.

    Rules applied in order:
      1. Strip leading/trailing whitespace
      2. Lowercase everything
      3. Replace runs of spaces, hyphens, and dots with a single underscore
      4. Remove characters that are not alphanumeric or underscore
      5. Collapse consecutive underscores
      6. Strip leading/trailing underscores

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with renamed columns.
    renamed : dict[str, str]
        Mapping of ``{original_name: new_name}`` for every column that changed.

    Examples
    --------
    >>> df.columns = ["Order ID", "Sales Rep.", "Unit-Price"]
    >>> df, renamed = normalize_column_names(df)
    >>> df.columns.tolist()
    ['order_id', 'sales_rep', 'unit_price']
    """
    df = df.copy()
    renamed: Dict[str, str] = {}

    new_names = []
    for col in df.columns:
        normalized = str(col).strip().lower()
        normalized = re.sub(r"[\s\-\.]+", "_", normalized)   # spaces/hyphens/dots → _
        normalized = re.sub(r"[^\w]", "", normalized)         # drop non-word chars
        normalized = re.sub(r"_+", "_", normalized)           # collapse __ → _
        normalized = normalized.strip("_")

        if normalized != col:
            renamed[col] = normalized
        new_names.append(normalized)

    df.columns = new_names
    return df, renamed


# ── 2. Duplicate removal ──────────────────────────────────────────────────────

def drop_duplicate_rows(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate rows and return how many were dropped.

    Parameters
    ----------
    df : pd.DataFrame
    subset : list[str] or None
        Columns to consider for duplicate detection.  None = all columns.

    Returns
    -------
    df : pd.DataFrame
        De-duplicated DataFrame.
    n_dropped : int
        Number of rows removed.
    """
    original_len = len(df)

    # Only pass subset columns that actually exist (guards against renames)
    if subset:
        valid_subset = [c for c in subset if c in df.columns]
        subset = valid_subset if valid_subset else None

    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    n_dropped = original_len - len(df)
    return df, n_dropped


# ── 3. Null handling ──────────────────────────────────────────────────────────

def handle_missing_values(
    df: pd.DataFrame,
    fill_config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], int]:
    """Fill or drop nulls according to per-column rules.

    Parameters
    ----------
    df : pd.DataFrame
    fill_config : dict[str, str | scalar]
        Maps column name → strategy or literal fill value.
        Columns not in ``fill_config`` are left unchanged.

        Strategies:
          ``"mean"``    → column mean   (numeric columns only)
          ``"median"``  → column median (numeric columns only)
          ``"mode"``    → most frequent value
          ``"drop"``    → drop rows where this column is null
          anything else → literal fill value

    Returns
    -------
    df : pd.DataFrame
        DataFrame with nulls handled.
    fills_applied : dict[str, Any]
        Maps column → the value that was used (useful for the report
        and for applying the same fills to new data later).
    rows_dropped : int
        Number of rows removed by ``"drop"`` strategies.
    """
    df = df.copy()
    fills_applied: Dict[str, Any] = {}
    rows_dropped = 0

    for col, strategy in fill_config.items():
        if col not in df.columns:
            continue

        if strategy == "drop":
            before = len(df)
            df = df.dropna(subset=[col]).reset_index(drop=True)
            rows_dropped += before - len(df)

        elif strategy == "mean":
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
                fills_applied[col] = round(fill_value, 4)
            else:
                # Fall back to mode for non-numeric columns
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
                if fill_value is not None:
                    df[col] = df[col].fillna(fill_value)
                    fills_applied[col] = fill_value

        elif strategy == "median":
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
                fills_applied[col] = round(fill_value, 4)
            else:
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
                if fill_value is not None:
                    df[col] = df[col].fillna(fill_value)
                    fills_applied[col] = fill_value

        elif strategy == "mode":
            fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
            if fill_value is not None:
                df[col] = df[col].fillna(fill_value)
                fills_applied[col] = fill_value

        else:
            # Treat as a literal constant
            df[col] = df[col].fillna(strategy)
            fills_applied[col] = strategy

    return df, fills_applied, rows_dropped


# ── 4. Type conversion ────────────────────────────────────────────────────────

def convert_dtypes(
    df: pd.DataFrame,
    type_config: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Convert column dtypes according to a config mapping.

    Conversions are applied with ``errors="coerce"`` so bad values become
    NaN/NaT rather than raising — consistent with production behaviour where
    upstream data quality is never guaranteed.

    Parameters
    ----------
    df : pd.DataFrame
    type_config : dict[str, str]
        Maps column name → target dtype string.

        Supported values:
          ``"datetime"``  → ``pd.to_datetime(..., errors="coerce")``
          ``"numeric"``   → ``pd.to_numeric(..., errors="coerce")``
          ``"int"``       → nullable ``Int64`` (handles NaNs gracefully)
          ``"float"``     → ``float64``
          ``"str"``       → ``object``

    Returns
    -------
    df : pd.DataFrame
        DataFrame with converted columns.
    applied : dict[str, str]
        Maps column → dtype string for columns that were successfully converted.
    """
    df = df.copy()
    applied: Dict[str, str] = {}

    converters = {
        "datetime": lambda s: pd.to_datetime(s, errors="coerce"),
        "numeric":  lambda s: pd.to_numeric(s, errors="coerce"),
        "float":    lambda s: pd.to_numeric(s, errors="coerce").astype("float64"),
        "str":      lambda s: s.astype(str),
        "int":      lambda s: pd.to_numeric(s, errors="coerce").astype("Int64"),
    }

    for col, dtype in type_config.items():
        if col not in df.columns:
            continue

        converter = converters.get(dtype.lower())
        if converter is None:
            # Unknown dtype — skip silently (don't crash the pipeline)
            continue

        try:
            df[col] = converter(df[col])
            applied[col] = dtype
        except Exception:
            # Conversion failed entirely — leave column untouched
            pass

    return df, applied


# ── 5. Column removal ─────────────────────────────────────────────────────────

def drop_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop the specified columns, ignoring any that do not exist.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Column names to remove.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the specified columns removed.
    actually_dropped : list[str]
        The subset of ``columns`` that actually existed and were dropped.
    """
    df = df.copy()
    existing = [c for c in columns if c in df.columns]
    df = df.drop(columns=existing)
    return df, existing


# ── 6. Null snapshot ──────────────────────────────────────────────────────────

def null_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Return the number of nulls per column as a plain dict.

    Used by DataCleaner to capture before/after snapshots for the report.
    Only columns with at least one null are included (keeps the report tidy).
    """
    counts = df.isnull().sum()
    return {col: int(cnt) for col, cnt in counts.items() if cnt > 0}
