"""
Normalize DataFrames before analysis.

Kept in a **standalone module** so Streamlit's script runner does not serve a
stale ``analysis.metrics`` object missing newer helpers (AttributeError on
``prepare_analysis_dataframe``) while the rest of the app reloads.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from cleaning.functions import dedupe_column_names


def stringify_nested_object_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Turn list/dict/ndarray cells in object columns into JSON strings.

    Nested JSON often yields dimensions like ``region`` as ``[\"North\"]`` per row.
    Pandas :meth:`DataFrame.groupby` then raises *Grouper … not 1-dimensional*.
    Plain strings and numbers are left unchanged.
    """
    df = df.copy()

    def to_scalar(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            try:
                return json.dumps(v.tolist(), sort_keys=True, default=str)
            except (TypeError, ValueError):
                return repr(v)
        if isinstance(v, (list, dict, set)):
            try:
                return json.dumps(v, sort_keys=True, default=str)
            except (TypeError, ValueError):
                return repr(v)
        return v

    for col in df.columns:
        chunk = df[col]
        if isinstance(chunk, pd.DataFrame):
            chunk = chunk.iloc[:, 0]
        if chunk.dtype != object:
            continue
        sample = chunk.dropna()
        if sample.empty:
            continue
        limit = min(500, len(sample))
        has_nested = False
        for v in sample.iloc[:limit]:
            if isinstance(v, (list, dict, set, np.ndarray)):
                has_nested = True
                break
        if not has_nested:
            continue
        df[col] = chunk.map(to_scalar)

    return df


def prepare_analysis_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce core numeric columns and make dimension columns groupby-safe."""
    # Local import avoids any circular import while ``metrics`` is still loading.
    from analysis import metrics as mm

    df = dedupe_column_names(df)
    df = mm.coerce_standard_numeric_columns(df)
    df = stringify_nested_object_cells(df)
    return df
