"""
SchemaWizard — two-step schema mapping tool.

Step 1 — Auto-detect
    The wizard inspects every column in the DataFrame (name, dtype, sample
    values, cardinality) and scores it against each semantic role.  It picks
    the highest-scoring candidate per role as the automatic suggestion.

Step 2 — Interactive confirmation (optional)
    When run in interactive mode the wizard presents each mapping to the user
    one column at a time.  The user can:

      • Press Enter to accept the suggestion
      • Type a role number to reassign the column to a different role
      • Type 's' to skip (mark the role as absent)
      • Type '?' to see what each role means

    At the end the wizard saves the resulting SchemaConfig to a JSON file.

Non-interactive use
    Call ``wizard.detect(df)`` to get an auto-detected SchemaConfig without
    any user prompts.  Useful for automated pipelines or testing.

Usage
-----
    wizard = SchemaWizard()

    # Fully automatic (no prompts)
    schema = wizard.detect(df)

    # Interactive (guided confirmation)
    schema = wizard.run(df, save_path="config/my_schema.json")
"""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.schema import (
    ALL_ROLES,
    OPTIONAL_ROLES,
    REQUIRED_ROLES,
    ROLE_DESCRIPTIONS,
    STANDARD_NAMES,
    SchemaConfig,
)


# ── Scoring rules ─────────────────────────────────────────────────────────────
# Each rule is (role, name_patterns, dtype_kinds, cardinality_hint)
# cardinality_hint: "high" = likely unique/ID, "low" = few distinct values,
#                   "any"  = no constraint

_NAME_PATTERNS: Dict[str, List[str]] = {
    "order_id":   ["order_id", "transaction_id", "txn_id", "receipt",
                   "invoice", "id", "order", "sale_id", "record_id"],
    "date":       ["date", "time", "day", "created", "timestamp",
                   "purchase_date", "sale_date", "order_date", "transaction_date"],
    "revenue":    ["revenue", "total", "amount", "sales", "income",
                   "net", "gross", "price_total", "total_sales", "sale_amount",
                   "value", "amt"],
    "product":    ["product", "item", "sku", "article", "goods",
                   "product_name", "item_name", "description"],
    "category":   ["category", "type", "group", "segment", "dept",
                   "department", "class", "product_type", "product_group"],
    "region":     ["region", "area", "territory", "zone", "location",
                   "country", "state", "branch", "market", "geo"],
    "sales_rep":  ["rep", "agent", "salesperson", "seller", "employee",
                   "staff", "associate", "sales_rep", "account_manager",
                   "manager"],
    "quantity":   ["quantity", "qty", "units", "count", "volume",
                   "num_units", "amount_units"],
    "unit_price": ["unit_price", "price", "rate", "cost", "unit_cost",
                   "price_per_unit", "selling_price", "list_price"],
    "discount":   ["discount", "disc", "reduction", "markdown",
                   "discount_pct", "discount_rate"],
}

_DTYPE_KINDS: Dict[str, List[str]] = {
    "order_id":   ["categorical"],
    "date":       ["datetime", "categorical"],   # dates often stored as strings
    "revenue":    ["numeric"],
    "product":    ["categorical"],
    "category":   ["categorical"],
    "region":     ["categorical"],
    "sales_rep":  ["categorical"],
    "quantity":   ["numeric"],
    "unit_price": ["numeric"],
    "discount":   ["numeric"],
}

# Expected cardinality range as fraction of total rows (min, max)
_CARDINALITY_RANGE: Dict[str, Tuple[float, float]] = {
    "order_id":   (0.80, 1.00),   # nearly unique
    "date":       (0.05, 1.00),   # varies widely
    "revenue":    (0.10, 1.00),   # many distinct values
    "product":    (0.001, 0.20),  # small catalogue
    "category":   (0.001, 0.10),  # very few categories
    "region":     (0.001, 0.10),  # small set
    "sales_rep":  (0.001, 0.15),  # small team
    "quantity":   (0.001, 0.30),  # small range of integers
    "unit_price": (0.001, 0.30),  # limited price points
    "discount":   (0.001, 0.20),  # limited discount tiers
}


# ── SchemaWizard ──────────────────────────────────────────────────────────────

class SchemaWizard:
    """Detects and interactively confirms the schema mapping for a DataFrame.

    Parameters
    ----------
    min_confidence : float
        Minimum score (0–10) for auto-detection to propose a suggestion.
        Columns below this threshold are shown as "uncertain" to the user.
        Defaults to 2.0.
    """

    def __init__(self, min_confidence: float = 2.0) -> None:
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def detect(self, df: pd.DataFrame) -> SchemaConfig:
        """Auto-detect a SchemaConfig without any user prompts.

        Each semantic role is matched to the highest-scoring column.
        A column can only be assigned to one role (greedy assignment in
        order of confidence score, highest first).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        SchemaConfig
            Roles that could not be confidently matched are set to None.
        """
        scores = self._score_all(df)
        mapping = self._assign_greedy(scores)
        return SchemaConfig(**{role: col for role, col in mapping.items()})

    def run(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
        silent: bool = False,
    ) -> SchemaConfig:
        """Run the interactive schema mapping wizard.

        Shows the auto-detected mapping and lets the user confirm or
        correct each role before saving.

        Parameters
        ----------
        df : pd.DataFrame
        save_path : str, optional
            If provided, the resulting SchemaConfig is saved as JSON here.
        silent : bool
            If True, skip all printing and use auto-detection only.
            Equivalent to calling ``detect()``.  Useful for testing.

        Returns
        -------
        SchemaConfig
        """
        if silent:
            schema = self.detect(df)
            if save_path:
                schema.to_json(save_path)
            return schema

        scores  = self._score_all(df)
        mapping = self._assign_greedy(scores)

        self._print_header(df)
        self._print_columns_table(df)

        mapping = self._interactive_loop(df, mapping, scores)

        schema = SchemaConfig(**{role: col for role, col in mapping.items()})
        self._print_summary(schema)

        if save_path:
            schema.to_json(save_path)
            print(f"\n  Schema saved → {save_path}")
            print("  Load it again with: SchemaConfig.from_json('" + save_path + "')\n")

        return schema

    # ------------------------------------------------------------------ #
    # Scoring                                                             #
    # ------------------------------------------------------------------ #

    def _score_all(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Return scores[role][column] = float score."""
        n_rows     = len(df)
        col_info   = self._build_col_info(df, n_rows)
        scores: Dict[str, Dict[str, float]] = {role: {} for role in ALL_ROLES}

        for role in ALL_ROLES:
            for col in df.columns:
                scores[role][col] = self._score_column(col, col_info[col], role, n_rows)

        return scores

    def _score_column(
        self,
        col: str,
        info: dict,
        role: str,
        n_rows: int,
    ) -> float:
        score = 0.0
        col_lower = col.lower().replace(" ", "_")

        # ── Name match ────────────────────────────────────────────────
        # Exact match scores highest
        if col_lower == role or col_lower == role.replace("_", ""):
            score += 5.0
        else:
            for pattern in _NAME_PATTERNS.get(role, []):
                if pattern in col_lower:
                    score += 3.0
                    break
            else:
                # Partial word match (e.g. "sale" matches "total_sales")
                role_words = set(re.split(r"[_\s]", role))
                col_words  = set(re.split(r"[_\s]", col_lower))
                if role_words & col_words:
                    score += 1.0

        # ── dtype match ───────────────────────────────────────────────
        if info["kind"] in _DTYPE_KINDS.get(role, []):
            score += 2.0

        # Special: detect date-like strings for the "date" role
        if role == "date" and info["kind"] == "categorical" and info["date_like"]:
            score += 3.0

        # ── Cardinality match ─────────────────────────────────────────
        if n_rows > 0:
            cardinality = info["unique_count"] / n_rows
            lo, hi = _CARDINALITY_RANGE.get(role, (0.0, 1.0))
            if lo <= cardinality <= hi:
                score += 1.5

        # ── Value range hints for numeric roles ───────────────────────
        if role == "discount" and info["kind"] == "numeric":
            if info.get("max") is not None and info["max"] <= 1.0:
                score += 2.0   # looks like a 0-1 fraction
            elif info.get("max") is not None and info["max"] <= 100.0:
                score += 1.0   # could be percentage

        return round(score, 2)

    def _assign_greedy(
        self,
        scores: Dict[str, Dict[str, float]],
    ) -> Dict[str, Optional[str]]:
        """Assign each role to its highest-scoring unassigned column.

        Required roles are assigned first.  A role is set to None if no
        column exceeds ``min_confidence``.
        """
        assigned_cols: set = set()
        mapping: Dict[str, Optional[str]] = {role: None for role in ALL_ROLES}

        role_order = list(REQUIRED_ROLES) + list(OPTIONAL_ROLES)
        for role in role_order:
            candidates = sorted(
                scores[role].items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            for col, score in candidates:
                if col not in assigned_cols and score >= self.min_confidence:
                    mapping[role] = col
                    assigned_cols.add(col)
                    break

        return mapping

    # ------------------------------------------------------------------ #
    # Column info helper                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_col_info(df: pd.DataFrame, n_rows: int) -> dict:
        info = {}
        for col in df.columns:
            series     = df[col]
            non_null   = series.dropna()
            unique_cnt = int(non_null.nunique())

            kind = SchemaWizard._infer_kind(series)
            date_like = False
            if kind == "categorical" and len(non_null) > 0:
                try:
                    pd.to_datetime(non_null.head(10).astype(str), infer_datetime_format=True)
                    date_like = True
                except Exception:
                    pass

            col_info: dict = {
                "dtype":        str(series.dtype),
                "kind":         kind,
                "unique_count": unique_cnt,
                "null_count":   int(series.isnull().sum()),
                "sample":       [str(v) for v in non_null.unique()[:3].tolist()],
                "date_like":    date_like,
            }
            if kind == "numeric":
                numeric = pd.to_numeric(non_null, errors="coerce").dropna()
                if not numeric.empty:
                    col_info["min"] = float(numeric.min())
                    col_info["max"] = float(numeric.max())

            info[col] = col_info
        return info

    @staticmethod
    def _infer_kind(series: pd.Series) -> str:
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        return "categorical"

    # ------------------------------------------------------------------ #
    # Interactive CLI                                                     #
    # ------------------------------------------------------------------ #

    def _interactive_loop(
        self,
        df: pd.DataFrame,
        mapping: Dict[str, Optional[str]],
        scores: Dict[str, Dict[str, float]],
    ) -> Dict[str, Optional[str]]:
        """Walk the user through each role and let them confirm or override."""
        W = 65
        print(f"\n{'─'*W}")
        print("  STEP 2 — Confirm or correct the role assignments")
        print(f"  Press Enter to accept  |  type a number to reassign  |  's' to skip")
        print(f"{'─'*W}\n")

        for i, role in enumerate(ALL_ROLES, 1):
            suggested = mapping[role]
            required  = role in REQUIRED_ROLES
            req_label = " [REQUIRED]" if required else " [optional]"

            print(f"  Role {i:>2}/{len(ALL_ROLES)}: {role}{req_label}")
            print(f"  {ROLE_DESCRIPTIONS[role]}")

            if suggested:
                col_info = self._build_col_info(df, len(df)).get(suggested, {})
                sample   = ", ".join(col_info.get("sample", []))
                dtype    = col_info.get("dtype", "?")
                print(f"  Suggested  : '{suggested}'  ({dtype})  samples: {sample}")
            else:
                # Find runner-up suggestions
                top = sorted(scores[role].items(), key=lambda kv: kv[1], reverse=True)[:3]
                suggestions_str = "  |  ".join(
                    f"'{c}' (score {s:.1f})" for c, s in top if s > 0
                )
                print(f"  Suggested  : — no confident match found")
                if suggestions_str:
                    print(f"  Candidates : {suggestions_str}")

            # Show numbered list of all columns for manual selection
            columns = list(df.columns)

            answer = input(
                f"\n  [Enter]=accept  [s]=skip  [?]=show all columns  "
                f"[1–{len(columns)}]=pick manually\n  > "
            ).strip().lower()

            if answer == "":
                pass   # keep suggested
            elif answer == "s":
                mapping[role] = None
                print(f"  ✓ Skipped — '{role}' will not be used in analysis.")
            elif answer == "?":
                self._print_numbered_columns(df)
                pick = input(f"  Enter column number (1–{len(columns)}) > ").strip()
                if pick.isdigit() and 1 <= int(pick) <= len(columns):
                    mapping[role] = columns[int(pick) - 1]
                    print(f"  ✓ Assigned: '{mapping[role]}' → role '{role}'")
            elif answer.isdigit() and 1 <= int(answer) <= len(columns):
                mapping[role] = columns[int(answer) - 1]
                print(f"  ✓ Assigned: '{mapping[role]}' → role '{role}'")
            else:
                print("  ⚠  Unrecognised input — keeping suggestion.")

            print()

        return mapping

    # ------------------------------------------------------------------ #
    # Print helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _print_header(df: pd.DataFrame) -> None:
        W = 65
        print(f"\n{'═'*W}")
        print("  SCHEMA SETUP WIZARD")
        print(f"  Dataset: {len(df):,} rows × {len(df.columns)} columns")
        print(f"{'═'*W}")

    @staticmethod
    def _print_columns_table(df: pd.DataFrame) -> None:
        print("\n  STEP 1 — Your dataset's columns\n")
        print(f"  {'#':>3}  {'Column name':<28}  {'Type':<12}  Sample values")
        print(f"  {'─'*3}  {'─'*28}  {'─'*12}  {'─'*20}")
        for i, col in enumerate(df.columns, 1):
            series   = df[col].dropna()
            dtype    = str(df[col].dtype)
            sample   = ", ".join(str(v) for v in series.unique()[:3])
            if len(sample) > 30:
                sample = sample[:27] + "..."
            print(f"  {i:>3}  {col:<28}  {dtype:<12}  {sample}")
        print()

    @staticmethod
    def _print_numbered_columns(df: pd.DataFrame) -> None:
        print()
        for i, col in enumerate(df.columns, 1):
            series = df[col].dropna()
            sample = ", ".join(str(v) for v in series.unique()[:2])
            print(f"    {i:>3}. {col:<28}  ({sample})")
        print()

    @staticmethod
    def _print_summary(schema: SchemaConfig) -> None:
        W = 65
        print(f"\n{'─'*W}")
        print("  FINAL SCHEMA MAPPING\n")
        for role in ALL_ROLES:
            actual  = getattr(schema, role)
            status  = f"→  '{actual}'" if actual else "→  (not mapped)"
            req     = "*" if role in REQUIRED_ROLES else " "
            print(f"  {req} {role:<14} {status}")
        print(f"\n  (* = required)")
        print(f"{'═'*W}")
