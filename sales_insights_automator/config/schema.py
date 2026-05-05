"""
SchemaConfig — maps your dataset's actual column names to the semantic roles
the analysis layer needs.

Problem
-------
Every company names columns differently.  One dataset calls it ``"revenue"``,
another calls it ``"total_sales"``, a third calls it ``"amt"``.  Hardcoding
column names breaks the tool the moment you switch datasets.

Solution
--------
``SchemaConfig`` stores a mapping:  semantic role  →  actual column name.

  order_id   →  "transaction_id"
  date       →  "sale_date"
  revenue    →  "total_sales"
  product    →  "item_name"
  ...

The ``SalesAnalyzer`` calls ``schema.rename_to_standard(df)`` once at the top
of ``analyze()``.  This renames the user's columns to the standard names the
rest of the pipeline expects.  Nothing else in the pipeline needs to change.

Roles
-----
Required (analysis cannot run without them):
  order_id   — unique identifier per transaction
  date       — transaction date
  revenue    — monetary value per transaction

Optional (analysis skips gracefully if absent):
  product    — product name / SKU
  category   — product category / group
  region     — sales territory / area
  sales_rep  — salesperson name or ID
  quantity   — units sold per transaction
  unit_price — price per unit
  discount   — discount applied (0.0–1.0 or percentage)

Usage
-----
    # Load from the wizard-generated file
    schema = SchemaConfig.from_json("config/my_dataset_schema.json")
    analyzer = SalesAnalyzer(schema=schema)
    result   = analyzer.analyze(raw_df)

    # Or build inline
    schema = SchemaConfig(revenue="total_sales", date="sale_date")
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ── Role metadata ─────────────────────────────────────────────────────────────

REQUIRED_ROLES = ("order_id", "date", "revenue")

OPTIONAL_ROLES = ("product", "category", "region", "sales_rep",
                  "quantity", "unit_price", "discount",
                  "customer_id", "gender", "age",
                  # financial enrichment
                  "profit", "cost",
                  # transaction attributes
                  "channel", "payment_method",
                  # customer attributes
                  "customer_segment", "return_flag", "rating")

ALL_ROLES = REQUIRED_ROLES + OPTIONAL_ROLES

# Maps role name → the column name the analysis layer expects after renaming.
# All roles map to themselves EXCEPT "discount", whose standard column name
# is "discount_pct" (matching the COL_DISCOUNT constant in metrics.py).
STANDARD_NAMES: dict = {role: role for role in ALL_ROLES}
STANDARD_NAMES["discount"] = "discount_pct"

# Human-readable labels shown in the UI (wizard dropdowns, mapping table, dashboard).
# Always prefer these over raw role keys when displaying text to users.
ROLE_LABELS = {
    "order_id":         "Order ID",
    "date":             "Date",
    "revenue":          "Revenue",
    "product":          "Product",
    "category":         "Product Category",
    "region":           "Region",
    "sales_rep":        "Salesperson",
    "quantity":         "Quantity",
    "unit_price":       "Unit Price",
    "discount":         "Discount",
    "customer_id":      "Customer ID",
    "gender":           "Gender",
    "age":              "Age",
    "profit":           "Profit",
    "cost":             "Cost",
    "channel":          "Sales Channel",
    "payment_method":   "Payment Method",
    "customer_segment": "Customer Segment",
    "return_flag":      "Return Status",
    "rating":           "Rating",
}

ROLE_DESCRIPTIONS = {
    "order_id":     "Unique identifier per transaction (e.g. order ID, receipt number)",
    "date":         "Transaction date / timestamp",
    "revenue":      "Monetary value per transaction (sales amount, total, amount paid)",
    "product":      "Product name, SKU, or item description",
    "category":     "Product category, group, or department",
    "region":       "Sales territory, area, zone, or branch",
    "sales_rep":    "Salesperson name or employee ID",
    "quantity":     "Number of units sold per transaction",
    "unit_price":   "Price per individual unit (before discount)",
    "discount":     "Discount applied (0.0–1.0 fraction, or percentage)",
    "customer_id":      "Unique identifier per customer (enables repeat-purchase analysis)",
    "gender":           "Customer gender (e.g. Male / Female) — used for demographic breakdowns",
    "age":              "Customer age in years (numeric) — used for age-group breakdowns",
    "profit":           "Profit amount per transaction (revenue minus cost)",
    "cost":             "Cost of goods sold per transaction",
    "channel":          "Sales channel (e.g. Online, In-Store, Phone, App)",
    "payment_method":   "Payment type (e.g. Credit Card, Cash, PayPal, Bank Transfer)",
    "customer_segment": "Customer tier or loyalty segment (e.g. Gold, Silver, New)",
    "return_flag":      "Return indicator — whether the transaction was returned (0/1 or Yes/No)",
    "rating":           "Product or service rating / review score (typically 1–5 scale)",
}


# ── SchemaConfig ──────────────────────────────────────────────────────────────

@dataclass
class SchemaConfig:
    """Maps semantic roles → actual column names in your specific dataset.

    Set a role to ``None`` to mark it as absent from the dataset.
    Required roles (order_id, date, revenue) must not be None when passed
    to ``SalesAnalyzer``.

    Parameters (all default to the standard column name)
    -------------------------------------------------------
    order_id, date, revenue : str — required
    product, category, region, sales_rep, quantity,
    unit_price, discount     : Optional[str] — optional
    """

    # Required
    order_id:    Optional[str] = "order_id"
    date:        Optional[str] = "date"
    revenue:     Optional[str] = "revenue"

    # Optional — sales dimensions
    product:     Optional[str] = "product"
    category:    Optional[str] = "category"
    region:      Optional[str] = "region"
    sales_rep:   Optional[str] = "sales_rep"
    quantity:    Optional[str] = "quantity"
    unit_price:  Optional[str] = "unit_price"
    discount:    Optional[str] = "discount_pct"

    # Optional — customer demographics
    customer_id:      Optional[str] = "customer_id"
    gender:           Optional[str] = "gender"
    age:              Optional[str] = "age"

    # Optional — financial enrichment
    profit:           Optional[str] = "profit"
    cost:             Optional[str] = "cost"

    # Optional — transaction attributes
    channel:          Optional[str] = "channel"
    payment_method:   Optional[str] = "payment_method"

    # Optional — customer attributes
    customer_segment: Optional[str] = "customer_segment"
    return_flag:      Optional[str] = "return_flag"
    rating:           Optional[str] = "rating"

    # ------------------------------------------------------------------ #
    # Core API                                                            #
    # ------------------------------------------------------------------ #

    def rename_to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with columns renamed to standard names.

        Only renames columns that:
          1. Have a non-None mapping in this config
          2. Actually exist in the DataFrame
          3. Are not already named with the standard name

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Copy with renamed columns.  Original DataFrame is not modified.
        """
        rename_map: dict = {}
        for role in ALL_ROLES:
            actual_name  = getattr(self, role)
            standard_name = STANDARD_NAMES[role]   # what metrics.py expects
            if actual_name is None:
                continue
            if actual_name == standard_name:
                continue                           # already the standard name
            if actual_name in df.columns:
                rename_map[actual_name] = standard_name

        return df.rename(columns=rename_map) if rename_map else df.copy()

    def validate(self, df: pd.DataFrame) -> List[str]:
        """Check that required roles map to columns that exist in ``df``.

        Returns
        -------
        List[str]
            List of error messages.  Empty list means the config is valid.
        """
        errors: List[str] = []
        for role in REQUIRED_ROLES:
            actual = getattr(self, role)
            if actual is None:
                errors.append(
                    f"Required role '{role}' is not mapped. "
                    f"Assign a column from your dataset."
                )
            elif actual not in df.columns:
                errors.append(
                    f"Required role '{role}' is mapped to '{actual}', "
                    f"but that column does not exist in the DataFrame. "
                    f"Available: {list(df.columns)}"
                )
        return errors

    def mapped_roles(self) -> dict:
        """Return only the roles that have a non-None mapping."""
        return {role: getattr(self, role) for role in ALL_ROLES
                if getattr(self, role) is not None}

    def summary(self) -> str:
        """Return a human-readable summary of the current mapping."""
        lines = ["Schema Configuration:"]
        for role in ALL_ROLES:
            actual = getattr(self, role)
            required = " (required)" if role in REQUIRED_ROLES else " (optional)"
            status = f"'{actual}'" if actual else "— not mapped"
            lines.append(f"  {role:<12}{required:<12}: {status}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save this config to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "SchemaConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "SchemaConfig":
        """Load a SchemaConfig from a JSON file."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def __repr__(self) -> str:
        mapped = sum(1 for r in ALL_ROLES if getattr(self, r) is not None)
        return f"SchemaConfig({mapped}/{len(ALL_ROLES)} roles mapped)"
