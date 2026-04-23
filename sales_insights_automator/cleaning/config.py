"""
CleaningConfig — declarative, data-driven rules for the cleaning layer.

Instead of hard-coding cleaning logic per dataset, callers describe *what*
they want cleaned via a config object.  The DataCleaner reads this config
and applies each rule in a defined order.

Config can be built three ways:
  1. Programmatically:  CleaningConfig(drop_duplicates=True, ...)
  2. From a dict:       CleaningConfig.from_dict({...})
  3. From a JSON file:  CleaningConfig.from_json("config/default_cleaning.json")
  4. From a YAML file:  CleaningConfig.from_yaml("config/default_cleaning.yaml")
     (requires PyYAML: pip install pyyaml)

This is the "config-based cleaning rules" design — every cleaning decision
is expressed as data, not code, so non-engineers can tune cleaning behaviour
by editing a JSON file without touching Python.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Supported null-fill strategies ───────────────────────────────────────────
#
# Used as the values in CleaningConfig.fill_missing.
# Any value that is NOT one of these keywords is treated as a literal constant.
#
STATISTICAL_STRATEGIES = {"mean", "median", "mode"}
NULL_STRATEGIES = STATISTICAL_STRATEGIES | {"drop"}


@dataclass
class CleaningConfig:
    """Declarative rules that drive the DataCleaner.

    Attributes
    ----------
    normalize_columns : bool
        Lowercase all column names and replace spaces / special characters
        with underscores.  Defaults to True.

    drop_duplicates : bool
        Remove duplicate rows.  Defaults to True.

    duplicate_subset : list[str] or None
        Columns to consider when identifying duplicates.
        None means all columns.

    fill_missing : dict[str, str | int | float]
        Maps column name → fill strategy or constant value.

        Strategies:
          - ``"mean"``    fill with column mean  (numeric only)
          - ``"median"``  fill with column median (numeric only)
          - ``"mode"``    fill with most frequent value
          - ``"drop"``    drop rows where this column is null
          - anything else is used as a literal fill value
                          e.g. ``{"region": "Unknown", "revenue": 0}``

    type_conversions : dict[str, str]
        Maps column name → target dtype string.

        Supported types:
          - ``"datetime"``  pandas to_datetime (coerce on error)
          - ``"numeric"``   pandas to_numeric  (coerce on error)
          - ``"int"``       integer (nullable Int64 to survive NaNs)
          - ``"float"``     float64
          - ``"str"``       object / string

    drop_columns : list[str]
        Columns to remove entirely before any other step.

    Examples
    --------
    >>> config = CleaningConfig(
    ...     normalize_columns=True,
    ...     drop_duplicates=True,
    ...     fill_missing={"region": "Unknown", "revenue": "median"},
    ...     type_conversions={"date": "datetime", "quantity": "int"},
    ... )
    """

    normalize_columns: bool = True
    drop_duplicates: bool = True
    duplicate_subset: Optional[List[str]] = None
    fill_missing: Dict[str, Any] = field(default_factory=dict)
    type_conversions: Dict[str, str] = field(default_factory=dict)
    drop_columns: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Alternative constructors                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: dict) -> "CleaningConfig":
        """Build a CleaningConfig from a plain dictionary.

        Unknown keys are silently ignored so configs can evolve without
        breaking older code.
        """
        known = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "CleaningConfig":
        """Load cleaning rules from a JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON config file.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: str) -> "CleaningConfig":
        """Load cleaning rules from a YAML file.

        Requires PyYAML:  pip install pyyaml

        Parameters
        ----------
        path : str
            Path to the YAML config file.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML config loading. "
                "Install it with:  pip install pyyaml"
            ) from exc

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    # ------------------------------------------------------------------ #
    # Convenience                                                         #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Serialise this config back to a plain dictionary."""
        return {
            "normalize_columns": self.normalize_columns,
            "drop_duplicates": self.drop_duplicates,
            "duplicate_subset": self.duplicate_subset,
            "fill_missing": self.fill_missing,
            "type_conversions": self.type_conversions,
            "drop_columns": self.drop_columns,
        }

    def to_json(self, path: str) -> None:
        """Persist this config as a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def __repr__(self) -> str:
        return (
            f"CleaningConfig("
            f"normalize_columns={self.normalize_columns}, "
            f"drop_duplicates={self.drop_duplicates}, "
            f"fill_missing={list(self.fill_missing.keys())}, "
            f"type_conversions={list(self.type_conversions.keys())}, "
            f"drop_columns={self.drop_columns})"
        )
