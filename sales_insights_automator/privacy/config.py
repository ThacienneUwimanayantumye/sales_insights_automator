"""
PrivacyConfig — declarative privacy rules for the AI layer.

Controls exactly what data is allowed to leave the local machine and
be sent to the OpenAI API. Every field has a privacy-safe default so
the system is secure out of the box without any configuration.

The config can be loaded from a JSON file so privacy rules can be
adjusted by a compliance team without touching Python code:

    config = PrivacyConfig.from_json("config/privacy_config.json")

Fields
------
mask_rep_names : bool
    Replace real sales rep names (e.g. "Alice Martin") with generic
    labels ("Sales Rep A", "Sales Rep B", ...) before sending to the AI.
    Protects employee identity. Default: True.

mask_product_names : bool
    Replace product names with generic labels ("Product A", ...).
    Enable if product names are commercially sensitive. Default: False.

mask_region_names : bool
    Replace region names with generic labels ("Region A", ...).
    Enable if regional data reveals sensitive business structure. Default: False.

round_revenue_to : int
    Round all revenue figures to the nearest N before sending.
    E.g. 1000 turns $956,745 → $957,000, reducing exact precision.
    0 = no rounding. Default: 1000.

request_no_training : bool
    Prepend a data-use instruction to the system prompt asking the model
    not to use this data for training or any purpose beyond the current
    request. Default: True.

strip_exact_dates : bool
    Replace exact date values with relative labels ("Month 1", "Month 2")
    so the precise business period is not disclosed. Default: False.

enable_audit_log : bool
    Write a metadata-only record of every API call to data/audit/.
    The log never contains prompt content — only shape, tokens, and
    which fields were masked. Default: True.
"""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PrivacyConfig:
    """Declarative privacy rules applied before any data leaves the system."""

    mask_rep_names:      bool = True
    mask_product_names:  bool = False
    mask_region_names:   bool = False
    round_revenue_to:    int  = 1000
    request_no_training: bool = True
    strip_exact_dates:   bool = False
    enable_audit_log:    bool = True

    # ------------------------------------------------------------------ #
    # Constructors                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: dict) -> "PrivacyConfig":
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, path: str) -> "PrivacyConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def maximum(cls) -> "PrivacyConfig":
        """Return a config with every privacy protection enabled."""
        return cls(
            mask_rep_names      = True,
            mask_product_names  = True,
            mask_region_names   = True,
            round_revenue_to    = 1000,
            request_no_training = True,
            strip_exact_dates   = True,
            enable_audit_log    = True,
        )

    @classmethod
    def minimum(cls) -> "PrivacyConfig":
        """Return a config with no masking (use only for local/dev data)."""
        return cls(
            mask_rep_names      = False,
            mask_product_names  = False,
            mask_region_names   = False,
            round_revenue_to    = 0,
            request_no_training = True,   # always request no training
            strip_exact_dates   = False,
            enable_audit_log    = True,   # always audit
        )

    # ------------------------------------------------------------------ #
    # Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def summary(self) -> str:
        """One-line human-readable summary of active protections."""
        active = []
        if self.mask_rep_names:     active.append("rep names masked")
        if self.mask_product_names: active.append("product names masked")
        if self.mask_region_names:  active.append("region names masked")
        if self.round_revenue_to:   active.append(f"revenue rounded to {self.round_revenue_to}")
        if self.request_no_training:active.append("no-training requested")
        if self.strip_exact_dates:  active.append("dates stripped")
        if self.enable_audit_log:   active.append("audit logging on")
        return " | ".join(active) if active else "no protections active"

    def __repr__(self) -> str:
        return f"PrivacyConfig({self.summary()})"
