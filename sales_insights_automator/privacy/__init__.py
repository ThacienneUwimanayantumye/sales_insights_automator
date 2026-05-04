"""
Privacy layer for the Sales Insights Automator.

Ensures that sensitive business data is anonymised before leaving
the local machine and that every AI API call is audited.
"""

from privacy.config import PrivacyConfig
from privacy.anonymizer import DataAnonymizer
from privacy.audit_log import AuditLogger

__all__ = ["PrivacyConfig", "DataAnonymizer", "AuditLogger"]
