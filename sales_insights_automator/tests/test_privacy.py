"""
Unit tests for the privacy layer.

Tests:
  - TestPrivacyConfig    (config.py)
  - TestDataAnonymizer   (anonymizer.py)
  - TestAuditLogger      (audit_log.py)
  - TestPrivacyIntegration (end-to-end with InsightGenerator)

Run with:
    pytest tests/test_privacy.py -v
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from privacy.config import PrivacyConfig
from privacy.anonymizer import DataAnonymizer
from privacy.audit_log import AuditLogger
from analysis.insight_builder import AnalysisResult
from analysis.analyzer import SalesAnalyzer
from ai.insight_generator import InsightGenerator
from ai.llm_client import LLMResponse


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def analysis_result() -> AnalysisResult:
    import random
    random.seed(7)
    rows = []
    for i in range(1, 21):
        rows.append({
            "order_id":    f"ORD-{i:03d}",
            "date":        pd.Timestamp(date(2024, 1, 1) + timedelta(days=random.randint(0, 364))),
            "product":     random.choice(["Laptop Pro 15", "Keyboard", "Monitor 27in"]),
            "category":    random.choice(["Computers", "Peripherals", "Displays"]),
            "region":      random.choice(["North", "South", "East", "West"]),
            "sales_rep":   random.choice(["Alice Martin", "Bob Chen", "Carla Diaz"]),
            "quantity":    random.randint(1, 5),
            "unit_price":  random.choice([1299.0, 89.99, 449.0]),
            "discount_pct":round(random.uniform(0, 0.1), 2),
            "revenue":     round(random.uniform(90, 5000), 2),
        })
    return SalesAnalyzer().analyze(pd.DataFrame(rows))


# ── TestPrivacyConfig ─────────────────────────────────────────────────────────

class TestPrivacyConfig:

    def test_default_masks_rep_names(self):
        assert PrivacyConfig().mask_rep_names is True

    def test_default_rounds_revenue(self):
        assert PrivacyConfig().round_revenue_to == 1000

    def test_default_requests_no_training(self):
        assert PrivacyConfig().request_no_training is True

    def test_default_enables_audit_log(self):
        assert PrivacyConfig().enable_audit_log is True

    def test_maximum_masks_everything(self):
        config = PrivacyConfig.maximum()
        assert config.mask_rep_names     is True
        assert config.mask_product_names is True
        assert config.mask_region_names  is True
        assert config.strip_exact_dates  is True

    def test_minimum_masks_nothing(self):
        config = PrivacyConfig.minimum()
        assert config.mask_rep_names     is False
        assert config.mask_product_names is False
        assert config.mask_region_names  is False
        assert config.round_revenue_to   == 0

    def test_minimum_still_requests_no_training(self):
        assert PrivacyConfig.minimum().request_no_training is True

    def test_from_dict(self):
        config = PrivacyConfig.from_dict({"mask_rep_names": False, "round_revenue_to": 500})
        assert config.mask_rep_names   is False
        assert config.round_revenue_to == 500

    def test_from_json_roundtrip(self, tmp_path):
        config = PrivacyConfig.maximum()
        path = str(tmp_path / "privacy.json")
        config.to_json(path)
        loaded = PrivacyConfig.from_json(path)
        assert loaded.mask_rep_names     == config.mask_rep_names
        assert loaded.mask_product_names == config.mask_product_names

    def test_summary_describes_active_rules(self):
        config = PrivacyConfig(mask_rep_names=True, round_revenue_to=1000)
        assert "rep names masked" in config.summary()
        assert "revenue rounded" in config.summary()

    def test_summary_empty_when_nothing_active(self):
        config = PrivacyConfig.minimum()
        config.request_no_training = False
        config.enable_audit_log    = False
        assert config.summary() == "no protections active"


# ── TestDataAnonymizer ────────────────────────────────────────────────────────

class TestDataAnonymizer:

    def test_rep_names_replaced(self, analysis_result):
        config = PrivacyConfig(mask_rep_names=True)
        anon   = DataAnonymizer(config)
        safe   = anon.anonymize(analysis_result)
        reps   = safe.revenue_by_sales_rep["sales_rep"].tolist()
        for rep in reps:
            assert rep.startswith("Sales Rep ")

    def test_real_rep_names_not_in_safe_result(self, analysis_result):
        config = PrivacyConfig(mask_rep_names=True)
        anon   = DataAnonymizer(config)
        safe   = anon.anonymize(analysis_result)
        summary = safe.text_summary()
        assert "Alice Martin" not in summary
        assert "Bob Chen"     not in summary
        assert "Carla Diaz"   not in summary

    def test_product_names_replaced_when_enabled(self, analysis_result):
        config = PrivacyConfig(mask_rep_names=False, mask_product_names=True)
        anon   = DataAnonymizer(config)
        safe   = anon.anonymize(analysis_result)
        products = safe.revenue_by_product["product"].tolist()
        for p in products:
            assert p.startswith("Product ")

    def test_product_names_retained_when_disabled(self, analysis_result):
        config = PrivacyConfig(mask_product_names=False)
        anon   = DataAnonymizer(config)
        safe   = anon.anonymize(analysis_result)
        summary = safe.text_summary()
        assert "Laptop Pro 15" in summary or "Keyboard" in summary

    def test_region_names_replaced_when_enabled(self, analysis_result):
        config = PrivacyConfig(mask_region_names=True)
        anon   = DataAnonymizer(config)
        safe   = anon.anonymize(analysis_result)
        regions = safe.revenue_by_region["region"].tolist()
        for r in regions:
            assert r.startswith("Region ")

    def test_revenue_rounded(self, analysis_result):
        config = PrivacyConfig(mask_rep_names=False, round_revenue_to=1000)
        anon   = DataAnonymizer(config)
        safe   = anon.anonymize(analysis_result)
        for rev in safe.revenue_by_region["total_revenue"]:
            assert rev % 1000 == 0

    def test_original_not_mutated(self, analysis_result):
        original_reps = analysis_result.revenue_by_sales_rep["sales_rep"].tolist()
        config = PrivacyConfig.maximum()
        anon   = DataAnonymizer(config)
        anon.anonymize(analysis_result)
        assert analysis_result.revenue_by_sales_rep["sales_rep"].tolist() == original_reps

    def test_mappings_populated_after_anonymize(self, analysis_result):
        config = PrivacyConfig(mask_rep_names=True)
        anon   = DataAnonymizer(config)
        anon.anonymize(analysis_result)
        assert len(anon.mappings) > 0
        assert "Alice Martin" in anon.mappings or "Bob Chen" in anon.mappings

    def test_dates_stripped_when_enabled(self, analysis_result):
        config = PrivacyConfig(strip_exact_dates=True)
        anon   = DataAnonymizer(config)
        safe   = anon.anonymize(analysis_result)
        assert safe.date_range["from"] == "Period Start"
        assert safe.date_range["to"]   == "Period End"

    def test_anonymize_text_replaces_known_values(self, analysis_result):
        config = PrivacyConfig(mask_rep_names=True)
        anon   = DataAnonymizer(config)
        anon.anonymize(analysis_result)
        text   = "Top rep: Alice Martin generated $50,000"
        result = anon.anonymize_text(text)
        assert "Alice Martin" not in result
        assert "Sales Rep" in result

    def test_masked_fields_reflects_config(self):
        config = PrivacyConfig(mask_rep_names=True, mask_product_names=True)
        anon   = DataAnonymizer(config)
        assert "sales_rep" in anon.masked_fields
        assert "product"   in anon.masked_fields

    def test_index_to_letter(self):
        assert DataAnonymizer._index_to_letter(0)  == "A"
        assert DataAnonymizer._index_to_letter(25) == "Z"
        assert DataAnonymizer._index_to_letter(26) == "AA"


# ── TestAuditLogger ───────────────────────────────────────────────────────────

class TestAuditLogger:

    def test_log_creates_file(self, tmp_path):
        logger = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        logger.log_api_call(
            template="full_report", model="gpt-4o-mini",
            prompt_tokens=800, completion_tokens=300,
            row_count=500, date_range={"from": "2024-01-01", "to": "2024-12-30"},
            masked_fields=["sales_rep"], latency_ms=1200.0,
            privacy_summary="rep names masked",
        )
        assert (tmp_path / "audit.jsonl").exists()

    def test_log_entry_is_valid_json(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=path)
        logger.log_api_call(
            template="full_report", model="gpt-4o-mini",
            prompt_tokens=800, completion_tokens=300,
            row_count=500, date_range={"from": "2024-01-01", "to": "2024-12-30"},
            masked_fields=["sales_rep"], latency_ms=1200.0,
            privacy_summary="rep names masked",
        )
        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["template"]       == "full_report"
        assert entry["content_logged"] is False

    def test_log_never_stores_content(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=path)
        logger.log_api_call(
            template="full_report", model="gpt-4o-mini",
            prompt_tokens=800, completion_tokens=300,
            row_count=500, date_range={},
            masked_fields=[], latency_ms=500.0,
            privacy_summary="",
        )
        raw = (tmp_path / "audit.jsonl").read_text()
        assert "prompt" not in raw.lower() or "prompt_tokens" in raw
        assert "insights" not in raw

    def test_read_log_returns_list(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=path)
        logger.log_api_call(
            template="full_report", model="gpt-4o-mini",
            prompt_tokens=100, completion_tokens=50,
            row_count=100, date_range={},
            masked_fields=[], latency_ms=200.0,
            privacy_summary="",
        )
        entries = logger.read_log()
        assert isinstance(entries, list)
        assert len(entries) == 1

    def test_read_log_returns_empty_for_missing_file(self, tmp_path):
        logger = AuditLogger(log_path=str(tmp_path / "nonexistent.jsonl"))
        assert logger.read_log() == []

    def test_multiple_entries_appended(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=path)
        for _ in range(3):
            logger.log_api_call(
                template="full_report", model="gpt-4o-mini",
                prompt_tokens=100, completion_tokens=50,
                row_count=100, date_range={},
                masked_fields=[], latency_ms=200.0,
                privacy_summary="",
            )
        assert len(logger.read_log()) == 3


# ── TestPrivacyIntegration ────────────────────────────────────────────────────

class TestPrivacyIntegration:
    """End-to-end: InsightGenerator uses privacy layer correctly."""

    @pytest.fixture()
    def mock_llm_response(self):
        return MagicMock(spec=LLMResponse,
            content="Strong Q4 performance driven by Furniture category.",
            model="gpt-4o-mini", prompt_tokens=900, completion_tokens=350,
            total_tokens=1250, finish_reason="stop", latency_ms=1300.0,
            was_truncated=False,
        )

    def test_rep_names_not_sent_to_llm(self, analysis_result, mock_llm_response):
        mock_client = MagicMock()
        mock_client.validate.return_value = True
        mock_client.model                 = "gpt-4o-mini"
        mock_client.complete.return_value = mock_llm_response

        config    = PrivacyConfig(mask_rep_names=True)
        generator = InsightGenerator(client=mock_client, privacy_config=config)
        generator.generate(analysis_result)

        call_args  = mock_client.complete.call_args
        user_prompt = call_args[0][1] if call_args[0] else call_args[1]["user_prompt"]
        assert "Alice Martin" not in user_prompt
        assert "Bob Chen"     not in user_prompt

    def test_privacy_instruction_in_system_prompt(self, analysis_result, mock_llm_response):
        mock_client = MagicMock()
        mock_client.validate.return_value = True
        mock_client.model                 = "gpt-4o-mini"
        mock_client.complete.return_value = mock_llm_response

        config    = PrivacyConfig(request_no_training=True)
        generator = InsightGenerator(client=mock_client, privacy_config=config)
        generator.generate(analysis_result)

        call_args     = mock_client.complete.call_args
        system_prompt = call_args[0][0] if call_args[0] else call_args[1]["system_prompt"]
        assert "CONFIDENTIALITY" in system_prompt

    def test_audit_log_written_on_live_call(self, analysis_result, mock_llm_response, tmp_path):
        mock_client = MagicMock()
        mock_client.validate.return_value = True
        mock_client.model                 = "gpt-4o-mini"
        mock_client.complete.return_value = mock_llm_response

        config    = PrivacyConfig(enable_audit_log=True)
        generator = InsightGenerator(client=mock_client, privacy_config=config)
        generator.audit_logger = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        generator.generate(analysis_result)

        entries = generator.audit_logger.read_log()
        assert len(entries) == 1
        assert entries[0]["content_logged"] is False

    def test_dry_run_also_writes_audit_log(self, analysis_result, tmp_path):
        config    = PrivacyConfig(enable_audit_log=True)
        generator = InsightGenerator(privacy_config=config, dry_run=True)
        generator.audit_logger = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
        generator.generate(analysis_result)

        entries = generator.audit_logger.read_log()
        assert len(entries) == 1
        assert entries[0]["event"] == "dry_run"

    def test_no_audit_when_disabled(self, analysis_result, tmp_path):
        config    = PrivacyConfig(enable_audit_log=False)
        generator = InsightGenerator(privacy_config=config, dry_run=True)
        log_path  = str(tmp_path / "audit.jsonl")
        generator.audit_logger = AuditLogger(log_path=log_path)
        generator.generate(analysis_result)
        assert not Path(log_path).exists()
