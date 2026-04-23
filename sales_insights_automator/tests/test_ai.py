"""
Unit tests for the AI layer.

The OpenAI client is always mocked — no real API calls are made.

Test groups:
  - TestPromptBuilder       (prompt_builder.py)
  - TestLLMClient           (llm_client.py)
  - TestInsightReport       (insight_generator.py — dataclass)
  - TestInsightGenerator    (insight_generator.py — orchestrator)

Run with:
    pytest tests/test_ai.py -v
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.prompt_builder import PromptBuilder, PromptPayload, TEMPLATES
from ai.llm_client import LLMClient, LLMResponse, LLMClientError
from ai.insight_generator import InsightGenerator, InsightReport
from analysis.insight_builder import AnalysisResult
from analysis.analyzer import SalesAnalyzer


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def analysis_result() -> AnalysisResult:
    """Minimal AnalysisResult built from a small synthetic DataFrame."""
    import random
    from datetime import date, timedelta
    random.seed(42)

    rows = []
    for i in range(1, 31):
        rows.append({
            "order_id":    f"ORD-{i:03d}",
            "date":        pd.Timestamp(date(2024, 1, 1) + timedelta(days=random.randint(0, 364))),
            "product":     random.choice(["Laptop", "Keyboard", "Monitor"]),
            "category":    random.choice(["Computers", "Peripherals", "Displays"]),
            "region":      random.choice(["North", "South", "East", "West"]),
            "sales_rep":   random.choice(["Alice", "Bob", "Carla"]),
            "quantity":    random.randint(1, 5),
            "unit_price":  random.choice([1299.0, 89.99, 449.0]),
            "discount_pct":round(random.uniform(0, 0.1), 2),
            "revenue":     round(random.uniform(90, 5000), 2),
        })

    df = pd.DataFrame(rows)
    return SalesAnalyzer().analyze(df)


@pytest.fixture()
def mock_llm_response() -> LLMResponse:
    """A realistic mock LLMResponse."""
    return LLMResponse(
        content           = "Overall revenue reached $956k driven by strong Furniture sales.",
        model             = "gpt-4o-mini",
        prompt_tokens     = 850,
        completion_tokens = 312,
        total_tokens      = 1162,
        finish_reason     = "stop",
        latency_ms        = 1240.5,
    )


# ── TestPromptBuilder ─────────────────────────────────────────────────────────

class TestPromptBuilder:

    def test_build_returns_prompt_payload(self, analysis_result):
        builder = PromptBuilder()
        payload = builder.build(analysis_result)
        assert isinstance(payload, PromptPayload)

    def test_all_templates_are_buildable(self, analysis_result):
        builder = PromptBuilder()
        for template in TEMPLATES:
            payload = builder.build(analysis_result, template=template)
            assert payload.template_name == template

    def test_invalid_template_raises(self, analysis_result):
        builder = PromptBuilder()
        with pytest.raises(ValueError, match="Unknown template"):
            builder.build(analysis_result, template="nonexistent")

    def test_system_prompt_is_non_empty(self, analysis_result):
        payload = PromptBuilder().build(analysis_result)
        assert len(payload.system) > 100

    def test_user_prompt_contains_data(self, analysis_result):
        payload = PromptBuilder().build(analysis_result)
        # The user prompt must contain the period and revenue data
        assert "SALES ANALYSIS REPORT" in payload.user

    def test_user_prompt_contains_focus_instruction(self, analysis_result):
        payload = PromptBuilder().build(analysis_result, template="recommendations")
        assert "recommendation" in payload.user.lower()

    def test_estimated_tokens_is_positive(self, analysis_result):
        payload = PromptBuilder().build(analysis_result)
        assert payload.estimated_tokens > 0

    def test_estimated_chars_matches_lengths(self, analysis_result):
        payload = PromptBuilder().build(analysis_result)
        assert payload.estimated_chars == len(payload.system) + len(payload.user)

    def test_context_truncated_when_too_long(self, analysis_result):
        builder = PromptBuilder(max_context_chars=100)
        payload = builder.build(analysis_result)
        assert "truncated" in payload.user

    def test_available_templates_returns_dict(self):
        templates = PromptBuilder.available_templates()
        assert isinstance(templates, dict)
        assert "full_report" in templates
        assert "recommendations" in templates

    def test_template_name_stored_on_payload(self, analysis_result):
        builder = PromptBuilder()
        payload = builder.build(analysis_result, template="anomalies")
        assert payload.template_name == "anomalies"


# ── TestLLMClient ─────────────────────────────────────────────────────────────

class TestLLMClient:

    def test_validate_returns_false_when_no_key(self):
        client = LLMClient(api_key="")
        assert client.validate() is False

    def test_validate_returns_false_for_placeholder_key(self):
        client = LLMClient(api_key="sk-...")
        assert client.validate() is False

    def test_validate_returns_true_for_real_looking_key(self):
        client = LLMClient(api_key="sk-realkey123456789abcdef")
        assert client.validate() is True

    def test_complete_returns_llm_response(self, mock_llm_response):
        client = LLMClient(api_key="sk-test-key")

        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content  = mock_llm_response.content
        mock_choice.finish_reason    = mock_llm_response.finish_reason

        mock_usage = MagicMock()
        mock_usage.prompt_tokens     = mock_llm_response.prompt_tokens
        mock_usage.completion_tokens = mock_llm_response.completion_tokens
        mock_usage.total_tokens      = mock_llm_response.total_tokens

        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]
        mock_api_response.usage   = mock_usage
        mock_api_response.model   = mock_llm_response.model

        mock_openai.chat.completions.create.return_value = mock_api_response
        client._client = mock_openai

        result = client.complete("system msg", "user msg")

        assert isinstance(result, LLMResponse)
        assert result.content           == mock_llm_response.content
        assert result.total_tokens      == mock_llm_response.total_tokens
        assert result.finish_reason     == "stop"

    def test_was_truncated_property(self):
        r = LLMResponse("text", "gpt-4o-mini", 100, 50, 150, "length", 500.0)
        assert r.was_truncated is True

    def test_not_truncated_when_stop(self):
        r = LLMResponse("text", "gpt-4o-mini", 100, 50, 150, "stop", 500.0)
        assert r.was_truncated is False

    def test_repr_contains_model(self):
        client = LLMClient(api_key="sk-test", model="gpt-4o-mini")
        assert "gpt-4o-mini" in repr(client)

    def test_complete_raises_on_import_error(self):
        """LLMClientError raised if openai package is not installed."""
        client = LLMClient(api_key="sk-test")
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(LLMClientError, match="not installed"):
                client._ensure_client()


# ── TestInsightReport ─────────────────────────────────────────────────────────

class TestInsightReport:

    @pytest.fixture()
    def report(self) -> InsightReport:
        return InsightReport(
            insights_text     = "Revenue was strong in Q4.",
            template_used     = "full_report",
            model             = "gpt-4o-mini",
            prompt_tokens     = 800,
            completion_tokens = 300,
            total_tokens      = 1100,
            latency_ms        = 1250.0,
            analysis_period   = {"from": "2024-01-01", "to": "2024-12-30"},
            row_count         = 500,
        )

    def test_to_markdown_is_string(self, report):
        md = report.to_markdown()
        assert isinstance(md, str)

    def test_to_markdown_contains_period(self, report):
        md = report.to_markdown()
        assert "2024-01-01" in md

    def test_to_markdown_contains_insights(self, report):
        md = report.to_markdown()
        assert "Revenue was strong in Q4." in md

    def test_to_markdown_shows_dry_run_note(self):
        r = InsightReport(insights_text="placeholder", is_dry_run=True)
        assert "dry run" in r.to_markdown().lower()

    def test_to_markdown_shows_truncation_warning(self):
        r = InsightReport(insights_text="text", was_truncated=True)
        assert "truncated" in r.to_markdown().lower()

    def test_to_dict_is_json_serialisable(self, report):
        d = report.to_dict()
        json.dumps(d)  # must not raise

    def test_to_dict_contains_insights_text(self, report):
        d = report.to_dict()
        assert d["insights_text"] == "Revenue was strong in Q4."

    def test_to_json_roundtrip(self, report, tmp_path):
        path = str(tmp_path / "report.json")
        report.to_json(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["total_tokens"] == 1100

    def test_repr_contains_template(self, report):
        assert "full_report" in repr(report)


# ── TestInsightGenerator ──────────────────────────────────────────────────────

class TestInsightGenerator:

    def test_dry_run_returns_insight_report(self, analysis_result):
        generator = InsightGenerator(dry_run=True)
        report    = generator.generate(analysis_result)
        assert isinstance(report, InsightReport)

    def test_dry_run_sets_flag(self, analysis_result):
        report = InsightGenerator(dry_run=True).generate(analysis_result)
        assert report.is_dry_run is True

    def test_dry_run_model_is_dry_run(self, analysis_result):
        report = InsightGenerator(dry_run=True).generate(analysis_result)
        assert report.model == "dry-run"

    def test_dry_run_prompt_appears_in_text(self, analysis_result):
        report = InsightGenerator(dry_run=True).generate(analysis_result)
        assert "DRY RUN" in report.insights_text

    def test_dry_run_all_templates(self, analysis_result):
        generator = InsightGenerator(dry_run=True)
        for template in PromptBuilder.available_templates():
            report = generator.generate(analysis_result, template=template)
            assert report.template_used == template

    def test_invalid_template_raises(self, analysis_result):
        generator = InsightGenerator(dry_run=True)
        with pytest.raises(ValueError):
            generator.generate(analysis_result, template="bad_template")

    def test_live_run_calls_llm_client(self, analysis_result, mock_llm_response):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.validate.return_value = True
        mock_client.model                 = "gpt-4o-mini"
        mock_client.complete.return_value = mock_llm_response

        generator = InsightGenerator(client=mock_client, dry_run=False)
        report    = generator.generate(analysis_result)

        mock_client.complete.assert_called_once()
        assert report.insights_text    == mock_llm_response.content
        assert report.total_tokens     == mock_llm_response.total_tokens
        assert report.is_dry_run is False

    def test_live_run_raises_without_api_key(self, analysis_result):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.validate.return_value = False

        generator = InsightGenerator(client=mock_client, dry_run=False)
        with pytest.raises(LLMClientError, match="API key"):
            generator.generate(analysis_result)

    def test_report_has_correct_period(self, analysis_result, mock_llm_response):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.validate.return_value = True
        mock_client.model                 = "gpt-4o-mini"
        mock_client.complete.return_value = mock_llm_response

        report = InsightGenerator(client=mock_client).generate(analysis_result)
        assert report.analysis_period == analysis_result.date_range

    def test_report_has_correct_row_count(self, analysis_result, mock_llm_response):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.validate.return_value = True
        mock_client.model                 = "gpt-4o-mini"
        mock_client.complete.return_value = mock_llm_response

        report = InsightGenerator(client=mock_client).generate(analysis_result)
        assert report.row_count == analysis_result.row_count

    def test_repr_contains_model(self):
        client = MagicMock(spec=LLMClient)
        client.model = "gpt-4o-mini"
        gen = InsightGenerator(client=client)
        assert "gpt-4o-mini" in repr(gen)
