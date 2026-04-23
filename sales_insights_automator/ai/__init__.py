"""
AI insight generation layer for the Sales Insights Automator.
"""

from ai.insight_generator import InsightGenerator, InsightReport
from ai.prompt_builder import PromptBuilder, PromptPayload
from ai.llm_client import LLMClient, LLMResponse, LLMClientError

__all__ = [
    "InsightGenerator",
    "InsightReport",
    "PromptBuilder",
    "PromptPayload",
    "LLMClient",
    "LLMResponse",
    "LLMClientError",
]
