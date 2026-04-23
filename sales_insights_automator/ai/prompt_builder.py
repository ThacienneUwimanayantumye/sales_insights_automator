"""
PromptBuilder — constructs structured prompts from an AnalysisResult.

Design philosophy
-----------------
The prompt is split into two parts, matching the OpenAI chat format:

  system  — defines the AI's role, output format, and constraints.
            This stays stable across calls and sets the "personality".
  user    — injects the actual data (from AnalysisResult.text_summary())
            and specifies what the AI should focus on for this run.

Multiple prompt *templates* are supported so callers can request
different types of output without changing any code:

  "executive_summary"  — C-suite narrative, 3–4 paragraphs
  "recommendations"    — 5 specific, prioritised action items
  "anomalies"          — focus on unusual patterns and what they signal
  "full_report"        — comprehensive analysis (default)

Adding a new template means adding one entry to TEMPLATES — nothing else
in the pipeline changes.

Usage
-----
    builder = PromptBuilder()
    payload = builder.build(result, template="recommendations")
    print(payload.system)
    print(payload.user)
"""

from dataclasses import dataclass
from typing import Dict

from analysis.insight_builder import AnalysisResult


# ── Prompt templates ──────────────────────────────────────────────────────────
# Each entry has a "focus" instruction that is appended to the user prompt.
# The system prompt is shared across all templates.

TEMPLATES: Dict[str, Dict[str, str]] = {
    "full_report": {
        "label": "Full Report",
        "focus": (
            "Provide a comprehensive business analysis covering:\n"
            "1. Overall performance summary (2–3 sentences)\n"
            "2. Top strengths — what is working well and why\n"
            "3. Key concerns or underperformance areas\n"
            "4. Notable trends (growth, decline, seasonality)\n"
            "5. Discount strategy assessment\n"
            "6. Top 3 prioritised, actionable recommendations\n\n"
            "Be specific — reference actual numbers from the data."
        ),
    },
    "executive_summary": {
        "label": "Executive Summary",
        "focus": (
            "Write a concise executive summary (3–4 paragraphs) for a non-technical "
            "business audience. Cover: overall revenue performance, the strongest and "
            "weakest areas, the most important trend, and one clear recommendation. "
            "Use plain business language. Do not use bullet points — write in prose."
        ),
    },
    "recommendations": {
        "label": "Action Recommendations",
        "focus": (
            "Identify and explain exactly 5 specific, actionable recommendations "
            "based on the data. For each recommendation:\n"
            "  - State the recommendation clearly in one sentence\n"
            "  - Cite the specific metric or pattern that justifies it\n"
            "  - Estimate the expected impact if acted upon\n\n"
            "Order them by potential business impact (highest first). "
            "Be direct and concrete — avoid generic advice."
        ),
    },
    "anomalies": {
        "label": "Anomaly & Pattern Analysis",
        "focus": (
            "Identify and explain the most significant anomalies, outliers, and "
            "unexpected patterns in the data. For each finding:\n"
            "  - Describe what the anomaly is and what makes it unusual\n"
            "  - Reference the specific numbers\n"
            "  - Suggest a likely business explanation\n"
            "  - Recommend how to investigate or respond\n\n"
            "Look at: month-over-month swings, regional imbalances, "
            "discount outliers, rep performance gaps, and day-of-week patterns."
        ),
    },
}

SYSTEM_PROMPT = """\
You are a senior data analyst and business intelligence expert with deep experience \
in B2B sales performance analysis.

Your role is to interpret quantitative sales data and translate it into clear, \
actionable business insights for company leadership.

Guidelines:
- Ground every statement in the data provided — never speculate without evidence
- Be specific: cite actual numbers, percentages, and time periods
- Use professional but accessible language — avoid jargon
- Be direct: state conclusions first, then support them
- Flag risks and opportunities with equal honesty
- Do not repeat raw tables back — interpret and synthesise them
"""


@dataclass
class PromptPayload:
    """A fully assembled prompt ready to send to the LLM.

    Attributes
    ----------
    system : str
        The system message that defines the AI's role.
    user : str
        The user message containing the data context and task.
    template_name : str
        Which template was used to build this payload.
    estimated_chars : int
        Rough total character count (system + user).
        Useful for estimating token cost before sending.
    """

    system:           str
    user:             str
    template_name:    str
    estimated_chars:  int = 0

    def __post_init__(self) -> None:
        self.estimated_chars = len(self.system) + len(self.user)

    @property
    def estimated_tokens(self) -> int:
        """Very rough token estimate: ~4 characters per token."""
        return self.estimated_chars // 4


class PromptBuilder:
    """Constructs PromptPayloads from an AnalysisResult.

    Parameters
    ----------
    max_context_chars : int
        Maximum characters allowed in the data context section of the
        user prompt.  Longer summaries are truncated with a note.
        Defaults to 12_000 (~3,000 tokens), comfortably under GPT-4o-mini's
        128k context window.

    Examples
    --------
    >>> builder = PromptBuilder()
    >>> payload = builder.build(result, template="full_report")
    >>> payload.estimated_tokens
    850
    """

    def __init__(self, max_context_chars: int = 12_000) -> None:
        self.max_context_chars = max_context_chars

    def build(
        self,
        result: AnalysisResult,
        template: str = "full_report",
    ) -> PromptPayload:
        """Build a PromptPayload for the given AnalysisResult.

        Parameters
        ----------
        result : AnalysisResult
            Output of ``SalesAnalyzer.analyze()``.
        template : str
            One of: ``"full_report"``, ``"executive_summary"``,
            ``"recommendations"``, ``"anomalies"``.
            Defaults to ``"full_report"``.

        Returns
        -------
        PromptPayload

        Raises
        ------
        ValueError
            If ``template`` is not a recognised template name.
        """
        if template not in TEMPLATES:
            valid = list(TEMPLATES.keys())
            raise ValueError(
                f"Unknown template '{template}'. Valid options: {valid}"
            )

        tmpl = TEMPLATES[template]
        data_context = self._build_data_context(result)
        user_prompt  = self._build_user_prompt(data_context, tmpl["focus"])

        return PromptPayload(
            system        = SYSTEM_PROMPT,
            user          = user_prompt,
            template_name = template,
        )

    @staticmethod
    def available_templates() -> Dict[str, str]:
        """Return a mapping of template name → human-readable label."""
        return {k: v["label"] for k, v in TEMPLATES.items()}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_data_context(self, result: AnalysisResult) -> str:
        """Extract the data context string from the result, truncating if needed."""
        context = result.text_summary()

        if len(context) > self.max_context_chars:
            context = (
                context[: self.max_context_chars]
                + "\n\n[... data truncated to fit context window ...]"
            )

        return context

    @staticmethod
    def _build_user_prompt(data_context: str, focus_instruction: str) -> str:
        """Assemble the final user message from data + focus instruction."""
        return (
            f"Below is the sales analysis data for the period under review.\n\n"
            f"{data_context}\n\n"
            f"---\n\n"
            f"Your task:\n{focus_instruction}"
        )
