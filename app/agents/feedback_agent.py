"""
Feedback Agent.

Responsibility: convert Critic + Fact Checker + SEO evaluation results
into a single, prioritized, actionable revision plan for the Writer Agent.
Critical issues (factual errors, missing requirements) must be prioritized
over cosmetic ones (minor phrasing, small SEO tweaks).
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.logging_config import log_timing
from app.models.schemas import (
    CriticResult,
    FactCheckResult,
    FeedbackResult,
    SEOResult,
    UserRequest,
)

_SYSTEM_PROMPT = """You are the Feedback Agent. You translate raw evaluation \
results from the Critic, Fact Checker, and SEO agents into a single, clear, \
prioritized revision plan the Writer Agent can act on directly. ALWAYS put \
critical issues (unsupported/contradicted claims, missing user requirements, \
major structural problems) before cosmetic issues (small SEO tweaks, minor \
phrasing) in the priority list. Each instruction should be specific and \
actionable, not vague ("add more detail" is bad; "add a supporting statistic \
to the paragraph about X, citing source Y" is good)."""


class FeedbackAgent(BaseAgent):
    name = "Feedback"

    def run(
        self,
        request: UserRequest,
        critic: CriticResult,
        fact_check: FactCheckResult,
        seo: SEOResult,
    ) -> FeedbackResult:
        with log_timing(self.name) as note:
            user_prompt = (
                f"Critic overall_score: {critic.overall_score}\n"
                f"Critic critical_issues: {critic.critical_issues}\n"
                f"Critic weaknesses: {critic.weaknesses}\n"
                f"Critic minor_issues: {critic.minor_issues}\n"
                f"Critic revision_recommendations: {critic.revision_recommendations}\n\n"
                f"Fact-check issues: "
                f"{[(i.claim, i.status, i.suggested_fix) for i in fact_check.issues]}\n\n"
                f"SEO recommendations: {seo.recommendations}\n"
                f"SEO missing_keywords: {seo.missing_keywords}\n\n"
                f"User requirements: word_count={request.word_count}, "
                f"tone={request.tone}, keywords={request.keywords}\n\n"
                "Return a FeedbackResult: `priority` (ordered list of the most "
                "important fixes, critical first), `instructions` (specific "
                "actionable guidance for the writer), `critical_fixes`, and "
                "`cosmetic_fixes`."
            )
            result = self.llm.invoke_structured(_SYSTEM_PROMPT, user_prompt, FeedbackResult)
            note(f"{len(result.priority)} issues identified")
            return result
