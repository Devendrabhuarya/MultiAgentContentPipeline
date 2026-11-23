"""
Critic Agent.

The most important evaluation component: scores the complete draft across
8 dimensions (factuality, relevance, quality, structure, readability, SEO,
evidence, requirement compliance) plus an overall_score, and produces
strengths/weaknesses/critical vs minor issues/revision recommendations.
This is the single evaluation the Quality Decision node acts on.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.logging_config import log_timing
from app.models.schemas import (
    CriticResult,
    DraftContent,
    FactCheckResult,
    ResearchResult,
    SEOResult,
    UserRequest,
)

_SYSTEM_PROMPT = """You are the Critic Agent -- the most important evaluator in \
this pipeline. Holistically assess the draft article across exactly these \
dimensions, each scored 0-100: factuality, relevance, quality, structure, \
readability, seo, evidence, requirement_compliance. Weigh the Fact Checker's \
findings heavily for `factuality` and the SEO Agent's score heavily for `seo`. \
Compute overall_score as a thoughtful weighted synthesis (factuality and \
requirement_compliance matter most; treat unresolved critical fact issues as a \
hard cap preventing a high overall_score). Be a genuinely critical, precise \
reviewer, not a rubber stamp -- if this is the first draft, expect real room \
for improvement unless it is truly excellent. List concrete strengths, \
weaknesses, critical_issues (must-fix), minor_issues (nice-to-fix), and \
specific revision_recommendations."""


class CriticAgent(BaseAgent):
    name = "Critic"

    def run(
        self,
        draft: DraftContent,
        request: UserRequest,
        research: ResearchResult,
        fact_check: FactCheckResult,
        seo: SEOResult,
    ) -> CriticResult:
        with log_timing(self.name) as note:
            user_prompt = (
                f"Draft:\nTitle: {draft.title}\n{draft.body_markdown}\n\n"
                f"User requirements: content_type={request.content_type.value}, "
                f"tone={request.tone}, audience={request.target_audience}, "
                f"word_count_target={request.word_count}, actual_word_count={draft.word_count}, "
                f"keywords={request.keywords}, include_references={request.include_references}\n\n"
                f"Fact-check summary: {fact_check.supported}/{fact_check.claims_checked} "
                f"claims supported, issues={[i.claim for i in fact_check.issues]}\n"
                f"SEO score: {seo.seo_score}, recommendations={seo.recommendations}\n"
                f"Research evidence available: {research.key_points}\n\n"
                "Return a complete CriticResult with all 8 dimension scores, "
                "overall_score, strengths, weaknesses, critical_issues, "
                "minor_issues, and revision_recommendations."
            )
            result = self.llm.invoke_structured(_SYSTEM_PROMPT, user_prompt, CriticResult)
            note(f"Overall Score: {result.overall_score}")
            return result
