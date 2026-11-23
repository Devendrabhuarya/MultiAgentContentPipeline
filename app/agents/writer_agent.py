"""
Writer Agent.

Responsibility: generate the article from the ContentPlan + ResearchResult
(DRAFT mode), or improve an existing draft using targeted feedback
(REWRITE mode). Rewrite mode is deliberately given the *previous draft* so
the model edits/improves it rather than regenerating an unrelated article
from scratch -- this is what makes the feedback loop actually converge.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.logging_config import log_timing
from app.models.schemas import (
    ContentPlan,
    CriticResult,
    DraftContent,
    FactCheckResult,
    FeedbackResult,
    ResearchResult,
    SEOResult,
    UserRequest,
)

_DRAFT_SYSTEM_PROMPT = """You are a skilled Writer Agent in a multi-agent content \
pipeline. Write the full article strictly following the given outline. Maintain \
the requested tone, stay within the target word count, avoid unsupported claims \
not present in the research evidence, and include citations/references where \
appropriate. Write in Markdown."""

_REWRITE_SYSTEM_PROMPT = """You are a skilled Writer Agent operating in REWRITE \
mode. You are given a previous draft and specific, prioritized feedback from a \
Critic, a Fact Checker, and an SEO Agent. Improve the EXISTING draft by directly \
addressing every piece of feedback -- do not discard it and write an unrelated \
new article. Preserve what already works. Write in Markdown."""


class WriterAgent(BaseAgent):
    name = "Writer"

    def write_draft(
        self,
        request: UserRequest,
        research: ResearchResult,
        plan: ContentPlan,
        version: int = 1,
    ) -> DraftContent:
        with log_timing(self.name, action="Draft generation"):
            outline_text = "\n".join(
                f"- {s.heading}: {', '.join(s.key_points_to_cover)} "
                f"(~{s.estimated_word_count} words, citation={s.requires_citation})"
                for s in plan.sections
            )
            user_prompt = (
                f"Title: {plan.title}\n"
                f"Introduction plan: {plan.introduction_plan}\n"
                f"Sections:\n{outline_text}\n"
                f"Conclusion plan: {plan.conclusion_plan}\n\n"
                f"Facts available: {research.facts}\n"
                f"Key points: {research.key_points}\n"
                f"Sources: {[s.title for s in research.sources]}\n\n"
                f"Target word count: {request.word_count}\n"
                f"Tone: {request.tone}\n"
                f"Audience: {request.target_audience}\n"
                f"Keywords to naturally include: {request.keywords}\n"
                f"Include references section: {request.include_references}\n\n"
                "Produce a DraftContent with title, body_markdown, meta_description, "
                "word_count, and references."
            )
            draft = self.llm.invoke_structured(_DRAFT_SYSTEM_PROMPT, user_prompt, DraftContent)
            draft.version = version
            draft.word_count = draft.word_count or _word_count(draft.body_markdown)
            return draft

    def rewrite(
        self,
        request: UserRequest,
        previous_draft: DraftContent,
        critic: CriticResult,
        fact_check: FactCheckResult,
        seo: SEOResult,
        feedback: FeedbackResult,
        version: int,
    ) -> DraftContent:
        with log_timing(self.name, action=f"Revision {version - 1}"):
            user_prompt = (
                f"Previous draft (version {previous_draft.version}):\n"
                f"Title: {previous_draft.title}\n"
                f"{previous_draft.body_markdown}\n\n"
                f"--- FEEDBACK TO ADDRESS (priority order) ---\n"
                + "\n".join(f"{i+1}. {p}" for i, p in enumerate(feedback.priority))
                + "\n\nDetailed instructions:\n"
                + "\n".join(f"- {ins}" for ins in feedback.instructions)
                + f"\n\nCritic critical issues: {critic.critical_issues}\n"
                f"Critic weaknesses: {critic.weaknesses}\n"
                f"Fact-check issues: {[i.claim + ' -> ' + i.status for i in fact_check.issues]}\n"
                f"SEO recommendations: {seo.recommendations}\n\n"
                f"Target word count: {request.word_count}\n"
                f"Tone: {request.tone}\n\n"
                "Produce an improved DraftContent (title, body_markdown, "
                "meta_description, word_count, references) that fixes every issue above."
            )
            draft = self.llm.invoke_structured(_REWRITE_SYSTEM_PROMPT, user_prompt, DraftContent)
            draft.version = version
            draft.word_count = draft.word_count or _word_count(draft.body_markdown)
            return draft


def _word_count(markdown_text: str) -> int:
    return len(markdown_text.split())
