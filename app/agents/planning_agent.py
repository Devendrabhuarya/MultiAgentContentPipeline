"""
Planning Agent.

Responsibility: turn ResearchResult + UserRequest into a concrete
ContentPlan -- title, section structure, evidence-to-section mapping,
citation requirements, and per-section word-count budget that sums to
roughly the user's requested word count.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.logging_config import log_timing
from app.models.schemas import ContentPlan, ResearchResult, UserRequest

_SYSTEM_PROMPT = """You are a Planning Agent in a multi-agent content pipeline. \
Given research notes and the user's content requirements, design a detailed, \
logically ordered content outline. Map research evidence to the sections that \
will use it, mark which sections require citations, and give each section a \
realistic estimated word count so the sections sum close to the requested \
total word count. Respect the requested tone and audience."""


class PlanningAgent(BaseAgent):
    name = "Planning"

    def run(self, request: UserRequest, research: ResearchResult) -> ContentPlan:
        with log_timing(self.name):
            research_summary = (
                f"Key points: {research.key_points}\n"
                f"Facts: {research.facts}\n"
                f"Important claims: {research.important_claims}\n"
                f"Uncertainties: {research.uncertainties}\n"
            )
            user_prompt = (
                f"Topic: {request.topic}\n"
                f"Content type: {request.content_type.value}\n"
                f"Target audience: {request.target_audience}\n"
                f"Tone: {request.tone}\n"
                f"Target word count: {request.word_count}\n"
                f"Include references: {request.include_references}\n\n"
                f"Research notes:\n{research_summary}\n\n"
                "Produce a ContentPlan: title, introduction_plan, sections (each with "
                "heading, subsections, key_points_to_cover, evidence_refs, "
                "estimated_word_count, requires_citation), conclusion_plan, "
                "evidence_mapping, and estimated_total_word_count."
            )
            plan = self.llm.invoke_structured(_SYSTEM_PROMPT, user_prompt, ContentPlan)
            if not plan.estimated_total_word_count:
                plan.estimated_total_word_count = sum(
                    s.estimated_word_count for s in plan.sections
                )
            return plan
