"""
SEO Agent.

Responsibility: evaluate the draft's SEO quality (title, headings, keyword
placement, readability, meta description, structure, search-intent
alignment) and return a score + actionable recommendations. It must NOT
rewrite the article -- only critique it, in the same spirit as the Fact
Checker and Critic agents.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.logging_config import log_timing
from app.models.schemas import DraftContent, SEOResult, UserRequest

_SYSTEM_PROMPT = """You are an SEO Agent. Evaluate the draft article for search \
engine optimization quality: title strength, heading structure, keyword \
relevance and placement, readability, meta description quality, content \
structure, and alignment with likely search intent for the topic. Do NOT \
rewrite the article. Return a numeric seo_score (0-100), detected and missing \
keywords, brief assessments, and concrete, actionable recommendations."""


class SEOAgent(BaseAgent):
    name = "SEO"

    def run(self, draft: DraftContent, request: UserRequest) -> SEOResult:
        with log_timing(self.name) as note:
            user_prompt = (
                f"Title: {draft.title}\n"
                f"Meta description: {draft.meta_description}\n"
                f"Body:\n{draft.body_markdown}\n\n"
                f"Target keywords: {request.keywords}\n"
                f"Topic: {request.topic}\n"
                f"Content type: {request.content_type.value}\n\n"
                "Return an SEOResult: seo_score, detected_keywords, "
                "missing_keywords, title_assessment, heading_assessment, "
                "readability_assessment, meta_description_assessment, and "
                "recommendations (a list of specific, actionable fixes)."
            )
            result = self.llm.invoke_structured(_SYSTEM_PROMPT, user_prompt, SEOResult)
            note(f"Score: {result.seo_score}")
            return result
