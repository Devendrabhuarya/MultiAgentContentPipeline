"""
Research Agent.

Responsibility: understand the topic, gather source evidence via a
ResearchProvider, and use the LLM to synthesize that raw evidence into
structured research notes (key points, facts, claims, uncertainties).

Explicitly does NOT generate final article content -- see schemas.py:
ResearchResult has no "content"/"draft" field, only structured evidence.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.logging_config import log_timing
from app.models.schemas import ResearchResult, UserRequest
from app.research.base import ResearchProvider

_SYSTEM_PROMPT = """You are a meticulous Research Agent in a multi-agent content \
pipeline. Your job is ONLY to research and organize evidence -- you must NEVER \
write final article prose. Given a topic, target audience, and a set of raw \
source snippets, produce structured research notes: the most important \
subtopics, key points, concrete facts, important claims worth citing, and any \
claims that are uncertain or controversial and should be handled carefully by \
downstream writers."""


class ResearchAgent(BaseAgent):
    name = "Research"

    def __init__(self, llm_client, research_provider: ResearchProvider):
        super().__init__(llm_client)
        self.research_provider = research_provider

    def run(self, request: UserRequest) -> ResearchResult:
        with log_timing(self.name) as note:
            queries = self._build_queries(request)
            sources = self.research_provider.search_many(queries, max_results_per_query=3)
            note(f"Retrieved {len(sources)} candidate sources")

            source_text = "\n".join(
                f"- {s.title} ({s.url or 'no url'}): {s.snippet or ''}" for s in sources
            ) or "No external sources were retrieved; rely on general domain knowledge."

            user_prompt = (
                f"Topic: {request.topic}\n"
                f"Target audience: {request.target_audience}\n"
                f"Content type: {request.content_type.value}\n\n"
                f"Raw source snippets:\n{source_text}\n\n"
                "Produce structured research notes covering: subtopics, key_points, "
                "facts, important_claims, and uncertainties. Populate `sources` with "
                "the most relevant snippets above (reuse their titles/urls)."
            )

            result = self.llm.invoke_structured(_SYSTEM_PROMPT, user_prompt, ResearchResult)

            # Guarantee provenance even if the LLM didn't echo every source back.
            if not result.sources and sources:
                result.sources = sources
            result.topic = result.topic or request.topic
            return result

    @staticmethod
    def _build_queries(request: UserRequest) -> list[str]:
        base = [request.topic]
        for kw in request.keywords[:3]:
            base.append(f"{request.topic} {kw}")
        return base
