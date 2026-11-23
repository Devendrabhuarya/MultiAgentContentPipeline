"""
Fact Checker Agent.

Responsibility: extract factual claims from the draft, compare them
against the research evidence, and return structured counts + a list of
specific issues (unsupported / contradicted / uncertain / hallucinated
claims) with confidence scoring. Never returns free-form prose.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.logging_config import log_timing
from app.models.schemas import DraftContent, FactCheckResult, ResearchResult

_SYSTEM_PROMPT = """You are a rigorous Fact Checker Agent. Extract the factual \
claims made in the draft article and check each one against the provided \
research evidence (facts, key points, sources). Classify every claim as \
supported, unsupported, uncertain, or contradicted. Flag anything that looks \
fabricated or hallucinated -- i.e. not traceable to any provided evidence -- as \
an issue even if it sounds plausible. Be conservative: when evidence is thin, \
prefer 'uncertain' over 'supported'."""


class FactCheckerAgent(BaseAgent):
    name = "FactChecker"

    def run(self, draft: DraftContent, research: ResearchResult) -> FactCheckResult:
        with log_timing(self.name) as note:
            evidence_text = (
                f"Facts: {research.facts}\n"
                f"Key points: {research.key_points}\n"
                f"Important claims (pre-vetted): {research.important_claims}\n"
                f"Known uncertainties: {research.uncertainties}\n"
                f"Sources: {[s.title for s in research.sources]}\n"
            )
            user_prompt = (
                f"Draft article:\n{draft.title}\n{draft.body_markdown}\n\n"
                f"Research evidence available:\n{evidence_text}\n\n"
                "Return a FactCheckResult with claims_checked, supported, "
                "unsupported, uncertain, contradicted counts (these should sum to "
                "claims_checked), a list of `issues` for every non-supported claim "
                "(claim, status, explanation, suggested_fix), and an overall "
                "confidence_score (0-100)."
            )
            result = self.llm.invoke_structured(_SYSTEM_PROMPT, user_prompt, FactCheckResult)
            note(f"{result.supported}/{result.claims_checked or 'N/A'} claims supported")
            return result
