"""Shared pytest fixtures/helpers for full-workflow tests."""
from __future__ import annotations

import json

from app.llm.client import MockLLMClient
from app.research.mock import MockResearchProvider


def _research_response(topic: str) -> str:
    return json.dumps({
        "topic": topic, "key_points": ["Key point A", "Key point B"],
        "facts": ["Fact 1", "Fact 2"], "sources": [],
        "important_claims": ["Claim 1"], "uncertainties": ["Uncertainty 1"],
        "subtopics": ["Subtopic A"],
    })


def _planning_response(topic: str) -> str:
    return json.dumps({
        "title": f"{topic}: An Overview", "introduction_plan": "intro",
        "sections": [{
            "heading": "Background", "subsections": [], "key_points_to_cover": ["Key point A"],
            "evidence_refs": [], "estimated_word_count": 300, "requires_citation": True,
        }],
        "conclusion_plan": "wrap up", "evidence_mapping": {}, "estimated_total_word_count": 300,
    })


def _draft_response(version: int, word_count: int = 300) -> str:
    return json.dumps({
        "title": "Draft Title", "body_markdown": f"# Draft v{version}\n\nContent body.",
        "meta_description": "desc", "word_count": word_count, "references": [], "version": version,
    })


def _fact_check_response(score: float) -> str:
    supported = 9 if score >= 80 else 6
    return json.dumps({
        "claims_checked": 10, "supported": supported, "unsupported": 10 - supported,
        "uncertain": 0, "contradicted": 0,
        "issues": [] if score >= 80 else [{
            "claim": "Unsupported claim", "status": "unsupported", "explanation": "no evidence",
            "suggested_fix": "cite a source",
        }],
        "confidence_score": score,
    })


def _seo_response(score: float) -> str:
    return json.dumps({
        "seo_score": score, "detected_keywords": ["keyword"], "missing_keywords": [],
        "title_assessment": "ok", "heading_assessment": "ok", "readability_assessment": "ok",
        "meta_description_assessment": "ok",
        "recommendations": [] if score >= 80 else ["Improve keyword density"],
    })


def _critic_response(score: float) -> str:
    return json.dumps({
        "factuality": score, "relevance": score, "quality": score, "structure": score,
        "readability": score, "seo": score, "evidence": score, "requirement_compliance": score,
        "overall_score": score,
        "strengths": ["Clear structure"],
        "weaknesses": [] if score >= 80 else ["Needs more evidence"],
        "critical_issues": [] if score >= 80 else ["Unsupported claim present"],
        "minor_issues": ["Minor phrasing"],
        "revision_recommendations": [] if score >= 80 else ["Add a citation to support the main claim"],
    })


def _feedback_response() -> str:
    return json.dumps({
        "priority": ["Fix unsupported claim", "Improve keyword density"],
        "instructions": ["Add a citation for the main claim", "Work 'keyword' into the introduction"],
        "critical_fixes": ["Fix unsupported claim"],
        "cosmetic_fixes": ["Improve keyword density"],
    })


def build_mock_llm_for_scores(topic: str, scores: list[float]) -> MockLLMClient:
    """
    Build a MockLLMClient whose canned responses drive the pipeline through
    exactly len(scores) draft/evaluate cycles, with the critic (and
    fact-check/SEO) overall score for cycle i equal to scores[i].

    E.g. scores=[62, 74, 86] simulates: draft v1 scores 62 -> revise ->
    draft v2 scores 74 -> revise -> draft v3 scores 86 -> done.
    """
    responses: list[str] = [_research_response(topic), _planning_response(topic)]
    for i, score in enumerate(scores, start=1):
        responses.append(_draft_response(version=i))
        responses.append(_fact_check_response(score))
        responses.append(_seo_response(score))
        responses.append(_critic_response(score))
        is_last = i == len(scores)
        if not is_last:
            responses.append(_feedback_response())
    return MockLLMClient(responses=responses)


def mock_research_provider() -> MockResearchProvider:
    return MockResearchProvider()
