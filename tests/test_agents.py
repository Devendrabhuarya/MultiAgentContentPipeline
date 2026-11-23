import json

import pytest

from app.agents.critic_agent import CriticAgent
from app.agents.fact_checker_agent import FactCheckerAgent
from app.agents.feedback_agent import FeedbackAgent
from app.agents.planning_agent import PlanningAgent
from app.agents.research_agent import ResearchAgent
from app.agents.seo_agent import SEOAgent
from app.agents.writer_agent import WriterAgent
from app.llm.client import LLMError, MockLLMClient
from app.models.schemas import (
    ContentPlan,
    ContentType,
    CriticResult,
    DraftContent,
    FactCheckResult,
    ResearchResult,
    SEOResult,
    SectionPlan,
    UserRequest,
)
from app.research.mock import MockResearchProvider


def make_request(**overrides) -> UserRequest:
    defaults = dict(topic="Federated Learning in Healthcare", word_count=800)
    defaults.update(overrides)
    return UserRequest(**defaults)


def test_research_agent_parses_structured_output():
    canned = json.dumps({
        "topic": "Federated Learning in Healthcare",
        "key_points": ["Privacy-preserving training"],
        "facts": ["FL trains models across decentralized data"],
        "sources": [],
        "important_claims": ["FL reduces data-sharing risk"],
        "uncertainties": ["Regulatory frameworks vary by country"],
        "subtopics": ["Privacy", "Regulation"],
    })
    llm = MockLLMClient(responses=[canned])
    agent = ResearchAgent(llm, MockResearchProvider())
    result = agent.run(make_request())

    assert isinstance(result, ResearchResult)
    assert "Privacy-preserving training" in result.key_points
    # Sources should be backfilled from the research provider since the LLM returned none.
    assert len(result.sources) > 0


def test_research_agent_retries_on_malformed_json_then_succeeds():
    bad = "not json at all"
    good = json.dumps({
        "topic": "X", "key_points": [], "facts": [], "sources": [],
        "important_claims": [], "uncertainties": [], "subtopics": [],
    })
    llm = MockLLMClient(responses=[bad, good])
    agent = ResearchAgent(llm, MockResearchProvider())
    result = agent.run(make_request(topic="X"))
    assert result.topic == "X"
    assert len(llm.calls) == 2  # confirms a retry happened


def test_research_agent_raises_llm_error_when_retries_exhausted():
    llm = MockLLMClient(responses=["still not json", "still not json"])
    agent = ResearchAgent(llm, MockResearchProvider())
    with pytest.raises(LLMError):
        agent.run(make_request())


def test_planning_agent_returns_content_plan():
    canned = json.dumps({
        "title": "FL in Healthcare",
        "introduction_plan": "intro",
        "sections": [{
            "heading": "Background", "subsections": [], "key_points_to_cover": ["what is FL"],
            "evidence_refs": [], "estimated_word_count": 300, "requires_citation": True,
        }],
        "conclusion_plan": "wrap up",
        "evidence_mapping": {},
        "estimated_total_word_count": 300,
    })
    llm = MockLLMClient(responses=[canned])
    agent = PlanningAgent(llm)
    research = ResearchResult(topic="FL")
    plan = agent.run(make_request(), research)
    assert isinstance(plan, ContentPlan)
    assert plan.sections[0].heading == "Background"


def test_writer_agent_draft_mode_sets_version_1():
    canned = json.dumps({
        "title": "FL Article", "body_markdown": "# FL\n\nBody text here.",
        "meta_description": "desc", "word_count": 0, "references": [],
    })
    llm = MockLLMClient(responses=[canned])
    agent = WriterAgent(llm)
    research = ResearchResult(topic="FL")
    plan = ContentPlan(
        title="FL Article", introduction_plan="intro",
        sections=[SectionPlan(heading="H1")], conclusion_plan="end",
    )
    draft = agent.write_draft(make_request(), research, plan, version=1)
    assert draft.version == 1
    assert draft.word_count > 0  # computed since LLM returned 0


def test_writer_agent_rewrite_mode_increments_version():
    canned = json.dumps({
        "title": "FL Article v2", "body_markdown": "# FL\n\nImproved body.",
        "meta_description": "desc", "word_count": 50, "references": [],
    })
    llm = MockLLMClient(responses=[canned])
    agent = WriterAgent(llm)
    prev_draft = DraftContent(title="FL Article", body_markdown="old body", word_count=20, version=1)
    critic = CriticResult(
        factuality=60, relevance=60, quality=60, structure=60, readability=60,
        seo=60, evidence=60, requirement_compliance=60, overall_score=60,
        critical_issues=["fix claim"],
    )
    fact_check = FactCheckResult(claims_checked=5, supported=3)
    seo = SEOResult(seo_score=60, recommendations=["add keyword"])
    from app.models.schemas import FeedbackResult
    feedback = FeedbackResult(priority=["fix claim"], instructions=["add citation"])

    draft = agent.rewrite(
        make_request(), prev_draft, critic, fact_check, seo, feedback, version=2
    )
    assert draft.version == 2
    assert draft.title == "FL Article v2"


def test_fact_checker_agent_parses_result():
    canned = json.dumps({
        "claims_checked": 10, "supported": 8, "unsupported": 1, "uncertain": 1,
        "contradicted": 0, "issues": [{"claim": "X causes Y", "status": "unsupported",
                                        "explanation": "no evidence found"}],
        "confidence_score": 80,
    })
    llm = MockLLMClient(responses=[canned])
    agent = FactCheckerAgent(llm)
    draft = DraftContent(title="T", body_markdown="body", word_count=10)
    research = ResearchResult(topic="T")
    result = agent.run(draft, research)
    assert result.supported == 8
    assert result.issues[0].status == "unsupported"


def test_seo_agent_parses_result():
    canned = json.dumps({
        "seo_score": 72, "detected_keywords": ["federated learning"], "missing_keywords": ["privacy"],
        "title_assessment": "ok", "heading_assessment": "ok", "readability_assessment": "ok",
        "meta_description_assessment": "ok", "recommendations": ["add 'privacy' keyword"],
    })
    llm = MockLLMClient(responses=[canned])
    agent = SEOAgent(llm)
    draft = DraftContent(title="T", body_markdown="body", word_count=10)
    result = agent.run(draft, make_request())
    assert result.seo_score == 72
    assert "privacy" in result.missing_keywords


def test_critic_agent_parses_result():
    canned = json.dumps({
        "factuality": 85, "relevance": 90, "quality": 78, "structure": 82,
        "readability": 88, "seo": 75, "evidence": 80, "requirement_compliance": 90,
        "overall_score": 82, "strengths": ["clear"], "weaknesses": ["thin evidence"],
        "critical_issues": [], "minor_issues": ["typo"], "revision_recommendations": ["add source"],
    })
    llm = MockLLMClient(responses=[canned])
    agent = CriticAgent(llm)
    draft = DraftContent(title="T", body_markdown="body", word_count=10)
    research = ResearchResult(topic="T")
    fact_check = FactCheckResult(claims_checked=5, supported=5)
    seo = SEOResult(seo_score=75)
    result = agent.run(draft, make_request(), research, fact_check, seo)
    assert result.overall_score == 82


def test_feedback_agent_prioritizes_critical_issues():
    canned = json.dumps({
        "priority": ["Fix unsupported claim about X", "Add missing keyword"],
        "instructions": ["Cite a source for claim X", "Include 'privacy' in the intro"],
        "critical_fixes": ["Fix unsupported claim about X"],
        "cosmetic_fixes": ["Add missing keyword"],
    })
    llm = MockLLMClient(responses=[canned])
    agent = FeedbackAgent(llm)
    critic = CriticResult(
        factuality=60, relevance=70, quality=70, structure=70, readability=70,
        seo=70, evidence=60, requirement_compliance=70, overall_score=65,
        critical_issues=["Unsupported claim about X"],
    )
    fact_check = FactCheckResult(claims_checked=5, supported=4)
    seo = SEOResult(seo_score=70, missing_keywords=["privacy"])
    result = agent.run(make_request(), critic, fact_check, seo)
    assert result.priority[0].lower().startswith("fix unsupported")
