import pytest
from pydantic import ValidationError

from app.models.schemas import (
    ContentPlan,
    CriticResult,
    DraftContent,
    FactCheckResult,
    FeedbackResult,
    FinalResult,
    PipelineStatus,
    ResearchResult,
    RevisionRecord,
    SEOResult,
    SectionPlan,
    UserRequest,
)


def test_user_request_defaults():
    req = UserRequest(topic="Test Topic")
    assert req.word_count == 1200
    assert req.max_iterations == 3
    assert req.quality_threshold == 80.0


def test_user_request_invalid_word_count_rejected():
    with pytest.raises(ValidationError):
        UserRequest(topic="Test", word_count=0)


def test_research_result_defaults_are_empty_lists():
    r = ResearchResult(topic="X")
    assert r.key_points == []
    assert r.sources == []


def test_content_plan_with_sections():
    plan = ContentPlan(
        title="T",
        introduction_plan="intro",
        sections=[SectionPlan(heading="H1", estimated_word_count=300)],
        conclusion_plan="conclusion",
    )
    assert plan.sections[0].heading == "H1"
    assert plan.sections[0].estimated_word_count == 300


def test_fact_check_result_bounds():
    fc = FactCheckResult(claims_checked=10, supported=8, unsupported=2, confidence_score=90)
    assert fc.confidence_score == 90
    with pytest.raises(ValidationError):
        FactCheckResult(confidence_score=150)


def test_seo_result_score_bounds():
    with pytest.raises(ValidationError):
        SEOResult(seo_score=101)
    seo = SEOResult(seo_score=75)
    assert seo.seo_score == 75


def test_critic_result_computes_overall_score_if_missing():
    critic = CriticResult(
        factuality=80, relevance=80, quality=80, structure=80,
        readability=80, seo=80, evidence=80, requirement_compliance=80,
        overall_score=None,
    )
    assert critic.overall_score == 80.0


def test_critic_result_explicit_overall_score_respected():
    critic = CriticResult(
        factuality=50, relevance=50, quality=50, structure=50,
        readability=50, seo=50, evidence=50, requirement_compliance=50,
        overall_score=42,
    )
    assert critic.overall_score == 42


def test_feedback_result_defaults():
    fb = FeedbackResult()
    assert fb.priority == []
    assert fb.instructions == []


def test_final_result_requires_nested_models():
    draft = DraftContent(title="T", body_markdown="body", word_count=2)
    critic = CriticResult(
        factuality=80, relevance=80, quality=80, structure=80,
        readability=80, seo=80, evidence=80, requirement_compliance=80, overall_score=80,
    )
    final = FinalResult(
        topic="Topic", final_content=draft, quality_score=critic,
        iterations_used=1, status=PipelineStatus.COMPLETED,
    )
    assert final.status == PipelineStatus.COMPLETED
    assert final.revision_history == []


def test_revision_record_roundtrip():
    draft = DraftContent(title="T", body_markdown="body", word_count=2)
    fc = FactCheckResult(claims_checked=1, supported=1)
    seo = SEOResult(seo_score=80)
    critic = CriticResult(
        factuality=80, relevance=80, quality=80, structure=80,
        readability=80, seo=80, evidence=80, requirement_compliance=80, overall_score=80,
    )
    record = RevisionRecord(version=1, draft=draft, fact_check=fc, seo_analysis=seo, critic_evaluation=critic)
    assert record.version == 1
    assert record.draft.title == "T"
