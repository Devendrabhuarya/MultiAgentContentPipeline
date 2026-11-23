import json

import pytest

from app.graph.nodes import quality_decision
from app.graph.workflow import run_pipeline
from app.llm.client import LLMError, MockLLMClient
from app.models.schemas import CriticResult, PipelineStatus, UserRequest
from tests.conftest import build_mock_llm_for_scores, mock_research_provider


def _critic(score: float) -> CriticResult:
    return CriticResult(
        factuality=score, relevance=score, quality=score, structure=score,
        readability=score, seo=score, evidence=score, requirement_compliance=score,
        overall_score=score,
    )


# ---------------------------------------------------------------------------
# 1. Score below threshold -> revision occurs
# ---------------------------------------------------------------------------
def test_score_below_threshold_triggers_feedback():
    request = UserRequest(topic="T", quality_threshold=80, max_iterations=3)
    state = {"user_request": request, "quality_score": 65.0, "iteration_count": 0}
    assert quality_decision(state) == "feedback"


# ---------------------------------------------------------------------------
# 2. Score above threshold -> final output, no revision
# ---------------------------------------------------------------------------
def test_score_above_threshold_goes_to_final():
    request = UserRequest(topic="T", quality_threshold=80, max_iterations=3)
    state = {"user_request": request, "quality_score": 92.0, "iteration_count": 0}
    assert quality_decision(state) == "final"


def test_score_exactly_at_threshold_goes_to_final():
    request = UserRequest(topic="T", quality_threshold=80, max_iterations=3)
    state = {"user_request": request, "quality_score": 80.0, "iteration_count": 0}
    assert quality_decision(state) == "final"


# ---------------------------------------------------------------------------
# 3. Maximum iterations reached -> pipeline terminates (even below threshold)
# ---------------------------------------------------------------------------
def test_max_iterations_reached_forces_final_even_below_threshold():
    request = UserRequest(topic="T", quality_threshold=80, max_iterations=2)
    state = {"user_request": request, "quality_score": 55.0, "iteration_count": 2}
    assert quality_decision(state) == "final"


def test_full_pipeline_terminates_with_warning_at_max_iterations():
    request = UserRequest(topic="Test Topic", quality_threshold=80, max_iterations=1, word_count=300)
    llm = build_mock_llm_for_scores("Test Topic", scores=[62, 74])  # never reaches 80
    state = run_pipeline(request, llm, mock_research_provider())

    final = state["final_content"]
    assert final.status == PipelineStatus.COMPLETED_WITH_WARNING
    assert final.iterations_used == 1
    assert len(final.warnings) == 1
    assert "Maximum iterations" in final.warnings[0]
    # Never exceeds the configured bound.
    assert len(state["revision_history"]) == 2


def test_max_iterations_zero_means_no_revision_ever():
    request = UserRequest(topic="Test Topic", quality_threshold=80, max_iterations=0, word_count=300)
    llm = build_mock_llm_for_scores("Test Topic", scores=[55])
    state = run_pipeline(request, llm, mock_research_provider())
    final = state["final_content"]
    assert final.iterations_used == 0
    assert final.status == PipelineStatus.COMPLETED_WITH_WARNING


# ---------------------------------------------------------------------------
# 4. Invalid LLM response -> retry / controlled error handling
# ---------------------------------------------------------------------------
def test_invalid_llm_response_retries_before_succeeding():
    good_research = json.dumps({
        "topic": "T", "key_points": [], "facts": [], "sources": [],
        "important_claims": [], "uncertainties": [], "subtopics": [],
    })
    llm = MockLLMClient(responses=["garbage output {{{", good_research])
    from app.agents.research_agent import ResearchAgent
    agent = ResearchAgent(llm, mock_research_provider())
    result = agent.run(UserRequest(topic="T"))
    assert result.topic == "T"
    assert len(llm.calls) == 2


def test_invalid_llm_response_raises_controlled_error_after_retries_exhausted():
    llm = MockLLMClient(responses=["not json", "still not json", "nope"])
    from app.agents.research_agent import ResearchAgent
    agent = ResearchAgent(llm, mock_research_provider())
    with pytest.raises(LLMError):
        agent.run(UserRequest(topic="T"))
    # llm_max_retries default (2) => 3 total attempts, all consumed.
    assert len(llm.calls) == 3
