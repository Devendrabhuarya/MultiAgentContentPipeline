from app.graph.workflow import build_agents, build_graph, print_graph_structure, run_pipeline
from app.models.schemas import PipelineStatus, UserRequest
from tests.conftest import build_mock_llm_for_scores, mock_research_provider


def test_pipeline_terminates_immediately_when_first_draft_passes_threshold():
    request = UserRequest(topic="Test Topic", quality_threshold=80, max_iterations=3, word_count=300)
    llm = build_mock_llm_for_scores("Test Topic", scores=[86])
    state = run_pipeline(request, llm, mock_research_provider())

    final = state["final_content"]
    assert final.status == PipelineStatus.COMPLETED
    assert final.iterations_used == 0
    assert len(state["revision_history"]) == 1


def test_pipeline_revises_when_score_below_threshold_then_succeeds():
    request = UserRequest(topic="Test Topic", quality_threshold=80, max_iterations=3, word_count=300)
    llm = build_mock_llm_for_scores("Test Topic", scores=[62, 86])
    state = run_pipeline(request, llm, mock_research_provider())

    final = state["final_content"]
    assert final.status == PipelineStatus.COMPLETED
    assert final.iterations_used == 1
    history = state["revision_history"]
    assert len(history) == 2
    assert history[0].critic_evaluation.overall_score == 62
    assert history[1].critic_evaluation.overall_score == 86
    # Quality visibly improved across the revision.
    assert history[1].critic_evaluation.overall_score > history[0].critic_evaluation.overall_score


def test_pipeline_shows_multi_step_quality_improvement():
    """Mirrors the spec's example: Draft 1 -> 62, Draft 2 -> 74, Draft 3 -> 86 -> FINAL."""
    request = UserRequest(topic="Test Topic", quality_threshold=80, max_iterations=3, word_count=300)
    llm = build_mock_llm_for_scores("Test Topic", scores=[62, 74, 86])
    state = run_pipeline(request, llm, mock_research_provider())

    scores = [r.critic_evaluation.overall_score for r in state["revision_history"]]
    assert scores == [62, 74, 86]
    assert state["final_content"].iterations_used == 2
    assert state["final_content"].status == PipelineStatus.COMPLETED


def test_graph_structure_can_be_printed(capsys):
    llm = build_mock_llm_for_scores("Anything", scores=[90])
    agents = build_agents(llm, mock_research_provider())
    from app.config import settings
    compiled = build_graph(agents, settings)
    mermaid = print_graph_structure(compiled)
    assert isinstance(mermaid, str)
    assert len(mermaid) > 0
