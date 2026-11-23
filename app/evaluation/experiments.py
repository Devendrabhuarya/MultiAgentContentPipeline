"""
Experiment runner: baseline (no feedback loop) vs. agentic (full
feedback loop) comparison.

Baseline = Research -> Planning -> Writer -> (evaluate once with Fact
Checker/SEO/Critic for scoring only, never revise). This isolates the
value the closed feedback loop adds versus just running the writer once.

Agentic = the full compiled LangGraph pipeline (app.graph.workflow).

Results are always produced by actually executing the pipelines -- this
module never fabricates numbers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.config import Settings, settings as default_settings
from app.evaluation.metrics import AggregateReport, RunMetrics, aggregate, improvement_percentage
from app.graph.nodes import PipelineAgents
from app.graph.workflow import build_agents, run_pipeline
from app.llm.client import BaseLLMClient
from app.models.schemas import FinalResult, PipelineStatus, RevisionRecord, UserRequest
from app.research.base import ResearchProvider


def _run_baseline(
    request: UserRequest, agents: PipelineAgents
) -> tuple[FinalResult, float]:
    start = time.perf_counter()
    research = agents.research.run(request)
    plan = agents.planning.run(request, research)
    draft = agents.writer.write_draft(request, research, plan, version=1)

    # Evaluate once, purely for measurement -- baseline never revises.
    fact_check = agents.fact_checker.run(draft, research)
    seo = agents.seo.run(draft, request)
    critic = agents.critic.run(draft, request, research, fact_check, seo)

    elapsed = time.perf_counter() - start
    final = FinalResult(
        topic=request.topic,
        final_content=draft,
        quality_score=critic,
        iterations_used=0,
        status=PipelineStatus.COMPLETED,
        revision_history=[
            RevisionRecord(
                version=1, draft=draft, fact_check=fact_check,
                seo_analysis=seo, critic_evaluation=critic,
            )
        ],
        total_execution_time_seconds=round(elapsed, 2),
    )
    return final, elapsed


def _run_agentic(
    request: UserRequest,
    llm_client: BaseLLMClient,
    research_provider: ResearchProvider,
    pipeline_settings: Settings,
) -> tuple[FinalResult, float]:
    start = time.perf_counter()
    state = run_pipeline(request, llm_client, research_provider, pipeline_settings)
    elapsed = time.perf_counter() - start
    return state["final_content"], elapsed


@dataclass
class ComparisonReport:
    baseline: AggregateReport
    agentic: AggregateReport
    improvement_percentage: float


def run_comparison(
    topics: list[str],
    llm_client: BaseLLMClient,
    research_provider: ResearchProvider,
    pipeline_settings: Settings | None = None,
    request_overrides: dict | None = None,
) -> ComparisonReport:
    """Run both baseline and agentic pipelines for each topic and aggregate."""
    pipeline_settings = pipeline_settings or default_settings
    request_overrides = request_overrides or {}

    agents = build_agents(llm_client, research_provider)

    runs: list[RunMetrics] = []
    for topic in topics:
        request = UserRequest(topic=topic, **request_overrides)

        baseline_result, baseline_time = _run_baseline(request, agents)
        runs.append(
            RunMetrics(
                topic=topic,
                mode="baseline",
                overall_score=baseline_result.quality_score.overall_score,
                factuality=baseline_result.quality_score.factuality,
                relevance=baseline_result.quality_score.relevance,
                readability=baseline_result.quality_score.readability,
                seo_score=baseline_result.quality_score.seo,
                evidence=baseline_result.quality_score.evidence,
                iterations=0,
                execution_time_seconds=round(baseline_time, 2),
                word_count=baseline_result.final_content.word_count,
            )
        )

        agentic_result, agentic_time = _run_agentic(
            request, llm_client, research_provider, pipeline_settings
        )
        runs.append(
            RunMetrics(
                topic=topic,
                mode="agentic",
                overall_score=agentic_result.quality_score.overall_score,
                factuality=agentic_result.quality_score.factuality,
                relevance=agentic_result.quality_score.relevance,
                readability=agentic_result.quality_score.readability,
                seo_score=agentic_result.quality_score.seo,
                evidence=agentic_result.quality_score.evidence,
                iterations=agentic_result.iterations_used,
                execution_time_seconds=round(agentic_time, 2),
                word_count=agentic_result.final_content.word_count,
            )
        )

    baseline_agg = aggregate(runs, "baseline")
    agentic_agg = aggregate(runs, "agentic")
    return ComparisonReport(
        baseline=baseline_agg,
        agentic=agentic_agg,
        improvement_percentage=improvement_percentage(baseline_agg, agentic_agg),
    )
