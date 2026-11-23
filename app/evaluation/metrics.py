"""
Evaluation metrics.

Pulls quantitative signals out of a completed pipeline run (PipelineState /
FinalResult) and provides aggregation helpers used by the experiment
runner to compare baseline (no feedback loop) vs. agentic (full feedback
loop) content generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.models.schemas import FinalResult


@dataclass
class RunMetrics:
    topic: str
    mode: str  # "baseline" | "agentic"
    overall_score: float
    factuality: float
    relevance: float
    readability: float
    seo_score: float
    evidence: float
    iterations: int
    execution_time_seconds: float
    word_count: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None


def metrics_from_final_result(
    result: FinalResult, mode: str, execution_time_seconds: float
) -> RunMetrics:
    qs = result.quality_score
    return RunMetrics(
        topic=result.topic,
        mode=mode,
        overall_score=qs.overall_score,
        factuality=qs.factuality,
        relevance=qs.relevance,
        readability=qs.readability,
        seo_score=qs.seo,
        evidence=qs.evidence,
        iterations=result.iterations_used,
        execution_time_seconds=execution_time_seconds,
        word_count=result.final_content.word_count,
    )


@dataclass
class AggregateReport:
    mode: str
    n_runs: int
    avg_overall_score: float
    avg_factuality: float
    avg_readability: float
    avg_seo_score: float
    avg_iterations: float
    avg_execution_time_seconds: float
    runs: list[RunMetrics] = field(default_factory=list)


def aggregate(runs: list[RunMetrics], mode: str) -> AggregateReport:
    subset = [r for r in runs if r.mode == mode]
    n = len(subset)
    if n == 0:
        return AggregateReport(mode, 0, 0, 0, 0, 0, 0, 0, [])

    def avg(getter):
        return round(sum(getter(r) for r in subset) / n, 2)

    return AggregateReport(
        mode=mode,
        n_runs=n,
        avg_overall_score=avg(lambda r: r.overall_score),
        avg_factuality=avg(lambda r: r.factuality),
        avg_readability=avg(lambda r: r.readability),
        avg_seo_score=avg(lambda r: r.seo_score),
        avg_iterations=avg(lambda r: r.iterations),
        avg_execution_time_seconds=avg(lambda r: r.execution_time_seconds),
        runs=subset,
    )


def improvement_percentage(baseline: AggregateReport, agentic: AggregateReport) -> float:
    if baseline.avg_overall_score == 0:
        return 0.0
    return round(
        (agentic.avg_overall_score - baseline.avg_overall_score)
        / baseline.avg_overall_score
        * 100,
        2,
    )
