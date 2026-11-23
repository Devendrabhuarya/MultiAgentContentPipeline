"""
Shared LangGraph state.

A single TypedDict flows through every node. Nodes read what they need and
return a partial dict of updates (LangGraph merges these into state). This
keeps state transitions explicit and inspectable -- printing the state
after any node tells you exactly what has happened so far.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from app.models.schemas import (
    ContentPlan,
    CriticResult,
    DraftContent,
    FactCheckResult,
    FeedbackResult,
    FinalResult,
    ResearchResult,
    RevisionRecord,
    SEOResult,
    UserRequest,
)


class PipelineState(TypedDict, total=False):
    # Input
    user_request: UserRequest
    topic: str
    requirements: dict

    # Agent outputs (latest)
    research: Optional[ResearchResult]
    outline: Optional[ContentPlan]
    draft: Optional[DraftContent]
    fact_check: Optional[FactCheckResult]
    seo_analysis: Optional[SEOResult]
    critic_evaluation: Optional[CriticResult]
    feedback: Optional[FeedbackResult]

    # Loop control
    iteration_count: int
    quality_score: float
    status: str  # "running" | "completed" | "completed_with_warning" | "failed"

    # History (for UI / evaluation)
    revision_history: list[RevisionRecord]
    agent_log: list[str]
    warnings: list[str]

    # Output
    final_content: Optional[FinalResult]

    # Timing
    start_time: float
