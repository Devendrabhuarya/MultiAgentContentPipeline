"""
Pydantic schemas shared across agents, graph state, API, and UI.

Every agent-to-agent communication in this system uses one of these models
instead of free-form text. This is what lets the Critic/FactChecker/SEO
agents' output be parsed reliably and lets the graph make deterministic
routing decisions.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------
class ContentType(str, Enum):
    BLOG = "Blog"
    TECHNICAL_ARTICLE = "Technical article"
    RESEARCH_SUMMARY = "Research summary"
    PRODUCT_DESCRIPTION = "Product description"
    SOCIAL_MEDIA = "Social media article"


class UserRequest(BaseModel):
    topic: str
    content_type: ContentType = ContentType.BLOG
    target_audience: str = "general audience"
    tone: str = "professional"
    word_count: int = Field(default=1200, gt=0)
    keywords: list[str] = Field(default_factory=list)
    include_references: bool = True
    max_iterations: int = Field(default=3, ge=0, le=10)
    quality_threshold: float = Field(default=80.0, ge=0, le=100)


# ---------------------------------------------------------------------------
# Research Agent
# ---------------------------------------------------------------------------
class SourceReference(BaseModel):
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    reliability: Optional[str] = Field(
        default=None, description="e.g. 'high', 'medium', 'low' - source credibility"
    )


class ResearchResult(BaseModel):
    topic: str
    key_points: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    important_claims: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Planning Agent
# ---------------------------------------------------------------------------
class SectionPlan(BaseModel):
    heading: str
    subsections: list[str] = Field(default_factory=list)
    key_points_to_cover: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(
        default_factory=list, description="Indices/snippets of research facts this section relies on"
    )
    estimated_word_count: int = 200
    requires_citation: bool = False


class ContentPlan(BaseModel):
    title: str
    introduction_plan: str
    sections: list[SectionPlan] = Field(default_factory=list)
    conclusion_plan: str
    evidence_mapping: dict[str, list[str]] = Field(
        default_factory=dict, description="section heading -> list of supporting facts"
    )
    estimated_total_word_count: int = 0


# ---------------------------------------------------------------------------
# Writer Agent
# ---------------------------------------------------------------------------
class DraftContent(BaseModel):
    title: str
    body_markdown: str
    meta_description: Optional[str] = None
    word_count: int = 0
    references: list[str] = Field(default_factory=list)
    version: int = 1

    @field_validator("word_count", mode="before")
    @classmethod
    def _default_word_count(cls, v, info):
        return v or 0


# ---------------------------------------------------------------------------
# Fact Checker Agent
# ---------------------------------------------------------------------------
class FactIssue(BaseModel):
    claim: str
    status: str = Field(description="'unsupported' | 'contradicted' | 'uncertain' | 'hallucinated'")
    explanation: str
    suggested_fix: Optional[str] = None


class FactCheckResult(BaseModel):
    claims_checked: int = 0
    supported: int = 0
    unsupported: int = 0
    uncertain: int = 0
    contradicted: int = 0
    issues: list[FactIssue] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0, le=100)


# ---------------------------------------------------------------------------
# SEO Agent
# ---------------------------------------------------------------------------
class SEOResult(BaseModel):
    seo_score: float = Field(ge=0, le=100)
    detected_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    title_assessment: str = ""
    heading_assessment: str = ""
    readability_assessment: str = ""
    meta_description_assessment: str = ""
    recommendations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Critic Agent
# ---------------------------------------------------------------------------
class CriticResult(BaseModel):
    factuality: float = Field(ge=0, le=100)
    relevance: float = Field(ge=0, le=100)
    quality: float = Field(ge=0, le=100)
    structure: float = Field(ge=0, le=100)
    readability: float = Field(ge=0, le=100)
    seo: float = Field(ge=0, le=100)
    evidence: float = Field(ge=0, le=100)
    requirement_compliance: float = Field(ge=0, le=100)
    overall_score: Optional[float] = Field(default=None, ge=0, le=100, validate_default=True)

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    critical_issues: list[str] = Field(default_factory=list)
    minor_issues: list[str] = Field(default_factory=list)
    revision_recommendations: list[str] = Field(default_factory=list)

    @field_validator("overall_score", mode="before")
    @classmethod
    def _compute_overall_if_missing(cls, v, info):
        if v is not None:
            return v
        data = info.data
        dims = [
            "factuality", "relevance", "quality", "structure",
            "readability", "seo", "evidence", "requirement_compliance",
        ]
        vals = [data.get(d) for d in dims if data.get(d) is not None]
        return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Feedback Agent
# ---------------------------------------------------------------------------
class FeedbackResult(BaseModel):
    priority: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    critical_fixes: list[str] = Field(default_factory=list)
    cosmetic_fixes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Revision history tracking
# ---------------------------------------------------------------------------
class RevisionRecord(BaseModel):
    version: int
    draft: DraftContent
    fact_check: FactCheckResult
    seo_analysis: SEOResult
    critic_evaluation: CriticResult
    feedback: Optional[FeedbackResult] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------
class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNING = "completed_with_warning"
    FAILED = "failed"


class FinalResult(BaseModel):
    topic: str
    final_content: DraftContent
    quality_score: CriticResult
    iterations_used: int
    status: PipelineStatus
    revision_history: list[RevisionRecord] = Field(default_factory=list)
    total_execution_time_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
