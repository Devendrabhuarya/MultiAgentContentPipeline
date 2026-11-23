"""
LangGraph node functions.

Each node is a plain function(state) -> partial_state_update, as LangGraph
expects. Node functions are produced by `build_nodes(agents, settings)`
so that the actual agent instances (and therefore the LLM client /
research provider used) are injected rather than constructed globally --
this is what makes the graph swappable between real and mocked agents in
tests.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.config import Settings
from app.agents.critic_agent import CriticAgent
from app.agents.fact_checker_agent import FactCheckerAgent
from app.agents.feedback_agent import FeedbackAgent
from app.agents.planning_agent import PlanningAgent
from app.agents.research_agent import ResearchAgent
from app.agents.seo_agent import SEOAgent
from app.agents.writer_agent import WriterAgent
from app.graph.state import PipelineState
from app.models.schemas import (
    FinalResult,
    PipelineStatus,
    RevisionRecord,
    UserRequest,
)

logger = logging.getLogger("pipeline.Graph")


@dataclass
class PipelineAgents:
    research: ResearchAgent
    planning: PlanningAgent
    writer: WriterAgent
    fact_checker: FactCheckerAgent
    seo: SEOAgent
    critic: CriticAgent
    feedback: FeedbackAgent


def build_nodes(agents: PipelineAgents, settings: Settings) -> dict:
    """Return a dict of node_name -> callable(state) -> dict for graph wiring."""

    def research_node(state: PipelineState) -> dict:
        request: UserRequest = state["user_request"]
        result = agents.research.run(request)
        return {
            "research": result,
            "status": "running",
            "agent_log": state.get("agent_log", []) + ["Research: completed"],
        }

    def planning_node(state: PipelineState) -> dict:
        request: UserRequest = state["user_request"]
        plan = agents.planning.run(request, state["research"])
        return {
            "outline": plan,
            "agent_log": state.get("agent_log", []) + ["Planning: completed"],
        }

    def writing_node(state: PipelineState) -> dict:
        request: UserRequest = state["user_request"]
        iteration = state.get("iteration_count", 0)

        if iteration == 0 or state.get("draft") is None:
            draft = agents.writer.write_draft(
                request, state["research"], state["outline"], version=1
            )
        else:
            draft = agents.writer.rewrite(
                request,
                previous_draft=state["draft"],
                critic=state["critic_evaluation"],
                fact_check=state["fact_check"],
                seo=state["seo_analysis"],
                feedback=state["feedback"],
                version=iteration + 1,
            )
        return {
            "draft": draft,
            "agent_log": state.get("agent_log", []) + [f"Writer: draft v{draft.version} generated"],
        }

    def fact_check_node(state: PipelineState) -> dict:
        result = agents.fact_checker.run(state["draft"], state["research"])
        return {
            "fact_check": result,
            "agent_log": state.get("agent_log", [])
            + [f"FactChecker: {result.supported}/{result.claims_checked} claims supported"],
        }

    def seo_node(state: PipelineState) -> dict:
        result = agents.seo.run(state["draft"], state["user_request"])
        return {
            "seo_analysis": result,
            "agent_log": state.get("agent_log", []) + [f"SEO: score {result.seo_score}"],
        }

    def critic_node(state: PipelineState) -> dict:
        result = agents.critic.run(
            state["draft"],
            state["user_request"],
            state["research"],
            state["fact_check"],
            state["seo_analysis"],
        )

        history = list(state.get("revision_history", []))
        history.append(
            RevisionRecord(
                version=state["draft"].version,
                draft=state["draft"],
                fact_check=state["fact_check"],
                seo_analysis=state["seo_analysis"],
                critic_evaluation=result,
                feedback=state.get("feedback"),
            )
        )
        return {
            "critic_evaluation": result,
            "quality_score": result.overall_score,
            "revision_history": history,
            "agent_log": state.get("agent_log", []) + [f"Critic: overall score {result.overall_score}"],
        }

    def feedback_node(state: PipelineState) -> dict:
        result = agents.feedback.run(
            state["user_request"],
            state["critic_evaluation"],
            state["fact_check"],
            state["seo_analysis"],
        )
        return {
            "feedback": result,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "agent_log": state.get("agent_log", [])
            + [f"Feedback: {len(result.priority)} issues identified"],
        }

    def final_node(state: PipelineState) -> dict:
        threshold = state["user_request"].quality_threshold
        max_iter = state["user_request"].max_iterations
        score = state.get("quality_score", 0.0)
        iteration = state.get("iteration_count", 0)

        warnings = list(state.get("warnings", []))
        if score >= threshold:
            status = PipelineStatus.COMPLETED
        else:
            status = PipelineStatus.COMPLETED_WITH_WARNING
            warnings.append(
                f"Maximum iterations ({max_iter}) reached before quality threshold "
                f"({threshold}) was met. Final score: {score}."
            )

        elapsed = time.perf_counter() - state.get("start_time", time.perf_counter())
        final = FinalResult(
            topic=state["user_request"].topic,
            final_content=state["draft"],
            quality_score=state["critic_evaluation"],
            iterations_used=iteration,
            status=status,
            revision_history=state.get("revision_history", []),
            total_execution_time_seconds=round(elapsed, 2),
            warnings=warnings,
        )
        logger.info(f"[Decision] {'Quality threshold reached' if score >= threshold else 'Max iterations reached'}")
        logger.info("[Pipeline] Completed")
        return {
            "final_content": final,
            "status": status.value,
            "warnings": warnings,
            "agent_log": state.get("agent_log", []) + ["Pipeline: completed"],
        }

    return {
        "research": research_node,
        "planning": planning_node,
        "writing": writing_node,
        "fact_check": fact_check_node,
        "seo": seo_node,
        "critic": critic_node,
        "feedback": feedback_node,
        "final": final_node,
    }


def quality_decision(state: PipelineState) -> str:
    """
    Conditional edge function: decide whether to finalize or loop back
    for another feedback+revision cycle.

    Never allows infinite loops: bounded by user_request.max_iterations.
    """
    request: UserRequest = state["user_request"]
    score = state.get("quality_score", 0.0)
    iteration = state.get("iteration_count", 0)

    if score >= request.quality_threshold:
        return "final"
    if iteration >= request.max_iterations:
        return "final"
    return "feedback"
