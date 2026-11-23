"""
LangGraph workflow definition.

Wires the node functions from nodes.py into the graph described in the
spec:

    START -> research -> planning -> writing -> fact_check -> seo -> critic
          -> [quality_decision]
                 |-- "final"    --> final -> END
                 `-- "feedback" --> feedback -> writing (rewrite mode) -> ...loop...

The loop is bounded by `quality_decision`, which always terminates once
either the score threshold or max_iterations is reached (see nodes.py).
"""
from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph

from app.config import Settings, settings as default_settings
from app.graph.nodes import PipelineAgents, build_nodes, quality_decision
from app.graph.state import PipelineState
from app.llm.client import BaseLLMClient, OpenAICompatibleLLMClient
from app.models.schemas import UserRequest
from app.research.base import ResearchProvider
from app.research import get_research_provider

from app.agents.critic_agent import CriticAgent
from app.agents.fact_checker_agent import FactCheckerAgent
from app.agents.feedback_agent import FeedbackAgent
from app.agents.planning_agent import PlanningAgent
from app.agents.research_agent import ResearchAgent
from app.agents.seo_agent import SEOAgent
from app.agents.writer_agent import WriterAgent


def build_agents(
    llm_client: BaseLLMClient, research_provider: ResearchProvider
) -> PipelineAgents:
    return PipelineAgents(
        research=ResearchAgent(llm_client, research_provider),
        planning=PlanningAgent(llm_client),
        writer=WriterAgent(llm_client),
        fact_checker=FactCheckerAgent(llm_client),
        seo=SEOAgent(llm_client),
        critic=CriticAgent(llm_client),
        feedback=FeedbackAgent(llm_client),
    )


def build_graph(agents: PipelineAgents, pipeline_settings: Settings):
    """Construct and compile the LangGraph StateGraph."""
    nodes = build_nodes(agents, pipeline_settings)
    graph = StateGraph(PipelineState)

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "research")
    graph.add_edge("research", "planning")
    graph.add_edge("planning", "writing")
    graph.add_edge("writing", "fact_check")
    graph.add_edge("fact_check", "seo")
    graph.add_edge("seo", "critic")

    graph.add_conditional_edges(
        "critic",
        quality_decision,
        {"final": "final", "feedback": "feedback"},
    )

    graph.add_edge("feedback", "writing")
    graph.add_edge("final", END)

    return graph.compile()


def print_graph_structure(compiled_graph) -> str:
    """Return (and print) a human-readable/mermaid representation of the graph."""
    try:
        mermaid = compiled_graph.get_graph().draw_mermaid()
    except Exception:  # noqa: BLE001
        mermaid = "(mermaid rendering unavailable in this environment)"
    print(mermaid)
    return mermaid


def run_pipeline(
    request: UserRequest,
    llm_client: BaseLLMClient | None = None,
    research_provider: ResearchProvider | None = None,
    pipeline_settings: Settings | None = None,
) -> PipelineState:
    """
    High-level convenience entrypoint: build agents+graph and run to completion.

    If llm_client/research_provider aren't supplied, they're built from
    global settings (real OpenAI-compatible client + configured research
    provider). Tests and offline demos should pass a MockLLMClient and/or
    MockResearchProvider explicitly.
    """
    pipeline_settings = pipeline_settings or default_settings
    llm_client = llm_client or OpenAICompatibleLLMClient()
    research_provider = research_provider or get_research_provider()

    agents = build_agents(llm_client, research_provider)
    compiled = build_graph(agents, pipeline_settings)

    initial_state: PipelineState = {
        "user_request": request,
        "topic": request.topic,
        "requirements": request.model_dump(),
        "iteration_count": 0,
        "quality_score": 0.0,
        "status": "running",
        "revision_history": [],
        "agent_log": [],
        "warnings": [],
        "start_time": time.perf_counter(),
    }

    # recursion_limit guards against any unforeseen graph-wiring bug creating
    # a runaway loop, independent of the quality_decision bound.
    result_state = compiled.invoke(
        initial_state, config={"recursion_limit": 50}
    )
    return result_state
