"""
FastAPI backend for the Multi-Agent Content Pipeline.

Run with:
    uvicorn api.main:app --reload

Endpoints:
    POST /generate       Run the full pipeline synchronously, return FinalResult.
    GET  /health          Basic liveness check.
    GET  /graph            Return the graph structure (mermaid text).
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.graph.workflow import build_agents, build_graph, print_graph_structure, run_pipeline
from app.llm.client import LLMError, OpenAICompatibleLLMClient
from app.logging_config import configure_logging
from app.models.schemas import FinalResult, UserRequest
from app.research import get_research_provider

configure_logging()
logger = logging.getLogger("pipeline.API")

app = FastAPI(
    title="Multi-Agent Content Pipeline API",
    description="Research -> Planning -> Writing -> Fact Check -> SEO -> "
    "Critic -> Feedback -> Revision -> Final Content",
    version="1.0.0",
)


class GenerateResponse(BaseModel):
    result: FinalResult


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/graph")
def graph_structure() -> dict:
    llm_client = OpenAICompatibleLLMClient() if settings.openai_api_key else None
    if llm_client is None:
        return {"mermaid": "(set OPENAI_API_KEY to render live graph; structure is static regardless)"}
    provider = get_research_provider()
    agents = build_agents(llm_client, provider)
    compiled = build_graph(agents, settings)
    return {"mermaid": print_graph_structure(compiled)}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: UserRequest) -> GenerateResponse:
    try:
        state = run_pipeline(request)
    except LLMError as exc:
        logger.exception("LLM error during pipeline execution")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during pipeline execution")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}") from exc

    final = state.get("final_content")
    if final is None:
        raise HTTPException(status_code=500, detail="Pipeline finished without producing final content.")
    return GenerateResponse(result=final)
