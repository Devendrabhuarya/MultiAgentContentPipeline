"""
Base agent class.

Every agent gets an LLMClient injected (dependency injection, not a
global), which is what makes agents independently unit-testable with
MockLLMClient. Each agent implements `run(...)` with its own signature;
there's no forced common interface beyond having a name/logger, since
agents genuinely take different inputs (this is also the extension point
for Phase 2 persistent memory: a `self.memory` slot could be added here
without touching individual agents' run() signatures).
"""
from __future__ import annotations

from app.llm.client import BaseLLMClient
from app.logging_config import get_agent_logger


class BaseAgent:
    name: str = "Agent"

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.logger = get_agent_logger(self.name)
