"""Research provider package + factory."""
from __future__ import annotations

from app.config import settings
from app.research.base import ResearchProvider
from app.research.local import LocalDocumentProvider
from app.research.mock import MockResearchProvider
from app.research.web import WebResearchProvider

__all__ = [
    "ResearchProvider",
    "MockResearchProvider",
    "WebResearchProvider",
    "LocalDocumentProvider",
    "get_research_provider",
]


def get_research_provider(name: str | None = None) -> ResearchProvider:
    """Factory: build a ResearchProvider based on config or explicit name."""
    provider_name = (name or settings.research_provider).lower()
    if provider_name == "web":
        return WebResearchProvider()
    if provider_name == "local":
        return LocalDocumentProvider()
    return MockResearchProvider()
