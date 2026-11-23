"""
Abstract research/RAG provider interface.

Keeping this abstract lets the Research Agent stay decoupled from *how*
evidence is gathered. Today: mock/web/local. Later (Phase 4+): PDFs,
academic paper APIs, company knowledge bases, vector-store-backed
retrieval -- all pluggable behind this same interface without touching
the agent or graph code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import SourceReference


class RawResearchItem(dict):
    """A single retrieved snippet: {title, url, snippet, reliability}."""


class ResearchProvider(ABC):
    """Base interface all research providers must implement."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SourceReference]:
        """Return a list of source references relevant to `query`."""

    def search_many(self, queries: list[str], max_results_per_query: int = 3) -> list[SourceReference]:
        """Convenience helper: search multiple queries and dedupe by title/url."""
        seen: set[str] = set()
        results: list[SourceReference] = []
        for q in queries:
            for ref in self.search(q, max_results=max_results_per_query):
                key = ref.url or ref.title
                if key in seen:
                    continue
                seen.add(key)
                results.append(ref)
        return results
