"""
Mock research provider.

Returns synthetic-but-structured source references without any network
call. Used as the default provider (RESEARCH_PROVIDER=mock) so the whole
pipeline is runnable and testable with zero external dependencies, and as
a safe fallback if WebResearchProvider fails.
"""
from __future__ import annotations

from app.models.schemas import SourceReference
from app.research.base import ResearchProvider


class MockResearchProvider(ResearchProvider):
    """Deterministic offline research provider for tests/demos."""

    def search(self, query: str, max_results: int = 5) -> list[SourceReference]:
        results = []
        for i in range(1, max_results + 1):
            results.append(
                SourceReference(
                    title=f"Reference {i} on '{query}'",
                    url=f"https://example.com/{query.replace(' ', '-').lower()}-{i}",
                    snippet=(
                        f"This is a synthetic research snippet #{i} discussing '{query}', "
                        "provided for offline development and testing purposes."
                    ),
                    reliability="medium",
                )
            )
        return results
