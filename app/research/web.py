"""
Web research provider.

Uses an external search API (e.g. Tavily, Serper, Bing) over HTTP via
httpx. The exact provider is intentionally kept swappable: set
WEB_SEARCH_API_KEY and adjust `_SEARCH_ENDPOINT` / `_build_payload` if you
switch vendors. Network/API failures are caught and logged rather than
propagated, so a flaky search API degrades gracefully instead of crashing
the whole pipeline (the Research Agent can still work off whatever partial
results it received, or fall back to MockResearchProvider upstream).
"""
from __future__ import annotations

import logging

import httpx

from app.config import settings
from app.models.schemas import SourceReference
from app.research.base import ResearchProvider

logger = logging.getLogger("pipeline.WebResearchProvider")

_SEARCH_ENDPOINT = "https://api.tavily.com/search"


class WebResearchProvider(ResearchProvider):
    """Real web search backed research provider (requires WEB_SEARCH_API_KEY)."""

    def __init__(self, api_key: str | None = None, timeout: int | None = None):
        self.api_key = api_key or settings.web_search_api_key
        self.timeout = timeout or settings.api_timeout
        if not self.api_key:
            logger.warning(
                "WebResearchProvider initialized without WEB_SEARCH_API_KEY; "
                "searches will return no results."
            )

    def search(self, query: str, max_results: int = 5) -> list[SourceReference]:
        if not self.api_key:
            return []
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(_SEARCH_ENDPOINT, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.error("Web search failed for query %r: %s", query, exc)
            return []
        except (ValueError, KeyError) as exc:
            logger.error("Web search returned malformed response for %r: %s", query, exc)
            return []

        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                SourceReference(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content"),
                    reliability="unknown",
                )
            )
        return results
