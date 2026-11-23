"""
Local document research provider.

Reads plain-text/markdown files from a local directory and does simple
keyword-overlap scoring against the query. This is intentionally simple
(no embeddings) so it has zero extra dependencies; `use_vector_store` in
config is the extension point for swapping this scoring step for a real
FAISS/Chroma similarity search later without changing the provider
interface.
"""
from __future__ import annotations

import logging
import os
import re

from app.config import settings
from app.models.schemas import SourceReference
from app.research.base import ResearchProvider

logger = logging.getLogger("pipeline.LocalDocumentProvider")

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


class LocalDocumentProvider(ResearchProvider):
    """Keyword-overlap search over a local directory of .txt/.md documents."""

    def __init__(self, directory: str | None = None):
        self.directory = directory or settings.local_documents_dir

    def _iter_documents(self):
        if not os.path.isdir(self.directory):
            logger.warning("Local documents directory not found: %s", self.directory)
            return
        for root, _dirs, files in os.walk(self.directory):
            for fname in files:
                if fname.lower().endswith((".txt", ".md")):
                    path = os.path.join(root, fname)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            yield path, f.read()
                    except OSError as exc:
                        logger.error("Failed to read %s: %s", path, exc)

    def search(self, query: str, max_results: int = 5) -> list[SourceReference]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, str, str]] = []
        for path, content in self._iter_documents():
            doc_tokens = _tokenize(content)
            if not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens) / len(query_tokens)
            if overlap > 0:
                snippet = content[:300].replace("\n", " ").strip()
                scored.append((overlap, path, snippet))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            SourceReference(
                title=os.path.basename(path),
                url=f"file://{path}",
                snippet=snippet,
                reliability="internal",
            )
            for _score, path, snippet in scored[:max_results]
        ]
