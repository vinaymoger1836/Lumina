"""Semantic retrieval from Qdrant."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import settings
from app.rag.embeddings import embed_query
from app.rag.vectorstore import ensure_collection, get_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved chunk with its provenance and similarity score."""

    text: str
    source: str
    source_type: str
    title: str
    page: int | None
    score: float

    def citation(self) -> str:
        """Human-readable source label, e.g. 'report.pdf (p.3)'."""
        if self.source_type == "pdf" and self.page is not None:
            return f"{self.title} (p.{self.page})"
        return self.title or self.source


def search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Return the top-k most similar chunks for a query."""
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    k = top_k or settings.top_k
    collection = ensure_collection()
    vector = embed_query(query)

    hits = get_client().query_points(
        collection_name=collection,
        query=vector,
        limit=k,
        score_threshold=settings.min_relevance_score,
        with_payload=True,
    ).points

    results = [
        RetrievedChunk(
            text=h.payload.get("text", ""),
            source=h.payload.get("source", ""),
            source_type=h.payload.get("source_type", ""),
            title=h.payload.get("title", ""),
            page=h.payload.get("page"),
            score=float(h.score),
        )
        for h in hits
    ]
    logger.info("Retrieved %d chunks for query (len=%d chars)", len(results), len(query))
    return results
