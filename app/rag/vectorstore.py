"""Qdrant vector store: client factory and collection management.

Centralises all Qdrant access so credentials are read in exactly one place and
the collection's vector config stays consistent with the embedding model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.rag.embeddings import embedding_dimension

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredDocument:
    """A distinct ingested document and how many chunks it occupies."""

    source: str
    title: str
    chunk_count: int


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Return a cached Qdrant client, validating credentials first."""
    settings.require("qdrant_url", "qdrant_api_key")
    logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=30,
    )


def ensure_collection(collection: str | None = None) -> str:
    """Create the collection if missing; return its name.
    Vector size is taken from the embedding model so the two can never drift.
    """
    name = collection or settings.qdrant_collection
    client = get_client()
    if client.collection_exists(name):
        return name

    dim = embedding_dimension()
    logger.info("Creating Qdrant collection '%s' (dim=%d, cosine)", name, dim)
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=dim, distance=qmodels.Distance.COSINE
        ),
    )
    # Index on `source` so we can filter/delete by document later.
    client.create_payload_index(
        collection_name=name,
        field_name="source",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )
    return name


def list_documents(collection: str | None = None) -> list[StoredDocument]:
    """List the distinct ingested documents with their chunk counts.

    Scrolls the collection reading payloads only (no vectors) and groups points
    by their `source` value, preserving first-seen order. Powers the knowledge-
    base management view so users can see and prune what retrieval draws from.
    """
    name = ensure_collection(collection)
    client = get_client()
    counts: dict[str, int] = {}
    titles: dict[str, str] = {}
    order: list[str] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            p = r.payload or {}
            source = p.get("source", "")
            if not source:
                continue
            if source not in counts:
                counts[source] = 0
                titles[source] = p.get("title") or source
                order.append(source)
            counts[source] += 1
        if offset is None:  # no more pages
            break
    return [
        StoredDocument(source=s, title=titles[s], chunk_count=counts[s])
        for s in order
    ]


def delete_by_source(source: str, collection: str | None = None) -> None:
    """Remove all chunks belonging to one document, by its `source` value.

    Used to make re-ingestion idempotent: delete the prior version of a document
    before upserting the new one, so retries (e.g. from n8n) don't create
    duplicate chunks.
    """
    name = ensure_collection(collection)
    get_client().delete(
        collection_name=name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source", match=qmodels.MatchValue(value=source)
                    )
                ]
            )
        ),
    )
    logger.info("Deleted existing chunks for source '%s' from '%s'", source, name)
