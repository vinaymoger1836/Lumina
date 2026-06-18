"""Qdrant vector store: client factory and collection management.

Centralises all Qdrant access so credentials are read in exactly one place and
the collection's vector config stays consistent with the embedding model.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.rag.embeddings import embedding_dimension

logger = logging.getLogger(__name__)


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
