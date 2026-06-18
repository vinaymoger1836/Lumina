"""Local text embeddings via sentence-transformers.

The model is loaded lazily and cached as a module-level singleton so the weights
(~90 MB for all-MiniLM-L6-v2) are loaded once per process, not per request.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()


def _dim(model: SentenceTransformer) -> int:
    """Output vector dimension, tolerant of the sentence-transformers rename."""
    getter = getattr(model, "get_embedding_dimension", None) or (
        model.get_sentence_embedding_dimension
    )
    return int(getter())


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and cache the embedding model (thread-safe, once per process)."""
    with _model_lock:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model ready (dim=%d)", _dim(model))
        return model


def embedding_dimension() -> int:
    """Return the output vector dimension of the configured embedding model."""
    return _dim(_load_model())


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents into normalized vectors."""
    if not texts:
        return []
    vectors = _load_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string into one normalized vector."""
    if not text or not text.strip():
        raise ValueError("Cannot embed an empty query.")
    return embed_texts([text])[0]
