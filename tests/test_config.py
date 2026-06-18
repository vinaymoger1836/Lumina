"""Tests for central configuration and secret validation."""

from __future__ import annotations

import pytest

from app.config import ConfigError, Settings


def test_require_raises_when_secret_missing() -> None:
    s = Settings(groq_api_key=None)
    with pytest.raises(ConfigError) as exc:
        s.require("groq_api_key")
    assert "GROQ_API_KEY" in str(exc.value)


def test_require_passes_when_present() -> None:
    s = Settings(groq_api_key="sk-test", qdrant_url="http://x", qdrant_api_key="k")
    # Should not raise.
    s.require("groq_api_key", "qdrant_url", "qdrant_api_key")


def test_defaults_are_applied() -> None:
    s = Settings()
    assert s.qdrant_collection
    assert "MiniLM" in s.embedding_model
    assert s.chunk_overlap < s.chunk_size
