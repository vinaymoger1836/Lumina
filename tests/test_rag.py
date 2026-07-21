"""Tests for RAG ingestion validation, chunking, and retrieval helpers.

These avoid network and model loading: only pure logic is exercised.
"""

from __future__ import annotations

import pytest

from typing import Any

from app.rag import retriever as retriever_module
from app.rag.ingest import IngestionError, _splitter, _validate_pdf, _validate_url
from app.rag.retriever import RetrievedChunk, search

# --- PDF validation ---------------------------------------------------------

def test_validate_pdf_rejects_wrong_extension() -> None:
    with pytest.raises(IngestionError, match="not a .pdf"):
        _validate_pdf(b"%PDF-1.4 ...", "notes.txt")


def test_validate_pdf_rejects_bad_magic() -> None:
    with pytest.raises(IngestionError, match="valid PDF"):
        _validate_pdf(b"GIF89a not a pdf", "fake.pdf")


def test_validate_pdf_rejects_oversized(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from app.rag import ingest

    # Settings is a frozen dataclass, so swap in a modified copy rather than
    # mutating a field in place.
    monkeypatch.setattr(ingest, "settings", replace(ingest.settings, max_upload_mb=1))
    big = b"%PDF-" + b"0" * (2 * 1024 * 1024)
    with pytest.raises(IngestionError, match="limit is"):
        _validate_pdf(big, "big.pdf")


def test_validate_pdf_accepts_valid() -> None:
    _validate_pdf(b"%PDF-1.7\n%...", "ok.pdf")  # should not raise


# --- URL validation ---------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "ftp://x.com", "not a url", "javascript:alert(1)"])
def test_validate_url_rejects_bad(bad: str) -> None:
    with pytest.raises(IngestionError):
        _validate_url(bad)


def test_validate_url_accepts_https() -> None:
    assert _validate_url("  https://example.com/a  ") == "https://example.com/a"


# --- Chunking ---------------------------------------------------------------

def test_splitter_produces_overlapping_chunks() -> None:
    text = "Sentence one. " * 400  # well over one chunk
    chunks = _splitter().split_text(text)
    assert len(chunks) > 1
    assert all(chunks)


# --- Citation formatting ----------------------------------------------------

def test_citation_includes_page_for_pdf() -> None:
    c = RetrievedChunk(
        text="x", source="r.pdf", source_type="pdf", title="r.pdf", page=3, score=0.9
    )
    assert c.citation() == "r.pdf (p.3)"


def test_citation_uses_title_for_url() -> None:
    c = RetrievedChunk(
        text="x",
        source="https://e.com",
        source_type="url",
        title="Example",
        page=None,
        score=0.5,
    )
    assert c.citation() == "Example"


# --- Retrieval relevance threshold ------------------------------------------

class _FakePoint:
    def __init__(self, payload: dict[str, Any], score: float) -> None:
        self.payload = payload
        self.score = score


class _FakeQueryResult:
    def __init__(self, points: list[_FakePoint]) -> None:
        self.points = points


class _RecordingClient:
    """Captures the kwargs passed to query_points and returns preset points."""

    def __init__(self, points: list[_FakePoint]) -> None:
        self._points = points
        self.last_kwargs: dict[str, Any] = {}

    def query_points(self, **kwargs: Any) -> _FakeQueryResult:
        self.last_kwargs = kwargs
        return _FakeQueryResult(self._points)


def test_search_forwards_min_relevance_score(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingClient([_FakePoint({"text": "hit", "source": "a.pdf"}, 0.7)])
    monkeypatch.setattr(retriever_module, "get_client", lambda: client)
    monkeypatch.setattr(retriever_module, "ensure_collection", lambda: "c")
    monkeypatch.setattr(retriever_module, "embed_query", lambda q: [0.0])

    results = search("q")

    assert client.last_kwargs["score_threshold"] == retriever_module.settings.min_relevance_score
    assert len(results) == 1
    assert results[0].text == "hit"
