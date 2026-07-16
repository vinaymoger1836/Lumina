"""Tests for knowledge-base management: listing and deleting documents.

These avoid network by faking the Qdrant client's scroll and by stubbing the
store functions at the API boundary, exercising only pure grouping/routing logic.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.rag import vectorstore


class _FakeRecord:
    """Stand-in for a Qdrant scroll record carrying only a payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


class _FakeClient:
    """Returns preset pages of records, paginating like Qdrant's scroll()."""

    def __init__(self, pages: list[list[_FakeRecord]]) -> None:
        self._pages = pages
        self._i = 0

    def scroll(self, **_kwargs: Any) -> tuple[list[_FakeRecord], int | None]:
        page = self._pages[self._i] if self._i < len(self._pages) else []
        self._i += 1
        offset = self._i if self._i < len(self._pages) else None
        return page, offset


def _patch_store(monkeypatch: pytest.MonkeyPatch, pages: list[list[_FakeRecord]]) -> None:
    monkeypatch.setattr(vectorstore, "ensure_collection", lambda collection=None: "c")
    monkeypatch.setattr(vectorstore, "get_client", lambda: _FakeClient(pages))


# --- list_documents grouping ------------------------------------------------

def test_list_documents_groups_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store(
        monkeypatch,
        [
            [
                _FakeRecord({"source": "a.pdf", "title": "A", "chunk_index": 0}),
                _FakeRecord({"source": "a.pdf", "title": "A", "chunk_index": 1}),
            ],
            [_FakeRecord({"source": "https://x.com", "title": "X"})],
        ],
    )
    docs = {d.source: d for d in vectorstore.list_documents()}
    assert docs["a.pdf"].chunk_count == 2
    assert docs["a.pdf"].title == "A"
    assert docs["https://x.com"].chunk_count == 1


def test_list_documents_skips_blank_source(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store(
        monkeypatch,
        [[_FakeRecord({"title": "orphan"}), _FakeRecord({"source": ""})]],
    )
    assert vectorstore.list_documents() == []


def test_list_documents_falls_back_to_source_for_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_store(monkeypatch, [[_FakeRecord({"source": "no-title.pdf"})]])
    docs = vectorstore.list_documents()
    assert docs[0].title == "no-title.pdf"


# --- API endpoints ----------------------------------------------------------

def test_get_documents_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = [vectorstore.StoredDocument(source="a.pdf", title="A", chunk_count=3)]
    monkeypatch.setattr(main_module, "list_documents", lambda: docs)
    resp = TestClient(app).get("/documents")
    assert resp.status_code == 200
    assert resp.json()["documents"][0] == {
        "source": "a.pdf",
        "title": "A",
        "chunk_count": 3,
    }


def test_delete_document_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, str] = {}
    monkeypatch.setattr(
        main_module,
        "delete_by_source",
        lambda source, collection=None: called.__setitem__("source", source),
    )
    resp = TestClient(app).delete("/documents/a.pdf")
    assert resp.status_code == 200
    assert resp.json() == {"source": "a.pdf", "deleted": True}
    assert called["source"] == "a.pdf"
