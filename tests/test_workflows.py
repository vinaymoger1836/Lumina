"""Tests for Phase 3 workflow automation: digest assembly and webhook auth.

Network, Qdrant, and LLM calls are avoided: grouping/rendering logic is tested
directly, and the webhook endpoints are exercised with their external calls
monkeypatched out.
"""

from __future__ import annotations

import dataclasses

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.workflows import n8n_webhooks
from app.workflows.digest import (
    Digest,
    DocumentChunk,
    DocumentSummary,
    _group_by_document,
)

# --- Digest grouping logic --------------------------------------------------


def test_group_orders_documents_and_chunks() -> None:
    chunks = [
        DocumentChunk(source="a", title="A", chunk_index=1, text="second"),
        DocumentChunk(source="a", title="A", chunk_index=0, text="first"),
        DocumentChunk(source="b", title="B", chunk_index=0, text="b0"),
    ]
    grouped = _group_by_document(chunks, limit_docs=10, chunks_per_doc=10)
    assert [g[0] for g in grouped] == ["a", "b"]  # first-appearance order
    assert grouped[0][2] == "first\n\nsecond"  # within-doc, by chunk_index


def test_group_respects_limit_docs() -> None:
    chunks = [
        DocumentChunk(source=s, title=s, chunk_index=0, text="x")
        for s in ("a", "b", "c")
    ]
    grouped = _group_by_document(chunks, limit_docs=2, chunks_per_doc=10)
    assert [g[0] for g in grouped] == ["a", "b"]


def test_group_respects_chunks_per_doc() -> None:
    chunks = [
        DocumentChunk(source="a", title="A", chunk_index=i, text=f"t{i}")
        for i in range(5)
    ]
    grouped = _group_by_document(chunks, limit_docs=10, chunks_per_doc=2)
    assert grouped[0][2] == "t0\n\nt1"


# --- Digest rendering -------------------------------------------------------


def test_digest_to_text_empty() -> None:
    d = Digest(generated_at="X", document_count=0, summaries=[])
    assert "No documents" in d.to_text()


def test_digest_to_text_lists_documents() -> None:
    d = Digest(
        generated_at="X",
        document_count=1,
        summaries=[
            DocumentSummary(title="Report", source="r.pdf", summary="About energy.")
        ],
    )
    text = d.to_text()
    assert "Report" in text and "About energy." in text


# --- Webhook authentication -------------------------------------------------


def _set_token(monkeypatch: pytest.MonkeyPatch, token: str | None) -> None:
    """Swap in a Settings copy with the given webhook token (frozen dataclass)."""
    monkeypatch.setattr(
        n8n_webhooks,
        "settings",
        dataclasses.replace(n8n_webhooks.settings, n8n_webhook_token=token),
    )


def test_require_token_unset_raises_503(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_token(monkeypatch, None)
    with pytest.raises(HTTPException) as exc:
        n8n_webhooks.require_webhook_token(None)
    assert exc.value.status_code == 503


def test_require_token_wrong_raises_401(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_token(monkeypatch, "secret")
    with pytest.raises(HTTPException) as exc:
        n8n_webhooks.require_webhook_token("nope")
    assert exc.value.status_code == 401


def test_require_token_correct_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_token(monkeypatch, "secret")
    assert n8n_webhooks.require_webhook_token("secret") is None


# --- Webhook endpoints (external calls monkeypatched) -----------------------


def _client() -> TestClient:
    import app.main

    return TestClient(app.main.app)


def test_digest_endpoint_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_token(monkeypatch, "secret")
    monkeypatch.setattr(
        n8n_webhooks,
        "generate_digest",
        lambda: Digest(
            generated_at="2026-06-24 08:00 UTC",
            document_count=1,
            summaries=[DocumentSummary(title="t", source="s", summary="sum")],
        ),
    )
    r = _client().post("/webhooks/digest", headers={"X-Webhook-Token": "secret"})
    assert r.status_code == 200
    assert "Lumina daily digest" in r.json()["body"]


def test_digest_endpoint_rejects_bad_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_token(monkeypatch, "secret")
    r = _client().post("/webhooks/digest", headers={"X-Webhook-Token": "wrong"})
    assert r.status_code == 401


def test_ingest_endpoint_idempotent_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.rag.ingest import IngestResult

    _set_token(monkeypatch, "secret")
    monkeypatch.setattr(n8n_webhooks, "delete_by_source", lambda source: None)
    monkeypatch.setattr(
        n8n_webhooks,
        "ingest_url",
        lambda url: IngestResult(
            source=url, source_type="url", title="Title", chunks=3
        ),
    )
    r = _client().post(
        "/webhooks/ingest",
        headers={"X-Webhook-Token": "secret"},
        json={"url": "https://example.com/a"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"] == 3
    assert body["replaced"] is True
