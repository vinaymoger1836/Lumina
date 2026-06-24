"""Scheduled-summary digest: summarize the ingested documents into an email body.

Phase 3 (workflow automation) calls `generate_digest()` from an n8n cron
workflow each morning; n8n then sends the returned text via its Gmail node. The
work here is read-only over Qdrant plus one LLM summary per document.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.llm import build_chat_model
from app.rag.vectorstore import ensure_collection, get_client

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = SystemMessage(
    content="Summarize the document excerpts into 1-2 concise, faithful sentences "
    "capturing the main topic. Do not invent details beyond the excerpts."
)


@dataclass(frozen=True)
class DocumentChunk:
    """One stored chunk's payload, used to assemble per-document text."""

    source: str
    title: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class DocumentSummary:
    """A one-line summary of a single ingested document."""

    title: str
    source: str
    summary: str


@dataclass(frozen=True)
class Digest:
    """A full digest of all ingested documents, ready to email."""

    generated_at: str
    document_count: int
    summaries: list[DocumentSummary]

    def to_text(self) -> str:
        """Render the digest as a plain-text email body."""
        if not self.summaries:
            return (
                f"Lumina daily digest — {self.generated_at}\n\n"
                "No documents have been ingested yet."
            )
        lines = [
            f"Lumina daily digest — {self.generated_at}",
            f"{self.document_count} document(s) in the knowledge base.",
            "",
        ]
        for i, s in enumerate(self.summaries, start=1):
            lines.append(f"{i}. {s.title}\n   {s.summary}")
        return "\n".join(lines)


def _scroll_chunks(limit_docs: int) -> list[DocumentChunk]:
    """Read chunk payloads from Qdrant (no vectors), bounded for safety."""
    collection = ensure_collection()
    client = get_client()
    chunks: list[DocumentChunk] = []
    # Hard cap on points scanned so a huge collection can't blow up the digest.
    max_points = limit_docs * settings.digest_chunks_per_doc * 4
    offset = None
    while len(chunks) < max_points:
        records, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            p = r.payload or {}
            chunks.append(
                DocumentChunk(
                    source=p.get("source", ""),
                    title=p.get("title", "") or p.get("source", ""),
                    chunk_index=int(p.get("chunk_index", 0)),
                    text=p.get("text", ""),
                )
            )
        if offset is None:  # no more pages
            break
    return chunks


def _group_by_document(
    chunks: list[DocumentChunk], limit_docs: int, chunks_per_doc: int
) -> list[tuple[str, str, str]]:
    """Group chunks by source into (source, title, combined_text) per document.

    Documents are ordered by first appearance; within a document, chunks are
    ordered by `chunk_index` and the first `chunks_per_doc` are concatenated.
    """
    order: list[str] = []
    by_source: dict[str, list[DocumentChunk]] = {}
    for c in chunks:
        if c.source not in by_source:
            if len(order) >= limit_docs:
                continue
            order.append(c.source)
            by_source[c.source] = []
        by_source[c.source].append(c)

    grouped: list[tuple[str, str, str]] = []
    for source in order:
        doc_chunks = sorted(by_source[source], key=lambda c: c.chunk_index)
        title = doc_chunks[0].title if doc_chunks else source
        text = "\n\n".join(c.text for c in doc_chunks[:chunks_per_doc] if c.text)
        grouped.append((source, title, text))
    return grouped


def _summarize_document(title: str, text: str) -> str:
    """Summarize one document's excerpts into a short line."""
    if not text.strip():
        return "(no extractable text)"
    messages = [
        _SUMMARY_SYSTEM,
        HumanMessage(content=f"Document: {title}\n\nExcerpts:\n{text}"),
    ]
    response = build_chat_model(temperature=0.3).invoke(messages)
    content = response.content
    return content.strip() if isinstance(content, str) else str(content)


def generate_digest() -> Digest:
    """Summarize every ingested document into an emailable digest."""
    settings.require("groq_api_key", "qdrant_url", "qdrant_api_key")
    chunks = _scroll_chunks(settings.digest_max_docs)
    grouped = _group_by_document(
        chunks, settings.digest_max_docs, settings.digest_chunks_per_doc
    )

    summaries: list[DocumentSummary] = []
    for source, title, text in grouped:
        try:
            summary = _summarize_document(title, text)
        except Exception as exc:  # noqa: BLE001 - one bad doc shouldn't fail the digest
            logger.error("Failed to summarize '%s': %s", source, exc)
            summary = "(summary unavailable)"
        summaries.append(
            DocumentSummary(title=title, source=source, summary=summary)
        )

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    logger.info("Generated digest for %d document(s)", len(summaries))
    return Digest(
        generated_at=generated_at,
        document_count=len(summaries),
        summaries=summaries,
    )
