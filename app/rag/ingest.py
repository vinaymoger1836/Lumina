"""Document ingestion: PDF/URL -> text -> chunks -> embeddings -> Qdrant.

Entry points are `ingest_pdf` and `ingest_url`. Both validate their input,
split text into overlapping chunks, embed them locally, and upsert the vectors
into Qdrant with metadata used later for retrieval and citations.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from urllib.parse import urlparse

import fitz  # PyMuPDF
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models as qmodels

from app.config import settings
from app.rag.embeddings import embed_texts
from app.rag.vectorstore import delete_by_source, ensure_collection, get_client

logger = logging.getLogger(__name__)

# WebBaseLoader emits a warning if no User-Agent is set; provide a polite default.
os.environ.setdefault("USER_AGENT", "Lumina/0.1 (+https://github.com) research-agent")

_PDF_MAGIC = b"%PDF-"


class IngestionError(ValueError):
    """Raised when input is invalid or yields no usable text."""


@dataclass(frozen=True)
class IngestResult:
    """Summary of one ingestion run."""

    source: str
    source_type: str  # "pdf" | "url"
    title: str
    chunks: int


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _validate_pdf(data: bytes, filename: str) -> None:
    """Reject non-PDF or oversized uploads with a clear message."""
    if not filename.lower().endswith(".pdf"):
        raise IngestionError(f"'{filename}' is not a .pdf file.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise IngestionError(
            f"File is {size_mb:.1f} MB; limit is {settings.max_upload_mb} MB."
        )
    if not data.startswith(_PDF_MAGIC):
        raise IngestionError(f"'{filename}' does not look like a valid PDF.")


def _validate_url(url: str) -> str:
    """Validate and normalise an http(s) URL."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise IngestionError(f"'{url}' is not a valid http(s) URL.")
    return parsed.geturl()


def _extract_pdf_pages(data: bytes) -> list[tuple[int, str]]:
    """Return (page_number, text) for each non-empty page (1-indexed)."""
    pages: list[tuple[int, str]] = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append((i, text))
    except Exception as exc:  # PyMuPDF raises various low-level errors
        raise IngestionError(f"Failed to parse PDF: {exc}") from exc
    if not pages:
        raise IngestionError("No extractable text found (is the PDF scanned/empty?).")
    return pages


def _upsert(
    *,
    texts: list[str],
    source: str,
    source_type: str,
    title: str,
    pages: list[int | None],
) -> int:
    """Embed chunks and upsert them into Qdrant. Returns the chunk count.

    Re-ingesting the same source replaces its chunks rather than duplicating
    them: we delete any prior chunks for this `source` before upserting.
    """
    collection = ensure_collection()
    delete_by_source(source, collection)
    vectors = embed_texts(texts)
    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": text,
                "source": source,
                "source_type": source_type,
                "title": title,
                "chunk_index": idx,
                "page": page,
            },
        )
        for idx, (text, vector, page) in enumerate(zip(texts, vectors, pages, strict=True))
    ]
    get_client().upsert(collection_name=collection, points=points)
    logger.info("Upserted %d chunks from '%s' into '%s'", len(points), source, collection)
    return len(points)


def ingest_pdf(data: bytes, filename: str) -> IngestResult:
    """Ingest a PDF (raw bytes) and return an ingestion summary."""
    _validate_pdf(data, filename)
    pages = _extract_pdf_pages(data)
    splitter = _splitter()

    texts: list[str] = []
    page_nums: list[int | None] = []
    for page_num, page_text in pages:
        for chunk in splitter.split_text(page_text):
            texts.append(chunk)
            page_nums.append(page_num)

    if not texts:
        raise IngestionError(f"'{filename}' produced no chunks after splitting.")

    count = _upsert(
        texts=texts,
        source=filename,
        source_type="pdf",
        title=filename,
        pages=page_nums,
    )
    return IngestResult(source=filename, source_type="pdf", title=filename, chunks=count)


def ingest_url(url: str) -> IngestResult:
    """Ingest the readable text of a web page and return a summary."""
    clean_url = _validate_url(url)
    try:
        docs = WebBaseLoader(clean_url).load()
    except Exception as exc:
        raise IngestionError(f"Failed to fetch '{clean_url}': {exc}") from exc

    full_text = "\n\n".join(d.page_content for d in docs if d.page_content).strip()
    if not full_text:
        raise IngestionError(f"No readable text found at '{clean_url}'.")

    title = (docs[0].metadata.get("title") or clean_url) if docs else clean_url
    chunks = _splitter().split_text(full_text)
    if not chunks:
        raise IngestionError(f"'{clean_url}' produced no chunks after splitting.")

    count = _upsert(
        texts=chunks,
        source=clean_url,
        source_type="url",
        title=title,
        pages=[None] * len(chunks),
    )
    return IngestResult(source=clean_url, source_type="url", title=title, chunks=count)
