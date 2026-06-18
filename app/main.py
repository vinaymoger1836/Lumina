"""FastAPI entry point.

Exposes ingestion and Q&A over HTTP. The Streamlit UI calls the RAG layer
directly, but this API powers programmatic use and the n8n webhooks in Phase 3.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import ConfigError
from app.logging_config import configure_logging
from app.rag.ingest import IngestionError, ingest_pdf, ingest_url
from app.rag.pipeline import answer_question

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Lumina API", version="0.1.0")


class UrlIngestRequest(BaseModel):
    """Request body for URL ingestion."""

    url: str = Field(..., min_length=1, description="An http(s) URL to ingest.")


class IngestResponse(BaseModel):
    """Result of an ingestion call."""

    source: str
    source_type: str
    title: str
    chunks: int


class AskRequest(BaseModel):
    """Request body for a question."""

    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    """A cited source returned alongside an answer."""

    citation: str
    source: str
    score: float


class AskResponse(BaseModel):
    """An answer plus its grounding sources."""

    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf_endpoint(file: UploadFile = File(...)) -> IngestResponse:
    """Ingest an uploaded PDF file."""
    data = await file.read()
    try:
        result = ingest_pdf(data, file.filename or "upload.pdf")
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return IngestResponse(**result.__dict__)


@app.post("/ingest/url", response_model=IngestResponse)
def ingest_url_endpoint(req: UrlIngestRequest) -> IngestResponse:
    """Ingest the readable text of a web page."""
    try:
        result = ingest_url(req.url)
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return IngestResponse(**result.__dict__)


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    """Answer a question grounded in the ingested documents."""
    try:
        result = answer_question(req.question, top_k=req.top_k)
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    sources = [
        Source(citation=c.citation(), source=c.source, score=c.score)
        for c in result.sources
    ]
    return AskResponse(answer=result.text, sources=sources)
