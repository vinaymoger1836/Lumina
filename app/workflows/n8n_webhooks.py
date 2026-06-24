"""Webhook endpoints that n8n workflows call (Phase 3 — workflow automation).

Two automations are supported:
- POST /webhooks/ingest  — n8n's Google Drive trigger fires on a new file and
  posts its URL here for idempotent auto-ingestion.
- POST /webhooks/digest  — an n8n cron workflow calls this each morning to get a
  summary of all ingested documents, which it then emails via its Gmail node.

Both are protected by a shared secret sent in the `X-Webhook-Token` header and
compared against `N8N_WEBHOOK_TOKEN`, so only the configured n8n instance can
trigger them.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import ConfigError, settings
from app.rag.ingest import IngestionError, ingest_url
from app.rag.vectorstore import delete_by_source
from app.workflows.digest import generate_digest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def require_webhook_token(x_webhook_token: str | None = Header(default=None)) -> None:
    """Authenticate an incoming webhook via the shared secret header.

    Raises 503 if the server has no token configured, 401 if the caller's token
    is missing or wrong. Uses a constant-time compare to avoid leaking the token
    through timing.
    """
    if not settings.n8n_webhook_token:
        raise HTTPException(
            status_code=503,
            detail="Webhooks are not configured (N8N_WEBHOOK_TOKEN is unset).",
        )
    if not x_webhook_token or not hmac.compare_digest(
        x_webhook_token, settings.n8n_webhook_token
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing webhook token.")


class IngestWebhookRequest(BaseModel):
    """Payload n8n posts when a new document URL should be ingested."""

    url: str = Field(..., min_length=1, description="An http(s) URL to ingest.")


class IngestWebhookResponse(BaseModel):
    """Result of an automated ingestion."""

    source: str
    title: str
    chunks: int
    replaced: bool = Field(
        description="True if a prior version of this source was removed first."
    )


class DigestResponse(BaseModel):
    """A document digest ready to be emailed by n8n."""

    generated_at: str
    document_count: int
    body: str


@router.post(
    "/ingest",
    response_model=IngestWebhookResponse,
    dependencies=[Depends(require_webhook_token)],
)
def ingest_webhook(req: IngestWebhookRequest) -> IngestWebhookResponse:
    """Idempotently ingest a document URL pushed by an n8n trigger."""
    # Delete any prior chunks for this URL first so retries don't duplicate data.
    try:
        delete_by_source(req.url)
        replaced = True
    except Exception as exc:  # noqa: BLE001 - absence/first-time is not an error
        logger.info("No existing chunks to replace for '%s' (%s)", req.url, exc)
        replaced = False

    try:
        result = ingest_url(req.url)
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return IngestWebhookResponse(
        source=result.source,
        title=result.title,
        chunks=result.chunks,
        replaced=replaced,
    )


@router.post(
    "/digest",
    response_model=DigestResponse,
    dependencies=[Depends(require_webhook_token)],
)
def digest_webhook() -> DigestResponse:
    """Return a plain-text digest of all ingested documents for emailing."""
    try:
        digest = generate_digest()
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface a clean 502 to n8n
        logger.error("Digest generation failed: %s", exc)
        raise HTTPException(
            status_code=502, detail="Failed to generate the digest."
        ) from exc

    return DigestResponse(
        generated_at=digest.generated_at,
        document_count=digest.document_count,
        body=digest.to_text(),
    )
