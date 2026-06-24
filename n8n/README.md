# Lumina — n8n Workflow Automation (Phase 3)

This folder holds the exported [n8n](https://n8n.io/) workflows that automate
Lumina: event-driven document ingestion and a scheduled email digest. n8n calls
Lumina's FastAPI backend over HTTP.

## Architecture

```
Google Drive (new PDF) ──► n8n Drive Trigger ──► download ──► POST /ingest/pdf ──► Qdrant
Cron (daily 08:00)     ──► n8n Schedule Trigger ──► POST /webhooks/digest ──► Gmail send
```

## Workflows

| File | Trigger | What it does | Lumina endpoint |
|------|---------|--------------|-----------------|
| `workflows/drive_auto_ingest.json` | New file in a Google Drive folder | Downloads the file and ingests it | `POST /ingest/pdf` (multipart) |
| `workflows/daily_digest_email.json` | Cron `0 8 * * *` | Fetches a summary of all docs and emails it | `POST /webhooks/digest` |

The webhook receivers live in `app/workflows/n8n_webhooks.py`. There is also a
token-protected `POST /webhooks/ingest` (JSON `{"url": "..."}`) for ingesting
**web-page URLs** from any n8n trigger — see "Ingesting URLs" below.

## Prerequisites

1. **A running Lumina API** reachable from n8n (e.g. the FastAPI app deployed on
   Render). `uvicorn app.main:app` exposes the endpoints above.
2. **A running n8n instance** — self-host on the Render free tier (Docker image
   `n8nio/n8n`) or run locally with `npx n8n`.
3. **Credentials configured in n8n:**
   - *Google Drive OAuth2* (for the auto-ingest workflow)
   - *Gmail OAuth2* (for the digest email)

## Environment variables (set these in n8n → Settings → Variables, or the host)

| Variable | Used by | Example |
|----------|---------|---------|
| `LUMINA_API_BASE` | both workflows | `https://lumina-api.onrender.com` |
| `N8N_WEBHOOK_TOKEN` | digest workflow header | must equal Lumina's `N8N_WEBHOOK_TOKEN` |
| `DIGEST_RECIPIENT` | digest email | `you@example.com` |

> **Security:** `N8N_WEBHOOK_TOKEN` is the shared secret that authenticates n8n
> to Lumina's `/webhooks/*` endpoints. Generate one with
> `python -c "import secrets; print(secrets.token_urlsafe(32))"` and set the
> **same value** in both n8n and Lumina's environment. Never commit it.

## How to import

1. In n8n: **Workflows → Import from File** → choose a JSON file from
   `workflows/`.
2. Open each node marked `REPLACE_WITH_...` and select your real credential /
   Drive folder. (The IDs in the JSON are placeholders.)
3. Set the environment variables above.
4. Use **Execute Workflow** to test once, then toggle **Active** to enable the
   trigger/schedule.

## Ingesting URLs (optional, token-protected)

To auto-ingest **web articles** instead of Drive PDFs, point any n8n trigger
(RSS, Webhook, a Google Sheet of links, …) at the URL webhook:

```bash
curl -X POST "$LUMINA_API_BASE/webhooks/ingest" \
  -H "X-Webhook-Token: $N8N_WEBHOOK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

This endpoint is **idempotent**: re-posting the same URL deletes the prior
chunks for that source before re-ingesting, so n8n retries never create
duplicates.

## Notes

- The schedule uses cron `0 8 * * *` (08:00 in the n8n instance's timezone — set
  `GENERIC_TIMEZONE` on the n8n host).
- The digest call can take a while on large knowledge bases (one LLM summary per
  document); the HTTP nodes set a 120 s timeout.
