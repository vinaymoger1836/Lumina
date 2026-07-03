---
title: Lumina
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
app_file: ui/streamlit_app.py
pinned: false
short_description: RAG over your PDFs and URLs, answered by Llama 3.3
---

# Lumina — AI Research & Knowledge Agent

> The block above is Hugging Face Spaces configuration. It is ignored by GitHub
> rendering and required by HF to launch the Streamlit app.

An end-to-end AI app: **RAG** over your own PDFs/URLs, an **agentic layer** (LangGraph
tool use), and **workflow automation** (n8n). Built entirely on free-tier tools.

## What it does

1. **RAG (Phase 1)** — upload PDFs or paste URLs, ask questions, get answers grounded
   in those documents with source citations.
2. **Agentic layer (Phase 2)** — an "Agent Mode" that autonomously chooses between
   searching your documents, searching the web (Tavily), and summarizing, chaining
   steps with a LangGraph state machine.
3. **Workflow automation (Phase 3)** — n8n watches a Google Drive folder and
   auto-ingests new PDFs, and emails a daily digest of all documents via Gmail.

## Architecture

```
                    ┌──────────────────────── Streamlit UI (HF Spaces) ────────────────────────┐
                    │   PDF/URL upload · RAG Q&A · Agent Mode                                   │
                    └───────────────────────────────────┬──────────────────────────────────────┘
                                                         │ HTTP
                    ┌────────────────────────── FastAPI (AWS EC2) ─────────────────────────────┐
   PyMuPDF ────────►│  /ingest  ·  /ask  ·  /agent/ask  ·  /webhooks/{ingest,digest}           │
   WebBaseLoader    │      │            │            │                                          │
                    │      ▼            ▼            ▼                                          │
                    │  chunk+embed   retrieve    LangGraph agent (search_docs/web_search/…)     │
                    └──────┼────────────┼────────────┼──────────────────────────────────────────┘
                           ▼            ▼            ▼
                     Qdrant Cloud   Qdrant      Groq (Llama 3.3 70B) · Tavily
                    (embeddings, ANN cosine search)

   n8n (AWS EC2, Docker) ── Drive poll ──► POST /ingest/pdf ──► auto-ingest
                        └── cron 15:00 ──► POST /webhooks/digest ──► Gmail digest email
```

## Tech stack

| Layer | Tool | Notes |
|-------|------|-------|
| LLM | Groq API (Llama 3.3 70B) | Fast, generous free tier |
| Orchestration | LangChain + LangGraph | Chains + agent state machine |
| Vector DB | Qdrant Cloud | 1 GB free cluster, cosine ANN |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | 384-dim, runs locally |
| Web search | Tavily | Agent web-search tool |
| UI | Streamlit | Deployed on Hugging Face Spaces |
| Backend | FastAPI | RAG + agent + webhook endpoints |
| Automation | n8n (self-hosted) | Google Drive + Gmail workflows |
| Doc parsing | PyMuPDF | PDF text extraction |
| Deploy | Hugging Face Spaces (UI) · AWS EC2 (API + n8n) | Docker Compose on one t3.micro |

## Deployment

- **UI** → Hugging Face Spaces (Streamlit SDK, free CPU tier).
- **API + n8n** → one AWS EC2 `t3.micro` running both via `docker-compose.yml`. The
  n8n UI is reached over an SSH tunnel so Google OAuth works on a `localhost` redirect
  (no domain/TLS needed). Full runbook: [`DEPLOY_AWS.md`](./DEPLOY_AWS.md); workflow
  import + credentials: [`n8n/README.md`](./n8n/README.md).

## Quick start (local)

```bash
# 1. Create + activate a virtual environment (Python 3.12)
py -3.12 -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
copy .env.example .env            # then fill in your API keys

# 4. Run the UI
streamlit run ui/streamlit_app.py

# (optional) run the API
uvicorn app.main:app --reload
```

## Configuration

All secrets are read from environment variables (`.env` locally). See
[`.env.example`](./.env.example) for the full list. **Never commit `.env`.**

## Project layout

| Path | Purpose |
|------|---------|
| `app/config.py` | Central settings + secret loading (the only place env vars are read) |
| `app/rag/` | Ingestion, retrieval, and the RAG answer pipeline |
| `app/agent/` | LangGraph agent + tools (Phase 2) |
| `app/workflows/` | n8n webhook receivers + digest generator (Phase 3) |
| `ui/streamlit_app.py` | Streamlit frontend |
| `n8n/workflows/` | Exported n8n workflow JSON (Drive auto-ingest, daily digest) |
| `DEPLOY_AWS.md` | AWS EC2 deployment runbook |
| `tests/` | pytest suite |
