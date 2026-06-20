---
title: Lumina
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
app_file: ui/streamlit_app.py
pinned: false
short_description: RAG over your own PDFs and URLs, answered by Llama 3.3 via Groq
---

# Lumina — AI Research & Knowledge Agent

> The block above is Hugging Face Spaces configuration. It is ignored by GitHub
> rendering and required by HF to launch the Streamlit app.

An end-to-end AI app: **RAG** over your own PDFs/URLs, an **agentic layer** (LangGraph
tool use), and **workflow automation** (n8n). Built on free-tier tools, deployed to
Hugging Face Spaces.

See [`CLAUDE.md`](./CLAUDE.md) for the full architecture, phases, and standing rules.

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
| `app/workflows/` | n8n webhook receivers (Phase 3) |
| `ui/streamlit_app.py` | Streamlit frontend |
| `tests/` | pytest suite |

## Status

Phase 1 (RAG Core) — in progress.
