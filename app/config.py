"""Central configuration: loads secrets and settings from the environment.

This is the ONLY place the app reads raw environment variables. Every secret
comes from `.env` (local) or the host's secret store (HF Spaces / Render) — never
a hardcoded string. Import `settings` elsewhere instead of calling os.getenv.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env once at import time. In hosted environments (HF Spaces) there is no
# .env file and real env vars are used instead; load_dotenv is a no-op then.
load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _get(name: str, default: str | None = None) -> str | None:
    """Return an env var, treating empty/whitespace-only values as unset."""
    value = os.getenv(name, default)
    if value is not None:
        value = value.strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    """Typed view of all runtime configuration."""

    # --- Secrets (no defaults; required for the features that use them) ---
    groq_api_key: str | None = field(default_factory=lambda: _get("GROQ_API_KEY"))
    qdrant_url: str | None = field(default_factory=lambda: _get("QDRANT_URL"))
    qdrant_api_key: str | None = field(default_factory=lambda: _get("QDRANT_API_KEY"))
    tavily_api_key: str | None = field(default_factory=lambda: _get("TAVILY_API_KEY"))
    huggingface_token: str | None = field(
        default_factory=lambda: _get("HUGGINGFACE_TOKEN")
    )
    n8n_webhook_token: str | None = field(
        default_factory=lambda: _get("N8N_WEBHOOK_TOKEN")
    )

    # --- Tunables (safe defaults) ---
    qdrant_collection: str = field(
        default_factory=lambda: _get("QDRANT_COLLECTION", "lumina_docs")
    )
    embedding_model: str = field(
        default_factory=lambda: _get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    groq_model: str = field(
        default_factory=lambda: _get("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    # --- Ingestion / retrieval defaults ---
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    max_upload_mb: int = 25

    # --- Agent (Phase 2) ---
    web_search_results: int = 5
    agent_recursion_limit: int = 12

    # --- Workflow automation (Phase 3) ---
    digest_chunks_per_doc: int = 6  # chunks sampled per document when summarizing
    digest_max_docs: int = 25  # cap on documents included in a single digest

    def require(self, *names: str) -> None:
        """Raise ConfigError if any named secret attribute is unset.

        Call this at a feature's entry point so failures are explicit and early,
        with a clear message, instead of an opaque downstream API error.
        """
        missing = [n for n in names if not getattr(self, n, None)]
        if missing:
            env_names = {
                "groq_api_key": "GROQ_API_KEY",
                "qdrant_url": "QDRANT_URL",
                "qdrant_api_key": "QDRANT_API_KEY",
                "tavily_api_key": "TAVILY_API_KEY",
                "huggingface_token": "HUGGINGFACE_TOKEN",
                "n8n_webhook_token": "N8N_WEBHOOK_TOKEN",
            }
            pretty = ", ".join(env_names.get(n, n) for n in missing)
            raise ConfigError(
                f"Missing required configuration: {pretty}. "
                "Set it in your .env file (local) or host secrets."
            )


# Singleton imported across the app.
settings = Settings()
