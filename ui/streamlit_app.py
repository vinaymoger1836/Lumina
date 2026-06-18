"""Streamlit frontend for Lumina: upload documents and ask questions about them."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# Allow `import app...` when Streamlit runs this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.rag.ingest import IngestionError, ingest_pdf, ingest_url  # noqa: E402
from app.rag.pipeline import answer_question  # noqa: E402

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Lumina", page_icon="📚", layout="wide")


def _init_state() -> None:
    st.session_state.setdefault("ingested", [])  # list of human-readable labels
    st.session_state.setdefault("messages", [])  # chat history


def _sidebar() -> None:
    """Document ingestion controls."""
    with st.sidebar:
        st.header("📥 Add documents")

        uploaded = st.file_uploader(
            "Upload a PDF", type=["pdf"], accept_multiple_files=False
        )
        if uploaded is not None and st.button("Ingest PDF", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded.name}…"):
                try:
                    result = ingest_pdf(uploaded.getvalue(), uploaded.name)
                except (IngestionError, ConfigError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state.ingested.append(
                        f"📄 {result.title} ({result.chunks} chunks)"
                    )
                    st.success(f"Added {result.chunks} chunks from {result.title}.")

        st.divider()

        url = st.text_input("…or paste a URL", placeholder="https://example.com/article")
        if url and st.button("Ingest URL", use_container_width=True):
            with st.spinner(f"Fetching {url}…"):
                try:
                    result = ingest_url(url)
                except (IngestionError, ConfigError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state.ingested.append(
                        f"🔗 {result.title} ({result.chunks} chunks)"
                    )
                    st.success(f"Added {result.chunks} chunks from {result.title}.")

        if st.session_state.ingested:
            st.divider()
            st.caption("Ingested this session:")
            for label in st.session_state.ingested:
                st.write(label)


def _chat() -> None:
    """Q&A chat interface."""
    st.title("📚 Lumina")
    st.caption(
        f"Ask questions about your documents · model: `{settings.groq_model}`"
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(f"**[{i}]** {src['citation']} · score {src['score']:.3f}")

    prompt = st.chat_input("Ask a question about your documents…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = answer_question(prompt)
            except ConfigError as exc:
                st.error(str(exc))
                return
            except RuntimeError as exc:
                st.error(str(exc))
                return

        st.markdown(result.text)
        sources = [
            {"citation": c.citation(), "score": c.score} for c in result.sources
        ]
        if sources:
            with st.expander("Sources"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**[{i}]** {src['citation']} · score {src['score']:.3f}")

    st.session_state.messages.append(
        {"role": "assistant", "content": result.text, "sources": sources}
    )


def main() -> None:
    _init_state()
    _sidebar()
    _chat()


main()
