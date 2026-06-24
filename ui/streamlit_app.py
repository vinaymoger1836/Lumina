"""Streamlit frontend for Lumina: upload documents and ask questions about them."""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

import streamlit as st

# Allow `import app...` when Streamlit runs this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph import run_agent  # noqa: E402
from app.config import ConfigError, settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.rag.ingest import IngestionError, ingest_pdf, ingest_url  # noqa: E402
from app.rag.pipeline import answer_question  # noqa: E402

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Lumina", page_icon="📚", layout="wide")


def _init_state() -> None:
    st.session_state.setdefault("ingested", [])  # list of human-readable labels
    st.session_state.setdefault("messages", [])  # RAG chat history
    st.session_state.setdefault("agent_messages", [])  # agent chat history
    # Stable per-session id so the agent's checkpointer keeps conversation memory.
    st.session_state.setdefault("agent_thread_id", uuid.uuid4().hex)


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
    """RAG Q&A chat interface (answers strictly from ingested documents)."""
    st.caption(
        f"Ask questions answered strictly from your documents · "
        f"model: `{settings.groq_model}`"
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


_TOOL_LABELS = {
    "search_docs": "📄 search_docs",
    "web_search": "🌐 web_search",
    "summarize": "📝 summarize",
}


def _render_agent_sources(sources: list[dict]) -> None:
    """Show the agent's consulted sources, linking web results."""
    if not sources:
        return
    with st.expander("Sources"):
        for i, src in enumerate(sources, start=1):
            if src["kind"] == "web" and src["location"]:
                st.markdown(f"**[{i}]** 🌐 [{src['label']}]({src['location']})")
            else:
                st.markdown(f"**[{i}]** 📄 {src['label']}")


def _agent_chat() -> None:
    """Autonomous agent interface (doc search + web search + reasoning)."""
    st.caption(
        "The agent decides which tools to use — your documents, web search, "
        "and summarization — and can chain several steps."
    )

    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools_used"):
                tags = " ".join(
                    _TOOL_LABELS.get(t, t) for t in dict.fromkeys(msg["tools_used"])
                )
                st.caption(f"Tools used: {tags}")
            _render_agent_sources(msg.get("sources", []))

    prompt = st.chat_input("Ask the agent anything…", key="agent_input")
    if not prompt:
        return

    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning and using tools…"):
            try:
                result = run_agent(
                    prompt, thread_id=st.session_state.agent_thread_id
                )
            except (ConfigError, RuntimeError) as exc:
                st.error(str(exc))
                return

        st.markdown(result.text)
        if result.tools_used:
            tags = " ".join(
                _TOOL_LABELS.get(t, t) for t in dict.fromkeys(result.tools_used)
            )
            st.caption(f"Tools used: {tags}")
        sources = [
            {"label": s.label, "location": s.location, "kind": s.kind}
            for s in result.sources
        ]
        _render_agent_sources(sources)

    st.session_state.agent_messages.append(
        {
            "role": "assistant",
            "content": result.text,
            "sources": sources,
            "tools_used": result.tools_used,
        }
    )


def main() -> None:
    _init_state()
    st.title("📚 Lumina")
    _sidebar()
    qa_tab, agent_tab = st.tabs(["💬 Q&A", "🤖 Agent Mode"])
    with qa_tab:
        _chat()
    with agent_tab:
        _agent_chat()


main()
