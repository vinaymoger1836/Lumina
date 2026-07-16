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
from app.rag.vectorstore import delete_by_source, list_documents  # noqa: E402

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Lumina", page_icon="📚", layout="wide")


_CHAT_CSS = """
<style>
/* --- Tighten the top so the header sits cleanly at the top --- */
[data-testid="stMainBlockContainer"] {
    padding-top: 2.5rem;
}

/* --- ChatGPT/Claude-style chat bubbles --- */

/* Base bubble: constrain width, round the corners, add padding. */
[data-testid="stChatMessage"] {
    width: fit-content;
    max-width: 82%;
    border-radius: 18px;
    padding: 0.15rem 0.9rem;
    margin-bottom: 0.6rem;
}

/* Assistant (AI) → left side, neutral bubble. */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(128, 128, 128, 0.12);
    margin-right: auto;
}

/* User → right side, accent bubble, avatar flipped to the right. */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(59, 130, 246, 0.16);
    margin-left: auto;
    flex-direction: row-reverse;
}

/* Give the docked input bar a subtle top divider so it reads as a footer. */
[data-testid="stBottomBlockContainer"] {
    border-top: 1px solid rgba(128, 128, 128, 0.18);
}
</style>
"""


def _inject_css() -> None:
    """Style chat messages as left/right bubbles and pin the input to the bottom."""
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)


def _init_state() -> None:
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
                    st.success(f"Added {result.chunks} chunks from {result.title}.")

        _knowledge_base()


def _knowledge_base() -> None:
    """List the documents held in Qdrant and let the user delete any of them."""
    st.divider()
    st.subheader("📚 Knowledge base")
    try:
        docs = list_documents()
    except ConfigError as exc:
        st.info(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - surface Qdrant/connectivity errors in-UI
        st.error(f"Could not load the knowledge base: {exc}")
        return

    if not docs:
        st.caption("No documents ingested yet.")
        return

    for doc in docs:
        row, action = st.columns([0.82, 0.18])
        row.markdown(f"📄 {doc.title}")
        row.caption(f"{doc.chunk_count} chunks")
        if action.button("🗑", key=f"del_{doc.source}", help=f"Delete {doc.title}"):
            with st.spinner(f"Removing {doc.title}…"):
                try:
                    delete_by_source(doc.source)
                except Exception as exc:  # noqa: BLE001 - report deletion failure in-UI
                    st.error(f"Delete failed: {exc}")
                else:
                    st.rerun()


def _render_qa_sources(sources: list[dict]) -> None:
    """Show the RAG citations for one message inside an expander."""
    if not sources:
        return
    with st.expander("Sources"):
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**[{i}]** {src['citation']} · score {src['score']:.3f}")


def _render_qa_history() -> None:
    """Render the RAG Q&A conversation so far."""
    st.caption(
        f"Ask questions answered strictly from your documents · "
        f"model: `{settings.groq_model}`"
    )
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_qa_sources(msg.get("sources", []))


def _process_qa(prompt: str) -> None:
    """Answer a new RAG question and append both turns to history."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = answer_question(prompt)
            except (ConfigError, RuntimeError) as exc:
                st.error(str(exc))
                return

        st.markdown(result.text)
        sources = [
            {"citation": c.citation(), "score": c.score} for c in result.sources
        ]
        _render_qa_sources(sources)

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


def _render_tools_used(tools_used: list[str]) -> None:
    """Render the caption listing which tools the agent invoked."""
    if not tools_used:
        return
    tags = " ".join(_TOOL_LABELS.get(t, t) for t in dict.fromkeys(tools_used))
    st.caption(f"Tools used: {tags}")


def _render_agent_history() -> None:
    """Render the agent-mode conversation so far."""
    st.caption(
        "The agent decides which tools to use — your documents, web search, "
        "and summarization — and can chain several steps."
    )
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_tools_used(msg.get("tools_used", []))
            _render_agent_sources(msg.get("sources", []))


def _process_agent(prompt: str) -> None:
    """Run the agent on a new question and append both turns to history."""
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning and using tools…"):
            try:
                result = run_agent(prompt, thread_id=st.session_state.agent_thread_id)
            except (ConfigError, RuntimeError) as exc:
                st.error(str(exc))
                return

        st.markdown(result.text)
        _render_tools_used(result.tools_used)
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


_QA_MODE = "💬 Q&A"
_AGENT_MODE = "🤖 Agent Mode"


def main() -> None:
    _init_state()
    _inject_css()
    st.title("📚 Lumina")
    _sidebar()

    # A top-level toggle (not st.tabs) so the single chat input below stays
    # docked to the bottom of the viewport — chat_input only auto-docks when it
    # is a top-level element, which it cannot be inside a tab.
    mode = st.segmented_control(
        "Mode",
        options=[_QA_MODE, _AGENT_MODE],
        default=_QA_MODE,
        label_visibility="collapsed",
        key="chat_mode",
    )

    if mode == _AGENT_MODE:
        _render_agent_history()
        placeholder = "Ask the agent anything…"
    else:
        _render_qa_history()
        placeholder = "Ask a question about your documents…"

    prompt = st.chat_input(placeholder)
    if prompt:
        if mode == _AGENT_MODE:
            _process_agent(prompt)
        else:
            _process_qa(prompt)


main()
