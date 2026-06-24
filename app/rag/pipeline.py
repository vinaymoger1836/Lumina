"""RAG answer pipeline: retrieve -> build grounded prompt -> Groq LLM -> answer."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.llm import build_chat_model
from app.rag.retriever import RetrievedChunk, search

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are Lumina, a careful research assistant. Answer the user's question "
    "using ONLY the provided context. If the context does not contain the answer, "
    "say you don't know based on the available documents — do not invent facts. "
    "Cite the sources you used by their bracketed numbers, e.g. [1], [2]."
)


@dataclass(frozen=True)
class Answer:
    """An LLM answer plus the chunks that grounded it."""

    text: str
    sources: list[RetrievedChunk]


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {c.citation()})\n{c.text}")
    return "\n\n".join(blocks)


def _llm() -> ChatGroq:
    """Construct the Groq chat model, validating the API key first."""
    return build_chat_model(temperature=0.1)


def answer_question(question: str, top_k: int | None = None) -> Answer:
    """Answer a question grounded in the ingested documents."""
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    chunks = search(question, top_k=top_k)
    if not chunks:
        return Answer(
            text="No documents have been ingested yet, so I can't answer from sources. "
            "Upload a PDF or add a URL first.",
            sources=[],
        )

    context = _build_context(chunks)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ),
    ]

    try:
        response = _llm().invoke(messages)
    except Exception as exc:
        logger.error("Groq LLM call failed: %s", exc)
        raise RuntimeError(
            "The language model is currently unavailable. Please try again shortly."
        ) from exc

    text = response.content if isinstance(response.content, str) else str(response.content)
    return Answer(text=text.strip(), sources=chunks)
