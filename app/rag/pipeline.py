"""RAG answer pipeline: retrieve -> build grounded prompt -> Groq LLM -> answer."""

from __future__ import annotations

import logging
from collections.abc import Iterator
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


_NO_DOCS_MESSAGE = (
    "No documents have been ingested yet, so I can't answer from sources. "
    "Upload a PDF or add a URL first."
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


def _build_messages(question: str, chunks: list[RetrievedChunk]) -> list:
    """Assemble the system + grounded-user messages for a question."""
    context = _build_context(chunks)
    return [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]


def _llm() -> ChatGroq:
    """Construct the Groq chat model, validating the API key first."""
    return build_chat_model(temperature=0.1)


def answer_question(question: str, top_k: int | None = None) -> Answer:
    """Answer a question grounded in the ingested documents."""
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    chunks = search(question, top_k=top_k)
    if not chunks:
        return Answer(text=_NO_DOCS_MESSAGE, sources=[])

    messages = _build_messages(question, chunks)

    try:
        response = _llm().invoke(messages)
    except Exception as exc:
        logger.error("Groq LLM call failed: %s", exc)
        raise RuntimeError(
            "The language model is currently unavailable. Please try again shortly."
        ) from exc

    text = response.content if isinstance(response.content, str) else str(response.content)
    return Answer(text=text.strip(), sources=chunks)


def stream_answer(
    question: str, top_k: int | None = None
) -> tuple[list[RetrievedChunk], Iterator[str]]:
    """Answer a question, streaming the LLM's text back token by token.

    Retrieval happens eagerly so the sources are known before the first token;
    the returned iterator yields answer text deltas as Groq produces them. Any
    ConfigError from key validation is raised here (before streaming begins).
    """
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    chunks = search(question, top_k=top_k)
    if not chunks:
        return [], iter([_NO_DOCS_MESSAGE])

    messages = _build_messages(question, chunks)
    llm = _llm()  # validates the key up front, before we start streaming

    def _tokens() -> Iterator[str]:
        try:
            for part in llm.stream(messages):
                content = part.content
                if isinstance(content, str) and content:
                    yield content
        except Exception as exc:
            logger.error("Groq LLM stream failed: %s", exc)
            raise RuntimeError(
                "The language model is currently unavailable. Please try again shortly."
            ) from exc

    return chunks, _tokens()
