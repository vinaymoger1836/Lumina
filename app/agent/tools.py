"""Tools the LangGraph agent can call: document search, web search, summarize.

Each tool returns a string that the LLM reads. The two retrieval tools also
return a structured list of `SourceRef`s on the tool message's *artifact*
channel (LangChain's ``content_and_artifact`` response format), so citations can
be surfaced to the user without bloating the model's context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from app.config import settings
from app.llm import build_chat_model
from app.rag.retriever import RetrievedChunk, search

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceRef:
    """A source the agent consulted, for citation display."""

    label: str  # human-readable, e.g. 'report.pdf (p.3)' or an article title
    location: str  # filename/source id or URL
    kind: str  # 'doc' (ingested document) or 'web' (Tavily result)


def _format_doc_results(
    chunks: list[RetrievedChunk],
) -> tuple[str, list[SourceRef]]:
    """Render retrieved chunks into numbered LLM context + their SourceRefs."""
    if not chunks:
        return "No relevant passages were found in the ingested documents.", []
    blocks: list[str] = []
    sources: list[SourceRef] = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {c.citation()})\n{c.text}")
        sources.append(SourceRef(label=c.citation(), location=c.source, kind="doc"))
    return "\n\n".join(blocks), sources


def _format_web_results(results: list[dict]) -> tuple[str, list[SourceRef]]:
    """Render Tavily result dicts into numbered LLM context + their SourceRefs."""
    if not results:
        return "No web results were found for that query.", []
    blocks: list[str] = []
    sources: list[SourceRef] = []
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or r.get("url") or "Untitled").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        blocks.append(f"[{i}] {title} ({url})\n{content}")
        sources.append(SourceRef(label=title, location=url, kind="web"))
    return "\n\n".join(blocks), sources


@tool(response_format="content_and_artifact")
def search_docs(query: str) -> tuple[str, list[SourceRef]]:
    """Search the user's ingested documents (uploaded PDFs and URLs) for passages
    relevant to the query. Use this first for any question that could be answered
    from the user's own documents."""
    chunks = search(query)
    logger.info("Agent search_docs returned %d chunks", len(chunks))
    return _format_doc_results(chunks)


@tool(response_format="content_and_artifact")
def web_search(query: str) -> tuple[str, list[SourceRef]]:
    """Search the public web for current events or information that is not present
    in the ingested documents. Use this when the documents lack the answer or the
    question is about recent/external facts."""
    settings.require("tavily_api_key")
    # Imported lazily so the module loads even when Tavily isn't configured.
    from tavily import TavilyClient

    try:
        response = TavilyClient(api_key=settings.tavily_api_key).search(
            query=query,
            max_results=settings.web_search_results,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent as tool output
        logger.error("Tavily web search failed: %s", exc)
        raise RuntimeError(
            "Web search is currently unavailable. Try again shortly."
        ) from exc

    results = response.get("results", []) if isinstance(response, dict) else []
    logger.info("Agent web_search returned %d results", len(results))
    return _format_web_results(results)


@tool
def summarize(text: str) -> str:
    """Summarize a block of text into a few concise sentences. Use this to condense
    long passages gathered from documents or the web before giving a final answer."""
    if not text or not text.strip():
        return "There is nothing to summarize."
    messages = [
        SystemMessage(
            content="Summarize the user's text into a few concise, faithful "
            "sentences. Do not add information that is not present."
        ),
        HumanMessage(content=text),
    ]
    try:
        response = build_chat_model(temperature=0.3).invoke(messages)
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent as tool output
        logger.error("summarize tool LLM call failed: %s", exc)
        raise RuntimeError("Summarization is currently unavailable.") from exc
    content = response.content
    return content.strip() if isinstance(content, str) else str(content)


# Registered with the agent and its tool node, in this order.
TOOLS = [search_docs, web_search, summarize]
