"""Tests for the agentic layer: tool formatting and turn-source collection.

These avoid network, model loading, and live API calls: only pure logic is
exercised (result formatting and message bookkeeping).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import (
    _collect_turn,
    _dedupe_sources,
    _extract_failed_generation,
    _recover_tool_call,
)
from app.agent.tools import (
    SourceRef,
    _format_doc_results,
    _format_web_results,
)
from app.rag.retriever import RetrievedChunk

# --- Document result formatting ---------------------------------------------


def test_format_doc_results_empty() -> None:
    text, sources = _format_doc_results([])
    assert "No relevant passages" in text
    assert sources == []


def test_format_doc_results_numbers_and_cites() -> None:
    chunks = [
        RetrievedChunk(
            text="Solar output rose.",
            source="energy.pdf",
            source_type="pdf",
            title="energy.pdf",
            page=2,
            score=0.9,
        ),
        RetrievedChunk(
            text="Costs fell.",
            source="https://e.com",
            source_type="url",
            title="Energy News",
            page=None,
            score=0.7,
        ),
    ]
    text, sources = _format_doc_results(chunks)
    assert "[1]" in text and "[2]" in text
    assert "energy.pdf (p.2)" in text
    assert [s.kind for s in sources] == ["doc", "doc"]
    assert sources[0].label == "energy.pdf (p.2)"
    assert sources[1].location == "https://e.com"


# --- Web result formatting --------------------------------------------------


def test_format_web_results_empty() -> None:
    text, sources = _format_web_results([])
    assert "No web results" in text
    assert sources == []


def test_format_web_results_maps_fields() -> None:
    results = [
        {"title": "Title A", "url": "https://a.com", "content": "Body A"},
        {"url": "https://b.com", "content": "Body B"},  # missing title
    ]
    text, sources = _format_web_results(results)
    assert "Title A" in text and "https://a.com" in text
    assert sources[0] == SourceRef(label="Title A", location="https://a.com", kind="web")
    # Falls back to the URL when the title is absent.
    assert sources[1].label == "https://b.com"
    assert all(s.kind == "web" for s in sources)


# --- Source de-duplication --------------------------------------------------


def test_dedupe_sources_preserves_first_order() -> None:
    a = SourceRef(label="A", location="x", kind="doc")
    b = SourceRef(label="B", location="y", kind="web")
    a_dup = SourceRef(label="A", location="x", kind="doc")
    assert _dedupe_sources([a, b, a_dup]) == [a, b]


# --- Turn collection from message history -----------------------------------


def test_collect_turn_only_after_last_human() -> None:
    old_src = SourceRef(label="Old", location="old", kind="doc")
    new_src = SourceRef(label="New", location="new", kind="web")
    messages = [
        HumanMessage(content="first question"),
        ToolMessage(content="...", name="search_docs", tool_call_id="1", artifact=[old_src]),
        AIMessage(content="first answer"),
        HumanMessage(content="second question"),  # current turn starts here
        ToolMessage(content="...", name="web_search", tool_call_id="2", artifact=[new_src]),
        AIMessage(content="second answer"),
    ]
    sources, tools_used = _collect_turn(messages)
    assert sources == [new_src]
    assert tools_used == ["web_search"]


def test_collect_turn_handles_no_tools() -> None:
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    sources, tools_used = _collect_turn(messages)
    assert sources == []
    assert tools_used == []


# --- Malformed tool-call recovery (Groq tool_use_failed) --------------------


class _FakeGroqError(Exception):
    """Stands in for groq.BadRequestError, which carries a ``.body`` dict."""

    def __init__(self, failed_generation: str) -> None:
        self.body = {
            "error": {
                "message": "Failed to call a function.",
                "code": "tool_use_failed",
                "failed_generation": failed_generation,
            }
        }
        super().__init__(str(self.body))


def test_extract_failed_generation_from_body() -> None:
    exc = _FakeGroqError('<function=search_docs{"query": "notice period"}</function>')
    assert _extract_failed_generation(exc) == (
        '<function=search_docs{"query": "notice period"}</function>'
    )


def test_recover_tool_call_from_malformed_generation() -> None:
    # The exact Llama-native shape Groq rejected in the wild (no `>` after name).
    exc = _FakeGroqError('<function=search_docs{"query": "notice period"}</function>')
    message = _recover_tool_call(exc)
    assert message is not None
    assert len(message.tool_calls) == 1
    call = message.tool_calls[0]
    assert call["name"] == "search_docs"
    assert call["args"] == {"query": "notice period"}
    assert call["id"]


def test_recover_tool_call_rejects_unknown_tool() -> None:
    exc = _FakeGroqError('<function=not_a_real_tool{"x": 1}>')
    assert _recover_tool_call(exc) is None


def test_recover_tool_call_rejects_bad_json() -> None:
    exc = _FakeGroqError("<function=search_docs{not valid json}>")
    assert _recover_tool_call(exc) is None


def test_recover_tool_call_none_without_failed_generation() -> None:
    assert _recover_tool_call(ValueError("some other error")) is None
