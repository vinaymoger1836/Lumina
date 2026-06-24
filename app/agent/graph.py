"""LangGraph agent: autonomously chooses tools to answer a question.

Graph shape (matches the Phase 2 design):

    START -> agent -> (tools_condition router) -> tools -> agent -> ... -> END

The `agent` node is the Groq LLM bound to the tools; `tools_condition` routes to
the `tools` node whenever the model emits tool calls, otherwise to END. A
`MemorySaver` checkpointer keyed by `thread_id` provides short-term conversation
memory so follow-up questions retain context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.tools import TOOLS, SourceRef
from app.config import settings
from app.llm import build_chat_model

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are Lumina, an autonomous research assistant. You have three tools:\n"
        "- search_docs: search the user's ingested documents.\n"
        "- web_search: search the public web for current or external facts.\n"
        "- summarize: condense long text.\n\n"
        "Strategy: for questions that could be answered from the user's documents, "
        "call search_docs first. If the documents don't contain the answer, or the "
        "question concerns recent or external information, use web_search. You may "
        "chain tools across multiple steps. When you have enough information, give a "
        "clear final answer and cite the sources you used with their bracketed "
        "numbers, e.g. [1], [2]. If you cannot find an answer, say so honestly."
    )
)


def _agent_node(state: MessagesState) -> dict:
    """LLM step: decide on a tool call or produce the final answer."""
    llm = build_chat_model(temperature=0.0).bind_tools(TOOLS)
    response = llm.invoke([_SYSTEM_PROMPT, *state["messages"]])
    return {"messages": [response]}


def _build_graph() -> CompiledStateGraph:
    """Compile the agent StateGraph with an in-memory checkpointer."""
    graph = StateGraph(MessagesState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    # tools_condition routes to "tools" when the LLM emitted tool calls, else END.
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


# Compiled once and reused so the checkpointer's memory persists across turns
# within a single process (e.g. one Streamlit session / one API worker).
_GRAPH: CompiledStateGraph | None = None


def get_graph() -> CompiledStateGraph:
    """Return the lazily-compiled, process-wide agent graph."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


@dataclass(frozen=True)
class AgentResult:
    """An agent answer with the sources consulted and tools used this turn."""

    text: str
    sources: list[SourceRef]
    tools_used: list[str]


def _dedupe_sources(sources: list[SourceRef]) -> list[SourceRef]:
    """Drop duplicate sources, preserving first-seen order."""
    seen: set[tuple[str, str]] = set()
    unique: list[SourceRef] = []
    for s in sources:
        key = (s.label, s.location)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def _collect_turn(messages: list[AnyMessage]) -> tuple[list[SourceRef], list[str]]:
    """Pull sources and tool names from the messages added since the last question.

    Only tool messages that follow the final HumanMessage belong to the current
    turn, so follow-up questions don't re-surface earlier turns' citations.
    """
    last_human = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
        default=-1,
    )
    sources: list[SourceRef] = []
    tools_used: list[str] = []
    for m in messages[last_human + 1 :]:
        if isinstance(m, ToolMessage):
            if m.name:
                tools_used.append(m.name)
            if isinstance(m.artifact, list):
                sources.extend(s for s in m.artifact if isinstance(s, SourceRef))
    return _dedupe_sources(sources), tools_used


def run_agent(question: str, thread_id: str = "default") -> AgentResult:
    """Run the agent on a question, returning the answer plus sources and tools used."""
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    settings.require("groq_api_key")

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.agent_recursion_limit,
    }
    try:
        state = get_graph().invoke(
            {"messages": [HumanMessage(content=question)]}, config
        )
    except Exception as exc:
        logger.error("Agent run failed: %s", exc)
        raise RuntimeError(
            "The agent is currently unavailable. Please try again shortly."
        ) from exc

    messages = state["messages"]
    final = messages[-1].content if messages else ""
    text = final.strip() if isinstance(final, str) else str(final)
    sources, tools_used = _collect_turn(messages)
    logger.info("Agent answered using tools=%s, %d sources", tools_used, len(sources))
    return AgentResult(text=text, sources=sources, tools_used=tools_used)
