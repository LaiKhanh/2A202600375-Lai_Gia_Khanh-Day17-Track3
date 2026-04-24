"""
LangGraph graph wiring for the Multi-Memory Agent.

Graph topology (linear pipeline):
  route_query → retrieve_memory → trim_context → generate_response → update_memory → END
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import StateGraph, END

from memory import EpisodicMemory, LongTermMemory, SemanticMemory, ShortTermMemory
from .nodes import (
    DEFAULT_BUDGET,
    make_generate_response,
    make_retrieve_memory,
    make_update_memory,
    route_query,
    trim_context,
)
from .state import MemoryState


def build_graph(
    short_term: Optional[ShortTermMemory] = None,
    long_term: Optional[LongTermMemory] = None,
    episodic: Optional[EpisodicMemory] = None,
    semantic: Optional[SemanticMemory] = None,
    llm: Optional[Callable] = None,
):
    """
    Construct and compile the LangGraph StateGraph.

    Parameters
    ----------
    short_term, long_term, episodic, semantic:
        Memory backend instances.  Fresh defaults are created when *None*.
    llm:
        ``(system_prompt: str, user_message: str) -> str`` callable.
        When *None* the deterministic mock LLM is used.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready for ``.invoke()``.
    """
    st = short_term if short_term is not None else ShortTermMemory()
    lt = long_term if long_term is not None else LongTermMemory()
    ep = episodic if episodic is not None else EpisodicMemory()
    sem = semantic if semantic is not None else SemanticMemory()

    # build nodes
    retrieve_memory_node = make_retrieve_memory(st, lt, ep, sem)
    generate_response_node = make_generate_response(llm)
    update_memory_node = make_update_memory(st, lt, ep)

    graph = StateGraph(MemoryState)

    graph.add_node("route_query", route_query)
    graph.add_node("retrieve_memory", retrieve_memory_node)
    graph.add_node("trim_context", trim_context)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("update_memory", update_memory_node)

    graph.set_entry_point("route_query")
    graph.add_edge("route_query", "retrieve_memory")
    graph.add_edge("retrieve_memory", "trim_context")
    graph.add_edge("trim_context", "generate_response")
    graph.add_edge("generate_response", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile()


class MultiMemoryAgent:
    """
    High-level wrapper around the compiled LangGraph graph.

    Usage
    -----
    >>> agent = MultiMemoryAgent()
    >>> response = agent.chat("Tên tôi là Linh")
    >>> response = agent.chat("Tên tôi là gì?")
    'Tên của bạn là linh.'
    """

    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        episodic: Optional[EpisodicMemory] = None,
        semantic: Optional[SemanticMemory] = None,
        llm: Optional[Callable] = None,
        memory_budget: int = DEFAULT_BUDGET,
        redis_url: Optional[str] = None,
    ) -> None:
        self.short_term = short_term if short_term is not None else ShortTermMemory()
        self.long_term = long_term if long_term is not None else LongTermMemory(redis_url=redis_url)
        self.episodic = episodic if episodic is not None else EpisodicMemory()
        self.semantic = semantic if semantic is not None else SemanticMemory()
        self.memory_budget = memory_budget

        self._graph = build_graph(
            short_term=self.short_term,
            long_term=self.long_term,
            episodic=self.episodic,
            semantic=self.semantic,
            llm=llm,
        )

        self._state: MemoryState = {
            "messages": [],
            "user_profile": {},
            "episodes": [],
            "semantic_hits": [],
            "memory_budget": memory_budget,
            "query_intent": "general",
            "response": "",
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Send *user_message* and return the assistant's response."""
        # append user message to current state
        messages = list(self._state.get("messages", []))
        messages.append({"role": "user", "content": user_message})

        self._state = {
            **self._state,
            "messages": messages,
            "memory_budget": self.memory_budget,
        }

        self._state = self._graph.invoke(self._state)
        return self._state.get("response", "")

    def reset(self) -> None:
        """Clear all memory and reset state (useful between benchmark runs)."""
        self.short_term.clear()
        self.long_term.clear()
        self.episodic.clear()
        self.semantic.clear()
        self._state = {
            "messages": [],
            "user_profile": {},
            "episodes": [],
            "semantic_hits": [],
            "memory_budget": self.memory_budget,
            "query_intent": "general",
            "response": "",
        }

    def get_profile(self) -> Dict[str, Any]:
        return self.long_term.get_all()

    def add_knowledge(self, texts: List[str]) -> None:
        """Pre-load semantic memory with document chunks."""
        self.semantic.add_documents(texts)
