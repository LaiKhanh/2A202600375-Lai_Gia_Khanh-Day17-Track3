"""
LangGraph state definition for the Multi-Memory Agent.
"""
from __future__ import annotations

from typing import Any, Dict, List
from typing_extensions import TypedDict


class MemoryState(TypedDict, total=False):
    """
    Shared state that flows through every node in the LangGraph graph.

    Fields
    ------
    messages : list[dict]
        Full conversation history as ``{"role": ..., "content": ...}`` dicts.
    user_profile : dict
        Long-term facts about the user (name, preferences, allergies, …).
        Conflicts are resolved by *last-write-wins*: newer facts overwrite old.
    episodes : list[dict]
        Recent episodic memories retrieved for the current turn.
    semantic_hits : list[str]
        Relevant text chunks retrieved from semantic (vector) memory.
    memory_budget : int
        Remaining token budget for context injection.
        Nodes must not exceed this when adding context to the prompt.
    query_intent : str
        One of "profile_recall", "factual_recall", "experience_recall",
        "semantic_search", or "general".  Set by the memory router.
    response : str
        The assistant's reply for the current turn (set by the response node).
    """

    messages: List[Dict[str, Any]]
    user_profile: Dict[str, Any]
    episodes: List[Dict[str, Any]]
    semantic_hits: List[str]
    memory_budget: int
    query_intent: str
    response: str
