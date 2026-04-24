"""
Short-term memory – sliding-window conversation buffer.

Stores the most recent N messages in a plain list.  When the window is
exceeded the oldest messages are evicted first (FIFO).
"""
from __future__ import annotations

from typing import List


class ShortTermMemory:
    """Conversation buffer with a configurable sliding window."""

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages
        self._buffer: List[dict] = []

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a message and evict oldest if over the window size."""
        self._buffer.append({"role": role, "content": content})
        if len(self._buffer) > self.max_messages:
            self._buffer = self._buffer[-self.max_messages :]

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get_recent(self, n: int | None = None) -> List[dict]:
        """Return the last *n* messages (or all if *n* is None)."""
        if n is None:
            return list(self._buffer)
        return list(self._buffer[-n:])

    def get_all(self) -> List[dict]:
        return list(self._buffer)

    # ------------------------------------------------------------------
    # management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._buffer.clear()

    def trim_to_budget(self, token_budget: int, tokens_per_message: int = 50) -> None:
        """
        Evict oldest messages until the estimated token usage fits within
        *token_budget*.  Uses a very rough estimate: each message ≈
        *tokens_per_message* tokens.
        """
        while self._buffer and len(self._buffer) * tokens_per_message > token_budget:
            self._buffer.pop(0)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ShortTermMemory(messages={len(self._buffer)}, max={self.max_messages})"
