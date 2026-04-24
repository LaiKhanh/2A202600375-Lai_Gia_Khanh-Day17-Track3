"""
Episodic memory – append-only JSON log of completed tasks / experiences.

Each episode is a dict with at minimum:
  - timestamp  : ISO-8601 string
  - task        : short description of what happened
  - outcome     : result / lesson learned
  - tags        : list[str] for quick filtering

Episodes can be persisted to / loaded from a JSON file.  When no path is
given the log lives in memory only.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class EpisodicMemory:
    """Append-only episode log backed by an optional JSON file."""

    def __init__(self, filepath: Optional[str] = None) -> None:
        self._filepath = filepath
        self._log: List[Dict[str, Any]] = []

        if filepath and os.path.isfile(filepath):
            self._load()

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def add_episode(
        self,
        task: str,
        outcome: str,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a completed task / experience."""
        episode: Dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "task": task,
            "outcome": outcome,
            "tags": tags or [],
        }
        if extra:
            episode.update(extra)

        self._log.append(episode)
        if self._filepath:
            self._save()
        return episode

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._log)

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        return list(self._log[-n:])

    def search_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Return episodes whose tag list contains *tag* (case-insensitive)."""
        tag_lower = tag.lower()
        return [ep for ep in self._log if tag_lower in [t.lower() for t in ep.get("tags", [])]]

    def search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Return episodes that mention *keyword* in task or outcome text."""
        kw = keyword.lower()
        return [
            ep
            for ep in self._log
            if kw in ep.get("task", "").lower() or kw in ep.get("outcome", "").lower()
        ]

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        assert self._filepath is not None
        with open(self._filepath, "w", encoding="utf-8") as fh:
            json.dump(self._log, fh, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        assert self._filepath is not None
        with open(self._filepath, encoding="utf-8") as fh:
            content = fh.read().strip()
        if not content:
            return
        data = json.loads(content)
        if isinstance(data, list):
            self._log = data

    def clear(self) -> None:
        self._log.clear()
        if self._filepath and os.path.isfile(self._filepath):
            self._save()

    def __len__(self) -> int:
        return len(self._log)

    def __repr__(self) -> str:  # pragma: no cover
        return f"EpisodicMemory(episodes={len(self._log)}, file={self._filepath!r})"
