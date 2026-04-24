"""
Long-term memory – persistent user-profile store.

The interface is deliberately Redis-compatible (hset / hget / hgetall /
hdel) so it can be swapped for a real Redis client without changing
callers.  The default backend is an in-process dict; an optional
``redis_url`` switches to a real Redis instance.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class LongTermMemory:
    """Key-value profile store with a Redis-compatible interface.

    Parameters
    ----------
    redis_url:
        When provided the store delegates to a real Redis hash.  When
        *None* (default) an in-process ``dict`` is used instead.
    namespace:
        Hash-key prefix / Redis hash name.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        namespace: str = "user_profile",
    ) -> None:
        self.namespace = namespace
        self._redis = None

        if redis_url:
            try:
                import redis as redis_lib  # type: ignore

                self._redis = redis_lib.from_url(redis_url, decode_responses=True)
                self._redis.ping()  # fail fast if unreachable
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[LongTermMemory] Redis not available ({exc}); "
                    "falling back to in-process dict."
                )
                self._redis = None

        # in-process fallback
        self._store: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def set(self, field: str, value: Any) -> None:
        """Store / overwrite a single profile fact (conflict → new wins)."""
        if self._redis:
            import json

            self._redis.hset(self.namespace, field, json.dumps(value))
        else:
            self._store[field] = value

    def update(self, mapping: Dict[str, Any]) -> None:
        """Bulk-update multiple fields.  New values overwrite old ones."""
        for field, value in mapping.items():
            self.set(field, value)

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get(self, field: str, default: Any = None) -> Any:
        if self._redis:
            import json

            raw = self._redis.hget(self.namespace, field)
            return json.loads(raw) if raw is not None else default
        return self._store.get(field, default)

    def get_all(self) -> Dict[str, Any]:
        if self._redis:
            import json

            return {k: json.loads(v) for k, v in self._redis.hgetall(self.namespace).items()}
        return dict(self._store)

    # ------------------------------------------------------------------
    # management
    # ------------------------------------------------------------------

    def delete(self, field: str) -> None:
        if self._redis:
            self._redis.hdel(self.namespace, field)
        else:
            self._store.pop(field, None)

    def clear(self) -> None:
        if self._redis:
            self._redis.delete(self.namespace)
        else:
            self._store.clear()

    def __repr__(self) -> str:  # pragma: no cover
        backend = "redis" if self._redis else "dict"
        return f"LongTermMemory(backend={backend}, fields={len(self.get_all())})"
