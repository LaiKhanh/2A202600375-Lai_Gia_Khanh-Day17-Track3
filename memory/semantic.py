"""
Semantic memory – vector search (FAISS) with keyword-search fallback.

The backend is chosen automatically at instantiation time:
  1. FAISS + numpy  – if both packages are importable (fast approximate NN)
  2. Keyword / TF-IDF fallback  – always available, no extra deps

Each document is stored as a plain string ("chunk").  Documents can be
added individually or in bulk.  Retrieval returns the top-k most
similar chunks for a query string.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase word tokeniser."""
    return re.findall(r"[a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF]+", text.lower())


def _tf(tokens: List[str]) -> dict:
    counts: dict = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def _cosine(a: dict, b: dict) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[t] * b[t] for t in shared)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# FAISS backend (optional)
# ---------------------------------------------------------------------------

class _FaissBackend:
    def __init__(self) -> None:
        import numpy as np  # type: ignore  # noqa: F401
        import faiss  # type: ignore  # noqa: F401

        self._np = np
        self._faiss = faiss
        self._dim: Optional[int] = None
        self._index = None
        self._docs: List[str] = []

    def _embed(self, text: str) -> "np.ndarray":  # type: ignore[name-defined]
        """
        Word-hash bag-of-words embedding using two independent hash buckets
        per token to reduce collisions.  No model download required.
        """
        np = self._np
        dim = 128
        vec = np.zeros(dim, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())
        for token in tokens:
            # primary bucket: polynomial rolling hash
            h1 = 0
            for ch in token:
                h1 = (h1 * 31 + ord(ch)) & 0x7FFFFFFF
            vec[h1 % dim] += 1.0
            # secondary bucket: reversed token hash for n-gram flavour
            h2 = 0
            for ch in reversed(token):
                h2 = (h2 * 37 + ord(ch)) & 0x7FFFFFFF
            vec[h2 % dim] += 0.5
        norm = np.linalg.norm(vec) or 1.0
        return (vec / norm).reshape(1, -1)

    def add(self, text: str) -> None:
        vec = self._embed(text)
        if self._index is None:
            self._dim = vec.shape[1]
            self._index = self._faiss.IndexFlatIP(self._dim)  # inner-product = cosine on unit vecs
        self._index.add(vec)
        self._docs.append(text)

    def search(self, query: str, k: int) -> List[Tuple[float, str]]:
        if self._index is None or len(self._docs) == 0:
            return []
        q_vec = self._embed(query)
        actual_k = min(k, len(self._docs))
        scores, indices = self._index.search(q_vec, actual_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((float(score), self._docs[idx]))
        return results

    def clear(self) -> None:
        self._index = None
        self._docs.clear()


# ---------------------------------------------------------------------------
# Keyword / TF-IDF fallback backend
# ---------------------------------------------------------------------------

class _KeywordBackend:
    def __init__(self) -> None:
        self._docs: List[str] = []
        self._tfs: List[dict] = []

    def add(self, text: str) -> None:
        self._docs.append(text)
        self._tfs.append(_tf(_tokenize(text)))

    def search(self, query: str, k: int) -> List[Tuple[float, str]]:
        if not self._docs:
            return []
        q_tf = _tf(_tokenize(query))
        scored = [(_cosine(q_tf, doc_tf), doc) for doc_tf, doc in zip(self._tfs, self._docs)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def clear(self) -> None:
        self._docs.clear()
        self._tfs.clear()


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    Semantic memory that stores text chunks and retrieves the most
    relevant ones for a given query.

    Uses FAISS when available, otherwise falls back to TF-IDF cosine
    keyword search.
    """

    def __init__(self, use_faiss: bool = False) -> None:
        # Default to keyword (TF-IDF cosine) backend which gives reliable
        # lexical recall without collision issues from hash-based embeddings.
        # Pass use_faiss=True to activate the FAISS backend.
        if use_faiss:
            try:
                self._backend: _FaissBackend | _KeywordBackend = _FaissBackend()
                self._backend_name = "faiss"
            except Exception:  # noqa: BLE001
                self._backend = _KeywordBackend()
                self._backend_name = "keyword"
        else:
            self._backend = _KeywordBackend()
            self._backend_name = "keyword"

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def add_document(self, text: str) -> None:
        """Index a text chunk."""
        self._backend.add(text)

    def add_documents(self, texts: List[str]) -> None:
        """Index multiple chunks at once."""
        for text in texts:
            self._backend.add(text)

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 3) -> List[str]:
        """Return the top-k most relevant chunks for *query*."""
        results = self._backend.search(query, k)
        return [doc for _, doc in results]

    def search_with_scores(self, query: str, k: int = 3) -> List[Tuple[float, str]]:
        return self._backend.search(query, k)

    # ------------------------------------------------------------------
    # management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._backend.clear()

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def __repr__(self) -> str:  # pragma: no cover
        return f"SemanticMemory(backend={self._backend_name})"
