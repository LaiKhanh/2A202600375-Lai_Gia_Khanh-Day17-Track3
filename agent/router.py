"""
Memory router – classifies a user query into one of five intents so the
retrieval step knows which memory backends to activate.

Intent taxonomy
---------------
profile_recall    – user asking about stored preferences / personal info
factual_recall    – asking about a past fact, setting, or config
experience_recall – asking about a past task / debugging session / lesson
semantic_search   – asking for a document chunk / FAQ / reference material
general           – everything else → still retrieves all memories lightly
"""
from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# keyword patterns (lower-cased)
# ---------------------------------------------------------------------------

_PROFILE_PATTERNS: List[str] = [
    r"\btên\b",          # "tên tôi là"
    r"\bname\b",
    r"\bthích\b",        # "tôi thích"
    r"\bprefer",
    r"\bdị ứng\b",       # allergy
    r"\ballerg",
    r"\btuổi\b",         # age
    r"\bage\b",
    r"\bnghề\b",         # profession
    r"\bjob\b",
    r"\boccupat",
    r"\bprofile\b",
]

_FACTUAL_PATTERNS: List[str] = [
    r"\bcài đặt\b",      # setting
    r"\bsetting",
    r"\bcấu hình\b",     # config
    r"\bconfig",
    r"\bgiá trị\b",      # value
    r"\bvalue\b",
    r"\blần trước\b",    # "last time"
    r"\blast time\b",
    r"\btrước đây\b",
]

_EXPERIENCE_PATTERNS: List[str] = [
    r"\bbài học\b",      # lesson
    r"\blesson",
    r"\bkinh nghiệm\b",  # experience
    r"\bexperience",
    r"\bsửa lỗi\b",      # debug
    r"\bdebug",
    r"\bgiải quyết\b",   # resolve
    r"\bsolve",
    r"\bproject\b",
    r"\btask\b",
    r"\bvấn đề\b",       # problem
    r"\bproblem\b",
]

_SEMANTIC_PATTERNS: List[str] = [
    r"\btài liệu\b",     # document
    r"\bdocument",
    r"\bhướng dẫn\b",    # guide
    r"\bguide\b",
    r"\bhỏi đáp\b",      # FAQ
    r"\bfaq\b",
    r"\btìm kiếm\b",     # search
    r"\bsearch\b",
    r"\bchunk\b",
    r"\breference\b",
    r"\bdocs?\b",
    r"\bapi\b",
]


def classify_intent(query: str) -> str:
    """Return the best-matching intent label for *query*."""
    q = query.lower()

    scores = {
        "profile_recall": 0,
        "factual_recall": 0,
        "experience_recall": 0,
        "semantic_search": 0,
    }

    for pat in _PROFILE_PATTERNS:
        if re.search(pat, q):
            scores["profile_recall"] += 1

    for pat in _FACTUAL_PATTERNS:
        if re.search(pat, q):
            scores["factual_recall"] += 1

    for pat in _EXPERIENCE_PATTERNS:
        if re.search(pat, q):
            scores["experience_recall"] += 1

    for pat in _SEMANTIC_PATTERNS:
        if re.search(pat, q):
            scores["semantic_search"] += 1

    best_score = max(scores.values())
    if best_score == 0:
        return "general"

    # return the label with the highest score (first one wins on tie)
    return max(scores, key=lambda k: scores[k])
