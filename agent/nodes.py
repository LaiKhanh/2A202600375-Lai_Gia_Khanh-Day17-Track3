"""
LangGraph nodes for the Multi-Memory Agent.

Node execution order (graph.py wires these together):
  1. route_query       – classify intent, set query_intent
  2. retrieve_memory   – pull context from all four backends
  3. trim_context      – honour memory_budget
  4. generate_response – call LLM (or mock) and produce response
  5. update_memory     – persist new facts / episodes learned from the turn
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .router import classify_intent
from .state import MemoryState
from memory import EpisodicMemory, LongTermMemory, SemanticMemory, ShortTermMemory

# ---------------------------------------------------------------------------
# Token-budget constants
# ---------------------------------------------------------------------------

DEFAULT_BUDGET = 3000          # total tokens allocated to injected context
TOKENS_PER_CHAR = 0.25         # rough estimate: 1 token ≈ 4 chars
PROFILE_PRIORITY = 4           # highest  (evicted last)
SEMANTIC_PRIORITY = 3
EPISODIC_PRIORITY = 2
SHORT_TERM_PRIORITY = 1        # oldest messages evicted first


def _char_tokens(text: str) -> int:
    """Rough token estimate from character count."""
    return max(1, int(len(text) * TOKENS_PER_CHAR))


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _latest_user_message(state: MemoryState) -> str:
    """Return the text of the most recent user message."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Node 1 – route_query
# ---------------------------------------------------------------------------

def route_query(state: MemoryState) -> MemoryState:
    """Classify the latest user message and store the intent."""
    query = _latest_user_message(state)
    intent = classify_intent(query)
    return {**state, "query_intent": intent}


# ---------------------------------------------------------------------------
# Node 2 – retrieve_memory
# ---------------------------------------------------------------------------

def make_retrieve_memory(
    short_term: ShortTermMemory,
    long_term: LongTermMemory,
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
):
    """Factory: returns the retrieve_memory node bound to the four backends."""

    def retrieve_memory(state: MemoryState) -> MemoryState:
        query = _latest_user_message(state)
        intent = state.get("query_intent", "general")

        # --- long-term profile (always retrieved) ---
        user_profile = long_term.get_all()

        # --- episodic (full recall for experience, recent-only otherwise) ---
        if intent == "experience_recall":
            episodes = episodic.search_by_keyword(query) or episodic.get_recent(5)
        else:
            episodes = episodic.get_recent(3)

        # --- semantic (activated for semantic_search and general) ---
        if intent in ("semantic_search", "general", "factual_recall"):
            semantic_hits = semantic.search(query, k=3)
        else:
            semantic_hits = []

        # state["messages"] is NOT replaced here – it carries the current
        # user turn and must remain intact for downstream nodes.
        return {
            **state,
            "user_profile": user_profile,
            "episodes": episodes,
            "semantic_hits": semantic_hits,
        }

    return retrieve_memory


# ---------------------------------------------------------------------------
# Node 3 – trim_context
# ---------------------------------------------------------------------------

def trim_context(state: MemoryState) -> MemoryState:
    """
    Enforce memory_budget using a priority-based 4-level eviction hierarchy.

    Priority (highest kept longest):
      4 – user_profile
      3 – semantic_hits
      2 – episodes
      1 – oldest short-term messages (already handled by ShortTermMemory)
    """
    budget = state.get("memory_budget", DEFAULT_BUDGET)
    user_profile = dict(state.get("user_profile", {}))
    episodes = list(state.get("episodes", []))
    semantic_hits = list(state.get("semantic_hits", []))
    messages = list(state.get("messages", []))

    # calculate current usage
    def usage() -> int:
        total = 0
        total += sum(_char_tokens(str(v)) for v in user_profile.values())
        total += sum(_char_tokens(ep.get("task", "") + ep.get("outcome", "")) for ep in episodes)
        total += sum(_char_tokens(h) for h in semantic_hits)
        total += sum(_char_tokens(m.get("content", "")) for m in messages)
        return total

    # evict oldest short-term messages first (priority 1)
    while usage() > budget and messages:
        messages.pop(0)

    # evict episodes next (priority 2)
    while usage() > budget and episodes:
        episodes.pop(0)

    # evict semantic hits next (priority 3)
    while usage() > budget and semantic_hits:
        semantic_hits.pop(0)

    # user_profile (priority 4) is never evicted automatically

    return {
        **state,
        "messages": messages,
        "episodes": episodes,
        "semantic_hits": semantic_hits,
        "user_profile": user_profile,
        "memory_budget": budget,
    }


# ---------------------------------------------------------------------------
# Node 4 – generate_response
# ---------------------------------------------------------------------------

def _build_system_prompt(state: MemoryState) -> str:
    """Assemble the system prompt with injected memory context."""
    sections: List[str] = ["You are a helpful assistant with persistent memory.\n"]

    # --- profile section ---
    profile = state.get("user_profile", {})
    if profile:
        profile_lines = "\n".join(f"  - {k}: {v}" for k, v in profile.items())
        sections.append(f"## User Profile\n{profile_lines}\n")

    # --- episodic section ---
    episodes = state.get("episodes", [])
    if episodes:
        ep_lines = "\n".join(
            f"  [{ep.get('timestamp', '')[:10]}] {ep.get('task', '')} → {ep.get('outcome', '')}"
            for ep in episodes
        )
        sections.append(f"## Past Experiences\n{ep_lines}\n")

    # --- semantic section ---
    hits = state.get("semantic_hits", [])
    if hits:
        hit_lines = "\n".join(f"  • {h}" for h in hits)
        sections.append(f"## Relevant Knowledge\n{hit_lines}\n")

    # --- recent conversation section ---
    messages = state.get("messages", [])
    if messages:
        conv_lines = "\n".join(
            f"  {m['role'].upper()}: {m['content']}" for m in messages[:-1]  # exclude latest user msg
        )
        if conv_lines:
            sections.append(f"## Recent Conversation\n{conv_lines}\n")

    return "\n".join(sections)


def make_generate_response(llm=None):
    """
    Factory: returns the generate_response node.

    Parameters
    ----------
    llm:
        A callable ``llm(system_prompt, user_message) -> str``.
        When *None* a simple rule-based mock is used (no API key needed).
    """

    def generate_response(state: MemoryState) -> MemoryState:
        user_msg = _latest_user_message(state)
        system_prompt = _build_system_prompt(state)

        if llm is not None:
            response = llm(system_prompt, user_msg)
        else:
            response = _mock_llm(system_prompt, user_msg, state)

        return {**state, "response": response}

    return generate_response


def _mock_llm(system_prompt: str, user_msg: str, state: MemoryState) -> str:
    """
    Deterministic mock LLM used when no real LLM is configured.

    It echoes back key facts from memory so benchmarks can verify
    recall without requiring an API key.
    """
    profile = state.get("user_profile", {})
    episodes = state.get("episodes", [])
    semantic_hits = state.get("semantic_hits", [])
    user_msg_lower = user_msg.lower()

    # name recall
    if re.search(r"\btên\b|\bname\b", user_msg_lower):
        name = profile.get("name") or profile.get("tên")
        if name:
            return f"Tên của bạn là {name}."
        return "Tôi chưa biết tên của bạn."

    # allergy recall / conflict
    if re.search(r"\bdị ứng\b|\ballerg", user_msg_lower):
        allergy = profile.get("allergy") or profile.get("dị_ứng")
        if allergy:
            return f"Bạn bị dị ứng với: {allergy}."
        return "Tôi chưa lưu thông tin dị ứng của bạn."

    # experience / lesson recall
    if re.search(r"\bbài học\b|\blesson\b|\bdebug\b|\bkinh nghiệm\b", user_msg_lower):
        if episodes:
            ep = episodes[-1]
            return f"Bài học gần nhất: {ep.get('task')} → {ep.get('outcome')}."
        return "Tôi chưa có bài học nào được lưu."

    # semantic chunk recall
    if re.search(r"\btài liệu\b|\bdocument\b|\bfaq\b|\bguide\b", user_msg_lower):
        if semantic_hits:
            return f"Tôi tìm được: {semantic_hits[0]}"
        return "Không tìm thấy tài liệu liên quan."

    # token / budget info
    if re.search(r"\btoken\b|\bbudget\b|\btrim\b", user_msg_lower):
        budget = state.get("memory_budget", DEFAULT_BUDGET)
        return f"Ngân sách token hiện tại: {budget}."

    # generic profile dump
    if profile:
        summary = ", ".join(f"{k}={v}" for k, v in profile.items())
        return f"Hồ sơ của bạn: {summary}."

    return "Xin chào! Tôi có thể giúp gì cho bạn?"


# ---------------------------------------------------------------------------
# Node 5 – update_memory
# ---------------------------------------------------------------------------

# Patterns that signal a user is stating / updating a profile fact
_PROFILE_UPDATE_PATTERNS = [
    # name
    (r"tên\s+(?:tôi|của tôi)\s+(?:là|:)\s+(.+)", "name"),
    (r"(?:my\s+)?name\s+is\s+(.+)", "name"),
    (r"(?:call me|gọi tôi là)\s+(.+)", "name"),
    # allergy – order matters: check "nhầm" (correction) last
    (r"dị\s*ứng\s+(?:với\s+)?(.+)", "allergy"),
    (r"allerg(?:ic to|y to|y:)\s*(.+)", "allergy"),
    # age
    (r"(?:tôi\s+)?(\d+)\s+tuổi", "age"),
    (r"(?:i am|i'm)\s+(\d+)\s+years?\s+old", "age"),
    # job
    (r"(?:tôi là|i am|i'm)\s+(?:một\s+)?(.+(?:developer|engineer|designer|kỹ sư|lập trình viên))", "job"),
    # preference
    (r"(?:tôi\s+)?thích\s+(.+)", "preference"),
    (r"(?:i\s+)?(?:like|prefer|enjoy)\s+(.+)", "preference"),
]

# Correction signals – if present, the new fact replaces the old one
_CORRECTION_SIGNALS = [
    r"\bnhầm\b",         # "nhầm" = mistake in Vietnamese
    r"\bsai\b",          # "sai" = wrong
    r"\bkhông phải\b",   # "không phải" = not
    r"\bcorrect\b",
    r"\bactually\b",
    r"\bnot\b.+\bbut\b",
]


# Vietnamese/English question-word endings that signal a recall question, not a fact statement
_QUESTION_ENDINGS_RE = re.compile(
    r"\b(gì|nào|không|thế\s*nào|bao\s*nhiêu|ai|đâu|sao|như\s*thế\s*nào)\s*[?]?\s*$",
    re.IGNORECASE,
)


def _is_question_value(value: str) -> bool:
    """Return True if the extracted value looks like a question phrase, not a fact."""
    v = value.strip()
    if v.endswith("?"):
        return True
    if _QUESTION_ENDINGS_RE.search(v):
        return True
    return False


def _extract_profile_updates(user_msg: str) -> Dict[str, str]:
    """
    Parse a user message and return a dict of ``{field: new_value}``
    profile facts.  Handles direct statements and corrections.
    """
    updates: Dict[str, str] = {}
    msg_lower = user_msg.lower().strip()

    # detect if this is a correction statement
    is_correction = any(re.search(p, msg_lower) for p in _CORRECTION_SIGNALS)

    for pattern, field in _PROFILE_UPDATE_PATTERNS:
        m = re.search(pattern, msg_lower)
        if m:
            raw_value = m.group(1).strip().rstrip(".!,")
            # Strip correction-clause suffixes:
            #   "đậu nành chứ không phải sữa bò" → "đậu nành"
            #   "soybeans not dairy"              → "soybeans"
            raw_value = re.sub(r"\s+chứ\s+không\s+phải\s+.+$", "", raw_value).strip()
            raw_value = re.sub(r"\s+không\s+phải\s+.+$", "", raw_value).strip()
            raw_value = re.sub(r"\s+not\s+.+$", "", raw_value).strip()
            raw_value = re.sub(r"\s+but\s+not\s+.+$", "", raw_value).strip()
            # strip trailing noise words
            raw_value = re.sub(r"\s+(nhé|nhỉ|nha|ok|okay)$", "", raw_value).strip()

            # skip values that look like question phrases (recall queries, not fact statements)
            if _is_question_value(raw_value):
                continue

            if is_correction and field in updates:
                # prefer the corrected value
                updates[field] = raw_value
            else:
                updates[field] = raw_value

    return updates


def _should_record_episode(user_msg: str, response: str) -> Optional[Dict[str, str]]:
    """
    Decide whether the current turn should be recorded as an episode.
    Returns a dict with task/outcome/tags or None.
    """
    keywords = ["xong", "done", "hoàn thành", "solved", "fixed", "đã sửa",
                 "thành công", "success", "bài học", "lesson", "debug"]
    combined = (user_msg + " " + response).lower()
    if any(kw in combined for kw in keywords):
        return {
            "task": user_msg[:120],
            "outcome": response[:120],
            "tags": ["auto"],
        }
    return None


def make_update_memory(
    short_term: ShortTermMemory,
    long_term: LongTermMemory,
    episodic: EpisodicMemory,
):
    """Factory: returns the update_memory node bound to the backends."""

    def update_memory(state: MemoryState) -> MemoryState:
        user_msg = _latest_user_message(state)
        response = state.get("response", "")

        # 1. Always persist to short-term buffer
        short_term.add_message("user", user_msg)
        short_term.add_message("assistant", response)

        # 2. Add assistant response to the rolling state messages
        messages = list(state.get("messages", []))
        messages.append({"role": "assistant", "content": response})

        # 3. Extract and persist profile facts (conflict → new wins)
        updates = _extract_profile_updates(user_msg)
        if updates:
            long_term.update(updates)

        # 4. Record episodic memory when a task is clearly completed
        ep_data = _should_record_episode(user_msg, response)
        if ep_data:
            episodic.add_episode(
                task=ep_data["task"],
                outcome=ep_data["outcome"],
                tags=ep_data.get("tags", []),
            )

        # Return updated profile and messages so state stays in sync
        return {**state, "user_profile": long_term.get_all(), "messages": messages}

    return update_memory
