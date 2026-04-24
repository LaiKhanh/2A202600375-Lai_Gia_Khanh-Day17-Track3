"""
Benchmark runner – executes 10 multi-turn conversations against
both a memory-disabled agent and a memory-enabled agent, then
produces a comparison report.
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional

from agent.graph import MultiMemoryAgent
from .conversations import CONVERSATIONS


# ---------------------------------------------------------------------------
# No-memory agent
# ---------------------------------------------------------------------------

class NoMemoryAgent:
    """
    Minimal agent with *no* persistent memory: it only sees the current
    user message, nothing else.
    """

    def chat(self, user_message: str) -> str:
        msg = user_message.lower()

        # trivial keyword responses – mirrors mock_llm but without any memory
        import re

        if re.search(r"\btên\b|\bname\b", msg):
            return "Xin lỗi, tôi không biết tên của bạn."
        if re.search(r"\bdị ứng\b|\ballerg", msg):
            return "Xin lỗi, tôi không có thông tin về dị ứng của bạn."
        if re.search(r"\bbài học\b|\blesson\b|\bdebug\b|\bkinh nghiệm\b", msg):
            return "Xin lỗi, tôi không có thông tin về kinh nghiệm của bạn."
        if re.search(r"\btài liệu\b|\bdocument\b|\bfaq\b|\bguide\b|\b2fa\b|\blanggraph\b|\bsettings\b", msg):
            return "Xin lỗi, tôi không tìm thấy tài liệu liên quan."
        if re.search(r"\btoken\b|\bbudget\b|\btrim\b", msg):
            return "Xin lỗi, tôi không biết ngân sách token hiện tại."
        if re.search(r"\btuổi\b|\bage\b", msg):
            return "Xin lỗi, tôi không biết tuổi của bạn."
        if re.search(r"\bhồ sơ\b|\bprofile\b", msg):
            return "Xin lỗi, tôi không có hồ sơ của bạn."

        return "Xin chào! Tôi có thể giúp gì cho bạn?"

    def reset(self) -> None:
        pass  # stateless


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------

def _run_scenario(
    scenario: Dict[str, Any],
    agent: Any,
    is_memory_agent: bool,
) -> Dict[str, Any]:
    """
    Execute one scenario and return a result dict.

    Returns
    -------
    {
        "id": int,
        "scenario": str,
        "group": str,
        "responses": list[str],   # one per user turn
        "last_response": str,
        "pass": bool,             # True if expected substring found in last response
        "expected": str,          # expected substring for the key turn
    }
    """
    turns = scenario["turns"]
    knowledge = scenario.get("knowledge", [])

    # pre-load semantic knowledge when using the memory agent
    if is_memory_agent and knowledge:
        agent.add_knowledge(knowledge)

    responses: List[str] = []
    expected_keyword = ""

    for i, turn in enumerate(turns):
        if turn["role"] == "user":
            resp = agent.chat(turn["content"])
            responses.append(resp)
        elif turn["role"] == "expected":
            # the *next* user turn's expected substring
            expected_keyword = turn["content"]

    # the final expected substring refers to the very last user turn
    final_response = responses[-1] if responses else ""
    passed = (
        expected_keyword == ""
        or expected_keyword.lower() in final_response.lower()
    )

    # profile check
    profile_ok = True
    expected_profile = scenario.get("expected_profile")
    if expected_profile and is_memory_agent and hasattr(agent, "get_profile"):
        actual_profile = agent.get_profile()
        for field, expected_value in expected_profile.items():
            actual_value = str(actual_profile.get(field, "")).lower()
            if expected_value.lower() not in actual_value:
                profile_ok = False

    return {
        "id": scenario["id"],
        "scenario": scenario["scenario"],
        "group": scenario["group"],
        "responses": responses,
        "last_response": final_response,
        "pass": passed and profile_ok,
        "expected": expected_keyword,
    }


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def run_benchmark(verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run all 10 scenarios with and without memory.

    Returns
    -------
    list of result dicts with keys:
        id, scenario, group,
        no_memory_result, with_memory_result,
        pass
    """
    results = []

    no_mem_agent = NoMemoryAgent()
    mem_agent = MultiMemoryAgent(memory_budget=2000)

    for scenario in CONVERSATIONS:
        # reset between runs
        no_mem_agent.reset()
        mem_agent.reset()

        no_mem_result = _run_scenario(scenario, no_mem_agent, is_memory_agent=False)
        mem_result = _run_scenario(scenario, mem_agent, is_memory_agent=True)

        combined = {
            "id": scenario["id"],
            "scenario": scenario["scenario"],
            "group": scenario["group"],
            "no_memory_result": no_mem_result["last_response"],
            "with_memory_result": mem_result["last_response"],
            "pass": mem_result["pass"],
            "expected": mem_result["expected"],
        }
        results.append(combined)

        if verbose:
            status = "✅ Pass" if combined["pass"] else "❌ Fail"
            print(f"[{combined['id']:2d}] {status} | {combined['scenario']}")
            print(f"      No-mem : {combined['no_memory_result'][:80]}")
            print(f"      W/ mem : {combined['with_memory_result'][:80]}")
            print()

    return results


# ---------------------------------------------------------------------------
# Markdown report generator
# ---------------------------------------------------------------------------

def generate_markdown_report(results: List[Dict[str, Any]]) -> str:
    """Convert benchmark results to a Markdown table."""
    header = (
        "| # | Scenario | Group "
        "| No-memory result | With-memory result | Pass? |\n"
        "|---|----------|-------"
        "|------------------|-------------------|-------|\n"
    )
    rows = []
    for r in results:
        no_mem = r["no_memory_result"].replace("\n", " ")[:60]
        with_mem = r["with_memory_result"].replace("\n", " ")[:60]
        status = "✅" if r["pass"] else "❌"
        rows.append(
            f"| {r['id']} | {r['scenario']} | {r['group']} "
            f"| {no_mem} | {with_mem} | {status} |"
        )

    pass_count = sum(1 for r in results if r["pass"])
    total = len(results)
    summary = (
        f"\n**Results: {pass_count}/{total} passed "
        f"({100 * pass_count // total}%)**\n"
    )

    return header + "\n".join(rows) + summary


if __name__ == "__main__":
    results = run_benchmark(verbose=True)
    md = generate_markdown_report(results)
    print("\n" + "=" * 70)
    print("MARKDOWN REPORT")
    print("=" * 70)
    print(md)
