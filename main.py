"""
main.py – command-line entry point for the Multi-Memory Agent.

Usage
-----
Interactive chat (mock LLM, no API key needed):
    python main.py

Run benchmark and update BENCHMARK.md:
    python main.py --benchmark

Use a real OpenAI LLM:
    OPENAI_API_KEY=sk-... python main.py
"""
from __future__ import annotations

import argparse
import os
import sys

from agent.graph import MultiMemoryAgent


# ---------------------------------------------------------------------------
# Optional real LLM wrapper (OpenAI)
# ---------------------------------------------------------------------------

def _build_openai_llm():
    """Return an OpenAI-backed LLM callable, or None if not configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)

        def llm(system_prompt: str, user_message: str) -> str:
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""

        return llm
    except ImportError:
        print("[main] openai package not installed; using mock LLM.")
        return None


# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

def interactive_chat(agent: MultiMemoryAgent) -> None:
    print("Multi-Memory Agent (type 'quit' or 'exit' to stop, 'reset' to clear memory)")
    print("=" * 60)
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("[Memory cleared]")
            continue
        if user_input.lower() == "profile":
            print("Profile:", agent.get_profile())
            continue

        response = agent.chat(user_input)
        print(f"\nAssistant: {response}")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark_and_write_md() -> None:
    from benchmark.runner import run_benchmark, generate_markdown_report
    import pathlib

    print("Running benchmark (10 multi-turn conversations)…\n")
    results = run_benchmark(verbose=True)
    md = generate_markdown_report(results)

    # Write (or refresh) BENCHMARK.md in the repo root
    output_path = pathlib.Path(__file__).parent / "BENCHMARK.md"

    # Read existing header if present, otherwise use default
    existing = ""
    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8")

    # Keep everything above the auto-generated results table
    separator = "<!-- AUTO-GENERATED RESULTS -->"
    if separator in existing:
        header = existing.split(separator)[0]
    else:
        header = _benchmark_header()

    output_path.write_text(header + separator + "\n\n" + md, encoding="utf-8")
    print(f"\nBENCHMARK.md updated → {output_path}")


def _benchmark_header() -> str:
    return """\
# Benchmark Report – Multi-Memory Agent

## Overview

Comparison of a **memory-enabled agent** (full 4-backend memory stack)
against a **no-memory baseline** on 10 multi-turn Vietnamese/English
conversations.

### Memory Stack
| Layer | Backend | Purpose |
|-------|---------|---------|
| Short-term | Sliding-window buffer (max 20 msgs) | Recent conversation context |
| Long-term | Dict / Redis-compatible KV store | Persistent user profile |
| Episodic | JSON append-only log | Past tasks & lessons |
| Semantic | FAISS / keyword TF-IDF fallback | Document / FAQ retrieval |

### Test Groups
- **profile_recall** – verify the agent remembers stated personal facts
- **conflict_update** – verify newer facts overwrite older contradicting ones
- **episodic_recall** – verify past tasks/lessons are retrieved correctly
- **semantic_retrieval** – verify relevant document chunks are surfaced
- **trim_token_budget** – verify context window stays within budget

## Results

"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Memory LangGraph Agent")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark and write BENCHMARK.md",
    )
    parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis URL for long-term memory (optional)",
    )
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark_and_write_md()
        return

    llm = _build_openai_llm()
    if llm:
        print("[main] Using OpenAI LLM.")
    else:
        print("[main] No OPENAI_API_KEY found – using deterministic mock LLM.")

    agent = MultiMemoryAgent(llm=llm, redis_url=args.redis_url)
    interactive_chat(agent)


if __name__ == "__main__":
    main()
