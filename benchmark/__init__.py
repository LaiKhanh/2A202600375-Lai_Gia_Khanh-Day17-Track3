"""
benchmark package – 10 multi-turn conversation scenarios.
"""
from .conversations import CONVERSATIONS
from .runner import run_benchmark

__all__ = ["CONVERSATIONS", "run_benchmark"]
