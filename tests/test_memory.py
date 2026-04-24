"""
Tests for all four memory backends.
"""
import json
import os
import tempfile

import pytest

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory


# ===========================================================================
# ShortTermMemory
# ===========================================================================

class TestShortTermMemory:
    def test_add_and_retrieve(self):
        mem = ShortTermMemory(max_messages=10)
        mem.add_message("user", "hello")
        mem.add_message("assistant", "hi")
        assert len(mem) == 2
        msgs = mem.get_recent()
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "hi"

    def test_sliding_window_eviction(self):
        mem = ShortTermMemory(max_messages=3)
        for i in range(5):
            mem.add_message("user", f"msg {i}")
        assert len(mem) == 3
        # oldest messages evicted; last 3 remain
        assert mem.get_all()[0]["content"] == "msg 2"
        assert mem.get_all()[-1]["content"] == "msg 4"

    def test_get_recent_n(self):
        mem = ShortTermMemory()
        for i in range(10):
            mem.add_message("user", str(i))
        recent = mem.get_recent(3)
        assert len(recent) == 3
        assert recent[-1]["content"] == "9"

    def test_trim_to_budget(self):
        mem = ShortTermMemory()
        for _ in range(10):
            mem.add_message("user", "x" * 40)
        # 10 messages * ~10 tokens each = 100 tokens; budget 50 forces eviction
        mem.trim_to_budget(token_budget=50, tokens_per_message=10)
        assert len(mem) <= 5

    def test_clear(self):
        mem = ShortTermMemory()
        mem.add_message("user", "hello")
        mem.clear()
        assert len(mem) == 0


# ===========================================================================
# LongTermMemory
# ===========================================================================

class TestLongTermMemory:
    def test_set_and_get(self):
        mem = LongTermMemory()
        mem.set("name", "Linh")
        assert mem.get("name") == "Linh"

    def test_conflict_resolution_new_wins(self):
        mem = LongTermMemory()
        mem.set("allergy", "sữa bò")
        mem.set("allergy", "đậu nành")  # overwrite
        assert mem.get("allergy") == "đậu nành"

    def test_update_bulk(self):
        mem = LongTermMemory()
        mem.update({"name": "An", "age": 25})
        assert mem.get("name") == "An"
        assert mem.get("age") == 25

    def test_get_all(self):
        mem = LongTermMemory()
        mem.set("x", 1)
        mem.set("y", 2)
        all_data = mem.get_all()
        assert all_data == {"x": 1, "y": 2}

    def test_delete(self):
        mem = LongTermMemory()
        mem.set("key", "value")
        mem.delete("key")
        assert mem.get("key") is None

    def test_default_value(self):
        mem = LongTermMemory()
        assert mem.get("missing", "default_val") == "default_val"

    def test_clear(self):
        mem = LongTermMemory()
        mem.set("a", 1)
        mem.clear()
        assert mem.get_all() == {}


# ===========================================================================
# EpisodicMemory
# ===========================================================================

class TestEpisodicMemory:
    def test_add_and_retrieve(self):
        mem = EpisodicMemory()
        ep = mem.add_episode("debug docker", "used service name", tags=["docker"])
        assert ep["task"] == "debug docker"
        assert "timestamp" in ep
        assert len(mem) == 1

    def test_get_recent(self):
        mem = EpisodicMemory()
        for i in range(10):
            mem.add_episode(f"task {i}", f"outcome {i}")
        recent = mem.get_recent(3)
        assert len(recent) == 3
        assert recent[-1]["task"] == "task 9"

    def test_search_by_tag(self):
        mem = EpisodicMemory()
        mem.add_episode("deploy", "success", tags=["docker", "ci"])
        mem.add_episode("test", "passed", tags=["pytest"])
        results = mem.search_by_tag("docker")
        assert len(results) == 1
        assert results[0]["task"] == "deploy"

    def test_search_by_keyword(self):
        mem = EpisodicMemory()
        mem.add_episode("debug connection", "fixed with env var")
        mem.add_episode("write tests", "all pass")
        results = mem.search_by_keyword("debug")
        assert len(results) == 1

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        try:
            mem1 = EpisodicMemory(filepath=path)
            mem1.add_episode("task A", "outcome A")
            # reload
            mem2 = EpisodicMemory(filepath=path)
            assert len(mem2) == 1
            assert mem2.get_all()[0]["task"] == "task A"
        finally:
            os.unlink(path)

    def test_clear(self):
        mem = EpisodicMemory()
        mem.add_episode("t", "o")
        mem.clear()
        assert len(mem) == 0


# ===========================================================================
# SemanticMemory
# ===========================================================================

class TestSemanticMemory:
    def test_add_and_search(self):
        mem = SemanticMemory()
        mem.add_document("Reset password via Settings Security panel.")
        mem.add_document("Enable 2FA in security settings.")
        results = mem.search("how to reset password", k=1)
        assert len(results) == 1
        assert "password" in results[0].lower() or "settings" in results[0].lower()

    def test_search_returns_top_k(self):
        mem = SemanticMemory()
        docs = [f"document {i} about topic {i}" for i in range(10)]
        mem.add_documents(docs)
        results = mem.search("topic 5", k=3)
        assert len(results) == 3

    def test_empty_search(self):
        mem = SemanticMemory()
        results = mem.search("anything")
        assert results == []

    def test_clear(self):
        mem = SemanticMemory()
        mem.add_document("test doc")
        mem.clear()
        results = mem.search("test")
        assert results == []

    def test_search_with_scores(self):
        mem = SemanticMemory()
        mem.add_document("Python programming language")
        mem.add_document("Cooking recipes for dinner")
        scored = mem.search_with_scores("python code", k=2)
        assert len(scored) == 2
        # scores should be numbers
        for score, _ in scored:
            assert isinstance(score, float)
