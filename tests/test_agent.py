"""
Tests for the agent layer: router, nodes, and the full graph.
"""
import pytest

from agent.router import classify_intent
from agent.nodes import _extract_profile_updates, _should_record_episode
from agent.graph import MultiMemoryAgent


# ===========================================================================
# Router
# ===========================================================================

class TestRouter:
    def test_profile_recall_tên(self):
        assert classify_intent("Tên tôi là gì?") == "profile_recall"

    def test_profile_recall_allergy(self):
        assert classify_intent("Tôi bị dị ứng với gì?") == "profile_recall"

    def test_experience_recall(self):
        intent = classify_intent("Nhắc lại bài học debug của tôi")
        assert intent == "experience_recall"

    def test_semantic_search(self):
        intent = classify_intent("Tìm kiếm tài liệu về API")
        assert intent in ("semantic_search", "factual_recall", "general")

    def test_general_fallback(self):
        assert classify_intent("Xin chào bạn!") == "general"

    def test_factual_recall(self):
        intent = classify_intent("Cài đặt lần trước của tôi là gì?")
        assert intent == "factual_recall"


# ===========================================================================
# Profile update extraction
# ===========================================================================

class TestProfileExtraction:
    def test_name_extraction(self):
        updates = _extract_profile_updates("Tên tôi là Linh.")
        assert updates.get("name") == "linh"

    def test_age_extraction(self):
        updates = _extract_profile_updates("Tôi 25 tuổi.")
        assert "age" in updates
        assert "25" in updates["age"]

    def test_allergy_extraction(self):
        updates = _extract_profile_updates("Tôi dị ứng sữa bò.")
        assert "allergy" in updates
        assert "sữa bò" in updates["allergy"]

    def test_allergy_correction(self):
        # First set sữa bò, then correct to đậu nành
        updates = _extract_profile_updates("À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.")
        assert "allergy" in updates
        assert "đậu nành" in updates["allergy"]

    def test_no_updates(self):
        updates = _extract_profile_updates("Thời tiết hôm nay thế nào?")
        assert updates == {}

    def test_preference_extraction(self):
        updates = _extract_profile_updates("Tôi thích Python.")
        assert "preference" in updates


# ===========================================================================
# Episode trigger detection
# ===========================================================================

class TestEpisodeTrigger:
    def test_triggers_on_solved(self):
        result = _should_record_episode("Debug lỗi kết nối.", "Đã solved bằng service name.")
        assert result is not None

    def test_triggers_on_done(self):
        result = _should_record_episode("Task xong rồi.", "OK.")
        assert result is not None

    def test_no_trigger_on_normal_chat(self):
        result = _should_record_episode("Thời tiết hôm nay?", "Hôm nay trời đẹp.")
        assert result is None


# ===========================================================================
# Full agent integration tests
# ===========================================================================

class TestMultiMemoryAgent:
    def setup_method(self):
        self.agent = MultiMemoryAgent(memory_budget=3000)

    def test_chat_returns_string(self):
        resp = self.agent.chat("Xin chào!")
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_name_recall_after_multiple_turns(self):
        self.agent.chat("Tên tôi là Linh.")
        self.agent.chat("Hôm nay trời đẹp.")
        self.agent.chat("Tôi đang học Python.")
        resp = self.agent.chat("Tên tôi là gì vậy?")
        assert "linh" in resp.lower()

    def test_allergy_conflict_resolution(self):
        self.agent.chat("Tôi dị ứng sữa bò.")
        self.agent.chat("À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.")
        profile = self.agent.get_profile()
        assert "đậu nành" in str(profile.get("allergy", "")).lower()
        # Old value must not be present
        assert "sữa bò" not in str(profile.get("allergy", "")).lower()

    def test_semantic_memory_retrieval(self):
        self.agent.add_knowledge([
            "Đặt lại mật khẩu: Settings → Security → Reset Password.",
            "Bật 2FA: Settings → Security → Two-Factor Authentication.",
        ])
        resp = self.agent.chat("Tìm kiếm tài liệu về reset mật khẩu.")
        # Should retrieve something from semantic memory
        assert isinstance(resp, str)

    def test_reset_clears_all_memory(self):
        self.agent.chat("Tên tôi là Test.")
        self.agent.reset()
        assert self.agent.get_profile() == {}
        assert self.agent.short_term.get_all() == []

    def test_episodic_memory_recorded(self):
        self.agent.chat("Tôi đã debug thành công lỗi import. Đã solved.")
        episodes = self.agent.episodic.get_all()
        assert len(episodes) >= 1

    def test_multiple_profile_facts(self):
        self.agent.chat("Tên tôi là Minh.")
        self.agent.chat("Tôi 25 tuổi.")
        profile = self.agent.get_profile()
        assert "name" in profile
        assert "age" in profile

    def test_token_budget_respected(self):
        """Long conversations should not crash; trim must work."""
        agent = MultiMemoryAgent(memory_budget=200)
        for i in range(20):
            resp = agent.chat(f"Tin nhắn thử nghiệm số {i} " + "a" * 50)
            assert isinstance(resp, str)


# ===========================================================================
# Benchmark sanity: all 10 conversations should run without errors
# ===========================================================================

class TestBenchmarkSanity:
    def test_benchmark_runs(self):
        from benchmark.runner import run_benchmark
        results = run_benchmark(verbose=False)
        assert len(results) == 10
        for r in results:
            assert "scenario" in r
            assert "no_memory_result" in r
            assert "with_memory_result" in r
            assert isinstance(r["pass"], bool)
