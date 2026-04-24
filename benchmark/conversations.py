"""
10 multi-turn benchmark conversations.

Each entry is a dict:
  id          : int (1-10)
  scenario    : short English description
  group       : test group (profile_recall / conflict_update /
                            episodic_recall / semantic_retrieval /
                            trim_token_budget)
  knowledge   : list[str] – document chunks to pre-load into semantic memory
                (empty for most scenarios)
  turns       : list[{"role": "user"|"assistant", "content": ...}]
                Only "user" turns are played; "assistant" entries are the
                *expected* substrings that must appear in the actual response.
  expected_profile : dict – the user_profile state expected after the
                      conversation (None = not checked)
  check_fn    : optional name of a special check to run after the conversation
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Turn = Dict[str, str]
Scenario = Dict[str, Any]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _turns(*pairs: tuple) -> List[Turn]:
    """
    Build a turn list from (user_msg, expected_substring) pairs.
    ``expected_substring`` may be "" if we only care about no crash.
    """
    result = []
    for user_msg, expected in pairs:
        result.append({"role": "user", "content": user_msg})
        result.append({"role": "expected", "content": expected})
    return result


# ---------------------------------------------------------------------------
# Conversations 1-10
# ---------------------------------------------------------------------------

CONVERSATIONS: List[Scenario] = [
    # -------------------------------------------------------------------
    # 1 – Profile recall: remember name after 6 turns
    # -------------------------------------------------------------------
    {
        "id": 1,
        "scenario": "Recall user name after 6 turns",
        "group": "profile_recall",
        "knowledge": [],
        "turns": _turns(
            ("Tên tôi là Linh.", ""),
            ("Bạn có thể giúp tôi học Python không?", ""),
            ("Tôi muốn tìm hiểu về list comprehension.", ""),
            ("Giải thích vòng lặp for trong Python.", ""),
            ("Cảm ơn, rất hữu ích!", ""),
            ("Tên tôi là gì vậy?", "linh"),
        ),
        "expected_profile": {"name": "linh"},
    },

    # -------------------------------------------------------------------
    # 2 – Conflict update: allergy correction
    # -------------------------------------------------------------------
    {
        "id": 2,
        "scenario": "Allergy conflict update (sữa bò → đậu nành)",
        "group": "conflict_update",
        "knowledge": [],
        "turns": _turns(
            ("Tôi dị ứng sữa bò.", ""),
            ("Bạn có thể gợi ý thực đơn không?", ""),
            ("À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.", ""),
            ("Bạn nhớ tôi dị ứng với gì không?", "đậu nành"),
        ),
        "expected_profile": {"allergy": "đậu nành"},
    },

    # -------------------------------------------------------------------
    # 3 – Episodic recall: remember a past debugging lesson
    # -------------------------------------------------------------------
    {
        "id": 3,
        "scenario": "Recall previous debug lesson (docker service name)",
        "group": "episodic_recall",
        "knowledge": [],
        "turns": _turns(
            ("Tôi đã debug lỗi kết nối DB trong Docker. Đã solved: dùng service name thay vì localhost.", ""),
            ("Tốt, cảm ơn bạn đã ghi nhớ bài học đó.", ""),
            ("Bạn nhớ bài học debug Docker của tôi không?", "docker"),
        ),
        "expected_profile": None,
    },

    # -------------------------------------------------------------------
    # 4 – Semantic retrieval: FAQ chunk retrieval
    # -------------------------------------------------------------------
    {
        "id": 4,
        "scenario": "Retrieve FAQ chunk about password reset",
        "group": "semantic_retrieval",
        "knowledge": [
            "Để đặt lại mật khẩu, vào Settings → Security → Reset Password.",
            "Chính sách hoàn tiền: liên hệ support trong 30 ngày.",
            "Cài đặt 2FA: Settings → Security → Two-Factor Authentication.",
        ],
        "turns": _turns(
            ("Làm thế nào để đặt lại mật khẩu?", ""),
            ("Xin hỏi về tài liệu reset mật khẩu.", "settings"),
        ),
        "expected_profile": None,
    },

    # -------------------------------------------------------------------
    # 5 – Token budget / trim: long conversation truncated correctly
    # -------------------------------------------------------------------
    {
        "id": 5,
        "scenario": "Token budget trim after 8 turns",
        "group": "trim_token_budget",
        "knowledge": [],
        "turns": _turns(
            ("Tên tôi là An.", ""),
            ("Tin nhắn 1: Đây là nội dung dài để kiểm tra trim " + "x" * 200, ""),
            ("Tin nhắn 2: Đây là nội dung dài để kiểm tra trim " + "x" * 200, ""),
            ("Tin nhắn 3: Đây là nội dung dài để kiểm tra trim " + "x" * 200, ""),
            ("Tin nhắn 4: Đây là nội dung dài để kiểm tra trim " + "x" * 200, ""),
            ("Tin nhắn 5: Đây là nội dung dài để kiểm tra trim " + "x" * 200, ""),
            ("Tên tôi là gì?", "an"),
            ("Ngân sách token còn bao nhiêu?", "token"),
        ),
        "expected_profile": {"name": "an"},
    },

    # -------------------------------------------------------------------
    # 6 – Profile recall: age and job after several turns
    # -------------------------------------------------------------------
    {
        "id": 6,
        "scenario": "Recall user age and job after profile setup",
        "group": "profile_recall",
        "knowledge": [],
        "turns": _turns(
            ("Tôi 28 tuổi.", ""),
            ("Tôi là một developer.", ""),
            ("Giúp tôi chọn framework phù hợp.", ""),
            ("Cảm ơn!", ""),
            ("Tên tôi là gì?", ""),
            ("Bạn còn nhớ tôi bao nhiêu tuổi không?", "28"),
        ),
        "expected_profile": {"age": "28"},
    },

    # -------------------------------------------------------------------
    # 7 – Multiple profile facts accumulated
    # -------------------------------------------------------------------
    {
        "id": 7,
        "scenario": "Multiple profile facts accumulated across turns",
        "group": "profile_recall",
        "knowledge": [],
        "turns": _turns(
            ("Tên tôi là Minh.", ""),
            ("Tôi 25 tuổi.", ""),
            ("Tôi thích lập trình Python.", ""),
            ("Nhắc lại hồ sơ của tôi.", "minh"),
        ),
        "expected_profile": {"name": "minh", "age": "25"},
    },

    # -------------------------------------------------------------------
    # 8 – Semantic retrieval: 2FA guide
    # -------------------------------------------------------------------
    {
        "id": 8,
        "scenario": "Retrieve 2FA setup guide from knowledge base",
        "group": "semantic_retrieval",
        "knowledge": [
            "Cài đặt 2FA: Settings → Security → Two-Factor Authentication.",
            "Để đặt lại mật khẩu, vào Settings → Security → Reset Password.",
            "Liên hệ hỗ trợ qua email support@example.com.",
        ],
        "turns": _turns(
            ("Tôi muốn bật xác thực hai yếu tố.", ""),
            ("Hướng dẫn tài liệu về 2FA ở đâu?", "2fa"),
        ),
        "expected_profile": None,
    },

    # -------------------------------------------------------------------
    # 9 – Episodic recall: remember completed project
    # -------------------------------------------------------------------
    {
        "id": 9,
        "scenario": "Recall completed ML project from episodic memory",
        "group": "episodic_recall",
        "knowledge": [],
        "turns": _turns(
            ("Tôi vừa hoàn thành dự án phân loại ảnh với ResNet. Đã thành công đạt 95% accuracy.", ""),
            ("Tốt quá! Đã ghi nhớ kết quả.", ""),
            ("Nhắc lại bài học kinh nghiệm về dự án ML của tôi.", "resnet"),
        ),
        "expected_profile": None,
    },

    # -------------------------------------------------------------------
    # 10 – Combined: profile + episodic + semantic in one conversation
    # -------------------------------------------------------------------
    {
        "id": 10,
        "scenario": "Combined: profile recall + episodic + semantic retrieval",
        "group": "profile_recall",
        "knowledge": [
            "LangChain là framework để xây dựng ứng dụng AI với LLM.",
            "LangGraph cho phép tạo agent có trạng thái với đồ thị.",
        ],
        "turns": _turns(
            ("Tên tôi là Hoa.", ""),
            ("Tôi đã debug thành công lỗi import LangChain. Lesson: pip install -U langchain.", ""),
            ("Hỏi về tài liệu LangGraph là gì?", "langgraph"),
            ("Bạn nhớ tên tôi không?", "hoa"),
        ),
        "expected_profile": {"name": "hoa"},
    },
]
