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

<!-- AUTO-GENERATED RESULTS -->

| # | Scenario | Group | No-memory result | With-memory result | Pass? |
|---|----------|-------|------------------|-------------------|-------|
| 1 | Recall user name after 6 turns | profile_recall | Xin lỗi, tôi không biết tên của bạn. | Tên của bạn là linh. | ✅ |
| 2 | Allergy conflict update (sữa bò → đậu nành) | conflict_update | Xin lỗi, tôi không có thông tin về dị ứng của bạn. | Bạn bị dị ứng với: đậu nành. | ✅ |
| 3 | Recall previous debug lesson (docker service name) | episodic_recall | Xin lỗi, tôi không có thông tin về kinh nghiệm của bạn. | Bài học gần nhất: Tốt, cảm ơn bạn đã ghi nhớ bài học đó. → B | ✅ |
| 4 | Retrieve FAQ chunk about password reset | semantic_retrieval | Xin lỗi, tôi không tìm thấy tài liệu liên quan. | Tôi tìm được: Để đặt lại mật khẩu, vào Settings → Security → | ✅ |
| 5 | Token budget trim after 8 turns | trim_token_budget | Xin lỗi, tôi không biết ngân sách token hiện tại. | Ngân sách token hiện tại: 2000. | ✅ |
| 6 | Recall user age and job after profile setup | profile_recall | Xin lỗi, tôi không biết tuổi của bạn. | Hồ sơ của bạn: age=28, job=một developer. | ✅ |
| 7 | Multiple profile facts accumulated across turns | profile_recall | Xin lỗi, tôi không có hồ sơ của bạn. | Hồ sơ của bạn: name=minh, age=25, preference=lập trình pytho | ✅ |
| 8 | Retrieve 2FA setup guide from knowledge base | semantic_retrieval | Xin lỗi, tôi không tìm thấy tài liệu liên quan. | Tôi tìm được: Cài đặt 2FA: Settings → Security → Two-Factor  | ✅ |
| 9 | Recall completed ML project from episodic memory | episodic_recall | Xin lỗi, tôi không có thông tin về kinh nghiệm của bạn. | Bài học gần nhất: Tôi vừa hoàn thành dự án phân loại ảnh với | ✅ |
| 10 | Combined: profile recall + episodic + semantic retrieval | profile_recall | Xin lỗi, tôi không biết tên của bạn. | Tên của bạn là hoa. | ✅ |
**Results: 10/10 passed (100%)**
