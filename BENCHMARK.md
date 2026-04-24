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
- **Profile Recall**
  - Scenarios: #1, #6, #7, #10
  - Agent successfully remembers user attributes (name, age, preferences) across multiple turns.

- **Conflict Update**
  - Scenario: #2
  - Demonstrates overwrite logic:
    - Old fact: allergy = sữa bò
    - New fact: allergy = đậu nành
    - Final memory correctly reflects latest user correction.

- **Episodic Recall**
  - Scenarios: #3, #9
  - Agent retrieves past experiences/tasks stored in episodic memory.

- **Semantic Retrieval**
  - Scenarios: #4, #8
  - Agent retrieves relevant knowledge chunks from vector/keyword search.

- **Token Budget / Context Trimming**
  - Scenario: #5
  - Confirms that context is trimmed and managed within token limits.

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


## Reflection: Privacy & Limitations

### 1. Which memory helps the agent the most?

- **Long-term profile memory** provides the most consistent improvement.
- It enables personalization (name, preferences, constraints) across conversations.
- Without it, the agent behaves stateless and generic.

### 2. Which memory is most risky?

- **Long-term profile memory is the most sensitive**, as it may contain:
  - Personal identifiable information (PII)
  - Preferences, habits, or private user data

- If incorrectly retrieved or leaked, it can cause:
  - Privacy violations
  - Incorrect personalization
  - Loss of user trust

### 3. Privacy Risks (PII)

Potential risks include:

- Storing sensitive user data without consent
- Retrieving outdated or incorrect personal facts
- Exposing private data in unrelated contexts

### 4. Mitigation Strategies

To reduce risks, the system should include:

- **TTL (Time-to-Live)** for profile and episodic memory
- **User-controlled deletion**
  - Example: delete profile, clear history
- **Consent mechanism**
  - Ask user before storing sensitive information
- **Scoped retrieval**
  - Only retrieve relevant memory for the current query

### 5. Memory Deletion

If a user requests deletion:

- **Short-term memory** → clear buffer
- **Long-term profile** → delete from KV store / Redis
- **Episodic memory** → remove entries from log
- **Semantic memory** → remove embeddings / documents

Deletion must be consistent across all backends.

### 6. Risks of Incorrect Retrieval

- Retrieving wrong semantic chunk → hallucinated answer
- Using outdated profile → incorrect personalization
- Mixing unrelated episodic events → confusion

### 7. Technical Limitations

Current system limitations include:

- **Heuristic routing**
  - Intent classification is rule-based → may misroute queries
- **No embedding update strategy**
  - Semantic memory may become stale over time
- **Token budget is approximate**
  - Uses word/character count instead of exact tokenizer
- **Scalability**
  - Memory growth (episodic logs, vector DB) may degrade performance
- **No ranking/reranking**
  - Semantic retrieval may return suboptimal results

### 8. Failure at Scale

The system may fail when:

- Too many users → memory storage explosion
- Too many episodes → slow retrieval
- Large vector DB → latency increases
- Poor memory filtering → noisy context → worse LLM output