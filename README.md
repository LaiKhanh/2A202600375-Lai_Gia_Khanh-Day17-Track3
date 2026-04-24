# 2A202600375-Lai_Gia_Khanh-Day17-Track3

## 🚀 Overview

This repository implements a **multi-memory AI agent** that combines four memory types to improve multi-turn reasoning and personalization:

* **Short-term memory** → recent conversation context
* **Long-term profile memory** → persistent user facts
* **Episodic memory** → past tasks / experiences
* **Semantic memory** → knowledge retrieval (vector / keyword)

The system follows a **LangGraph-style pipeline** (can run with or without LangGraph):

```
User Input → Route → Retrieve Memory → Trim → Generate → Update Memory
```

The goal is to demonstrate how different memory layers interact and how they improve performance compared to a **no-memory baseline**.

---

## 🧠 Key Features

* Full **4-layer memory stack**
* Memory **router + state management**
* **Prompt injection** with structured memory sections
* **Conflict handling** (latest fact overrides old)
* **Episodic learning** from past interactions
* **Semantic retrieval** (FAISS / keyword fallback)
* **Token budget trimming**
* Benchmark with **10 multi-turn conversations**

---

## 📁 Repository Structure

```
.
├── agent/
│   ├── graph.py              # Main pipeline (route → retrieve → generate → update)
│   ├── state.py              # MemoryState definition
│   ├── router.py             # Intent classification / routing logic
│   └── prompt.py             # Prompt template + memory injection
│
├── memory/
│   ├── short_term.py         # Sliding window conversation buffer
│   ├── long_term.py          # User profile (KV store / Redis-like)
│   ├── episodic.py           # Task history / experience log
│   └── semantic.py           # Vector search / keyword retrieval
│
├── retrieval/
│   ├── embedder.py           # Embedding logic
│   └── index.py              # FAISS / search index
│
├── data/
│   ├── knowledge_base/       # Documents / FAQ for semantic memory
│   └── episodes.json         # Stored episodic memory
│
├── benchmark/
│   ├── scenarios.py          # 10 multi-turn test cases
│   └── runner.py             # Benchmark execution
│
├── BENCHMARK.md              # Benchmark results + analysis
├── README.md                 # This file
├── requirements.txt
└── main.py                   # Entry point
```

---

## 🔄 System Flow

1. **Route**

   * Classify user intent (profile / semantic / episodic / general)

2. **Retrieve Memory**

   * Fetch relevant data from:

     * Profile store
     * Episodic logs
     * Semantic index
     * Recent messages

3. **Trim (Token Budget)**

   * Prioritize important memory
   * Remove low-value context if needed

4. **Generate Response**

   * Inject structured memory into prompt:

```
[User Profile]
...

[Past Episodes]
...

[Relevant Knowledge]
...

[Recent Conversation]
...
```

5. **Update Memory**

   * Save new profile facts
   * Log episodic events
   * Update short-term buffer

---

## 📊 Benchmark

* 10 **multi-turn conversations**
* Compare:

  * ❌ No-memory baseline
  * ✅ Full memory agent

Covered scenarios:

* Profile recall
* Conflict update
* Episodic recall
* Semantic retrieval
* Token budget trimming

👉 See details in **`BENCHMARK.md`**

---

## ⚠️ Limitations

* Rule-based routing (not LLM-based classification)
* Approximate token counting (word-based)
* No reranking for semantic retrieval
* Memory growth may affect scalability

---

## 🔐 Privacy Considerations

* Long-term memory may contain **PII**
* Requires:

  * User consent
  * Memory deletion support
  * TTL (time-to-live)

---

## 🛠️ How to Run

```bash
pip install -r requirements.txt
python main.py
```

---
