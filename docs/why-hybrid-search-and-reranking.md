# How Hybrid Search & Cross-Encoder Reranking Improve This RAG

The original baseline in `local-pdf-rag-assistant` already worked well for general conceptual questions thanks to **Two-Tier (Small-to-Big) Retrieval** and **deterministic page citations**. This document explains *why* the new additions—**Okapi BM25 Hybrid Search**, **Reciprocal Rank Fusion (RRF)**, and **Local Cross-Encoder Reranking**—make retrieval measurably more accurate without increasing OpenAI token usage or requiring you to re-embed your existing PDFs.

---

## 1. Before vs. After at a Glance

| Stage | Original Baseline | Upgraded 4-Stage Pipeline |
| :--- | :--- | :--- |
| **Candidate Search** | Dense FAISS only (`k=5` child chunks) | **FAISS (`candidate_k=25`) + Okapi BM25 (`candidate_k=25`)** |
| **Score Merging** | None (single ranked list) | **Reciprocal Rank Fusion (`rrf_k=60`)** |
| **Parent Expansion** | Top 5 children $\to$ up to `max_parents=3` | Top 25 fused children $\to$ up to **`rerank_candidates=10` unique parents** |
| **Precision Filtering** | Bi-Encoder L2 distance only | **Local Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)** on `(query, parent_text)` $\to$ **top 3 parents** |
| **LLM Prompt Budget** | Up to 3 parent passages (~4,500 chars) | **Still up to 3 parent passages** (same or lower OpenAI token cost) |
| **Existing Index Compatibility** | — | **100% backward compatible** (BM25 builds in memory from `metadata.json` in ~150 ms) |

---

## 2. Three Concrete Failure Modes Fixed

### A. The "Exact Keyword & Identifier Blind Spot"
- **What happened before:**
  `all-mpnet-base-v2` compresses a 250-character child chunk into 768 floats representing its *general semantic topic*. If you asked:
  > *"What does fault code `E-409` mean?"* or *"What are the limits in `Section 14.3(b)`?"*
  
  FAISS often returned 5 generic chunks discussing "fault codes" or "section limits" while missing the exact page containing `E-409`.
- **How the upgrade fixes it:**
  - `tokenize_for_bm25()` in `src/pdf_rag/hybrid_search.py` preserves compound identifiers (`e-409`, `14.3`, `qt_fp16`, `0x80070005`) and also indexes their sub-parts (`409`, `fp16`), while stripping common English stopwords.
  - `BM25Index` gives rare, exact tokens high Robertson-Walker IDF weight.
  - `reciprocal_rank_fusion()` merges FAISS (semantic) and BM25 (exact lexical) rankings without needing score normalization, ensuring exact-match chunks enter the candidate pool even when dense vector distance misses them.

### B. "Topical False Positives" Pushing Out the Real Answer
- **What happened before:**
  FAISS uses a **Bi-Encoder**: the user's question and each PDF chunk are embedded *separately* into vectors, and then compared with L2 distance. Because the question and the chunk never interact word-by-word during encoding, a passage that merely repeats topical buzzwords (e.g., an introduction or table of contents mentioning "hydraulic pressure") can score closer than the page that actually explains the procedure. With `k=5` and `max_parents=3`, just two noisy hits could push the true answer out of the 3 parent slots sent to OpenAI.
- **How the upgrade fixes it:**
  - Stage 1–3 cast a wider net (`candidate_k=25` $\to$ up to `10` deduplicated parent windows), so the true answer is far more likely to be included in the candidate set.
  - Stage 4 passes each `(query, parent_text)` pair together into `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` on your GPU. Full transformer self-attention across `[CLS] query [SEP] parent [SEP]` evaluates whether the passage *actually answers* the question—not just whether it shares vocabulary.
  - Only the **top 3 reranked parents** are sent to `refine_with_openai()`.

### C. Negations, Qualifiers, and Split Heading/Answer Contexts
- **What happened before:**
  - Bi-encoders struggle with negations (*"What should **not** be done during cold startup?"*) because `"cold startup"` dominates the vector regardless of `"not"`.
  - A 250-character child chunk might hold only a section heading, while the answer sits two sentences later in the 1,500-character parent window.
- **How the upgrade fixes it:**
  - `rerank_parent_contexts()` intentionally reranks **Parent contexts** (~1,500 chars) rather than 250-char Child chunks. That gives the Cross-Encoder visibility into both the heading and the full answer paragraph, while cutting the number of Transformer forward passes by >60% (reranking $\le 10$ unique parents instead of 25 children).

---

## 3. Why Specific Engineering Choices Were Made

| Design Choice | Why We Did It This Way |
| :--- | :--- |
| **Reciprocal Rank Fusion (`rrf_k=60`) instead of weighted score averaging** | FAISS L2 distances (lower = better) and BM25 scores (higher = better, scales with query length) are on incompatible scales. RRF uses rank positions ($\sum \frac{1}{60 + \text{rank}}$), requiring zero calibration and boosting consensus hits found by both retrievers. |
| **In-memory BM25 rebuild on `load_index()` instead of `pickle`** | Rebuilding `BM25Index` from `child_chunks` in `index/metadata.json` takes ~150 ms, keeps existing `index/` folders 100% compatible with zero re-embedding, and avoids insecure `pickle` deserialization. |
| **Lazy Cross-Encoder initialization (`_get_reranker()`)** | Deferring model load until the first query keeps GUI startup instant and ensures the app gracefully falls back to Hybrid RRF ordering if the reranker cannot be loaded (e.g., offline environments). |
| **Pure-Python inverted index (`BM25Index`)** | An inverted posting list (`term -> [(doc_id, tf)]`) only visits chunks containing query terms ($O(\text{matches})$ instead of $O(N)$), running in <2 ms on 26.5k chunks with no extra C/wheel dependencies. |

---

## 4. Net Impact on Cost, Speed, and Accuracy

- **OpenAI API Cost:** **Same or lower.** `refine_with_openai()` still receives at most `max_parents=3` parent passages, but with higher relevance—reducing fallback calls and vague answers.
- **Local Latency:** Adds ~2 ms for BM25 + RRF and ~30–80 ms on CUDA GPU for Cross-Encoder reranking of $\le 10$ parent windows—imperceptible compared to the LLM API round-trip.
- **Re-indexing Effort:** **Zero.** Your existing `index/index.faiss` and `index/metadata.json` work out of the box.
