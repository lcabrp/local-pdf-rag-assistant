# Local-First LLM & Mobile GPU Configuration Guide

This document summarizes the complete **`local-pdf-rag-assistant`** architecture built so far, the **local-first / cloud-fallback** LLM design, and the recommended **Liquid AI (LFM2.5)** models and `.env` configurations for mobile GPUs (**4 GB A1000**, **6 GB A2000**, and **8 GB RTX 2000 / A4000**).

---

## 1. What We Have Built So Far

`local-pdf-rag-assistant` is a **100% local-first**, framework-free RAG system organized in a standard `src/` layout:

1. **Two-Tier (Small-to-Big) Storage (`src/pdf_rag/database_builder.py`):**
   - **Child Chunks (~250 chars, word-boundary safe):** Embedded with `all-mpnet-base-v2` into a 16-bit quantized FAISS index (`QT_fp16`, cutting index size by 50%) and indexed into an in-memory **Okapi BM25** inverted index.
   - **Parent Store (~1,500 chars, 200-char overlap):** Stored in `index/metadata.json` with `source` filename and 1-based `page` number.
2. **4-Stage Local Retrieval Pipeline (`retrieve_for_query`):**
   - **Optional Pre-Stage:** **Expand query (Multi-Query)** (`expand_query()`) generates 2 alternative queries (paraphrase + concept/solution query) to bridge vocabulary gaps before search.
   - **Stage 1:** Dense FAISS (`candidate_k=50`) + Lexical Okapi BM25 with light inflection stemming (`candidate_k=50` via `src/pdf_rag/hybrid_search.py`).
   - **Stage 2:** Scale-invariant **Reciprocal Rank Fusion (`rrf_k=60`)**.
   - **Stage 3:** Deduplicated Parent expansion (`rerank_candidates=15` unique 1,500-char windows).
   - **Stage 4:** Local GPU **Cross-Encoder Reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, max-scoring across any Multi-Query variants) $\to$ **top 4 parents**.
3. **Grounding, Inspection & Evaluation (`src/pdf_rag/app.py`, `scripts/eval_retrieval.py`):**
   - **Fast Unchanged-Folder Load:** Checks MD5 file hashes *before* PyMuPDF parsing so unchanged directories skip PDF extraction completely.
   - **Documents only (refuse if not found):** Defaults to `True` so the app refuses to guess from general LLM weights when no PDF passage matches.
   - **Show retrieved passages:** Optional GUI checkbox to inspect the exact 4 parent passages before the LLM answer (separating retrieval errors from generation errors).
   - **Expand query (Multi-Query):** Optional GUI checkbox to print and search 2 LLM-generated query variants alongside your original question.
   - **Verified Page Citations:** Maps `[1]..[4]` labels back to `<filename>.pdf, page <N>` and drops any ungrounded label.
   - **Defensive Small-Model JSON Parsing:** `_parse_confidence()` and `_resolve_sources()` gracefully handle string confidence levels (`"High"` $\to$ `85%`) and string `"sources"` fields emitted by compact local models.
   - **Local Retrieval Benchmark (`scripts/eval_retrieval.py`):** Evaluates Dense-only vs. Hybrid+Rerank Hit@3 latency and accuracy locally with zero OpenAI calls.
4. **Local-First LLM Generation with No-CPU-Spill Guard (`src/pdf_rag/llm_client.py`):**
   - Calls local **Ollama (`/api/chat`)** first with `"think": false`, `"format": "json"`, a fixed `"num_ctx": 4096`, and `num_predict: 2048` (`1024` for query expansion) so reasoning GGUFs like `LFM2.5-8B-A1B` never truncate before emitting `message.content`.
   - Inspects Ollama's **`/api/ps`** endpoint (`size_vram` vs `size`). If any layers spill to CPU (`size_vram < 99%` of `size`) or Ollama is unreachable/malformed, it automatically falls back to **OpenAI (`gpt-6-luna`)**.
5. **Python Environment Note:**
   - Pinned to **Python 3.14.10** (`.python-version`, `requires-python = ">=3.13,<3.15"` in `pyproject.toml`) because `faiss-cpu==1.16.0` does not yet ship Windows (`win_amd64`) wheels for Python 3.15.

---

## 2. VRAM Budget on Mobile GPUs

In this application, PyTorch shares the GPU with Ollama:

| Component | Approx. VRAM Usage |
| :--- | :---: |
| `SentenceTransformer("all-mpnet-base-v2")` | ~0.42 GB |
| `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` | ~0.09 GB |
| PyTorch CUDA Context | ~0.30 GB |
| **RAG Retrieval Baseline (before LLM)** | **~0.81 GB** |
| Windows Display / Desktop Overhead | ~0.40 – 0.80 GB |

That leaves roughly **~2.4 GB free on a 4 GB A1000**, **~4.4 GB free on a 6 GB A2000**, and **~6.4 GB free on an 8 GB 2000/A4000**.

---

## 3. Advised Liquid AI (LFM) & Fallback Models

### Recommended Models

| Model | Pull Command | Approx. Size | Target GPUs | Notes & Test Results |
| :--- | :--- | :---: | :--- | :--- |
| **`LiquidAI/lfm2.5-2.6b`** *(Recommended)* | `ollama pull LiquidAI/lfm2.5-2.6b` | **~1.6–2.0 GB** | **A1000 (4 GB), A2000 (6 GB), 2000 (8 GB)** | **Tested on 11 PDFs:** 100% GPU residency, 90–100% confidence on grounded answers, accurate page citations, and clean refusals when topics were absent. |
| **`hf.co/LiquidAI/LFM3-1.7B-GGUF`** | `ollama pull hf.co/LiquidAI/LFM3-1.7B-GGUF` | **~1.1–1.3 GB** | **A1000 (4 GB), A2000 (6 GB), 2000 (8 GB)** | Newer **LFM3** tri-hybrid architecture (gated convolutions + sliding-window + global attention, 256K context) built for local agents and instruction following. |
| **`hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF`** | `ollama pull hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | **< 1.0 GB** | **4 GB A1000** *(tightest budget)* | Non-thinking Instruct variant of LFM2.5-1.2B. Good ultra-light fallback if a 4 GB laptop has heavy display/browser VRAM usage. |
| **`hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M`** | `ollama pull hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M` | **~4.8–5.0 GB** | **8 GB RTX 2000 / A4000** | 8.3B Mixture-of-Experts (only **1.5B active** params per token). Fast inference with 8B-class knowledge. *Requires Ollama v0.17.1+ for `lfm2moe` tensor mapping.* |
| **`gemma4:e4b-it-qat`** | `ollama pull gemma4:e4b-it-qat` | **6.1 GB** | **8 GB+** *(borderline with PyTorch)* | Strong 4B QAT model, but sits close to the 8 GB ceiling alongside PyTorch (~0.81 GB). May trigger CPU-spill fallback depending on Windows VRAM usage. |

### Models to Avoid for This Pipeline

| Model | Why to Avoid |
| :--- | :--- |
| **`lfm2.5-thinking:1.2b`** *(731 MB)* | Although it fits 100% in VRAM (`0.79/0.79 GB`), it is a reasoning (`-thinking`) model. Forcing `"think": false` + `"format": "json"` over 3 parent passages caused it to echo the user's question back at `0%` confidence. Use **Instruct** variants instead. |
| **`gemma4:12b`** *(7.6 GB)* | Exceeds available VRAM on 4 GB, 6 GB, and 8 GB GPUs once PyTorch + KV cache + OS overhead are included, causing CPU spill. |

---

## 4. `.env` Configurations

Copy `.env.example` to `.env` and configure the target model for the laptop you are testing on.

### A. 4 GB / 6 GB / 8 GB Laptops (Recommended Default — Liquid 2.6B)
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=LiquidAI/lfm2.5-2.6b

# Cloud fallback (only called if Ollama is offline or spills to CPU)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-6-luna
```

### B. Ultra-Low-VRAM 4 GB Laptop (Liquid 1.2B Instruct)
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF

OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-6-luna
```

### C. 8 GB Laptop MoE Option (Liquid 8B-A1B)
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M

OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-6-luna
```

### D. 100% Offline Mode (No Cloud Key)
Omit `OPENAI_API_KEY` (or leave it unset). As long as Ollama is running and the model fits 100% in VRAM, `LocalFirstLLMClient` never instantiates the OpenAI client.

---

## 5. Live Benchmark Results & Takeaways (11-PDF Corpus, 8 GB RTX A4000)

All three Liquid AI models were benchmarked side-by-side on the exact same retrieved parent passages (`50 -> 15 -> 4` + light stemming) across 3 queries (2 grounded Python/Pandas questions, 1 out-of-scope Kubernetes refusal):

| Model | VRAM Used | Warm Latency | Q1 (Set Duplicates) | Q2 (Pandas `read_csv`) | Q3 (Kubernetes Refusal) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`hf.co/LiquidAI/LFM3-1.7B-GGUF`** | **1.09 GB** (100% GPU) | 2.8–3.4 s *(8.1 s cold)* | 75% `['[1]', '[2]']` *(Accurate)* | **0% `[]` *(Missed passage)*** | 0% `[]` *(Clean refusal)* |
| **`LiquidAI/lfm2.5-2.6b`** | **1.70 GB** (100% GPU) | **2.9–3.1 s** *(5.0 s cold)* | **90%** `['[2]']` *(Accurate)* | **90%** `['[1]', '[2]']` *(Accurate)* | 0% `[]` *(Clean refusal)* |
| **`hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M`** | **4.90 GB** (100% GPU) | 3.9–5.8 s *(29.1 s cold)* | **100%** `['[1]']` *(Crisp)* | **100%** `[3]` *(Crisp)* | 0% `[]` *(Clean refusal)* |

### Key Takeaways
1. **Best Overall Across All Target Laptops (4 GB / 6 GB / 8 GB): `LiquidAI/lfm2.5-2.6b`**
   - Uses only **1.70 GB VRAM**, has the fastest warm response time (**~3.0 s**), and scored **3/3** on grounded accuracy and refusals. Safe for **A1000 (4 GB)**, **A2000 (6 GB)**, and **2000/A4000 (8 GB)**.
2. **Best Quality on 8 GB Laptops: `hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M`**
   - Uses **4.90 GB VRAM** (100% GPU on 8 GB, too large for 4 GB A1000 and tight on 6 GB A2000).
   - Delivered `100%` confidence, crisp answers, and single-source citations (`[1]` and `[3]`).
3. **Skip `hf.co/LiquidAI/LFM3-1.7B-GGUF` when `lfm2.5-2.6b` fits:**
   - Although `LFM3-1.7B` uses only 1.09 GB, it missed `pd.read_csv("Cars.csv")` in Passage `[2]`/`[3]`, whereas `lfm2.5-2.6b` costs only +0.61 GB and answered all 3 reliably.
