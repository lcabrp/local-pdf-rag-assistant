# Local PDF RAG Assistant

**Local-Hardware Multi-Stage PDF RAG Application** — Built from scratch in native Python using **PyMuPDF**, **PyTorch GPU embeddings & Cross-Encoder reranking** (`CUDA`/`MPS`), **FAISS `QT_fp16` + Okapi BM25 Hybrid Search (RRF)**, **optional Multi-Query expansion**, and **local-first Ollama (`LiquidAI/lfm2.5`) with OpenAI (`gpt-6-luna`) cloud fallback** and verified page-level citations.

---

## 1. Project Goal & Design Philosophy

Rather than relying on heavy orchestration frameworks (like LangChain or LlamaIndex) or sending entire documents to cloud embedding APIs, **`local-pdf-rag-assistant`** runs the entire RAG pipeline—**indexing, hybrid search, Cross-Encoder reranking, and answer synthesis**—locally on your GPU at \$0 API cost, falling back to OpenAI (`gpt-6-luna`) automatically if local Ollama is offline or spills to CPU.

### Key Highlights
1. **Zero Framework Bloat:** Written in clean, well-commented, inspectable Python (`src/pdf_rag/`).
2. **Two-Tier (Small-to-Big) Dictionary Retrieval:**
   - **Child Chunks (~250 chars, word-boundary safe):** Embedded into FAISS and indexed into an in-memory Okapi BM25 inverted index for sharp, high-precision matching.
   - **Parent Store (`{parent_id: {"text", "source", "page"}}`):** Stores ~1,500-character overlapping page windows in a Python dictionary. When child chunks match, `retrieve_parent_contexts()` deduplicates their `parent_id`s and expands them into full surrounding parent sections.
3. **Hybrid Search (FAISS Dense Vector + Okapi BM25 with Light Stemming + RRF):**
   - Pure embeddings understand paraphrased concepts but blur exact technical strings (`E-409`, `Clause 14.3(b)`).
   - `hybrid_search.py` pairs FAISS L2 vector search with an inverted-index Okapi BM25 engine (with lightweight inflection stemming `_light_stem()` so `remove` matches `removed`) and merges candidates via **Reciprocal Rank Fusion (RRF)**.
4. **Optional Multi-Query Expansion & Local Cross-Encoder Reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`):**
   - When **Expand query (Multi-Query)** is checked, `expand_query()` generates 2 alternative queries (a paraphrase + a solution/concept query) before running Hybrid Search + RRF.
   - `rerank_parent_contexts()` jointly scores up to 15 candidate parent windows against the query (and any expanded variants via max-scoring) on GPU/CPU to pick the **top 4 parents**.
5. **Local-First Ollama Generation with No-CPU-Spill Guard & OpenAI Fallback:**
   - `LocalFirstLLMClient` (`src/pdf_rag/llm_client.py`) calls local Ollama (`/api/chat` with `format: "json"`, `num_ctx: 4096`) first and checks `/api/ps` (`size_vram` vs. `size`). If Ollama is unreachable or spills layers onto the CPU, it falls back seamlessly to OpenAI (`gpt-6-luna`).
6. **Anti-Hallucination Page Citations, Refusal Mode & Inspection Controls:**
   - **Documents only (refuse if not found)** defaults to `True` so the model refuses rather than guessing from general weights.
   - **Show retrieved passages** displays the exact top-4 parent snippets sent to the LLM so you can distinguish retrieval errors from generation errors.
   - `_resolve_sources()` deterministically maps `"[1]"`–`"[4]"` back to `<filename>.pdf, page <N>` in Python and drops any unretrieved labels so citations can never be invented.
7. **Portable Hardware Acceleration (`CUDA` / `MPS` / `CPU`) & `QT_fp16` Quantization:**
   - Automatically detects `cuda` (any NVIDIA GPU), `mps` (Apple Silicon), or `cpu`.
   - Uses `faiss.IndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_fp16)` (`float16`, 2 bytes/dim), cutting `index/index.faiss` in half (`77.67 MB` $\to$ **`38.84 MB`** for 26.5k chunks).
   - Checks MD5 file hashes *before* PyMuPDF parsing so unchanged folders skip PDF extraction and load immediately.

---

## 2. Architecture & Order of Operations

The application operates in two distinct phases: **Phase 1: Ingestion** (run once when a folder changes) and **Phase 2: Querying** (run on every user question).

```
========================= PHASE 1: INGESTION (One-Time Per Folder) =========================
1. Curated PDF Folder (e.g., 2–5 related PDFs)
      │
      ▼ [PyMuPDF: extract_pages_from_pdf() in document_handler.py]
2. Per-Page Records: [{"source": "Book.pdf", "page": 187, "text": "..."}]
   (Pure text extracted per page; images/graphs skipped, table grids flattened)
      │
      ▼ [build_embeddings() in database_builder.py]
3. Two-Tier Split:
   ├─► Parent Store Dict: {parent_id: {"text": ~1500 chars, "source": "Book.pdf", "page": 187}}
   └─► Child Chunks List: ~250-char word-safe slices linked to parent_id
      │
      ▼ [PyTorch + SentenceTransformer ("all-mpnet-base-v2") on CUDA GPU (batch_size=64)]
4. Dense Child Vectors (768 dimensions per child chunk)
      │
      ▼ [faiss.IndexScalarQuantizer (QT_fp16 - 16-bit float compression)]
5. Disk Persistence:
   ├─► index/index.faiss   (Compacted 16-bit vector index: ~38.8 MB for 26.5k chunks)
   └─► index/metadata.json (MD5 file hashes + child_chunks + child_metadata + parent_store)


========================= PHASE 2: QUERYING (Per User Question) ============================
1. User Enters Question in Tkinter GUI (app.py)
      │
      ▼ [Optional: Expand query (Multi-Query) -> expand_query() generates 2 synonym/concept queries]
      ├─► [Stage 1A - Dense Search]: PyTorch SentenceTransformer ("all-mpnet-base-v2") + FAISS (Top 50)
      └─► [Stage 1B - Lexical Search]: Okapi BM25 (with light stemming) over child_chunks (Top 50)
      │
      ▼ [Stage 2 - Reciprocal Rank Fusion (RRF) in hybrid_search.py]
2. Top-50 Fused Child Chunk Indices (combining semantic similarity + exact/stemmed keyword matches)
      │
      ▼ [Stage 3 - Two-Tier Parent Expansion: retrieve_parent_contexts()]
3. Up to 15 Deduplicated Parent Contexts (~1,500 chars each, with source filename & page number)
      │
      ▼ [Stage 4 - Cross-Encoder Reranking: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") on GPU]
4. Top-4 Reranked Parent Contexts (jointly scored against query + any expanded variants)
      │
      ▼ [Stage 5 - Grounded Synthesis: LocalFirstLLMClient -> Ollama (100% VRAM guard) or OpenAI fallback]
5. Validated JSON Reply: {"answer": "...", "sources": ["[1]"], "confidence": 0.95}
      │
      ▼ [_resolve_sources(): maps "[1]" -> "Book.pdf, page 187"]
6. GUI Displays Clean Answer + Model Used + Confidence % + Page Citations
```

---

## 3. Setup & Usage

> **Python Requirement:** Pinned to **Python 3.14.10** in `.python-version` (`requires-python = ">=3.13,<3.15"`). Python 3.15 is currently excluded because `faiss-cpu==1.16.0` does not yet publish Windows `win_amd64` wheels for 3.15.

```powershell
# 1. Install dependencies (including CUDA-enabled PyTorch on Windows/Linux)
uv sync

# 2. Pull a local Ollama model (recommended 1.7 GB model fits 4 GB / 6 GB / 8 GB GPUs)
ollama pull LiquidAI/lfm2.5-2.6b

# 3. Configure local Ollama model & optional OpenAI cloud fallback
Copy-Item .env.example .env
# Edit .env and set OLLAMA_MODEL=LiquidAI/lfm2.5-2.6b (and optional OPENAI_API_KEY=...)

# 4. Launch the desktop GUI
uv run pdf-rag

# 5. Run unit tests
uv run python -m unittest discover -s tests -v
```

1. Click **Browse** and select a folder containing your PDFs (e.g., `data/pdfs/`).
2. The app checks MD5 file hashes *first*. On unchanged folders it skips PyMuPDF parsing and loads the cached `index/` + in-memory BM25 index immediately; when files change it extracts pages, embeds on your GPU, and saves `index/index.faiss` and `index/metadata.json`.
3. Enter a question, toggle **Documents only**, **Show retrieved passages**, or **Expand query (Multi-Query)** as needed, and click **Get Answer**.

For detailed VRAM budgets, Liquid AI benchmarks, and 4 GB / 6 GB / 8 GB laptop `.env` profiles, see [`docs/local-llm-and-hardware-guide.md`](docs/local-llm-and-hardware-guide.md).

---

## 4. Repository Layout

```
local-pdf-rag-assistant/
├── src/
│   └── pdf_rag/                # Core installable package
│       ├── __init__.py
│       ├── app.py              # Tkinter desktop GUI, prompt builder, and source resolver
│       ├── database_builder.py # Two-Tier chunker, FAISS + BM25 hybrid retriever, and Cross-Encoder reranker
│       ├── document_handler.py # PyMuPDF directory scanner and per-page text extractor
│       ├── hybrid_search.py    # Inverted-index Okapi BM25 and Reciprocal Rank Fusion (RRF)
│       └── llm_client.py       # Local-first Ollama client (no-CPU-spill guard) + OpenAI fallback
├── scripts/
│   ├── eval_retrieval.py       # Benchmark Dense-only vs. Hybrid+Rerank Hit@3 accuracy locally
│   └── preview_pdfs.py         # CLI utility to inspect extracted text from PDFs
├── docs/
│   ├── local-llm-and-hardware-guide.md               # Local-first Ollama / Liquid AI models & 4-8 GB GPU guide
│   ├── rag-landscape-and-enterprise-architecture.md  # RAG storage, retrieval, and enterprise architecture guide
│   ├── sibling-repo-comparison-pdf-rag-assistant.md  # Side-by-side comparison & future upgrade notes for pdf-rag-assistant
│   ├── why-hybrid-search-and-reranking.md            # Before vs. After analysis of Hybrid BM25 + Cross-Encoder improvements
│   └── reference/
│       └── ngxson-simple-rag/                        # Reference 77-line RAG from scratch (demo.py + cat-facts.txt)
├── data/
│   └── pdfs/                   # Local PDF input directory (contents gitignored)
├── tests/
│   ├── __init__.py
│   ├── eval_questions.example.json # Template for 10-question local retrieval evaluation
│   ├── test_hybrid_and_rerank.py   # Unit tests for BM25 tokenization, exact ID search, and RRF
│   └── test_llm_client.py          # Unit tests for local-first Ollama + CPU-spill cloud fallback
├── index/                      # Persisted QT_fp16 FAISS index and UTF-8 metadata dictionary
├── pyproject.toml              # Dependencies & `pdf-rag` entry point managed via uv
├── .env.example                # Environment variable template
└── .gitignore                  # Git ignore rules
```

