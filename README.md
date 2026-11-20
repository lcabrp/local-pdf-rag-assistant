# Local PDF RAG Assistant

**Local-Hardware Multi-Stage PDF RAG Application** — Built from scratch in native Python using **PyMuPDF**, **PyTorch GPU embeddings & Cross-Encoder reranking** (`CUDA`/`MPS`), **FAISS `QT_fp16` + Okapi BM25 Hybrid Search (RRF)**, and **OpenAI `gpt-6-luna`** with verified page-level citations.

---

## 1. Project Goal & Design Philosophy

Rather than relying on heavy orchestration frameworks (like LangChain or LlamaIndex) or sending entire documents to cloud embedding APIs, **`local-pdf-rag-assistant`** leverages **local hardware (NVIDIA GPU / Apple Silicon / CPU)** for fast, zero-cost hybrid indexing, Reciprocal Rank Fusion, and Cross-Encoder reranking, while using `gpt-6-luna` strictly to synthesize the final answer from the top 3 reranked parent passages.

### Key Highlights
1. **Zero Framework Bloat:** Written in clean, well-commented, inspectable Python (`src/pdf_rag/`).
2. **Two-Tier (Small-to-Big) Dictionary Retrieval:**
   - **Child Chunks (~250 chars, word-boundary safe):** Embedded into FAISS and indexed into an in-memory Okapi BM25 inverted index for sharp, high-precision matching.
   - **Parent Store (`{parent_id: {"text", "source", "page"}}`):** Stores ~1,500-character page windows in a Python dictionary. When child chunks match, `retrieve_parent_contexts()` deduplicates their `parent_id`s and expands them into full surrounding parent sections.
3. **Hybrid Search (FAISS Dense Vector + Okapi BM25 Lexical + RRF):**
   - Pure embeddings understand paraphrased concepts ("renal failure" $\leftrightarrow$ "kidney disease") but blur exact technical strings (error codes like `E-409`, part numbers, or `Clause 14.3(b)`).
   - `hybrid_search.py` pairs FAISS L2 vector search with an inverted-index Okapi BM25 engine and merges their top candidates via **Reciprocal Rank Fusion (RRF)**.
4. **Local Cross-Encoder Reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`):**
   - Bi-encoder embeddings encode the question and chunk separately. Before calling OpenAI, `rerank_parent_contexts()` jointly scores `(query, parent_text)` pairs through a local Cross-Encoder on GPU/CPU to filter up to 10 candidate parent windows down to the **true top 3**—increasing precision without increasing OpenAI prompt tokens.
5. **Anti-Hallucination Page Citations & Confidence Scores:**
   - `extract_pages_from_pdf()` extracts text page-by-page with 1-based page numbers and filenames.
   - `refine_with_openai()` numbers retrieved parent passages (`[1]`, `[2]`, `[3]`) and uses OpenAI JSON mode (`response_format={"type": "json_object"}`) to enforce `{"answer": "...", "sources": ["[1]"], "confidence": 0.0}`.
   - `_resolve_sources()` deterministically maps `"[1]"` back to `<filename>.pdf, page <N>` in Python and drops any unretrieved labels so citations can never be invented.
6. **Portable Hardware Acceleration (`CUDA` / `MPS` / `CPU`) & `QT_fp16` Quantization:**
   - Automatically detects `cuda` (any NVIDIA GPU), `mps` (Apple Silicon), or falls back to `cpu`.
   - Uses `faiss.IndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_fp16)` (`float16`, 2 bytes/dim), cutting `index/index.faiss` in half (`77.67 MB` $\to$ **`38.84 MB`** for 26.5k chunks) with near-zero accuracy loss.

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
      ├─► [Stage 1A - Dense Search]: PyTorch SentenceTransformer ("all-mpnet-base-v2") + FAISS (Top 25)
      └─► [Stage 1B - Lexical Search]: Okapi BM25 Inverted Index over child_chunks (Top 25)
      │
      ▼ [Stage 2 - Reciprocal Rank Fusion (RRF) in hybrid_search.py]
2. Top-25 Fused Child Chunk Indices (combining semantic similarity + exact keyword matches)
      │
      ▼ [Stage 3 - Two-Tier Parent Expansion: retrieve_parent_contexts()]
3. Up to 10 Deduplicated Parent Contexts (~1,500 chars each, with source filename & page number)
      │
      ▼ [Stage 4 - Cross-Encoder Reranking: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") on GPU]
4. Top-3 Reranked Parent Contexts (jointly scored against the query via full cross-attention)
      │
      ▼ [Stage 5 - Grounded Synthesis: refine_with_openai() formats [1]..[3] + calls gpt-6-luna JSON mode]
5. Validated JSON Reply: {"answer": "...", "sources": ["[1]"], "confidence": 0.95}
      │
      ▼ [_resolve_sources(): maps "[1]" -> "Book.pdf, page 187"]
6. GUI Displays Clean Answer + Confidence % + Page Citations
```

---

## 3. Setup & Usage

```powershell
# 1. Install dependencies (including CUDA-enabled PyTorch on Windows/Linux)
uv sync

# 2. Configure your OpenAI API key
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY=...

# 3. Launch the desktop GUI
uv run pdf-rag

# 4. Run unit tests
uv run python -m unittest discover -s tests -v
```

1. Click **Browse** and select a folder containing your PDFs (e.g., `data/pdfs/`).
2. On first load (or when files change), the app extracts pages, embeds child chunks on your GPU, and saves `index/index.faiss` and `index/metadata.json`. On subsequent runs with unchanged files, it loads the cached index and builds the in-memory BM25 index immediately.
3. Enter a question and click **Get Answer**.

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
│   └── why-hybrid-search-and-reranking.md            # Before vs. After analysis of Hybrid BM25 + Cross-Encoder improvements
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

