# Local PDF RAG Assistant

**Local-Hardware Two-Tier PDF RAG Application** — Built from scratch in native Python using **PyMuPDF**, **PyTorch GPU embeddings** (`CUDA`/`MPS`), **FAISS `QT_fp16` scalar quantization**, and **OpenAI `gpt-6-luna`** with verified page-level citations.

---

## 1. Project Goal & Design Philosophy

Rather than relying on heavy orchestration frameworks (like LangChain or LlamaIndex) or sending entire documents to cloud embedding APIs, **`local-pdf-rag-assistant`** leverages **local hardware (NVIDIA GPU / Apple Silicon / CPU)** for fast, zero-cost vector indexing and similarity search, while using `gpt-6-luna` strictly to synthesize the final answer from the top retrieved passages.

### Key Highlights
1. **Zero Framework Bloat:** Written in ~400 lines of clean, inspectable Python (`app.py`, `database_builder.py`, `document_handler.py`).
2. **Two-Tier (Small-to-Big) Dictionary Retrieval:**
   - **Child Chunks (~250 chars, word-boundary safe):** Embedded into FAISS for sharp, high-precision semantic search.
   - **Parent Store (`{parent_id: {"text", "source", "page"}}`):** Stores ~1,500-character page windows in a Python dictionary. When FAISS matches small child chunks, `retrieve_parent_contexts()` deduplicates their `parent_id`s and passes the full surrounding parent sections to the LLM.
3. **Anti-Hallucination Page Citations & Confidence Scores:**
   - `extract_pages_from_pdf()` extracts text page-by-page with 1-based page numbers and filenames.
   - `refine_with_openai()` numbers retrieved parent passages (`[1]`, `[2]`, `[3]`) and uses OpenAI JSON mode (`response_format={"type": "json_object"}`) to enforce `{"answer": "...", "sources": ["[1]"], "confidence": 0.0}`.
   - `_resolve_sources()` deterministically maps `"[1]"` back to `<filename>.pdf, page <N>` in Python and drops any unretrieved labels so citations can never be invented.
4. **Portable Hardware Acceleration (`CUDA` / `MPS` / `CPU`):**
   - Automatically detects `cuda` (any NVIDIA GPU), `mps` (Apple Silicon), or falls back to `cpu`.
   - With `batch_size=64` on an **NVIDIA RTX A4000 Laptop GPU**, embeds **26,512 child chunks** (across **6,413 parent contexts**) in **2 minutes 5 seconds** (~212 chunks/sec).
5. **Compacted FAISS Index (`QT_fp16` Scalar Quantization):**
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
      ▼ [PyTorch + SentenceTransformer ("all-mpnet-base-v2") on GPU/CPU]
2. Question Vector (768 numbers in the exact same vector space)
      │
      ▼ [faiss.Index.search(query_vector, k=5)]
3. Top-5 Child Chunk Indices Matched by L2 Distance
      │
      ▼ [retrieve_parent_contexts(): looks up parent_id in self.parent_store dict]
4. Top-3 Deduplicated Parent Contexts (with source filename & page number)
      │
      ▼ [refine_with_openai(): formats [1]..[3] + calls gpt-6-luna in JSON mode]
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
uv run python app.py
```

1. Click **Browse** and select a folder containing your PDFs.
2. On first load (or when files change), the app extracts pages, embeds child chunks on your GPU, and saves `index/index.faiss` and `index/metadata.json`. On subsequent runs with unchanged files, it loads the cached index immediately.
3. Enter a question and click **Get Answer**.

---

## 4. Repository Layout

```
local-pdf-rag-assistant/
├── app.py                  # Tkinter desktop GUI, prompt builder, and source resolver
├── database_builder.py     # Two-Tier chunker, GPU SentenceTransformer, and QT_fp16 FAISS store
├── document_handler.py     # PyMuPDF directory scanner and per-page text extractor
├── pyproject.toml          # Dependencies managed via uv (with pytorch-cu128 index)
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
└── index/                  # Persisted QT_fp16 FAISS index and UTF-8 metadata dictionary
```

