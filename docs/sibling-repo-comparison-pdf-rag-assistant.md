# Comparison with Sibling Repo (`pdf-rag-assistant`)

This document records the architectural differences between **`local-pdf-rag-assistant`** (this repository) and the sibling repository **`pdf-rag-assistant`** (`../pdf-rag-assistant`) as a reference for future improvements—without modifying any code in `pdf-rag-assistant`.

---

## 1. Side-by-Side Architecture Comparison

| Dimension | `pdf-rag-assistant` (Sibling) | `local-pdf-rag-assistant` (This Repo) |
| :--- | :--- | :--- |
| **UI / Entry Points** | Streamlit web UI (`app.py`) + CLI (`main.py`) | Tkinter desktop GUI (`src/pdf_rag/app.py`, `uv run pdf-rag`) |
| **PDF Extractor** | `pypdf` (`rag/pdf_loader.py`) | `pymupdf` (`src/pdf_rag/document_handler.py`) |
| **Embeddings** | OpenAI `text-embedding-3-small` API (`rag/ingest.py`) | Local `SentenceTransformer("all-mpnet-base-v2")` on CUDA/MPS/CPU |
| **Vector Store** | ChromaDB persistent collection (`./chroma_db` via `chromadb.PersistentClient`) | FAISS `IndexScalarQuantizer(QT_fp16)` (`index/index.faiss`) + `index/metadata.json` |
| **Chunking Strategy** | Fixed 500-char slices (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`, can split mid-word) | Word-boundary-safe Two-Tier: 250-char children $\to$ 1,500-char overlapping parents |
| **Lexical / Hybrid Search** | **None** — dense-only `collection.query(..., n_results=5)` (`rag/retrieve.py`) | **Okapi BM25 + FAISS** merged via Reciprocal Rank Fusion (`rrf_k=60`) |
| **Reranking** | **None** — top 5 Chroma hits go directly to prompt | **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks up to 10 parents $\to$ top 3 |
| **Grounding & Citations** | Prompt-level refusal + `[n]` label $\to$ `source, page` mapping + Pydantic schema (`rag/answer.py`) | Code-level refusal toggle (`Documents only`) + `[n]` label $\to$ `source, page` mapping |
| **Debugging & Evals** | None | `Show retrieved passages` GUI toggle + `scripts/eval_retrieval.py` Hit@3 benchmark |
| **Offline Capability** | Requires OpenAI API for both indexing and querying | Indexing, BM25, FAISS, and Cross-Encoder reranking run 100% offline (\$0 API cost) |

---

## 2. Why `pdf-rag-assistant` Often Replies "Nothing Was Found"

1. **Strict Grounding Prompt (`rag/prompt.py`):**
   `gpt-6-luna` is instructed to answer *only* from the numbered passages and to say plainly when the passages do not contain the answer.
2. **Narrow, Dense-Only 500-Character Window (`rag/retrieve.py`):**
   Each question only fetches `n_results=5` fixed 500-character chunks (~2,500 characters total), with no BM25 keyword search, no parent-window expansion, and no reranking.
   - If the answer relies on an exact identifier, acronym, or clause number that dense embeddings miss, or if the answer is split across a 500-character slice boundary, none of the 5 chunks contain the complete fact—so the LLM correctly refuses.

---

## 3. Why ChromaDB Appears Very Fast in `pdf-rag-assistant`

1. **Remote API Embeddings (`text-embedding-3-small`):**
   There is no local PyTorch Transformer (`all-mpnet-base-v2`, ~420 MB) loaded into VRAM at startup or run locally during batch ingestion.
2. **Single-PDF Upload Workflow:**
   Streamlit ingests one uploaded PDF at a time (`ingest_pdf`), rather than scanning and embedding an entire multi-book directory (tens of thousands of chunks) in one pass.
3. **In-Process Cached HNSW (`@cache` on `get_collection()`):**
   ChromaDB keeps its `hnswlib` index open in memory across queries, so the local nearest-neighbor lookup takes ~1–2 ms (comparable to FAISS). The trade-off is ongoing OpenAI embedding API latency and cost on every upload and every query.

---

## 4. Future Upgrade Checklist for `pdf-rag-assistant`

If we revisit `pdf-rag-assistant` later, the highest-impact upgrades to port from this repo—without changing its ChromaDB + Streamlit stack—are:

1. **Word-Boundary & Two-Tier (Parent-Child) Chunking:**
   Store small child chunks in ChromaDB for sharp search, and store parent window text (or `parent_id`) in Chroma metadata so the LLM receives full ~1,500-character paragraphs instead of 500-character fragments.
2. **BM25 Hybrid Search + RRF:**
   Build an in-memory `BM25Index` from `collection.get(include=["documents", "metadatas"])` and fuse its top-25 results with Chroma's top-25 results via `reciprocal_rank_fusion()` so exact identifiers and policy names never miss.
3. **Cross-Encoder Reranking:**
   Retrieve 20–25 candidates instead of 5, deduplicate parents, and rerank down to the top 3–5 passages before calling `gpt-6-luna`.
4. **Passage Inspection & Retrieval Eval:**
   Add an optional Streamlit expander showing the retrieved passages (`source`, `page`, snippet) and a local Hit@3 evaluation script alongside `chroma_db/`.
