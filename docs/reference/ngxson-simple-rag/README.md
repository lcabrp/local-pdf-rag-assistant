# Reference: "Code a Simple RAG from Scratch" (`ngxson`)

Saved verbatim for educational reference from Xuan-Son Nguyen's (`ngxson`) Hugging Face blog post and companion repository:

- **Article:** https://huggingface.co/blog/ngxson/make-your-own-rag
- **Source Repo:** https://huggingface.co/ngxson/demo_simple_rag_py
- **Files Saved Here:**
  - [`demo.py`](demo.py) — Complete 77-line RAG pipeline in pure Python + Ollama
  - [`cat-facts.txt`](cat-facts.txt) — Sample line-per-fact knowledge base

---

## Key Concepts Highlighted in the Article

1. **Embeddings as Numeric Meaning:**
   Each text chunk is passed through an embedding model (`hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` via `ollama.embed`) to produce a dense vector of floats (`[0.1, 0.04, -0.34, ...]`).
2. **A Vector Database at Its Core (`VECTOR_DB`):**
   At the simplest level, a vector store is just a list of `(chunk, embedding)` tuples in memory. Dedicated stores (FAISS, ChromaDB, pgvector) add persistence, quantization, metadata filtering, and fast approximate nearest-neighbor indexes (HNSW/IVF) on top of this exact idea.
3. **Cosine Similarity from First Principles:**
   $$\text{cosine\_similarity}(a, b) = \frac{a \cdot b}{\|a\| \, \|b\|}$$
   Measures the angle between the query vector and every stored chunk vector (`1.0` = identical direction, `0.0` = orthogonal).
4. **Retrieval Before Generation (Grounding):**
   The script retrieves the top $N=3$ chunks, prints them with their similarity scores for inspection, and injects them into the `system` prompt (*"Use only the following pieces of context to answer the question. Don't make up any new information."*).
5. **Streaming Local Generation:**
   Calls `ollama.chat(..., stream=True)` with a compact 1B local model (`hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF`) to print tokens as they are generated.

---

## How `local-pdf-rag-assistant` Builds Beyond This Baseline

- Replaces 1-line `cat-facts.txt` with **PyMuPDF per-page extraction** and **Two-Tier (Parent-Child) chunking**.
- Replaces ephemeral in-memory brute-force search with persistent **FAISS `QT_fp16` + Okapi BM25 (light stemming) + Reciprocal Rank Fusion + GPU Cross-Encoder reranking + optional Multi-Query expansion**.
- Adds **local-first Ollama with a `/api/ps` no-CPU-spill guard + OpenAI fallback** and **deterministic `[n]` $\to$ `<file>.pdf, page <N>` citations**.
