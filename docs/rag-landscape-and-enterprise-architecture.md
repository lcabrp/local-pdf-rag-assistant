# The RAG Landscape: Storage Engines, Retrieval Architectures & Enterprise Production

This document compares the different ways to build a Retrieval-Augmented Generation (RAG) system—contrasting our local **FAISS** implementation in `local-pdf-rag-assistant` with **ChromaDB** (used in `pdf-rag-assistant`) and with **Enterprise Production RAG** architectures—and outlines the highest-impact local upgrades for this repository.

---

## 1. Two Axes of RAG Design

When comparing RAG implementations, it helps to separate two independent layers:

1. **The Storage & Search Engine:** *Where* vectors, text chunks, and metadata live (e.g., FAISS, ChromaDB, PostgreSQL `pgvector`, Qdrant, Azure AI Search).
2. **The Retrieval Architecture:** *How* candidate passages are searched, filtered, scored, and expanded before being handed to the LLM (e.g., Naive Dense Search, Two-Tier Parent-Child, Hybrid BM25 + Vector, Cross-Encoder Reranking, Agentic RAG, GraphRAG).

> **Key Insight:** Swapping one vector store for another (e.g., FAISS $\leftrightarrow$ ChromaDB) while keeping the same embedding model and chunking strategy yields nearly identical answer quality. What changes between storage engines is **operational ergonomics** (metadata filtering, persistence, concurrency, scale), whereas **answer quality** is driven by your **Retrieval Architecture**.

---

## 2. Storage & Vector Search Engines (5 Categories)

| Category | Representative Tools | How It Works | Strengths & Trade-Offs |
| :--- | :--- | :--- | :--- |
| **1. In-Process Vector Libraries** *(Used in this repo)* | **FAISS**, **USearch**, **Annoy**, **hnswlib** | Pure C++/Python linear-algebra & nearest-neighbor search library (`index/index.faiss`). Stores only raw/quantized vectors; has **no built-in document store or `WHERE` filtering**—paired with a custom sidecar (`index/metadata.json`). | **Fastest local execution**, explicit control over quantization (`QT_fp16` cuts RAM/disk by 50%), zero server/daemon overhead. Requires managing metadata synchronization yourself. |
| **2. Embedded Vector Databases** *(Used in sibling repo)* | **ChromaDB**, **LanceDB**, **DuckDB (`vss`)**, **SQLite-vec** | Runs in-process and persists to a local directory/file, but provides a full database API: stores vectors, raw text, and metadata together and supports metadata filtering (`WHERE page > 10`). | Great developer ergonomics for local apps and prototypes. Slightly more storage/serialization overhead than bare FAISS. |
| **3. Relational DBs + Vector Extensions** | **PostgreSQL (`pgvector` / `pgvectorscale`)**, **SQL Server 2025 / Azure SQL**, **MongoDB Atlas**, **Redis** | Adds HNSW or DiskANN vector indexes directly inside a standard transactional database. | **Top Enterprise choice** when application data already lives in SQL/Postgres. Gives ACID transactions, relational joins, backups, and row-level security in one system. |
| **4. Dedicated Distributed Vector DBs** | **Qdrant**, **Milvus / Zilliz**, **Weaviate**, **Pinecone**, **Turbopuffer** | Purpose-built vector database servers (typically written in Rust/Go/C++) supporting tens of millions to billions of vectors, sharding, replication, and filtered HNSW search. | Built for high-scale multi-tenant SaaS workloads and low-latency pre-filtered vector search at massive scale. |
| **5. Enterprise Search Engines (Lexical + Vector)** | **Elasticsearch / OpenSearch**, **Azure AI Search**, **Vespa**, **Meilisearch** | Mature full-text search engines (BM25 inverted indexes) extended with native vector indexing, hybrid scoring, and security trimming. | **Top Enterprise choice** for internal knowledge bases, legal/financial document repositories, and SharePoint/Confluence search. |

---

## 3. Retrieval Architectures (From Basic to Advanced)

1. **Naive Dense RAG**
   - Fixed-size text chunks $\to$ dense embeddings $\to$ top-$k$ cosine/L2 search $\to$ LLM prompt.
   - *Weakness:* Chunks either lose surrounding context (if too small) or dilute the embedding vector (if too large), and dense embeddings frequently miss exact keywords, part numbers, or acronyms.

2. **Two-Tier / Small-to-Big Retrieval *(Implemented in this repo)***
   - Embeds small, focused **child chunks (~250 chars)** into FAISS for sharp semantic matching, then maps matched children back to deduplicated **parent windows (~1,500 chars)** in `parent_store` so the LLM reads complete paragraphs.

3. **Hybrid Search (Dense Vectors + Sparse Lexical BM25)**
   - Runs **Dense Vector Search** (semantic meaning) and **BM25 Keyword Search** (exact tokens, identifiers, numbers, names) in parallel, combining their rankings via **Reciprocal Rank Fusion (RRF)**.

4. **Two-Stage Retrieval with Cross-Encoder Reranking**
   - Stage 1 (Bi-Encoder / Hybrid) retrieves a wide net of candidates (e.g., `top_k = 30` fast).
   - Stage 2 passes each `(query, candidate_text)` pair together through a **Cross-Encoder Reranker** (e.g., `BAAI/bge-reranker-v2-m3` or `ms-marco-MiniLM-L-6-v2`) to compute deep token-by-token relevance and selects the true top 3–5 parents.

5. **Contextual Retrieval (Anthropic Pattern)**
   - Prepends a short document/section header (e.g., `[Source: Manual.pdf | Page 42 | Section: Hydraulic Pressure Limits]`) to every child chunk before embedding and BM25 indexing so pronouns and isolated numbers retain their subject.

6. **Agentic RAG (Tool-Using Retrieval)**
   - Exposes retrieval as tools (`search_chunks`, `keyword_search`, `read_page`) so the model can decompose multi-part questions, run follow-up searches, or inspect adjacent pages when initial results are insufficient.

7. **GraphRAG**
   - Extracts entities and relationships into a knowledge graph alongside vector chunks to answer corpus-wide or multi-hop relational queries.

---

## 4. What Enterprise RAG Actually Uses in Production

Enterprise production systems rarely use bare FAISS or local ChromaDB. Instead, they converge on a five-stage pipeline designed around **access control, table/layout fidelity, and hybrid precision**:

```text
PDFs / SharePoint / Confluence / Databases
  │
  ▼  1. Layout-Aware Document Parsing (Docling, Azure Document Intelligence, Unstructured)
     Preserves tables as Markdown/HTML, section hierarchies, headings, and page bounding boxes
  │
  ▼  2. Hierarchical Chunking + ACL Metadata Tagging
     Attaches document permissions (user/group ACLs), effective dates, version, and parent IDs
  │
  ▼  3. Unified Hybrid Store (PostgreSQL pgvector + tsvector, Azure AI Search, OpenSearch, or Qdrant)
     Stores both HNSW/DiskANN dense vectors and BM25 inverted keyword indexes
  │
  ▼  4. Multi-Stage Query Pipeline:
     ├─► Security Pre-Filter (WHERE user_roles && document_acl)
     ├─► Hybrid Search: Dense Vector + BM25 merged via Reciprocal Rank Fusion (Top 30–50)
     ├─► Cross-Encoder Reranker: Scores (query, chunk) pairs together (Top 30 ──► Top 5)
     └─► Parent-Window Expansion + Grounded JSON Generation with Verified Citations
```

### Why Enterprises Standardize on This Stack
- **Security Trimming (RBAC / ACL Pre-Filtering):** Permissions must be enforced *inside* the retrieval query before chunks ever reach the reranker or LLM.
- **Layout & Table Preservation:** Standard PDF text extractors flatten multi-column layouts and financial/engineering tables into unreadable word soup. Layout-aware parsers (such as **Docling** locally) convert tables into structured Markdown so the LLM can read rows and columns accurately.
- **Hybrid + Reranking Over Raw Vector Similarity:** In enterprise evaluations, adding **BM25 + Cross-Encoder Reranking** reduces retrieval failures by **30–50%**.

---

## 5. Benefits of Adding BM25 Hybrid Search & Cross-Encoder Reranking to This Repo

`local-pdf-rag-assistant` already has two major architectural advantages over basic tutorials: **Two-Tier (Small-to-Big) Parent-Child Retrieval** and **Deterministic Page Citations**.

Adding **1) BM25 Hybrid Search** and **2) a Local Cross-Encoder Reranker** directly addresses the two remaining blind spots of pure FAISS vector search while staying **100% local, zero-API-cost, and fast on your GPU**:

### Benefit 1: BM25 Hybrid Search Fixes the "Exact Keyword & Identifier Blind Spot"
- **The Problem with Pure Embeddings (`all-mpnet-base-v2` + FAISS):**
  Dense embedding models compress a 250-character chunk into 768 floating-point numbers representing *general semantic topic*. If you ask about a specific **part number, error code, statute number, chemical formula, acronym, function name, or exact figure** (e.g., `"What does error E-409 mean?"` or `"Clause 14.3(b)"`), dense embeddings often retrieve generic passages about errors or clauses while missing the exact page containing `E-409`.
- **What BM25 Adds:**
  BM25 scores exact term frequency and inverse document frequency (rare words like `E-409` get massive weight). By fusing FAISS top-20 results with BM25 top-20 results via **Reciprocal Rank Fusion (RRF)**, you get **semantic understanding** (when you paraphrase) *and* **exact keyword lock-on** (when you ask for a specific name, code, or number).
- **Cost in This Repo:**
  Pure Python (`rank-bm25`), builds in milliseconds alongside `self.child_chunks`, adds negligible RAM, and requires zero network calls.

### Benefit 2: Cross-Encoder Reranking Fixes "False-Positive Parent Contexts"
- **The Problem with Bi-Encoder Vector Distance (`k=5` $\to$ `max_parents=3`):**
  FAISS embeds the **question** and the **child chunks** *separately* (a "Bi-Encoder") and compares them with L2 distance. Because the question and passage never attend to each other token-by-token during embedding, chunks that merely share vocabulary with the question often outrank the chunk that actually *answers* the question. With `max_parents=3`, just two false-positive child matches can push the true answer out of the 3 parent slots sent to OpenAI.
- **What a Cross-Encoder Reranker Adds:**
  Instead of trusting the top 5 FAISS hits blindly, you retrieve the top **20–25 candidate child/parent passages** and pass `(query, passage)` pairs simultaneously through a local Cross-Encoder on your GPU (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-base`, already supported by `sentence-transformers` which is installed in this repo).
  - The Cross-Encoder performs full self-attention across the question and passage together, catching negations, subtle qualifiers, and direct question-answer alignment.
  - Only the **top 3 reranked parent contexts** are sent to OpenAI—improving answer accuracy *and* keeping your OpenAI token usage low.
- **Cost in This Repo:**
  Uses `sentence_transformers.CrossEncoder` (no new heavy framework needed—`sentence-transformers` is already in `pyproject.toml`), runs in **~30–80 ms on your CUDA GPU** for 20 candidates, and dramatically increases the precision of the 3 parent passages passed to `refine_with_openai()`.
