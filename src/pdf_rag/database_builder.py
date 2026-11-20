"""Two-Tier RAG index builder and multi-stage hybrid + cross-encoder retriever.

Architecture Overview:
----------------------
1. Two-Tier (Small-to-Big) Storage:
   - Child Chunks (~250 chars): Embedded into a 16-bit quantized FAISS index (`QT_fp16`)
     and indexed into an in-memory Okapi BM25 inverted index for sharp, granular matching.
   - Parent Store (~1,500 chars): Stored in a Python dictionary keyed by `parent_id` with
     provenance metadata (`source` filename and 1-based `page` number).

2. Four-Stage Retrieval Pipeline (`retrieve_for_query`):
   - Stage 1 (Dense + Lexical Retrieval): Queries FAISS (semantic similarity) and BM25
     (exact keywords/identifiers) for the top `candidate_k` child chunks each.
   - Stage 2 (Reciprocal Rank Fusion): Merges FAISS and BM25 rankings into a single
     scale-invariant candidate list using RRF.
   - Stage 3 (Two-Tier Parent Expansion): Maps matched child chunks back to their
     deduplicated ~1,500-character parent windows (`rerank_candidates`).
   - Stage 4 (Cross-Encoder Reranking): Jointly scores `(query, parent_text)` pairs on
     GPU/CPU using a Cross-Encoder to select the final `max_parents` passages for the LLM.
"""

from __future__ import annotations

import hashlib
import json
import os

import faiss
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from pdf_rag.hybrid_search import BM25Index, reciprocal_rank_fusion

DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class DatabaseBuilder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        index_dir: str = "index",
        reranker_model_name: str | None = DEFAULT_RERANKER_MODEL,
    ) -> None:
        """Initialize the embedding model, storage paths, BM25 index, and lazy reranker.

        Why the Cross-Encoder is loaded lazily (`self.reranker = None`):
        ----------------------------------------------------------------
        Loading two Transformer models into VRAM at GUI startup slows down window launch
        even before the user selects a folder or asks a question. Deferring Cross-Encoder
        initialization until the first rerank call keeps startup snappy and allows offline
        or unit-test environments to run without downloading a second model upfront.
        """
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 Using NVIDIA GPU for embeddings & reranking: {gpu_name}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            print("🚀 Using Apple Silicon GPU (MPS) for embeddings & reranking")
        else:
            self.device = "cpu"
            print("💻 Using CPU for embeddings & reranking")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.reranker_model_name = reranker_model_name
        self.reranker: CrossEncoder | None = None
        self._reranker_load_failed: bool = False

        self.index_dir = index_dir
        self.index: faiss.Index | None = None
        self.bm25_index = BM25Index()
        self.child_chunks: list[
            str
        ] = []  # Small text snippets embedded into FAISS + BM25
        self.child_metadata: list[
            dict
        ] = []  # [{"parent_id": int, "source": str, "page": int, "char_len": int}]
        self.parent_store: dict[
            int, dict
        ] = {}  # {parent_id: {"text": str, "source": str, "page": int}}
        self.metadata_file = os.path.join(self.index_dir, "metadata.json")
        self.index_file = os.path.join(self.index_dir, "index.faiss")

        os.makedirs(self.index_dir, exist_ok=True)

    def compute_hash(self, file_path: str) -> str | None:
        """Compute MD5 hash of a file's contents for change detection."""
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError as e:
            print(f"Error computing hash for {file_path}: {e}")
            return None

    def save_metadata(self, pdf_paths: list[str]) -> None:
        """Save metadata: file hashes, child chunks, child metadata, and parent store."""
        try:
            metadata = {
                "schema_version": 2,
                "files": {pdf: self.compute_hash(pdf) for pdf in pdf_paths},
                "child_chunks": self.child_chunks,
                "child_metadata": self.child_metadata,
                "parent_store": self.parent_store,
            }
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False)
        except OSError as e:
            print(f"Error saving metadata: {e}")

    def load_metadata(self) -> dict:
        """Load metadata from disk, returning an empty dict if missing or corrupted."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                print("Metadata file is corrupted. Rebuilding index.")
                return {}
        return {}

    def has_changes(self, pdf_paths: list[str]) -> bool:
        """Check if PDFs have changed, new files were added, or metadata schema upgraded."""
        existing_metadata = self.load_metadata()
        if existing_metadata.get("schema_version") != 2:
            return True

        existing_hashes = existing_metadata.get("files", {})
        if not existing_hashes:
            return True

        current_hashes = {}
        for pdf in pdf_paths:
            file_hash = self.compute_hash(pdf)
            if file_hash:
                current_hashes[pdf] = file_hash

        return current_hashes != existing_hashes

    def split_into_chunks(self, text: str, chunk_size: int = 250) -> list[str]:
        """Split text into word-boundary-safe child chunks for sharp embedding retrieval.

        Why word-boundary splitting matters:
        Slicing raw strings every N characters (`text[i:i+250]`) chops words in half
        (e.g., `"quantization"` -> `"quant"` + `"ization"`), which corrupts both
        subword tokenization in the embedding model and exact token matching in BM25.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected a string, but received {type(text)}")

        words = text.split()
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for w in words:
            if current_len + len(w) + 1 > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [w]
                current_len = len(w)
            else:
                current_chunk.append(w)
                current_len += len(w) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def build_embeddings(
        self,
        pages_or_texts: list[dict | str],
        pdf_paths: list[str],
        parent_window_size: int = 1500,
        child_chunk_size: int = 250,
    ) -> None:
        """Build a Two-Tier RAG index (FAISS QT_fp16 + BM25) with page-level provenance."""
        self.child_chunks = []
        self.child_metadata = []
        self.parent_store = {}

        parent_id = 0

        for item in pages_or_texts:
            if isinstance(item, dict):
                full_text = item.get("text", "")
                source_name = item.get("source", "unknown")
                page_num = item.get("page", 1)
            elif isinstance(item, str):
                full_text = item
                source_name = "document"
                page_num = 1
            else:
                continue

            if not full_text or not full_text.strip():
                continue

            # Step 1: Create overlapping Parent Windows (~1500 chars with 200-char overlap)
            # Why 200-char overlap: prevents sentences at the boundary of a long page
            # from being orphaned without their preceding context.
            step_size = max(parent_window_size - 200, 1)
            raw_parents = [
                full_text[i : i + parent_window_size]
                for i in range(0, len(full_text), step_size)
            ]

            for parent_text in raw_parents:
                clean_parent = parent_text.strip()
                if not clean_parent:
                    continue

                parent_id += 1
                self.parent_store[parent_id] = {
                    "text": clean_parent,
                    "source": source_name,
                    "page": page_num,
                }

                # Step 2: Create Child Chunks (~250 characters) linked to this Parent ID
                children = self.split_into_chunks(
                    clean_parent, chunk_size=child_chunk_size
                )

                for child in children:
                    clean_child = child.strip()
                    if (
                        len(clean_child) < 30
                    ):  # Skip trivial headers/footers/page numbers
                        continue

                    self.child_chunks.append(clean_child)
                    self.child_metadata.append(
                        {
                            "parent_id": parent_id,
                            "source": source_name,
                            "page": page_num,
                            "char_len": len(clean_child),
                        }
                    )

        if not self.child_chunks:
            print("⚠ No valid text chunks found to create embeddings.")
            return

        try:
            print(
                f"📊 Two-Tier Indexing: Created {len(self.parent_store)} Parents and {len(self.child_chunks)} Children."
            )
            print(
                f"🧠 Generating embeddings for {len(self.child_chunks)} child chunks on {self.device.upper()}..."
            )
            all_embeddings = self.model.encode(
                self.child_chunks, batch_size=64, show_progress_bar=True
            )

            emb_array = np.asarray(all_embeddings, dtype=np.float32)
            dimension = emb_array.shape[1]

            # Why QT_fp16 scalar quantization:
            # Compresses 32-bit floats (4 bytes/dim) to 16-bit floats (2 bytes/dim),
            # cutting index RAM and disk footprint by 50% with near-zero recall loss.
            self.index = faiss.IndexScalarQuantizer(
                dimension, faiss.ScalarQuantizer.QT_fp16
            )
            self.index.train(emb_array)
            self.index.add(emb_array)

            # Build in-memory BM25 inverted index over the exact same child_chunks list
            self.bm25_index.build(self.child_chunks)

            self.save_index()
            self.save_metadata(pdf_paths)
            print("✅ Two-Tier FAISS + BM25 index created successfully.")
        except Exception as e:  # noqa: BLE001
            print(f"⚠ Error creating FAISS index: {e}")

    def save_index(self) -> None:
        """Save FAISS index to disk if initialized."""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_file)
            else:
                print("Warning: Attempted to save an uninitialized index.")
        except Exception as e:  # noqa: BLE001
            print(f"Error saving FAISS index: {e}")

    def load_index(self) -> None:
        """Load FAISS index and two-tier dictionaries from disk, and build BM25 in memory.

        Why BM25 is rebuilt in memory on `load_index()` rather than pickled to disk:
        ----------------------------------------------------------------------------
        1. Backward compatibility: Existing `index/metadata.json` files already contain
           `child_chunks`, so upgrading to Hybrid Search requires zero re-embedding or
           schema migration.
        2. Speed & safety: Tokenizing 26,500 short child chunks takes ~150 ms in Python
           and avoids insecure `pickle` deserialization.
        """
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
                meta = self.load_metadata()
                self.child_chunks = meta.get("child_chunks", [])
                self.child_metadata = meta.get("child_metadata", [])
                # JSON keys are strings; convert parent_store keys back to int
                raw_parent = meta.get("parent_store", {})
                self.parent_store = {int(k): v for k, v in raw_parent.items()}
                self.bm25_index.build(self.child_chunks)
            except Exception as e:  # noqa: BLE001
                print(f"Error loading FAISS index: {e}")
                self.index = None
        else:
            print("FAISS index file missing. Rebuilding index is required.")

    def search_index(self, query: str, k: int = 5):
        """Dense-only FAISS vector search (kept for direct benchmarking & compatibility)."""
        if self.index is None:
            print(
                "Error: FAISS index not initialized. Try reloading or rebuilding the index."
            )
            return [], []

        try:
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(query_embedding, k)
            return distances, indices
        except Exception as e:  # noqa: BLE001
            print(f"Error searching index: {e}")
            return [], []

    def hybrid_search_index(self, query: str, candidate_k: int = 25):
        """Stage 1 & 2: Run Dense FAISS + Lexical BM25 search and fuse rankings via RRF.

        Why Hybrid Search outperforms Dense-only search:
        - FAISS (`all-mpnet-base-v2`) finds conceptually related chunks even when the
          user paraphrases the terminology.
        - BM25 (`self.bm25_index`) locks onto exact identifiers, acronyms, page/section
          numbers, and rare technical terms that dense embeddings blur together.
        - Reciprocal Rank Fusion (`reciprocal_rank_fusion`) merges both ranked lists
          without needing score calibration, promoting chunks that both systems agree on.
        """
        if self.index is None:
            print(
                "Error: FAISS index not initialized. Try reloading or rebuilding the index."
            )
            return [], []

        distances, faiss_indices = self.search_index(query, k=candidate_k)
        if faiss_indices is None or len(faiss_indices) == 0:
            return [], []

        faiss_ranked = [int(idx) for idx in faiss_indices[0] if idx >= 0]

        # Lazily ensure BM25 is built if child_chunks were populated outside load_index()
        if self.bm25_index.num_docs != len(self.child_chunks) and self.child_chunks:
            self.bm25_index.build(self.child_chunks)

        bm25_ranked = self.bm25_index.search(query, k=candidate_k)

        fused_ids = reciprocal_rank_fusion(
            [faiss_ranked, bm25_ranked],
            rrf_k=60,
            top_n=candidate_k,
        )
        if not fused_ids:
            return distances, faiss_indices

        # Return shape (1, len(fused_ids)) to match FAISS's 2D index array convention
        return distances, np.asarray([fused_ids], dtype=np.int64)

    def retrieve_parent_contexts(self, indices, max_parents: int = 3):
        """Stage 3: Map child chunk indices back to deduplicated parent contexts.

        Why we deduplicate by `parent_id`:
        A single 1,500-character parent passage typically spawns 5-7 child chunks.
        When a parent is highly relevant, multiple of its children will land in the top
        search results. Deduplicating by `parent_id` ensures we never send duplicate
        copies of the same paragraph to the reranker or LLM.
        """
        if indices is None or len(indices) == 0 or len(indices[0]) == 0:
            return [], []

        matched_children: list[str] = []
        retrieved_parents: list[dict] = []
        seen_parent_ids: set[int] = set()

        for idx in indices[0]:
            if 0 <= idx < len(self.child_chunks):
                child_text = self.child_chunks[idx]
                matched_children.append(child_text)

                pid = self.child_metadata[idx].get("parent_id")
                if pid and pid in self.parent_store and pid not in seen_parent_ids:
                    seen_parent_ids.add(pid)
                    retrieved_parents.append(self.parent_store[pid])
                    if len(retrieved_parents) >= max_parents:
                        break

        return matched_children, retrieved_parents

    def _get_reranker(self) -> CrossEncoder | None:
        """Lazily initialize the Cross-Encoder reranker on first use, degrading gracefully if unavailable."""
        if not self.reranker_model_name or self._reranker_load_failed:
            return None
        if self.reranker is None:
            try:
                print(
                    f"🎯 Loading Cross-Encoder reranker ({self.reranker_model_name}) on {self.device.upper()}..."
                )
                self.reranker = CrossEncoder(
                    self.reranker_model_name, device=self.device
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"⚠ Cross-Encoder unavailable ({e}); falling back to Hybrid RRF order."
                )
                self._reranker_load_failed = True
                return None
        return self.reranker

    def rerank_parent_contexts(
        self,
        query: str,
        parent_contexts: list[dict],
        max_parents: int = 3,
        extra_queries: list[str] | None = None,
    ) -> list[dict]:
        """Stage 4: Re-score candidate parent passages jointly with the query using a Cross-Encoder.

        Why Cross-Encoder reranking is performed on Parent contexts (not Child chunks):
        -------------------------------------------------------------------------------
        1. Full context visibility: A 250-char child chunk may contain a section heading
           or question while the actual answer sits two sentences later in the 1,500-char
           parent window. Scoring `(query, parent_text)` lets the Cross-Encoder attend
           across both the heading and the answer simultaneously.
        2. Compute efficiency: Deduplicating child hits into <=15 unique parent windows
           before reranking cuts Transformer forward passes by >60%.
        3. Multi-Query max-scoring: When `extra_queries` are supplied (Query Expansion),
           each parent is scored against the original query and each expanded variant,
           keeping its best (max) relevance score so a passage matching a synonym
           concept (e.g., "Sets" for "remove duplicates") is not penalized.
        """
        if not parent_contexts:
            return []
        if len(parent_contexts) <= 1:
            return parent_contexts[:max_parents]

        reranker = self._get_reranker()
        if reranker is None:
            return parent_contexts[:max_parents]

        try:
            all_queries = [query, *(extra_queries or [])]
            parent_texts = [
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in parent_contexts
            ]
            # Score every (query_variant, parent_text) pair in a single batched GPU pass
            pairs = [(q, text) for q in all_queries for text in parent_texts]
            raw_scores = reranker.predict(pairs, show_progress_bar=False)

            num_parents = len(parent_contexts)
            best_scores: list[float] = []
            for p_idx in range(num_parents):
                variant_scores = [
                    float(raw_scores[q_idx * num_parents + p_idx])
                    for q_idx in range(len(all_queries))
                ]
                best_scores.append(max(variant_scores))

            scored_parents = sorted(
                zip(best_scores, parent_contexts, strict=False),
                key=lambda pair: pair[0],
                reverse=True,
            )
            return [parent for _score, parent in scored_parents[:max_parents]]
        except Exception as e:  # noqa: BLE001
            print(f"⚠ Error during Cross-Encoder reranking ({e}); using RRF order.")
            return parent_contexts[:max_parents]

    def retrieve_for_query(
        self,
        query: str,
        candidate_k: int = 25,
        rerank_candidates: int = 10,
        max_parents: int = 3,
        extra_queries: list[str] | None = None,
    ) -> tuple[list[str], list[dict]]:
        """Execute the full 4-stage retrieval pipeline: Hybrid (FAISS + BM25 + RRF) -> Parent Expansion -> Cross-Encoder Rerank.

        Args:
            query: User's natural-language question.
            candidate_k: Number of top child chunks to retrieve from each of FAISS and BM25 (default: 25).
            rerank_candidates: Number of deduplicated parent windows to pass into the Cross-Encoder (default: 10).
            max_parents: Final number of top-ranked parent passages to send to the LLM (default: 3).
            extra_queries: Optional LLM-generated alternative queries for Multi-Query Expansion.

        Returns:
            Tuple of `(matched_children, top_reranked_parents)`.
        """
        if extra_queries:
            ranked_lists: list[list[int]] = []
            for q in [query, *extra_queries]:
                _dist, q_indices = self.hybrid_search_index(q, candidate_k=candidate_k)
                if q_indices is not None and len(q_indices) > 0:
                    ranked_lists.append([int(i) for i in q_indices[0] if i >= 0])
            fused = reciprocal_rank_fusion(ranked_lists, rrf_k=60, top_n=candidate_k)
            fused_indices = np.asarray([fused], dtype=np.int64) if fused else []
        else:
            _distances, fused_indices = self.hybrid_search_index(
                query, candidate_k=candidate_k
            )

        if (
            fused_indices is None
            or len(fused_indices) == 0
            or len(fused_indices[0]) == 0
        ):
            return [], []

        matched_children, candidate_parents = self.retrieve_parent_contexts(
            fused_indices,
            max_parents=max(rerank_candidates, max_parents),
        )
        if not candidate_parents:
            return matched_children, []

        top_parents = self.rerank_parent_contexts(
            query,
            candidate_parents,
            max_parents=max_parents,
            extra_queries=extra_queries,
        )
        return matched_children, top_parents
