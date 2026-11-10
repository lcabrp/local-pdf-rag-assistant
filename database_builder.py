# database_builder.py

import hashlib
import json
import os

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class DatabaseBuilder:
    # all-MiniLM-L6-v2
    def __init__(self, model_name="all-mpnet-base-v2", index_dir="index"):
        """
        Initializes the DatabaseBuilder with model name and index directory.
        Automatically selects CUDA GPU if available.
        """
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 Using NVIDIA GPU for embeddings: {gpu_name}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            print("🚀 Using Apple Silicon GPU (MPS) for embeddings")
        else:
            self.device = "cpu"
            print("💻 Using CPU for embeddings")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index_dir = index_dir
        self.index = None
        self.child_chunks = []  # Small text snippets embedded into FAISS
        self.child_metadata = []  # [{"parent_id": int, "source": str, "page": int, "char_len": int}]
        self.parent_store = {}  # {parent_id: {"text": str, "source": str, "page": int}}
        self.metadata_file = os.path.join(self.index_dir, "metadata.json")
        self.index_file = os.path.join(self.index_dir, "index.faiss")

        # Ensure index directory exists
        os.makedirs(self.index_dir, exist_ok=True)

    def compute_hash(self, file_path):
        """
        Compute MD5 hash of a file's contents for change detection.
        """
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f"Error computing hash for {file_path}: {e}")
            return None

    def save_metadata(self, pdf_paths):
        """
        Save metadata: file hashes, child chunk metadata, and parent store dictionary.
        """
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
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def load_metadata(self):
        """
        Load metadata from disk, return empty if missing.
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                print("Metadata file is corrupted. Rebuilding index.")
                return {}
        return {}

    def has_changes(self, pdf_paths):
        """
        Check if files have changed, new files have been added, or metadata schema upgraded.
        """
        existing_metadata = self.load_metadata()
        if existing_metadata.get("schema_version") != 2:
            return True

        existing_hashes = existing_metadata.get("files", {})
        current_hashes = {
            pdf: self.compute_hash(pdf) for pdf in pdf_paths if self.compute_hash(pdf)
        }

        if not existing_hashes:  # If metadata file was missing or empty, rebuild the index
            return True

        return current_hashes != existing_hashes

    def split_into_chunks(self, text, chunk_size=250):
        """
        Splits text into smaller child chunks for sharp embedding retrieval.
        Respects whitespace where possible rather than chopping words.
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected a string, but received {type(text)}")

        words = text.split()
        chunks = []
        current_chunk = []
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
        self, pages_or_texts, pdf_paths, parent_window_size=1500, child_chunk_size=250
    ):
        """
        Builds a Two-Tier RAG index with page-level provenance:
        - Parent Store: Page / large contextual windows saved in a Python dictionary
          ({parent_id: {"text": str, "source": str, "page": int}}).
        - Child Store: Small focused snippets (~250 chars) embedded into FAISS.
        """
        self.child_chunks = []
        self.child_metadata = []
        self.parent_store = {}
        all_embeddings = []

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

            # Step 1: Create Parent Windows (~1500 characters per page window)
            raw_parents = [
                full_text[i : i + parent_window_size]
                for i in range(0, len(full_text), max(parent_window_size - 200, 1))
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

                # Step 2: Create Child Chunks (~250 characters) from this Parent
                children = self.split_into_chunks(clean_parent, chunk_size=child_chunk_size)

                for child in children:
                    clean_child = child.strip()
                    if len(clean_child) < 30:  # Skip trivial fragments
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
            # Use 16-bit float scalar quantization (QT_fp16) to cut index size by 50%
            # (or change QT_fp16 -> QT_8bit for a 75% reduction)
            self.index = faiss.IndexScalarQuantizer(
                dimension, faiss.ScalarQuantizer.QT_fp16
            )
            self.index.train(emb_array)
            self.index.add(emb_array)

            self.save_index()
            self.save_metadata(pdf_paths)
            print("✅ Two-Tier embeddings and index created successfully.")
        except Exception as e:
            print(f"⚠ Error creating FAISS index: {e}")

    def save_index(self):
        """
        Save FAISS index to disk if initialized.
        """
        try:
            if self.index:
                faiss.write_index(self.index, self.index_file)
            else:
                print("Warning: Attempted to save an uninitialized index.")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")

    def load_index(self):
        """
        Load FAISS index and two-tier dictionaries from disk.
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
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                self.index = None  # Ensure that we don't use a broken index
        else:
            print("FAISS index file missing. Rebuilding index is required.")

    def search_index(self, query, k=5):
        """
        Search for similar embeddings and return indices and distances.
        """
        if self.index is None:
            print(
                "Error: FAISS index not initialized. Try reloading or rebuilding the index."
            )
            return [], []

        try:
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(query_embedding, k)
            return distances, indices
        except Exception as e:
            print(f"Error searching index: {e}")
            return [], []

    def retrieve_parent_contexts(self, indices, max_parents=3):
        """
        Takes FAISS child chunk indices, uses metadata to look up parent IDs in
        the parent_store dictionary, and returns the full parent contexts (deduplicated).
        """
        if indices is None or len(indices) == 0 or len(indices[0]) == 0:
            return [], []

        matched_children = []
        retrieved_parents = []
        seen_parent_ids = set()

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

