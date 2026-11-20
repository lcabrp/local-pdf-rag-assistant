import json
import os
import re
import tkinter as tk
from tkinter import filedialog, ttk

from dotenv import load_dotenv

from pdf_rag.database_builder import DatabaseBuilder
from pdf_rag.document_handler import DocumentHandler
from pdf_rag.llm_client import LocalFirstLLMClient

_LABEL_NUMBER = re.compile(r"\d+")

# Load environment variables from .env file (used for OpenAI cloud fallback and optional Ollama overrides)
load_dotenv()


class AIHelperApp:
    def __init__(self, root):
        """
        Initialize the GUI application.
        """
        self.root = root
        self.root.title("Local PDF RAG Assistant")
        self.setup_gui()

        # Initialize modules
        self.doc_handler = None
        self.db_builder = DatabaseBuilder(index_dir="index")
        self.llm_client = LocalFirstLLMClient()

    def setup_gui(self):
        """
        Sets up the GUI components.
        """
        tk.Label(self.root, text="Select Document Directory:").pack(pady=10)
        self.dir_button = tk.Button(
            self.root, text="Browse", command=self.load_directory
        )
        self.dir_button.pack(pady=5)

        self.query_label = tk.Label(self.root, text="Ask your question:")
        self.query_label.pack(pady=10)
        self.query_entry = tk.Entry(self.root, width=50)
        self.query_entry.pack(pady=5)

        # Grounding & retrieval-inspection controls:
        # - strict_docs_var: refuses to answer from general LLM weights when no PDF chunk matches
        # - show_chunks_var: prints the top retrieved parents so retrieval vs. generation errors are visible
        # - expand_query_var: uses the LLM to generate 2 synonym/concept queries before FAISS + BM25 search
        options_frame = tk.Frame(self.root)
        options_frame.pack(pady=2)
        self.strict_docs_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            options_frame,
            text="Documents only (refuse if not found)",
            variable=self.strict_docs_var,
        ).pack(side=tk.LEFT, padx=6)
        self.show_chunks_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            options_frame,
            text="Show retrieved passages",
            variable=self.show_chunks_var,
        ).pack(side=tk.LEFT, padx=6)
        self.expand_query_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            options_frame,
            text="Expand query (Multi-Query)",
            variable=self.expand_query_var,
        ).pack(side=tk.LEFT, padx=6)

        self.answer_button = tk.Button(
            self.root, text="Get Answer", command=self.ask_question
        )
        self.answer_button.pack(pady=10)

        self.answer_text = tk.Text(self.root, height=12, width=70)
        self.answer_text.pack(pady=10)

    def load_directory(self):
        """
        Loads and processes the selected directory with a progress bar.
        """
        dir_path = filedialog.askdirectory()
        if not dir_path:
            return

        self.answer_text.insert(tk.END, "📂 Browsing and loading directory...\n")
        self.doc_handler = DocumentHandler(base_dir=dir_path)

        # Set up a progress bar
        progress_bar = ttk.Progressbar(
            self.root, orient="horizontal", length=300, mode="determinate"
        )
        progress_bar.pack(pady=10)
        progress_bar["maximum"] = 3  # Three major steps

        # Step 1: Scanning for PDFs
        self.answer_text.insert(tk.END, "📄 Scanning for PDF files...\n")
        progress_bar["value"] = 1
        self.root.update_idletasks()  # Refresh GUI
        pdf_paths = self.doc_handler.get_all_pdf_paths()
        self.answer_text.insert(
            tk.END, f"✅ Found {len(pdf_paths)} PDFs in the directory.\n"
        )

        # Step 2: Check MD5 file hashes *before* parsing PDFs so unchanged folders
        # skip expensive PyMuPDF page extraction completely.
        self.answer_text.insert(tk.END, "📊 Checking for changes & managing index...\n")
        progress_bar["value"] = 2
        self.root.update_idletasks()

        missing_faiss = not os.path.exists(self.db_builder.index_file)
        missing_metadata = not os.path.exists(self.db_builder.metadata_file)

        if missing_faiss or missing_metadata or self.db_builder.has_changes(pdf_paths):
            self.answer_text.insert(
                tk.END, "🔄 Changes detected. Extracting pages & rebuilding index...\n"
            )
            progress_bar["value"] = 3
            self.root.update_idletasks()

            all_pages = []
            for pdf_path in pdf_paths:
                pages = self.doc_handler.extract_pages_from_pdf(pdf_path)
                all_pages.extend(pages)

            self.db_builder.build_embeddings(all_pages, pdf_paths)
            self.answer_text.insert(tk.END, "✅ Index rebuilt successfully.\n")
        else:
            progress_bar["value"] = 3
            self.root.update_idletasks()
            self.db_builder.load_index()
            self.answer_text.insert(
                tk.END, "✅ No changes detected. Loaded existing index.\n"
            )

        # Remove progress bar after completion
        progress_bar.destroy()

    def ask_question(self):
        """
        Processes the user's query, retrieves semantically relevant information from indexed documents,
        and fetches additional details from OpenAI if needed.
        """
        query = self.query_entry.get().strip()

        # Validate input
        if not query:
            self.answer_text.insert(tk.END, "❗ Please enter a valid question.\n")
            return

        if not self.doc_handler:
            self.answer_text.insert(
                tk.END, "❗ Please load documents before asking questions.\n"
            )
            return

        # Multi-Stage Retrieval Pipeline:
        #   Stage 1 & 2: Hybrid Search (FAISS + Okapi BM25 merged via RRF, top 50 children)
        #   Stage 3:     Two-Tier Parent Expansion (maps child hits -> up to 15 unique 1,500-char parents)
        #   Stage 4:     Cross-Encoder Reranking (scores (query, parent) pairs on GPU -> top 4 parents)
        # Why 50 -> 15 -> 4:
        #   Across multi-book corpora (10+ PDFs / 25k+ chunks), widening candidate recall
        #   to 50/15 gives indirect passages a chance to reach the Cross-Encoder, while
        #   capping the final prompt at 4 parents (~1,500 tokens) keeps small local
        #   Ollama models reliable inside a 4,096-token context window.
        try:
            extra_queries: list[str] = []
            if self.expand_query_var.get():
                extra_queries = self.llm_client.expand_query(query, max_variants=2)
                if extra_queries:
                    self.answer_text.insert(
                        tk.END,
                        f"\n🔀 Expanded queries: {' | '.join(extra_queries)}\n",
                    )

            matched_children, relevant_parents = self.db_builder.retrieve_for_query(
                query,
                candidate_k=50,
                rerank_candidates=15,
                max_parents=4,
                extra_queries=extra_queries or None,
            )

            if not matched_children or not relevant_parents:
                if self.strict_docs_var.get():
                    self.answer_text.insert(
                        tk.END,
                        "\n🚫 No relevant passages found in the loaded PDFs. "
                        "Refusing to guess outside your documents.\n",
                    )
                else:
                    self.answer_text.insert(
                        tk.END,
                        "❗ No relevant document found. Fetching answer from OpenAI...\n",
                    )
                    self.fetch_openai_answer(query)
                return

            if self.show_chunks_var.get():
                self.answer_text.insert(tk.END, "\n🔎 Retrieved Parent Passages:\n")
                for pos, item in enumerate(relevant_parents, start=1):
                    source = item.get("source", "document")
                    page = item.get("page", 1)
                    snippet = item.get("text", "")[:220].replace("\n", " ")
                    self.answer_text.insert(
                        tk.END, f"  [{pos}] {source}, p.{page}: {snippet}...\n"
                    )

            # Stage 5: Synthesize grounded JSON answer + verified page citations from top-3 parents
            self.refine_with_openai(query, relevant_parents)

        except Exception as e:  # noqa: BLE001
            self.answer_text.insert(tk.END, f"⚠ Error searching index: {e}\n")
            if not self.strict_docs_var.get():
                self.fetch_openai_answer(query)

    def fetch_openai_answer(self, query):
        """
        Fetches an answer using local Ollama first (with OpenAI cloud fallback)
        when 'Documents only' is unchecked and no relevant documents are found.
        """
        try:
            result = self.llm_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions.",
                    },
                    {"role": "user", "content": query},
                ],
                json_mode=False,
            )
            self.answer_text.insert(
                tk.END,
                f"\n✨ Answer ({result.provider}/{result.model}):\n{result.content}\n",
            )
        except Exception as e:  # noqa: BLE001
            self.answer_text.insert(
                tk.END, f"⚠ Error fetching response from LLM: {e}\n"
            )

    @staticmethod
    def _parse_confidence(value) -> float:
        """Parse confidence into [0.0, 1.0], handling small models that emit strings like 'High'."""
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        if isinstance(value, str):
            text = value.strip().lower().rstrip("%")
            named = {"high": 0.85, "medium": 0.55, "moderate": 0.55, "low": 0.25}
            if text in named:
                return named[text]
            try:
                num = float(text)
                return max(0.0, min(1.0, num / 100.0 if num > 1.0 else num))
            except ValueError:
                return 0.0
        return 0.0

    def _resolve_sources(self, labels, parent_records):
        """
        Map the model's "[1]" labels back to the real file and page of each parent context.
        Drops any label that does not point at a retrieved passage so citations cannot be invented.
        """
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, list):
            return []

        resolved = []
        for label in labels:
            match = _LABEL_NUMBER.search(str(label))
            if match is None:
                continue
            position = int(match.group())
            if not 1 <= position <= len(parent_records):
                continue
            item = parent_records[position - 1]
            if isinstance(item, dict):
                citation = (
                    f"{item.get('source', 'document')}, page {item.get('page', 1)}"
                )
            else:
                citation = f"Passage [{position}]"
            if citation not in resolved:
                resolved.append(citation)
        return resolved

    def refine_with_openai(self, query, relevant_parents):
        """
        Synthesizes an answer from the retrieved parent contexts using OpenAI JSON mode,
        returning the answer, a confidence score, and verified page-level citations.
        """
        try:
            passages = []
            for pos, item in enumerate(relevant_parents, start=1):
                if isinstance(item, dict):
                    text = item.get("text", "")
                    passages.append(f"[{pos}]\n{text}")
                else:
                    passages.append(f"[{pos}]\n{item}")
            context = "\n\n".join(passages)

            prompt = f"""Answer the question using only the numbered passages below.

Passages:
{context}

Question: {query}

Rules:
- Use only the passages. If they do not contain the answer, say so plainly and return an empty sources list.
- "sources" must repeat the bracket labels you used, exactly as written, for example ["[1]", "[2]"]. Never invent a label.
- "confidence" is a number between 0 and 1 saying how well the passages support the answer.
- Reply with a single JSON object and nothing else.

Response format:
{{"answer": "text", "sources": ["[1]"], "confidence": 0.0}}"""

            llm_result = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
            )
            raw = llm_result.content or "{}"
            parsed = json.loads(raw)

            answer = parsed.get("answer", raw)
            raw_sources = parsed.get("sources", [])
            confidence = self._parse_confidence(parsed.get("confidence", 0.0))
            sources = self._resolve_sources(raw_sources, relevant_parents)

            model_badge = f"{llm_result.provider}/{llm_result.model}"
            if llm_result.fallback_reason:
                model_badge += f" (fallback: {llm_result.fallback_reason})"

            self.answer_text.insert(tk.END, f"\n✨ Answer:\n{answer}\n")
            self.answer_text.insert(tk.END, f"🤖 Model: {model_badge}\n")
            self.answer_text.insert(tk.END, f"📊 Confidence: {confidence:.0%}\n")
            if sources:
                self.answer_text.insert(tk.END, f"📚 Sources: {'; '.join(sources)}\n")
            else:
                self.answer_text.insert(tk.END, "📚 Sources: None cited\n")
        except Exception as e:  # noqa: BLE001
            self.answer_text.insert(tk.END, f"⚠ Error refining answer with LLM: {e}\n")


def main():
    root = tk.Tk()
    AIHelperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
