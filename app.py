# app.py

import json
import os
import re
import tkinter as tk
from tkinter import filedialog, ttk

from database_builder import DatabaseBuilder
from document_handler import DocumentHandler
from dotenv import load_dotenv
from openai import OpenAI

_LABEL_NUMBER = re.compile(r"\d+")

# Load environment variables from .env file
load_dotenv()

# Set up the OpenAI client
client = OpenAI()


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

    def setup_gui(self):
        """
        Sets up the GUI components.
        """
        tk.Label(self.root, text="Select Document Directory:").pack(pady=10)
        self.dir_button = tk.Button(self.root, text="Browse", command=self.load_directory)
        self.dir_button.pack(pady=5)

        self.query_label = tk.Label(self.root, text="Ask your question:")
        self.query_label.pack(pady=10)
        self.query_entry = tk.Entry(self.root, width=50)
        self.query_entry.pack(pady=5)

        self.answer_button = tk.Button(self.root, text="Get Answer", command=self.ask_question)
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
        self.answer_text.insert(tk.END, f"✅ Found {len(pdf_paths)} PDFs in the directory.\n")

        # Step 2: Extracting text per page (with source & page numbers)
        self.answer_text.insert(tk.END, "🔍 Extracting pages from PDFs...\n")
        progress_bar["value"] = 2
        self.root.update_idletasks()
        all_pages = []
        for pdf_path in pdf_paths:
            pages = self.doc_handler.extract_pages_from_pdf(pdf_path)
            all_pages.extend(pages)

        # Step 3: Checking for changes & managing the index
        self.answer_text.insert(tk.END, "📊 Checking for changes & managing index...\n")
        progress_bar["value"] = 3
        self.root.update_idletasks()

        missing_faiss = not os.path.exists(self.db_builder.index_file)
        missing_metadata = not os.path.exists(self.db_builder.metadata_file)

        if missing_faiss or missing_metadata or self.db_builder.has_changes(pdf_paths):
            self.answer_text.insert(tk.END, "🔄 Changes detected. Rebuilding index...\n")
            self.db_builder.build_embeddings(all_pages, pdf_paths)
            self.answer_text.insert(tk.END, "✅ Index rebuilt successfully.\n")
        else:
            self.db_builder.load_index()
            self.answer_text.insert(tk.END, "✅ No changes detected. Loaded existing index.\n")

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
            self.answer_text.insert(tk.END, "❗ Please load documents before asking questions.\n")
            return

        # Step 1: Search for relevant content in the indexed PDFs (Two-Tier Retrieval)
        try:
            # Retrieve top 5 child matches from FAISS
            distances, indices = self.db_builder.search_index(query, k=5)

            # Defensive handling of empty or malformed indices
            if indices is None or len(indices) == 0 or len(indices[0]) == 0:
                self.answer_text.insert(
                    tk.END, "❗ No relevant document found. Fetching answer from OpenAI...\n"
                )
                self.fetch_openai_answer(query)
                return

            # Step 2: Use the dictionary to look up the large PARENT contexts
            _, relevant_parents = self.db_builder.retrieve_parent_contexts(
                indices, max_parents=3
            )

            if relevant_parents:
                # Refine response using OpenAI with the full parent contexts
                self.refine_with_openai(query, relevant_parents)
            else:
                self.answer_text.insert(
                    tk.END,
                    "❗ No relevant parent contexts found. Fetching answer from OpenAI...\n",
                )
                self.fetch_openai_answer(query)

        except Exception as e:
            self.answer_text.insert(tk.END, f"⚠ Error searching index: {e}\n")
            self.fetch_openai_answer(query)  # Call OpenAI as a fallback in case of errors

    def fetch_openai_answer(self, query):
        """
        Fetches an answer directly from OpenAI if no relevant documents are found.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-6-luna",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions.",
                    },
                    {"role": "user", "content": query},
                ],
            )
            openai_answer = response.choices[0].message.content
            self.answer_text.insert(tk.END, f"\n✨ Answer from OpenAI:\n{openai_answer}\n")
        except Exception as e:
            self.answer_text.insert(tk.END, f"⚠ Error fetching response from OpenAI: {e}\n")

    def _resolve_sources(self, labels, parent_records):
        """
        Map the model's "[1]" labels back to the real file and page of each parent context.
        Drops any label that does not point at a retrieved passage so citations cannot be invented.
        """
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
                citation = f"{item.get('source', 'document')}, page {item.get('page', 1)}"
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

            response = client.chat.completions.create(
                model="gpt-6-luna",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)

            answer = parsed.get("answer", raw)
            raw_sources = parsed.get("sources", [])
            confidence = float(parsed.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            sources = self._resolve_sources(raw_sources, relevant_parents)

            self.answer_text.insert(tk.END, f"\n✨ Answer:\n{answer}\n")
            self.answer_text.insert(tk.END, f"📊 Confidence: {confidence:.0%}\n")
            if sources:
                self.answer_text.insert(tk.END, f"📚 Sources: {'; '.join(sources)}\n")
            else:
                self.answer_text.insert(tk.END, "📚 Sources: None cited\n")
        except Exception as e:
            self.answer_text.insert(tk.END, f"⚠ Error refining answer with OpenAI: {e}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = AIHelperApp(root)
    root.mainloop()

