# document_handler.py

import os
from typing import List

import pymupdf  # PyMuPDF for PDF text extraction


class DocumentHandler:
    def __init__(self, base_dir: str):
        """Initialize with the base directory path where files are stored."""
        self.base_dir = base_dir

    def get_all_pdf_paths(self) -> List[str]:
        """Recursively finds all PDF files in the base directory."""
        pdf_paths = []
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_paths.append(os.path.join(root, file))
        return pdf_paths

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text from a single PDF file.
        Handles file and parsing errors gracefully.
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"File not found: {pdf_path}")
            if not pdf_path.lower().endswith(".pdf"):
                raise ValueError(f"Invalid file type: {pdf_path} is not a PDF")
            with pymupdf.open(pdf_path) as doc:
                return " ".join(page.get_text() for page in doc)
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""

    def extract_pages_from_pdf(self, pdf_path: str) -> List[dict]:
        """Extracts text per page with 1-based page numbers and source filename
        so retrieved parent contexts can cite the exact file and page.
        """
        pages: List[dict] = []
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"File not found: {pdf_path}")
            if not pdf_path.lower().endswith(".pdf"):
                raise ValueError(f"Invalid file type: {pdf_path} is not a PDF")
            source_name = os.path.basename(pdf_path)
            with pymupdf.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = (page.get_text() or "").strip()
                    if text:
                        pages.append(
                            {
                                "source": source_name,
                                "page": page_num,
                                "text": text,
                            }
                        )
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
            print(f"Error reading {pdf_path}: {e}")
        return pages


if __name__ == "__main__":
    import sys

    # Prevent charmap / cp1252 encoding crashes when printing exotic Unicode from PDFs on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    base_dir = r".\docs\test"
    doc_handler = DocumentHandler(base_dir)
    pdf_paths = doc_handler.get_all_pdf_paths()

    print(f"Found {len(pdf_paths)} total PDFs. Previewing first 10:\n")
    for pdf in pdf_paths[:10]:
        text = doc_handler.extract_text_from_pdf(pdf)
        preview = text[:400].strip()
        print(f"-> {pdf}\n   Text: {preview}...\n")

