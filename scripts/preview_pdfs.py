"""Preview extracted text from PDFs in a directory (defaults to data/pdfs)."""

import argparse
import sys

from pdf_rag.document_handler import DocumentHandler


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Preview extracted text from PDFs.")
    parser.add_argument(
        "directory",
        nargs="?",
        default="data/pdfs",
        help="Directory to scan for PDFs (default: data/pdfs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of PDFs to preview (default: 10)",
    )
    args = parser.parse_args()

    doc_handler = DocumentHandler(args.directory)
    pdf_paths = doc_handler.get_all_pdf_paths()

    print(
        f"Found {len(pdf_paths)} total PDFs in '{args.directory}'. Previewing first {args.limit}:\n"
    )
    for pdf in pdf_paths[: args.limit]:
        text = doc_handler.extract_text_from_pdf(pdf)
        preview = text[:400].strip()
        print(f"-> {pdf}\n   Text: {preview}...\n")


if __name__ == "__main__":
    main()
