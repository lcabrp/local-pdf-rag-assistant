"""Benchmark retrieval accuracy (Dense-only FAISS vs. Hybrid BM25+FAISS+Rerank).

Why evaluate retrieval separately from generation:
-------------------------------------------------
When a RAG answer is wrong, you need to know whether the *retriever* failed to
find the right page or the *LLM* misread the retrieved text. This script runs
your evaluation questions locally against `index/index.faiss` and
`index/metadata.json` with zero OpenAI API calls, comparing:

  1. Baseline: Dense-only FAISS (`k=5` -> `max_parents=3`)
  2. Upgraded: Hybrid FAISS + BM25 + RRF (`candidate_k=25`) -> Cross-Encoder Rerank (`top 3`)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pdf_rag.database_builder import DatabaseBuilder


def _is_hit(parents: list[dict], item: dict) -> bool:
    """Return True if any retrieved parent matches the expected source/page or keywords."""
    expected_source = item.get("expected_source")
    expected_page = item.get("expected_page")
    expected_keywords = [kw.lower() for kw in item.get("expected_keywords", [])]

    for parent in parents:
        source_ok = (
            not expected_source
            or parent.get("source", "").lower() == expected_source.lower()
        )
        page_ok = expected_page is None or parent.get("page") == expected_page
        text_lower = parent.get("text", "").lower()
        keywords_ok = not expected_keywords or all(
            kw in text_lower for kw in expected_keywords
        )
        if source_ok and page_ok and keywords_ok:
            return True
    return False


def evaluate(questions_path: Path, index_dir: str = "index") -> int:
    if not questions_path.exists():
        print(f"Eval file not found: {questions_path}")
        print(
            "Copy tests/eval_questions.example.json to data/eval_questions.json and fill in your PDF questions."
        )
        return 1

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not questions:
        print("Eval file is empty.")
        return 1

    db = DatabaseBuilder(index_dir=index_dir)
    db.load_index()
    if db.index is None or not db.parent_store:
        print(f"No valid index found in '{index_dir}'. Load PDFs in the app first.")
        return 1

    dense_hits = 0
    hybrid_hits = 0
    dense_ms_total = 0.0
    hybrid_ms_total = 0.0

    print(f"\nEvaluating {len(questions)} question(s) against '{index_dir}'...\n")
    for idx, item in enumerate(questions, start=1):
        query = item["question"]

        # 1. Baseline: Dense-only FAISS (k=5 -> top 3 parents)
        t0 = time.perf_counter()
        _dist, indices = db.search_index(query, k=5)
        _, dense_parents = db.retrieve_parent_contexts(indices, max_parents=3)
        dense_ms = (time.perf_counter() - t0) * 1000.0
        dense_ms_total += dense_ms
        dense_ok = _is_hit(dense_parents, item)
        dense_hits += int(dense_ok)

        # 2. Upgraded: Hybrid (FAISS + BM25 + RRF) -> Cross-Encoder Rerank (top 3 parents)
        t1 = time.perf_counter()
        _, hybrid_parents = db.retrieve_for_query(
            query, candidate_k=25, rerank_candidates=10, max_parents=3
        )
        hybrid_ms = (time.perf_counter() - t1) * 1000.0
        hybrid_ms_total += hybrid_ms
        hybrid_ok = _is_hit(hybrid_parents, item)
        hybrid_hits += int(hybrid_ok)

        print(
            f"[{idx:02d}] Dense: {'HIT ' if dense_ok else 'MISS'} ({dense_ms:5.1f} ms) | "
            f"Hybrid+Rerank: {'HIT ' if hybrid_ok else 'MISS'} ({hybrid_ms:5.1f} ms) | {query[:65]}"
        )

    total = len(questions)
    print("\n--- Summary ---")
    print(
        f"Dense-only Hit@3:    {dense_hits}/{total} ({dense_hits / total:.0%}) | Avg latency: {dense_ms_total / total:.1f} ms"
    )
    print(
        f"Hybrid+Rerank Hit@3: {hybrid_hits}/{total} ({hybrid_hits / total:.0%}) | Avg latency: {hybrid_ms_total / total:.1f} ms"
    )
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Evaluate Dense-only vs. Hybrid+Rerank retrieval accuracy (Hit@3)."
    )
    parser.add_argument(
        "questions_file",
        nargs="?",
        default="data/eval_questions.json",
        help="Path to evaluation JSON file (default: data/eval_questions.json)",
    )
    parser.add_argument(
        "--index-dir",
        default="index",
        help="Path to FAISS + metadata directory (default: index)",
    )
    args = parser.parse_args()
    raise SystemExit(evaluate(Path(args.questions_file), index_dir=args.index_dir))


if __name__ == "__main__":
    main()
