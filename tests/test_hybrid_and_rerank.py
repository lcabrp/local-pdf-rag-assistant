"""Unit tests for Okapi BM25 lexical search, Reciprocal Rank Fusion, and Cross-Encoder reranking."""

from __future__ import annotations

import unittest

from pdf_rag.hybrid_search import BM25Index, reciprocal_rank_fusion, tokenize_for_bm25


class TestHybridSearchAndReranking(unittest.TestCase):
    def test_tokenize_for_bm25_preserves_technical_identifiers(self) -> None:
        """Compound identifiers (error codes, clauses, quantizers) should emit both full and split tokens."""
        tokens = tokenize_for_bm25(
            "The error code ERR-409 occurs in section 14.3 with QT_fp16."
        )
        self.assertIn("err-409", tokens)
        self.assertIn("409", tokens)
        self.assertIn("14.3", tokens)
        self.assertIn("qt_fp16", tokens)
        self.assertIn("fp16", tokens)
        # Common stopwords like "the", "in", "with" must be filtered out
        self.assertNotIn("the", tokens)
        self.assertNotIn("in", tokens)
        self.assertNotIn("with", tokens)

    def test_bm25_finds_exact_identifier_among_similar_passages(self) -> None:
        """BM25 should rank the chunk containing an exact rare identifier (#1) above generic topical matches."""
        corpus = [
            "The hydraulic pump system may experience pressure fluctuations during cold startup.",
            "General maintenance of the hydraulic valve and pressure sensor requires calibration.",
            "Fault code E-409 indicates a critical relief valve blockage on manifold B.",
            "Troubleshooting system error codes and diagnostic warnings in the control panel.",
        ]
        bm25 = BM25Index()
        bm25.build(corpus)

        top_hits = bm25.search("What does fault code E-409 mean?", k=2)
        self.assertGreaterEqual(len(top_hits), 1)
        # Chunk index 2 contains the exact identifier "E-409"
        self.assertEqual(top_hits[0], 2)

    def test_bm25_matches_inflected_forms_via_light_stemming(self) -> None:
        """Querying 'remove duplicates' should match a passage phrased as 'duplicates are removed'."""
        corpus = [
            "Lists in Python are ordered, mutable sequences of items.",
            "When a sequence is made into a set, duplicate items are removed automatically.",
            "Dictionaries store key-value pairs for fast lookup.",
        ]
        bm25 = BM25Index()
        bm25.build(corpus)

        top_hits = bm25.search("how to remove duplicates from a list", k=2)
        self.assertGreaterEqual(len(top_hits), 1)
        self.assertEqual(top_hits[0], 1)

    def test_reciprocal_rank_fusion_promotes_consensus_hits(self) -> None:
        """A chunk ranked near the top of BOTH FAISS and BM25 should beat a single-retriever hit."""
        # Chunk 42 is #2 in both FAISS and BM25; Chunk 10 is #1 in FAISS only; Chunk 99 is #1 in BM25 only.
        faiss_ranked = [10, 42, 7, 8]
        bm25_ranked = [99, 42, 12, 15]

        fused = reciprocal_rank_fusion([faiss_ranked, bm25_ranked], rrf_k=60, top_n=5)
        # Score(42) = 1/62 + 1/62 = 2/62 = 0.03225
        # Score(10) = 1/61 = 0.01639
        # Score(99) = 1/61 = 0.01639
        self.assertEqual(fused[0], 42)
        self.assertEqual(
            len(fused), len(set(fused)), "Fused list must not contain duplicates"
        )


if __name__ == "__main__":
    unittest.main()
