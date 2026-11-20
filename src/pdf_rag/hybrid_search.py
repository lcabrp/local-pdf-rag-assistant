"""Hybrid lexical search (Okapi BM25) and Reciprocal Rank Fusion (RRF).

Why this module exists:
-----------------------
Dense vector embeddings (like `all-mpnet-base-v2` in FAISS) excel at matching
paraphrased concepts ("renal failure" <-> "kidney disease"), but compress every
250-character chunk into a single 768-float vector. That compression loses exact
character-level tokens such as part numbers, error codes (`0x80070005`), legal
clauses (`Section 14.3(b)`), acronyms, and rare proper nouns.

By pairing FAISS (dense semantic search) with Okapi BM25 (sparse lexical search)
and merging their ranked lists via Reciprocal Rank Fusion (RRF), the retrieval
pipeline captures both semantic meaning and exact keyword matches.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Match alphanumeric words and preserve hyphenated/dotted/underscored technical
# identifiers (e.g., "e-409", "14.3", "qt_fp16", "0x80070005").
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

# Lightweight English stopword set. Removing ultra-frequent function words
# keeps the inverted index small and prevents query words like "what", "is",
# or "the" from adding noise to BM25 scores.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


def _light_stem(word: str) -> str | None:
    """Return a lightweight morphological stem for plain alphabetic English words.

    Why we emit a stem *alongside* the exact word (instead of replacing it):
    ------------------------------------------------------------------------
    1. Bridges verb/noun inflections: A user asking "how to remove duplicates"
       now matches a book phrasing "duplicates are removed" (`remove` and
       `removed` both stem to `remov`; `duplicates` and `duplicate` both stem
       to `duplicat`).
    2. Preserves exact-match priority: Because the original token is still
       indexed, an exact word match scores higher than a stemmed-only match.
    3. Protects technical identifiers: Tokens containing digits or separators
       (`e-409`, `qt_fp16`, `14.3`) are never stemmed.
    """
    if len(word) < 5 or not word.isalpha():
        return None
    for suffix in ("ing", "ies", "es", "ed", "s", "e"):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if len(stem) >= 4:
                return stem
    return None


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text into normalized lexical terms for BM25 indexing and search.

    Why compound tokens and stems are both emitted:
    - If a document contains `err-409` and a user searches either `err-409` or `409`,
      emitting both the full compound token and its sub-parts ensures exact matches
      get a double IDF boost while partial queries still match.
    - Emitting a lightweight stem for plain words (`remove` / `removed` -> `remov`)
      bridges tense and plural differences without requiring an external stemmer.
    """
    if not text:
        return []

    tokens: list[str] = []
    for raw_match in _TOKEN_PATTERN.findall(text.lower()):
        if raw_match not in _STOPWORDS:
            tokens.append(raw_match)
            stem = _light_stem(raw_match)
            if stem and stem not in _STOPWORDS:
                tokens.append(stem)
        # If the token is a compound identifier (contains -, _, or .), also emit
        # its sub-parts so partial identifier searches still hit the posting list.
        if any(sep in raw_match for sep in ("-", "_", ".")):
            for part in re.split(r"[-_.]", raw_match):
                if part and len(part) > 1 and part not in _STOPWORDS:
                    tokens.append(part)
    return tokens


class BM25Index:
    """In-memory Okapi BM25 inverted index over child text chunks.

    Why an inverted index instead of a matrix scan:
    Most queries contain only 3-8 keywords. An inverted index (`term -> [(doc_id, tf)]`)
    only visits chunks that contain at least one query term, making search O(matches)
    rather than O(total_chunks). Across 26,500 child chunks, queries complete in <2 ms
    in pure Python with zero external C dependencies.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 hyperparameters.

        Args:
            k1: Term-frequency saturation parameter (typical range: 1.2 - 2.0).
                Controls how quickly repeated occurrences of a word hit diminishing
                returns. At k1=1.5, a word appearing 5 times scores much higher than
                once, but 50 times is barely higher than 5.
            b: Document-length normalization (0.0 = no normalization, 1.0 = full).
                Prevents longer chunks from scoring higher purely because they contain
                more total words.
        """
        self.k1 = k1
        self.b = b
        self.num_docs: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: list[int] = []
        # Inverted index mapping: token -> list of (chunk_index, term_frequency)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        # Precomputed Robertson-Walker IDF per token
        self.idf: dict[str, float] = {}

    def build(self, documents: list[str]) -> None:
        """Build the inverted index and precompute IDF weights from a list of chunks."""
        self.num_docs = len(documents)
        self.doc_lengths = [0] * self.num_docs
        self.postings = {}
        self.idf = {}

        if self.num_docs == 0:
            self.avg_doc_len = 0.0
            return

        total_tokens = 0
        for doc_id, text in enumerate(documents):
            tokens = tokenize_for_bm25(text)
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            total_tokens += doc_len

            term_counts = Counter(tokens)
            for term, freq in term_counts.items():
                self.postings.setdefault(term, []).append((doc_id, freq))

        self.avg_doc_len = total_tokens / self.num_docs if self.num_docs else 1.0
        if self.avg_doc_len == 0.0:
            self.avg_doc_len = 1.0

        # Precompute Robertson-Walker IDF:
        #   IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
        # Why the +1 inside ln(...):
        # Standard BM25 IDF can become negative if a word appears in >50% of chunks.
        # Adding 1 inside the logarithm guarantees non-negative weights for all terms.
        for term, posting_list in self.postings.items():
            df = len(posting_list)
            self.idf[term] = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 25) -> list[int]:
        """Return the top-k document indices ordered by descending BM25 score."""
        if self.num_docs == 0 or k <= 0:
            return []

        query_terms = tokenize_for_bm25(query)
        if not query_terms:
            return []

        scores: dict[int, float] = {}
        k1 = self.k1
        b = self.b
        avg_dl = self.avg_doc_len
        doc_lengths = self.doc_lengths

        # Accumulate BM25 scores only across documents present in the query terms' posting lists
        for term in set(query_terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings[term]:
                dl = doc_lengths[doc_id]
                # Okapi BM25 term score formula
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (dl / avg_dl))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (
                    numerator / denominator
                )

        if not scores:
            return []

        # Sort candidate doc_ids by BM25 score descending
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _score in ranked[:k]]


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    rrf_k: int = 60,
    top_n: int = 25,
) -> list[int]:
    """Merge multiple ranked index lists into a single unified ranking using RRF.

    Why Reciprocal Rank Fusion (RRF) instead of weighted score averaging:
    ---------------------------------------------------------------------
    1. Incompatible scales: FAISS L2 distances are unbounded [0, inf) where *lower*
       is better, whereas BM25 scores are unbounded [0, inf) where *higher* is better
       and scale linearly with the number of words in the user's question.
    2. Calibration-free: RRF ignores raw score magnitudes and scores each item by
       its ordinal rank position in each retriever:
           RRF_score(doc) = sum( 1 / (rrf_k + rank) )
    3. Consensus boost: With `rrf_k = 60` (the standard constant from Cormack et al.),
       a chunk that ranks near the top of *both* FAISS (semantic) and BM25 (exact
       keyword) outranks a chunk that only scored well in one retriever.

    Args:
        ranked_lists: List of ordered document ID lists (e.g., [faiss_ids, bm25_ids]).
        rrf_k: Smoothing constant (default 60).
        top_n: Maximum number of fused document IDs to return.

    Returns:
        Deduplicated list of document IDs ordered by descending RRF score.
    """
    fused_scores: dict[int, float] = {}

    for ranked_list in ranked_lists:
        # Deduplicate within a single retriever list while preserving first-seen rank
        seen_in_list: set[int] = set()
        for rank_idx, doc_id in enumerate(ranked_list, start=1):
            if doc_id < 0 or doc_id in seen_in_list:
                continue
            seen_in_list.add(doc_id)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (
                1.0 / (rrf_k + rank_idx)
            )

    ranked_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _score in ranked_docs[:top_n]]
