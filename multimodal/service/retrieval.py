"""Lexical retrieval primitives adapted from the StructMem service."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", re.UNICODE)


CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


def lexical_tokens(text: str, *, cjk_ngrams: bool = False) -> list[str]:
    tokens = [
        token
        for token in (
            match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)
        )
        if token not in STOP_WORDS
    ]
    # Whitespace-free CJK text otherwise becomes one sentence-long token, which
    # prevents BM25 from matching a question to an overlapping caption phrase.
    if cjk_ngrams:
        for span in CJK_PATTERN.findall(text):
            tokens.extend(span[index:index + 2] for index in range(len(span) - 1))
    return tokens


def bm25_scores(
    documents: list[str], query: str, *, k1: float = 1.2, b: float = 0.75,
    cjk_ngrams: bool | None = None,
) -> np.ndarray:
    scores = np.zeros(len(documents), dtype=np.float32)
    use_cjk_ngrams = bool(CJK_PATTERN.search(query)) if cjk_ngrams is None else cjk_ngrams
    query_terms = set(lexical_tokens(query, cjk_ngrams=use_cjk_ngrams))
    if not documents or not query_terms:
        return scores
    tokenized = [
        lexical_tokens(document, cjk_ngrams=use_cjk_ngrams) for document in documents
    ]
    average_length = sum(map(len, tokenized)) / len(tokenized)
    if average_length == 0:
        return scores
    frequencies = [Counter(tokens) for tokens in tokenized]
    document_frequency = Counter(
        term for counts in frequencies for term in query_terms.intersection(counts)
    )
    count = len(documents)
    for index, counts in enumerate(frequencies):
        length = len(tokenized[index])
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            inverse = math.log(
                (count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
                + 1.0
            )
            scores[index] += (
                inverse
                * frequency
                * (k1 + 1.0)
                / (frequency + k1 * (1.0 - b + b * length / average_length))
            )
    return scores
