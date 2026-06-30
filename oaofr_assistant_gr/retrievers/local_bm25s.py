from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np


def tokenize(value):
    if isinstance(value, str):
        return _tokens(value)
    return [_tokens(item) for item in value]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._avg_doc_length = 0.0

    def index(self, tokenized_corpus: Iterable[list[str]]) -> None:
        self._term_freqs = [Counter(tokens) for tokens in tokenized_corpus]
        self._doc_lengths = [sum(freqs.values()) for freqs in self._term_freqs]
        self._avg_doc_length = sum(self._doc_lengths) / max(len(self._doc_lengths), 1)

        doc_freqs: Counter[str] = Counter()
        for freqs in self._term_freqs:
            doc_freqs.update(freqs.keys())

        doc_count = len(self._term_freqs)
        self._idf = {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freqs.items()
        }

    def retrieve(self, query_tokens, k: int = 10):
        if query_tokens and isinstance(query_tokens[0], list):
            query_tokens = query_tokens[0]

        scores = np.asarray([self._score(query_tokens, index) for index in range(len(self._term_freqs))], dtype="float32")
        if len(scores) == 0:
            return np.asarray([[]], dtype="int64"), np.asarray([[]], dtype="float32")

        k = min(k, len(scores))
        order = np.argsort(-scores)[:k].astype("int64")
        return order.reshape(1, -1), scores[order].reshape(1, -1)

    def _score(self, query_tokens: list[str], doc_index: int) -> float:
        freqs = self._term_freqs[doc_index]
        doc_length = self._doc_lengths[doc_index] or 1
        score = 0.0
        for term in query_tokens:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_length / max(self._avg_doc_length, 1e-12))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9_]+|[=+\-*/^(){}\[\]<>≤≥]", str(text).lower())
