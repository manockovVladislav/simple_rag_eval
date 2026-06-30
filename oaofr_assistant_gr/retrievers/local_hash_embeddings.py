from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np


class HashSentenceTransformer:
    """Small deterministic encoder for local smoke tests without model weights."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def to(self, device: str):
        return self

    def encode(
        self,
        texts: Iterable[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        vectors = [self._encode_one(text) for text in texts]
        matrix = np.asarray(vectors, dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
        return matrix

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype="float32")
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return vector


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9_]+|[=+\-*/^(){}\[\]<>≤≥]", text.lower())
