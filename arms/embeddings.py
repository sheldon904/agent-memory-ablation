"""Pluggable, deterministic text embedders.

Two backends, both L2-normalised so cosine similarity is a dot product:

* ``minilm``, sentence-transformers ``all-MiniLM-L6-v2`` (384-d), CPU, the same
  model the system under ablation uses. Deterministic given the pinned model
  version. This is the default and the one the paper's numbers are produced with.

* ``hash``, a dependency-free, download-free hashing embedder (character
  n-grams -> signed feature hashing -> L2 norm). It exists so the harness stays
  runnable in a fully hermetic/offline environment (CI, an air-gapped box). It
  is a *different* embedding space, so it yields different absolute numbers, but
  because all three arms share whichever embedder is selected, the architectural
  comparison stays valid. Select it with ``AMA_EMBEDDER=hash``.

Selection precedence: explicit argument > ``$AMA_EMBEDDER`` > ``minilm``.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HASH_DIM = 384


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised row vectors."""


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (m / norms).astype(np.float32)


class MiniLMEmbedder(Embedder):
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # lazy, heavy import

        self.name = "minilm"
        self._model_name = model_name
        # Force CPU + deterministic single-threaded inference so runs match
        # across machines up to float determinism.
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return v.astype(np.float32)


class HashEmbedder(Embedder):
    """Deterministic char-n-gram signed feature-hashing embedder."""

    def __init__(self, dim: int = HASH_DIM, ngram_min: int = 3, ngram_max: int = 5) -> None:
        self.name = "hash"
        self.dim = dim
        self._nmin = ngram_min
        self._nmax = ngram_max

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        s = f"^{text.lower()}$"
        for n in range(self._nmin, self._nmax + 1):
            for i in range(0, max(0, len(s) - n + 1)):
                gram = s[i : i + n]
                h = hashlib.md5(gram.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if (h[4] & 1) else -1.0
                v[idx] += sign
        return v

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        m = np.vstack([self._vec(t) for t in texts])
        return _l2_normalize(m)


def get_embedder(name: str | None = None) -> Embedder:
    choice = (name or os.environ.get("AMA_EMBEDDER") or "minilm").lower()
    if choice == "hash":
        return HashEmbedder()
    if choice == "minilm":
        return MiniLMEmbedder()
    raise ValueError(f"unknown embedder: {choice!r} (use 'minilm' or 'hash')")
