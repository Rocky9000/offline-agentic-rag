"""Embedding backends. All return L2-normalised float32 matrices."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    """Dependency-free, fully offline embedding via signed feature hashing.

    Features are stemmed words plus character trigrams of each word, so it
    tolerates spelling variants and morphology that pure BM25 misses. It has
    no notion of synonyms; use a neural embedder for that. It exists so the
    whole system runs and is testable on an air-gapped machine with nothing
    but numpy.
    """

    name = "hashing"

    def __init__(self, dim: int = 1024, trigram_weight: float = 0.5):
        self.dim = dim
        self.trigram_weight = trigram_weight

    def _features(self, text: str):
        for word in tokenize(text):
            yield word, 1.0
            padded = f"#{word}#"
            for i in range(len(padded) - 2):
                yield padded[i : i + 3], self.trigram_weight

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, h % self.dim] += sign * weight
        return _normalise(out)


class SentenceTransformerEmbedder:
    """Local neural embeddings (e.g. ``BAAI/bge-small-en-v1.5``).

    Download the model once while online; afterwards it loads from the
    Hugging Face cache with no network access.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("pip install 'offline-agentic-rag[neural]'") from e
        self._model = SentenceTransformer(model)
        self.name = f"st:{model}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs model weights
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class OllamaEmbedder:
    """Embeddings from a local Ollama server (e.g. ``nomic-embed-text``)."""

    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self.name = f"ollama:{model}"
        self.dim = len(self.embed(["dimension probe"])[0])

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs server
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/embed", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        return _normalise(np.asarray(data["embeddings"], dtype=np.float32))


def make_embedder(spec: str) -> Embedder:
    """Build an embedder from a short spec: ``hashing``, ``st:<model>``, ``ollama:<model>``."""
    if spec == "hashing":
        return HashingEmbedder()
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    if spec.startswith("ollama:"):
        return OllamaEmbedder(spec[7:])
    raise ValueError(f"unknown embedder spec: {spec!r}")
