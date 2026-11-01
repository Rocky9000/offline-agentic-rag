"""Lexical (BM25), dense, and hybrid (reciprocal-rank-fusion) retrieval."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder, make_embedder
from .text import tokenize


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class BM25Index:
    """Okapi BM25 over stemmed tokens, implemented from scratch."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self._tfs: list[Counter] = []
        self._lens: list[int] = []
        self._df: Counter = Counter()

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            tf = Counter(tokenize(c.text))
            self.chunks.append(c)
            self._tfs.append(tf)
            self._lens.append(sum(tf.values()))
            self._df.update(tf.keys())

    def _idf(self, term: str) -> float:
        n, df = len(self.chunks), self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.chunks), dtype=np.float64)
        if not self.chunks:
            return out
        avgdl = sum(self._lens) / len(self._lens) or 1.0
        for term in set(tokenize(query)):
            if term not in self._df:
                continue
            idf = self._idf(term)
            for i, tf in enumerate(self._tfs):
                f = tf.get(term)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self._lens[i] / avgdl)
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        s = self.scores(query)
        order = [i for i in np.argsort(-s, kind="stable")[:k] if s[i] > 0]
        return [SearchHit(self.chunks[i], float(s[i])) for i in order]


class VectorIndex:
    """Exact cosine search over a dense matrix (fast enough to ~10^5 chunks)."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, embedder.dim), dtype=np.float32)

    def add(self, chunks: list[Chunk], vectors: np.ndarray | None = None) -> None:
        if not chunks:
            return
        if vectors is None:
            vectors = self.embedder.embed([c.text for c in chunks])
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        sims = self.vectors @ q
        order = np.argsort(-sims, kind="stable")[:k]
        return [SearchHit(self.chunks[i], float(sims[i])) for i in order]


def reciprocal_rank_fusion(rankings: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    """Fuse ranked lists: score(d) = sum over lists of 1 / (k + rank(d))."""
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1.0 / (k + rank)
            by_id[hit.chunk.id] = hit.chunk
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [SearchHit(by_id[cid], score) for cid, score in ordered]


class HybridIndex:
    """BM25 + dense retrieval fused with RRF. Persists to a directory."""

    MODES = ("bm25", "vector", "hybrid")

    def __init__(self, embedder: Embedder, rrf_k: int = 60, candidates: int = 20):
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.bm25 = BM25Index()
        self.vector = VectorIndex(embedder)

    def __len__(self) -> int:
        return len(self.bm25.chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return self.bm25.chunks

    def add(self, chunks: list[Chunk], vectors: np.ndarray | None = None) -> None:
        self.bm25.add(chunks)
        self.vector.add(chunks, vectors)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[SearchHit]:
        if mode == "bm25":
            return self.bm25.search(query, k)
        if mode == "vector":
            return self.vector.search(query, k)
        if mode != "hybrid":
            raise ValueError(f"mode must be one of {self.MODES}")
        n = max(k, self.candidates)
        fused = reciprocal_rank_fusion(
            [self.bm25.search(query, n), self.vector.search(query, n)], k=self.rrf_k
        )
        return fused[:k]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        np.save(p / "vectors.npy", self.vector.vectors)
        meta = {"embedder": self.embedder.name, "dim": self.embedder.dim, "rrf_k": self.rrf_k}
        (p / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> HybridIndex:
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        embedder = embedder or make_embedder(meta["embedder"])
        if embedder.dim != meta["dim"]:
            raise ValueError(
                f"index was built with {meta['embedder']} (dim {meta['dim']}), "
                f"got {embedder.name} (dim {embedder.dim})"
            )
        index = cls(embedder, rrf_k=meta.get("rrf_k", 60))
        with open(p / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        index.add(chunks, np.load(p / "vectors.npy"))
        return index
