"""Sentence-aware chunking with character overlap."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    start: int
    end: int
    metadata: dict = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(d["id"], d["doc_id"], d["text"], d["start"], d["end"], d.get("metadata", {}))


def _sentence_spans(text: str, max_chars: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in _SENTENCE.finditer(text):
        s, e = m.span()
        if not text[s:e].strip():
            continue
        # A single sentence longer than the budget is hard-split.
        while e - s > max_chars:
            spans.append((s, s + max_chars))
            s += max_chars
        spans.append((s, e))
    return spans


def chunk_text(
    text: str,
    doc_id: str,
    max_chars: int = 800,
    overlap: int = 120,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Pack whole sentences into chunks of at most ``max_chars`` characters.

    Consecutive chunks share trailing sentences that start within ``overlap``
    characters of the previous chunk's end, so a fact that straddles a
    boundary is still retrievable from one chunk. Every chunk advances by at
    least one sentence, so the loop always terminates.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be in [0, max_chars)")

    spans = _sentence_spans(text, max_chars)
    chunks: list[Chunk] = []
    i = 0
    while i < len(spans):
        start = spans[i][0]
        j = i
        while j + 1 < len(spans) and spans[j + 1][1] - start <= max_chars:
            j += 1
        end = spans[j][1]
        body = text[start:end].strip()
        if body:
            cid = hashlib.sha1(f"{doc_id}:{start}:{end}".encode()).hexdigest()[:12]
            chunks.append(Chunk(cid, doc_id, body, start, end, dict(metadata or {})))
        if j + 1 >= len(spans):
            break
        k = j + 1
        while k - 1 > i and end - spans[k - 1][0] <= overlap:
            k -= 1
        i = k
    return chunks
