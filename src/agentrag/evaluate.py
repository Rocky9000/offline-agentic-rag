"""Retrieval and end-to-end evaluation.

Dataset format (JSONL), one item per line::

    {"question": "...", "relevant_docs": ["doc-id", ...], "answer_keywords": ["...", ...]}

``relevant_docs`` drives retrieval metrics; ``answer_keywords`` (all must
appear, case-insensitive) is a cheap, reproducible proxy for answer
correctness. Questions with ``relevant_docs: []`` are unanswerable and the
system is scored on whether it abstains.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from .agents import extract_citations
from .index import HybridIndex
from .pipeline import AgenticRAG


@dataclass
class EvalItem:
    question: str
    relevant_docs: list[str]
    answer_keywords: list[str]


def load_dataset(path: str | Path) -> list[EvalItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            items.append(EvalItem(d["question"], d.get("relevant_docs", []), d.get("answer_keywords", [])))
    return items


def _ranked_docs(index: HybridIndex, question: str, k: int, mode: str) -> list[str]:
    docs: list[str] = []
    for hit in index.search(question, k=k * 4, mode=mode):
        if hit.chunk.doc_id not in docs:
            docs.append(hit.chunk.doc_id)
    return docs[:k]


def retrieval_metrics(index: HybridIndex, items: list[EvalItem], k: int = 3, mode: str = "hybrid") -> dict:
    """Document-level Recall@k, Hit@k and MRR over answerable questions."""
    answerable = [it for it in items if it.relevant_docs]
    recalls, hits, rrs = [], [], []
    for it in answerable:
        ranked = _ranked_docs(index, it.question, k, mode)
        rel = set(it.relevant_docs)
        recalls.append(len(rel & set(ranked)) / len(rel))
        hits.append(1.0 if rel & set(ranked) else 0.0)
        rrs.append(next((1.0 / r for r, d in enumerate(ranked, 1) if d in rel), 0.0))
    n = len(answerable) or 1
    return {
        "mode": mode,
        "k": k,
        "n": len(answerable),
        f"recall@{k}": sum(recalls) / n,
        f"hit@{k}": sum(hits) / n,
        "mrr": sum(rrs) / n,
    }


def end_to_end_metrics(rag: AgenticRAG, items: list[EvalItem]) -> dict:
    correct, grounded, valid_cites, abstain_ok, latencies, iters = [], [], [], [], [], []
    for it in items:
        r = rag.ask(it.question)
        latencies.append(r.latency_ms)
        iters.append(r.iterations)
        if not it.relevant_docs:
            abstain_ok.append(1.0 if r.abstained else 0.0)
            continue
        text = r.answer.lower()
        correct.append(1.0 if all(kw.lower() in text for kw in it.answer_keywords) else 0.0)
        grounded.append(1.0 if r.grounded else 0.0)
        cites = extract_citations(r.answer)
        valid_cites.append(1.0 if cites and len(r.sources) == len(set(cites)) else 0.0)

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    lat = sorted(latencies)
    return {
        "n": len(items),
        "keyword_accuracy": mean(correct),
        "grounded_rate": mean(grounded),
        "citation_validity": mean(valid_cites),
        "abstention_accuracy": mean(abstain_ok),
        "mean_iterations": mean([float(i) for i in iters]),
        "latency_p50_ms": statistics.median(lat) if lat else None,
        "latency_p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
    }


def markdown_table(rows: list[dict]) -> str:
    cols = list(rows[0].keys())
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else str(v))  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)
