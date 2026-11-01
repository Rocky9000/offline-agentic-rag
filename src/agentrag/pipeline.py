"""Orchestration: plan -> retrieve -> answer -> critique -> (re-retrieve) loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .agents import Answerer, Critic, Planner, extract_citations, is_abstention
from .index import HybridIndex, SearchHit, reciprocal_rank_fusion
from .llm import LLM


@dataclass
class Step:
    name: str
    detail: dict
    ms: float


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[SearchHit]
    grounded: bool
    abstained: bool
    iterations: int
    latency_ms: float
    trace: list[Step] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "grounded": self.grounded,
            "abstained": self.abstained,
            "iterations": self.iterations,
            "latency_ms": round(self.latency_ms, 1),
            "sources": [
                {"n": i, "doc_id": h.chunk.doc_id, "chunk_id": h.chunk.id, "text": h.chunk.text}
                for i, h in enumerate(self.sources, start=1)
            ],
            "trace": [{"step": s.name, "ms": round(s.ms, 1), **s.detail} for s in self.trace],
        }


class AgenticRAG:
    """Multi-agent RAG with a bounded self-correction loop.

    1. Planner decomposes the question into search queries.
    2. Each query hits the hybrid index; results are fused with RRF.
    3. Answerer writes a cited answer from the top passages.
    4. Critic checks citations deterministically, then asks the LLM to judge
       support. If ungrounded it may propose a follow-up query, which is
       added to the search and the loop repeats (up to ``max_iterations``).
    """

    def __init__(
        self,
        index: HybridIndex,
        llm: LLM,
        k: int = 5,
        max_context: int = 6,
        max_iterations: int = 2,
        use_planner: bool = True,
        use_critic: bool = True,
    ):
        self.index = index
        self.k = k
        self.max_context = max_context
        self.max_iterations = max(1, max_iterations)
        self.use_planner = use_planner
        self.use_critic = use_critic
        self.planner = Planner(llm)
        self.answerer = Answerer(llm)
        self.critic = Critic(llm)

    def retrieve(self, queries: list[str]) -> list[SearchHit]:
        return reciprocal_rank_fusion([self.index.search(q, self.k) for q in queries])

    def ask(self, question: str) -> RAGResult:
        t0 = time.perf_counter()
        trace: list[Step] = []

        def record(name: str, detail: dict, since: float) -> None:
            trace.append(Step(name, detail, (time.perf_counter() - since) * 1000))

        t = time.perf_counter()
        queries = self.planner.plan(question) if self.use_planner else [question]
        record("plan", {"queries": list(queries)}, t)

        t = time.perf_counter()
        context = self.retrieve(queries)[: self.max_context]
        record("retrieve", {"chunks": [h.chunk.id for h in context]}, t)

        answer, grounded, feedback, iterations = "", False, "", 0
        for iterations in range(1, self.max_iterations + 1):
            t = time.perf_counter()
            answer = self.answerer.answer(question, context, feedback)
            record("answer", {"iteration": iterations, "answer": answer}, t)

            if not self.use_critic:
                grounded = bool(extract_citations(answer)) and not is_abstention(answer)
                break

            t = time.perf_counter()
            verdict = self.critic.review(question, answer, context)
            record(
                "critique",
                {"iteration": iterations, "grounded": verdict.grounded, "reason": verdict.reason,
                 "follow_up_query": verdict.follow_up_query},
                t,
            )
            grounded = verdict.grounded
            if grounded or iterations == self.max_iterations:
                break
            feedback = verdict.reason
            if verdict.follow_up_query and verdict.follow_up_query not in queries:
                queries.append(verdict.follow_up_query)
                t = time.perf_counter()
                context = self.retrieve(queries)[: self.max_context]
                record("retrieve", {"chunks": [h.chunk.id for h in context]}, t)

        cited = sorted({c for c in extract_citations(answer) if 1 <= c <= len(context)})
        sources = [context[c - 1] for c in cited]
        return RAGResult(
            question=question,
            answer=answer,
            sources=sources,
            grounded=grounded,
            abstained=is_abstention(answer),
            iterations=iterations,
            latency_ms=(time.perf_counter() - t0) * 1000,
            trace=trace,
        )
