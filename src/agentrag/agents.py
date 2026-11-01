"""The three agents: Planner, Answerer, Critic.

Each agent is a thin, testable wrapper around one prompt. LLM output is
never trusted to be well formed: JSON is parsed defensively and every
agent has a deterministic fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .index import SearchHit
from .llm import LLM

ABSTAIN = "I don't know based on the provided documents."

PLANNER_SYSTEM = """You are a query planner for a document search engine.
Rewrite the user's question as at most 3 short, self-contained search queries.
Split multi-part questions into one query per part. A simple question needs one query.
Respond with JSON only, no prose: {"queries": ["...", "..."]}"""

ANSWER_SYSTEM = f"""You answer questions using ONLY the numbered context passages.
Cite every factual sentence with its passage number in square brackets, e.g. [2].
Never cite a number that is not in the context.
If the passages do not contain the answer, reply exactly: "{ABSTAIN}"
Be concise."""

CRITIC_SYSTEM = """You verify whether an answer is fully supported by numbered context passages.
An answer is grounded only if every claim appears in the passage it cites.
Respond with JSON only, no prose:
{"grounded": true or false, "reason": "one sentence", "follow_up_query": "a search query for the missing information, or an empty string"}"""

_CITATION = re.compile(r"\[(\d+)\]")


def parse_json_object(raw: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply (handles code fences and chatter)."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_citations(answer: str) -> list[int]:
    return [int(n) for n in _CITATION.findall(answer)]


def is_abstention(answer: str) -> bool:
    return "don't know based on the provided documents" in answer.lower()


def format_context(hits: list[SearchHit]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h.chunk.doc_id})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
    )


class Planner:
    def __init__(self, llm: LLM, max_queries: int = 3):
        self.llm, self.max_queries = llm, max_queries

    def plan(self, question: str) -> list[str]:
        data = parse_json_object(self.llm.complete(question, system=PLANNER_SYSTEM)) or {}
        raw = data.get("queries") if isinstance(data.get("queries"), list) else []
        queries = [question]  # the original question is always searched
        for q in raw:
            if isinstance(q, str) and q.strip() and q.strip() not in queries:
                queries.append(q.strip())
        return queries[: self.max_queries + 1]


class Answerer:
    def __init__(self, llm: LLM):
        self.llm = llm

    def answer(self, question: str, hits: list[SearchHit], feedback: str = "") -> str:
        if not hits:
            return ABSTAIN
        prompt = f"Context:\n{format_context(hits)}\n\nQuestion: {question}\n"
        if feedback:
            prompt += f"\nA reviewer rejected your previous answer: {feedback}\nFix it.\n"
        return self.llm.complete(prompt + "Answer:", system=ANSWER_SYSTEM).strip()


@dataclass
class Verdict:
    grounded: bool
    reason: str
    follow_up_query: str = ""
    invalid_citations: list[int] = field(default_factory=list)


class Critic:
    def __init__(self, llm: LLM):
        self.llm = llm

    def review(self, question: str, answer: str, hits: list[SearchHit]) -> Verdict:
        cites = extract_citations(answer)
        invalid = sorted({c for c in cites if not 1 <= c <= len(hits)})
        if invalid:
            return Verdict(False, f"cites passages that do not exist: {invalid}", "", invalid)
        if not cites and not is_abstention(answer):
            return Verdict(False, "answer has no citations")

        prompt = (
            f"Context:\n{format_context(hits)}\n\nQuestion: {question}\n\n"
            f"Answer to verify:\n{answer}"
        )
        data = parse_json_object(self.llm.complete(prompt, system=CRITIC_SYSTEM))
        if data is None:
            # An unparseable judge is not evidence of a bad answer; the
            # deterministic citation checks above already passed.
            return Verdict(True, "critic output unparseable; citation checks passed")
        grounded = data.get("grounded") is True
        if is_abstention(answer):
            grounded = False  # honest, but worth one more retrieval attempt
        follow_up = data.get("follow_up_query")
        return Verdict(
            grounded=grounded,
            reason=str(data.get("reason", "")),
            follow_up_query=follow_up.strip() if isinstance(follow_up, str) else "",
        )
