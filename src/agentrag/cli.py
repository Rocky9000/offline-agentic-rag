"""Command line: ingest, ask, eval, serve."""

from __future__ import annotations

import argparse
import json
import sys

from .embeddings import make_embedder
from .evaluate import end_to_end_metrics, load_dataset, markdown_table, retrieval_metrics
from .index import HybridIndex
from .ingest import load_directory
from .llm import OllamaLLM, OpenAICompatLLM
from .pipeline import AgenticRAG


def _llm(args):
    if args.backend == "openai":
        return OpenAICompatLLM(args.model, args.base_url)
    return OllamaLLM(args.model, args.host)


def _add_llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--index", default=".index")
    p.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    p.add_argument("--model", default="llama3.2:3b")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--base-url", default="http://localhost:8080/v1")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-iterations", type=int, default=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentrag", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="chunk and index a folder of documents")
    p.add_argument("path")
    p.add_argument("--index", default=".index")
    p.add_argument("--embedder", default="hashing", help="hashing | st:<model> | ollama:<model>")
    p.add_argument("--max-chars", type=int, default=800)
    p.add_argument("--overlap", type=int, default=120)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("--json", action="store_true")
    _add_llm_args(p)

    p = sub.add_parser("eval", help="evaluate retrieval (and optionally the full pipeline)")
    p.add_argument("dataset")
    p.add_argument("--index", default=".index")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--e2e", action="store_true", help="also run the LLM pipeline (needs a model)")
    p.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    p.add_argument("--model", default="llama3.2:3b")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--base-url", default="http://localhost:8080/v1")
    p.add_argument("--max-iterations", type=int, default=2)

    p = sub.add_parser("serve", help="run the HTTP API and web UI")
    p.add_argument("--port", type=int, default=8000)
    _add_llm_args(p)

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        chunks = load_directory(args.path, args.max_chars, args.overlap)
        index = HybridIndex(make_embedder(args.embedder))
        index.add(chunks)
        index.save(args.index)
        docs = len({c.doc_id for c in chunks})
        print(f"indexed {len(chunks)} chunks from {docs} documents -> {args.index}")
        return 0

    index = HybridIndex.load(args.index)

    if args.cmd == "eval":
        items = load_dataset(args.dataset)
        rows = [retrieval_metrics(index, items, args.k, m) for m in HybridIndex.MODES]
        print("## Retrieval\n\n" + markdown_table(rows))
        if args.e2e:
            rag = AgenticRAG(index, _llm(args), max_iterations=args.max_iterations)
            print("\n## End to end\n\n" + markdown_table([end_to_end_metrics(rag, items)]))
        return 0

    rag = AgenticRAG(index, _llm(args), k=args.k, max_iterations=args.max_iterations)

    if args.cmd == "ask":
        result = rag.ask(args.question)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(result.answer)
            for i, h in enumerate(result.sources, 1):
                print(f"  [{i}] {h.chunk.doc_id}")
            flag = "grounded" if result.grounded else "NOT verified"
            print(f"({flag}, {result.iterations} iteration(s), {result.latency_ms:.0f} ms)")
        return 0

    if args.cmd == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(rag), host="127.0.0.1", port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
