"""Offline multi-agent retrieval-augmented generation."""

from .chunking import Chunk, chunk_text
from .embeddings import HashingEmbedder, make_embedder
from .index import BM25Index, HybridIndex, SearchHit, VectorIndex, reciprocal_rank_fusion
from .llm import OllamaLLM, OpenAICompatLLM, ScriptedLLM
from .pipeline import AgenticRAG, RAGResult

__all__ = [
    "AgenticRAG",
    "BM25Index",
    "Chunk",
    "HashingEmbedder",
    "HybridIndex",
    "OllamaLLM",
    "OpenAICompatLLM",
    "RAGResult",
    "ScriptedLLM",
    "SearchHit",
    "VectorIndex",
    "chunk_text",
    "make_embedder",
    "reciprocal_rank_fusion",
]
__version__ = "0.1.0"
