"""Load documents from disk and turn them into chunks."""

from __future__ import annotations

from pathlib import Path

from .chunking import Chunk, chunk_text

TEXT_SUFFIXES = {".md", ".txt", ".rst"}


def read_document(path: Path) -> str:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("PDF support needs: pip install 'offline-agentic-rag[pdf]'") from e
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    raise ValueError(f"unsupported file type: {path.suffix}")


def load_directory(
    root: str | Path, max_chars: int = 800, overlap: int = 120
) -> list[Chunk]:
    root = Path(root)
    files = [root] if root.is_file() else sorted(
        p for p in root.rglob("*") if p.suffix.lower() in TEXT_SUFFIXES | {".pdf"}
    )
    chunks: list[Chunk] = []
    for path in files:
        doc_id = path.stem if root.is_file() else path.relative_to(root).with_suffix("").as_posix()
        chunks.extend(
            chunk_text(read_document(path), doc_id, max_chars, overlap, {"path": str(path)})
        )
    return chunks
