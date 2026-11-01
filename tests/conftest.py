from pathlib import Path

import pytest

from agentrag import HashingEmbedder, HybridIndex
from agentrag.ingest import load_directory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def handbook_index() -> HybridIndex:
    index = HybridIndex(HashingEmbedder())
    index.add(load_directory(ROOT / "data" / "handbook"))
    return index
