from itertools import pairwise

import pytest

from agentrag.chunking import chunk_text

TEXT = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(60))


def test_chunks_respect_budget():
    chunks = chunk_text(TEXT, "doc", max_chars=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_chunks_cover_every_sentence():
    chunks = chunk_text(TEXT, "doc", max_chars=200, overlap=50)
    joined = " ".join(c.text for c in chunks)
    for i in range(60):
        assert f"Sentence number {i} " in joined


def test_consecutive_chunks_overlap_and_advance():
    chunks = chunk_text(TEXT, "doc", max_chars=200, overlap=80)
    for a, b in pairwise(chunks):
        assert b.start > a.start  # always makes progress
        assert b.start < a.end  # shares text with the previous chunk


def test_zero_overlap_has_no_shared_text():
    chunks = chunk_text(TEXT, "doc", max_chars=200, overlap=0)
    for a, b in pairwise(chunks):
        assert b.start >= a.end


def test_overlong_sentence_is_hard_split():
    chunks = chunk_text("x" * 1000, "doc", max_chars=300, overlap=0)
    assert [len(c.text) for c in chunks] == [300, 300, 300, 100]


def test_ids_are_stable_and_unique():
    a = chunk_text(TEXT, "doc", max_chars=200, overlap=50)
    b = chunk_text(TEXT, "doc", max_chars=200, overlap=50)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


@pytest.mark.parametrize("max_chars,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_arguments(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_text(TEXT, "doc", max_chars=max_chars, overlap=overlap)


def test_empty_text():
    assert chunk_text("   \n  ", "doc") == []
