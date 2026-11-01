import numpy as np
import pytest

from agentrag import BM25Index, Chunk, HashingEmbedder, HybridIndex, SearchHit, reciprocal_rank_fusion
from agentrag.text import stem, tokenize


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(f"c{i}", f"d{i}", text, 0, len(text))


DOCS = [
    _chunk(0, "Charge lithium batteries on the balance charger at 1C."),
    _chunk(1, "Replace propellers every 25 flight hours."),
    _chunk(2, "Do not fly when wind exceeds 10 metres per second."),
]


def test_stemmer_folds_inflections():
    assert stem("batteries") == stem("battery")
    assert stem("charged") == stem("charging") == stem("charge") == stem("charges")
    assert tokenize("The battery IS charging") == ["battery", "charg"]


def test_bm25_ranks_matching_doc_first():
    idx = BM25Index()
    idx.add(DOCS)
    hits = idx.search("how often are propellers replaced", k=3)
    assert hits[0].chunk.id == "c1"


def test_bm25_returns_nothing_for_unknown_terms():
    idx = BM25Index()
    idx.add(DOCS)
    assert idx.search("zebra quantum", k=3) == []


def test_bm25_idf_prefers_rare_terms():
    idx = BM25Index()
    idx.add([_chunk(0, "drone drone drone"), _chunk(1, "drone propeller"), _chunk(2, "drone")])
    assert idx.search("propeller drone", k=1)[0].chunk.id == "c1"


def test_hashing_embedder_is_normalised_and_deterministic():
    emb = HashingEmbedder(dim=256)
    a = emb.embed(["battery storage voltage", ""])
    b = emb.embed(["battery storage voltage", ""])
    np.testing.assert_allclose(a, b)
    assert a.shape == (2, 256)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
    assert np.linalg.norm(a[1]) == 0.0  # empty text stays a zero vector


def test_hashing_embedder_similarity_is_meaningful():
    emb = HashingEmbedder()
    q, near, far = emb.embed(["charging batteries", "battery charger rules", "radio channel nine"])
    assert q @ near > q @ far


def test_rrf_rewards_agreement():
    a, b, c = DOCS
    fused = reciprocal_rank_fusion([[SearchHit(a, 9), SearchHit(b, 5)], [SearchHit(b, 0.9), SearchHit(c, 0.1)]])
    assert fused[0].chunk.id == "c1"  # b appears in both lists
    assert {h.chunk.id for h in fused} == {"c0", "c1", "c2"}


@pytest.mark.parametrize("mode", HybridIndex.MODES)
def test_all_modes_find_the_battery_doc(handbook_index, mode):
    hits = handbook_index.search("storage voltage for lithium packs", k=3, mode=mode)
    assert "battery-care" in {h.chunk.doc_id for h in hits}


def test_invalid_mode(handbook_index):
    with pytest.raises(ValueError):
        handbook_index.search("x", mode="magic")


def test_save_load_roundtrip(tmp_path, handbook_index):
    handbook_index.save(tmp_path / "idx")
    loaded = HybridIndex.load(tmp_path / "idx")
    assert len(loaded) == len(handbook_index)
    q = "what is the hotel limit in metro cities"
    assert [h.chunk.id for h in loaded.search(q)] == [h.chunk.id for h in handbook_index.search(q)]


def test_load_rejects_mismatched_embedder(tmp_path, handbook_index):
    handbook_index.save(tmp_path / "idx")
    with pytest.raises(ValueError):
        HybridIndex.load(tmp_path / "idx", embedder=HashingEmbedder(dim=64))
