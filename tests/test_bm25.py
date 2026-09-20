from app.models.schemas import ChunkRecord
from app.retrieval.bm25 import BM25Index, tokenize


def _chunk(chunk_id, text, chapter="Chapter I", section=None, page=1, seq=0):
    return ChunkRecord(
        chunk_id=chunk_id,
        document="Doc",
        text=text,
        chapter=chapter,
        chapter_title=None,
        section=section,
        start_page=page,
        end_page=page,
        token_count=len(text.split()),
        sequence=seq,
    )


def test_tokenize_keeps_joined_references_intact():
    tokens = tokenize("Quran verse 2:190 discusses fighting.")
    assert "2:190" in tokens


def test_bm25_search_finds_exact_keyword_match():
    chunks = [
        _chunk("c1", "The aardvark forages for insects at night.", page=1, seq=1),
        _chunk("c2", "The beaver builds dams out of wood and mud.", page=2, seq=2),
        _chunk("c3", "The cactus stores water in its thick stem.", page=3, seq=3),
    ]
    index = BM25Index.build(chunks)
    results = index.search("beaver dam", top_k=3)
    assert results
    assert results[0]["chunk_id"] == "c2"


def test_bm25_matching_chunk_ids_respects_filters():
    chunks = [
        _chunk("c1", "alpha content", chapter="Chapter I", page=5, seq=1),
        _chunk("c2", "beta content", chapter="Chapter II", page=15, seq=2),
    ]
    index = BM25Index.build(chunks)
    assert index.matching_chunk_ids(chapter="Chapter I") == {"c1"}
    assert index.matching_chunk_ids(page_min=10) == {"c2"}
    assert index.matching_chunk_ids(page_max=10) == {"c1"}


def test_bm25_save_and_load_round_trip(tmp_path):
    # Classic BM25 IDF is log((N - n + 0.5) / (n + 0.5)): it's exactly zero
    # when a term appears in precisely half the corpus, so this needs >=3
    # documents (with the term in a minority of them) for a positive score.
    chunks = [
        _chunk("c1", "unique aardvark text", seq=1),
        _chunk("c2", "completely unrelated content about rivers", page=2, seq=2),
        _chunk("c3", "yet another unrelated passage about mountains", page=3, seq=3),
    ]
    index = BM25Index.build(chunks)
    path = tmp_path / "bm25.pkl"
    index.save(str(path))

    loaded = BM25Index.load(str(path))
    assert loaded is not None
    results = loaded.search("aardvark", top_k=1)
    assert results[0]["chunk_id"] == "c1"


def test_bm25_load_missing_file_returns_none(tmp_path):
    assert BM25Index.load(str(tmp_path / "does_not_exist.pkl")) is None
