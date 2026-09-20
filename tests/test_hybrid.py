from app.models.schemas import ChunkRecord, QueryFilters
from app.retrieval import bm25 as bm25_module
from app.retrieval import vector_store
from app.retrieval.embeddings import embed_documents
from app.retrieval.hybrid import hybrid_search, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_formula():
    list_a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
    list_b = [{"chunk_id": "y"}, {"chunk_id": "z"}]
    scores = reciprocal_rank_fusion([list_a, list_b], k=10)
    assert scores["x"] == 1 / 11
    assert scores["z"] == 1 / 12
    assert scores["y"] == 1 / 12 + 1 / 11  # appears rank 2 in A, rank 1 in B
    assert scores["y"] > scores["x"] > scores["z"]


def _chunk(chunk_id, text, page, seq):
    return ChunkRecord(
        chunk_id=chunk_id, document="Doc", text=text, chapter="Chapter I", chapter_title=None,
        section=None, start_page=page, end_page=page, token_count=len(text.split()), sequence=seq,
    )


def test_hybrid_search_fuses_bm25_and_vector_results(patched_settings):
    chunks = [
        _chunk("c1", "The aardvark forages for insects at night in the African savanna.", 1, 1),
        _chunk("c2", "The stock market fell sharply amid persistent inflation fears.", 2, 2),
        _chunk("c3", "A beaver dam changes the flow of a river over many years.", 3, 3),
    ]
    embeddings = embed_documents([c.text for c in chunks])
    vector_store.reset_collection()
    vector_store.add_chunks(chunks, embeddings)
    bm25_module.build_and_save(chunks)

    results = hybrid_search("nocturnal African mammal aardvark", filters=QueryFilters(), settings=patched_settings)
    assert results
    assert results[0].chunk_id == "c1"
    assert results[0].fused_score is not None


def test_hybrid_search_applies_chapter_filter(patched_settings):
    chunks = [
        ChunkRecord(chunk_id="c1", document="Doc", text="foxes are clever animals in chapter one",
                    chapter="Chapter I", start_page=1, end_page=1, token_count=8, sequence=1),
        ChunkRecord(chunk_id="c2", document="Doc", text="foxes are clever animals in chapter two",
                    chapter="Chapter II", start_page=10, end_page=10, token_count=8, sequence=2),
    ]
    embeddings = embed_documents([c.text for c in chunks])
    vector_store.reset_collection()
    vector_store.add_chunks(chunks, embeddings)
    bm25_module.build_and_save(chunks)

    results = hybrid_search("foxes", filters=QueryFilters(chapter="Chapter II"), settings=patched_settings)
    assert results
    assert all(r.chapter == "Chapter II" for r in results)


def test_hybrid_search_trace_is_populated_when_requested(patched_settings):
    chunks = [_chunk("c1", "unique aardvark content", 1, 1), _chunk("c2", "totally different river content", 2, 2)]
    embeddings = embed_documents([c.text for c in chunks])
    vector_store.reset_collection()
    vector_store.add_chunks(chunks, embeddings)
    bm25_module.build_and_save(chunks)

    trace = {}
    hybrid_search("aardvark", settings=patched_settings, trace=trace)
    assert "bm25_results" in trace
    assert "vector_results" in trace
    assert "fused_results" in trace
