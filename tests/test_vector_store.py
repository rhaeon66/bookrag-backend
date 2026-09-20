from app.models.schemas import ChunkRecord
from app.retrieval import vector_store
from app.retrieval.embeddings import embed_documents


def _chunk(chunk_id, text, **kw):
    defaults = dict(document="Doc", chapter="Chapter I", chapter_title=None, section=None,
                     start_page=1, end_page=1, token_count=len(text.split()), sequence=1)
    defaults.update(kw)
    return ChunkRecord(chunk_id=chunk_id, text=text, **defaults)


def test_add_and_query_returns_best_semantic_match(patched_settings):
    chunks = [
        _chunk("c1", "The aardvark forages for insects at night in Africa.", start_page=1, end_page=1, sequence=1),
        _chunk("c2", "The stock market fell sharply amid inflation fears.", start_page=2, end_page=2, sequence=2),
    ]
    embeddings = embed_documents([c.text for c in chunks])

    vector_store.reset_collection()
    vector_store.add_chunks(chunks, embeddings)
    assert vector_store.count() == 2

    from app.retrieval.embeddings import embed_query
    query_vec = embed_query("Tell me about a nocturnal African mammal.")
    results = vector_store.query(query_vec, top_k=2)
    assert results[0]["chunk_id"] == "c1"


def test_query_filters_by_chapter(patched_settings):
    chunks = [
        _chunk("c1", "alpha content about foxes", chapter="Chapter I", start_page=1, end_page=1, sequence=1),
        _chunk("c2", "alpha content about foxes too", chapter="Chapter II", start_page=2, end_page=2, sequence=2),
    ]
    embeddings = embed_documents([c.text for c in chunks])
    vector_store.reset_collection()
    vector_store.add_chunks(chunks, embeddings)

    from app.retrieval.embeddings import embed_query
    query_vec = embed_query("foxes")
    results = vector_store.query(query_vec, top_k=5, filters={"chapter": "Chapter II", "page_min": None, "page_max": None})
    assert {r["chunk_id"] for r in results} == {"c2"}


def test_reset_collection_clears_existing_data(patched_settings):
    chunks = [_chunk("c1", "some content", start_page=1, end_page=1, sequence=1)]
    embeddings = embed_documents([c.text for c in chunks])
    vector_store.reset_collection()
    vector_store.add_chunks(chunks, embeddings)
    assert vector_store.count() == 1

    vector_store.reset_collection()
    assert vector_store.count() == 0


def test_query_on_empty_collection_returns_empty_list(patched_settings):
    vector_store.reset_collection()
    from app.retrieval.embeddings import embed_query
    results = vector_store.query(embed_query("anything"), top_k=5)
    assert results == []
