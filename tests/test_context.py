from app.generation.context import _citation_label, _dedupe, build_context
from app.models.schemas import ChunkRecord, RetrievedChunk
from app.retrieval import bm25 as bm25_module


def _retrieved(chunk_id, text, chapter="Chapter III", section="Some Section", page=27, seq=1,
                fused_score=1.0, rerank_score=None):
    return RetrievedChunk(
        chunk_id=chunk_id, text=text, document="Doc", chapter=chapter, section=section,
        start_page=page, end_page=page, sequence=seq, fused_score=fused_score, rerank_score=rerank_score,
    )


def test_citation_label_formats():
    with_section = _retrieved("c1", "text", chapter="Chapter III", section="Prophetic Mission in Mecca", page=27)
    assert _citation_label(with_section) == "Chapter III → Section: Prophetic Mission in Mecca → Page 27"

    no_section = _retrieved("c2", "text", chapter="Chapter III", section=None, page=27)
    assert _citation_label(no_section) == "Chapter III — Page 27"

    no_chapter = _retrieved("c3", "text", chapter=None, section=None, page=5)
    assert _citation_label(no_chapter) == "Page 5"

    spanning = RetrievedChunk(chunk_id="c4", text="t", document="Doc", chapter="Chapter I", section=None,
                               start_page=10, end_page=12, sequence=1)
    assert "Pages 10" in _citation_label(spanning)


def test_dedupe_drops_near_identical_chunks():
    a = _retrieved("a", "Muhammad went to Mecca and preached to the Quraysh tribe for many years.")
    b = _retrieved("b", "Muhammad went to Mecca and preached to the Quraysh tribe for many years too.")
    c = _retrieved("c", "The cactus is a plant adapted to arid desert climates.")
    kept = _dedupe([a, b, c], threshold=0.85)
    kept_ids = {k.chunk_id for k in kept}
    assert "c" in kept_ids
    assert len(kept_ids) == 2  # a and b collapse into one


def test_build_context_respects_max_chunks_and_produces_sources(isolated_settings):
    chunks = [_retrieved(f"c{i}", f"Distinct passage number {i} about totally different topics.", page=20 + i, seq=i) for i in range(5)]
    context_block, sources = build_context(chunks, isolated_settings, max_chunks=2)
    assert len(sources) == 2
    assert "[Source 1]" in context_block
    assert sources[0].page == sources[0].start_page


def test_build_context_min_rerank_score_drops_low_scoring_chunks(isolated_settings, monkeypatch):
    monkeypatch.setattr(isolated_settings, "context_min_rerank_score", 0.0, raising=False)
    monkeypatch.setattr(isolated_settings, "context_neighbor_expansion", False, raising=False)

    strong = _retrieved("strong", "Muhammad was allegedly driven out of Mecca by the Quraysh.", page=29, seq=1, rerank_score=7.9)
    weak = _retrieved("weak", "Unrelated passage about the Jews and Ishmael.", page=65, seq=2, rerank_score=-3.2)

    _, sources = build_context([strong, weak], isolated_settings)
    ids = {s.chunk_id for s in sources}
    assert ids == {"strong"}


def test_build_context_min_rerank_score_ignores_chunks_without_a_score(isolated_settings, monkeypatch):
    # Reranking disabled (or a neighbor add-on) -> no rerank_score -> the
    # threshold must not drop it just because it was never scored.
    monkeypatch.setattr(isolated_settings, "context_min_rerank_score", 5.0, raising=False)
    monkeypatch.setattr(isolated_settings, "context_neighbor_expansion", False, raising=False)

    unscored = _retrieved("unscored", "Some passage that was never reranked.", page=1, seq=1, rerank_score=None)

    _, sources = build_context([unscored], isolated_settings)
    assert {s.chunk_id for s in sources} == {"unscored"}


def test_build_context_neighbor_expansion_pulls_adjacent_chunk(isolated_settings, monkeypatch):
    monkeypatch.setattr("app.generation.context.get_settings", lambda: isolated_settings)
    monkeypatch.setattr(isolated_settings, "context_neighbor_expansion", True, raising=False)

    stored_chunks = {
        "top": {"text": "top ranked passage", "metadata": {"chapter": "Chapter I", "sequence": 5,
                                                             "start_page": 10, "end_page": 10, "document": "Doc"}},
        "next": {"text": "the very next passage right after it", "metadata": {"chapter": "Chapter I", "sequence": 6,
                                                                               "start_page": 11, "end_page": 11, "document": "Doc"}},
    }
    index = bm25_module.BM25Index(chunk_ids=list(stored_chunks), corpus_tokens=[[], []], chunks=stored_chunks)
    monkeypatch.setattr(bm25_module, "_index_cache", index)

    top_chunk = RetrievedChunk(chunk_id="top", text="top ranked passage", document="Doc", chapter="Chapter I",
                                start_page=10, end_page=10, sequence=5, fused_score=1.0)
    context_block, sources = build_context([top_chunk], isolated_settings)
    ids = {s.chunk_id for s in sources}
    assert "top" in ids
    assert "next" in ids
