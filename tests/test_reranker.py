from app.models.schemas import RetrievedChunk
from app.retrieval.reranker import rerank


def _retrieved(chunk_id, text):
    return RetrievedChunk(
        chunk_id=chunk_id, text=text, document="Doc", chapter="Chapter I",
        start_page=1, end_page=1, sequence=1,
    )


def test_rerank_reorders_by_relevance_to_query(isolated_settings):
    candidates = [
        _retrieved("off_topic", "The weather today is sunny with a light breeze."),
        _retrieved("on_topic", "Muhammad was allegedly driven out of Mecca by the Quraysh in 622 CE."),
    ]
    reranked = rerank("Was Muhammad driven out of Mecca?", candidates, isolated_settings, top_k=2)
    assert reranked[0].chunk_id == "on_topic"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score >= reranked[1].rerank_score


def test_rerank_respects_top_k_override(isolated_settings):
    candidates = [_retrieved(f"c{i}", f"passage number {i} about various topics") for i in range(5)]
    reranked = rerank("various topics", candidates, isolated_settings, top_k=2)
    assert len(reranked) == 2


def test_rerank_empty_candidates_returns_empty(isolated_settings):
    assert rerank("anything", [], isolated_settings) == []
