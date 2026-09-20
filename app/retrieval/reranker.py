"""Optional Cross-Encoder reranking (sentence-transformers CrossEncoder).

Pipeline: BM25 + Vector -> top ~20 fused candidates -> Cross-Encoder -> top
5-8 chunks -> LLM. Fully skippable via RERANK_ENABLED=false, in which case
the fused hybrid ranking is used as-is.
"""
from functools import lru_cache
from typing import List, Optional

from app.config import Settings, get_settings
from app.models.schemas import RetrievedChunk


@lru_cache
def get_reranker_model():
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    return CrossEncoder(settings.rerank_model, device=settings.rerank_device)


def rerank(
    query: str,
    candidates: List[RetrievedChunk],
    settings: Optional[Settings] = None,
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    settings = settings or get_settings()
    if not candidates:
        return candidates

    model = get_reranker_model()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    limit = top_k or settings.rerank_top_k
    reranked: List[RetrievedChunk] = []
    for chunk, score in scored[:limit]:
        chunk = chunk.model_copy(update={"rerank_score": float(score), "source": chunk.source})
        reranked.append(chunk)
    return reranked
