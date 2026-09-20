"""Hybrid retrieval: BM25 + vector search combined via Reciprocal Rank Fusion."""
from typing import Dict, List, Optional

from app.config import Settings, get_settings
from app.models.schemas import QueryFilters, RetrievedChunk
from app.retrieval import bm25 as bm25_module
from app.retrieval import vector_store
from app.retrieval.embeddings import embed_query


def reciprocal_rank_fusion(
    ranked_lists: List[List[dict]], k: int = 60
) -> Dict[str, float]:
    """RRF score for each chunk_id: sum over lists of 1 / (k + rank).

    `ranked_lists` is a list of result lists, each already sorted best-first,
    where each item has a "chunk_id" key. Returns {chunk_id: rrf_score}.
    """
    scores: Dict[str, float] = {}
    for results in ranked_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_search(
    query: str,
    filters: Optional[QueryFilters] = None,
    settings: Optional[Settings] = None,
    trace: Optional[dict] = None,
) -> List[RetrievedChunk]:
    """Run BM25 + vector search and fuse them with RRF.

    When a `trace` dict is passed, it is filled in-place with the raw BM25,
    vector and fused result lists (chunk_id + score only) for debug/
    observability purposes; tracing is otherwise a no-op with zero overhead.
    """
    settings = settings or get_settings()
    filters = filters or QueryFilters()

    bm25_index = bm25_module.get_bm25_index()
    allowed_ids = None
    vector_where = None
    if filters.chapter or filters.page_min is not None or filters.page_max is not None:
        vector_where = {"chapter": filters.chapter, "page_min": filters.page_min, "page_max": filters.page_max}
        if bm25_index is not None:
            allowed_ids = bm25_index.matching_chunk_ids(
                chapter=filters.chapter, page_min=filters.page_min, page_max=filters.page_max
            )

    bm25_results = bm25_index.search(query, settings.bm25_top_k, allowed_chunk_ids=allowed_ids) if bm25_index else []

    query_embedding = embed_query(query)
    vector_results = vector_store.query(query_embedding, settings.vector_top_k, filters=vector_where)

    fused_scores = reciprocal_rank_fusion([bm25_results, vector_results], k=settings.rrf_k)

    by_id: Dict[str, dict] = {}
    bm25_score_map: Dict[str, float] = {}
    vector_score_map: Dict[str, float] = {}
    for item in bm25_results:
        by_id.setdefault(item["chunk_id"], item)
        bm25_score_map[item["chunk_id"]] = item["score"]
    for item in vector_results:
        by_id.setdefault(item["chunk_id"], item)
        vector_score_map[item["chunk_id"]] = item["score"]

    ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
    ranked_ids = ranked_ids[: settings.hybrid_candidate_k]

    if trace is not None:
        trace["bm25_results"] = [{"chunk_id": i["chunk_id"], "score": i["score"]} for i in bm25_results]
        trace["vector_results"] = [{"chunk_id": i["chunk_id"], "score": i["score"]} for i in vector_results]
        trace["fused_results"] = [{"chunk_id": cid, "score": fused_scores[cid]} for cid in ranked_ids]

    results: List[RetrievedChunk] = []
    for chunk_id in ranked_ids:
        item = by_id[chunk_id]
        meta = item["metadata"]
        source = "hybrid" if chunk_id in bm25_score_map and chunk_id in vector_score_map else (
            "bm25" if chunk_id in bm25_score_map else "vector"
        )
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=item["text"],
                document=meta.get("document", ""),
                chapter=meta.get("chapter") or None,
                chapter_title=meta.get("chapter_title") or None,
                section=meta.get("section") or None,
                start_page=meta.get("start_page", meta.get("page", 0)),
                end_page=meta.get("end_page", meta.get("page", 0)),
                sequence=meta.get("sequence", -1),
                bm25_score=bm25_score_map.get(chunk_id),
                vector_score=vector_score_map.get(chunk_id),
                fused_score=fused_scores[chunk_id],
                source=source,
            )
        )
    return results
