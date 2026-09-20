"""Corpus introspection endpoints: stats, ingested documents, chapter index,
and the (safe) running configuration."""
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.models.schemas import ChapterInfo, ConfigResponse, DocumentInfo, StatsResponse
from app.services.rag_pipeline import get_manifest

router = APIRouter(tags=["meta"])


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    settings = get_settings()
    manifest = get_manifest(settings)
    if manifest is None:
        raise HTTPException(status_code=404, detail="No book has been ingested yet.")
    return StatsResponse(
        document=manifest["document"],
        total_pages=manifest["total_pages"],
        total_chunks=manifest["total_chunks"],
        total_chapters=len(manifest["chapters"]),
        total_sections=sum(len(c["sections"]) for c in manifest["chapters"]),
        avg_chunk_tokens=manifest["avg_chunk_tokens"],
        embedding_model=settings.embedding_model,
        rerank_enabled=settings.rerank_enabled,
        rerank_model=settings.rerank_model if settings.rerank_enabled else None,
        llm_model=settings.llm_model,
        ingested_at=manifest["ingested_at"],
    )


@router.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[DocumentInfo]:
    """List ingested documents. BookRAG indexes one book at a time, so this
    returns zero or one entry."""
    manifest = get_manifest()
    if manifest is None:
        return []
    return [
        DocumentInfo(
            document=manifest["document"],
            total_pages=manifest["total_pages"],
            total_chapters=len(manifest["chapters"]),
            total_chunks=manifest["total_chunks"],
            ingested_at=manifest["ingested_at"],
        )
    ]


@router.get("/chapters", response_model=list[ChapterInfo])
def chapters() -> list[ChapterInfo]:
    manifest = get_manifest()
    if manifest is None:
        raise HTTPException(status_code=404, detail="No book has been ingested yet.")
    return [ChapterInfo(**c) for c in manifest["chapters"]]


@router.get("/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    settings = get_settings()
    return ConfigResponse(
        book_title=settings.book_title,
        embedding_model=settings.embedding_model,
        embedding_device=settings.embedding_device,
        rerank_enabled=settings.rerank_enabled,
        rerank_model=settings.rerank_model,
        rerank_device=settings.rerank_device,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        vector_top_k=settings.vector_top_k,
        bm25_top_k=settings.bm25_top_k,
        hybrid_candidate_k=settings.hybrid_candidate_k,
        rrf_k=settings.rrf_k,
        rerank_top_k=settings.rerank_top_k,
        context_max_tokens=settings.context_max_tokens,
        context_max_chunks=settings.context_max_chunks,
        query_rewrite_enabled=settings.query_rewrite_enabled,
        debug_mode=settings.debug_mode,
    )
