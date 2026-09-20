"""Health check endpoint."""
from fastapi import APIRouter

from app.generation import llm
from app.models.schemas import HealthResponse
from app.retrieval import bm25 as bm25_module
from app.retrieval import vector_store

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    chroma_count = vector_store.count()
    bm25_index = bm25_module.get_bm25_index()
    ollama_ready = llm.is_available()

    return HealthResponse(
        status="ok" if (chroma_count > 0 and bm25_index is not None) else "not_ingested",
        chroma_ready=chroma_count > 0,
        bm25_ready=bm25_index is not None,
        ollama_ready=ollama_ready,
        indexed_chunks=chroma_count,
    )
