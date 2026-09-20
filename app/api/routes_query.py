"""Query endpoint: ask a question about the ingested book."""
from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse
from app.retrieval import vector_store
from app.services.rag_pipeline import answer_question

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty.")
    if vector_store.count() == 0:
        raise HTTPException(status_code=409, detail="The book has not been ingested yet. Call POST /api/ingest first.")
    try:
        return answer_question(request)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach the local LLM (Ollama). Make sure it is running and the configured model "
            f"is pulled. ({exc})",
        ) from exc
