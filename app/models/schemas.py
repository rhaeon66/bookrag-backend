"""Pydantic request/response and internal record models."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingestion records
# ---------------------------------------------------------------------------

class PageRecord(BaseModel):
    """One page of the source PDF after parsing, structure detection and cleaning."""

    document: str
    page: int
    raw_text: str
    cleaned_text: str
    chapter: Optional[str] = None
    chapter_title: Optional[str] = None
    section: Optional[str] = None


class ChunkRecord(BaseModel):
    """A chunk of cleaned text ready to be embedded and indexed."""

    chunk_id: str
    document: str
    text: str
    chapter: Optional[str] = None
    chapter_title: Optional[str] = None
    section: Optional[str] = None
    start_page: int
    end_page: int
    token_count: int
    sequence: int  # position of this chunk within the whole document, for neighbor lookups

    def to_metadata(self) -> dict:
        """Flat metadata dict for ChromaDB / BM25 storage (no nested/None-unfriendly values)."""
        return {
            "document": self.document,
            "chapter": self.chapter or "",
            "chapter_title": self.chapter_title or "",
            "section": self.section or "",
            "start_page": self.start_page,
            "end_page": self.end_page,
            "page": self.start_page,
            "chunk_id": self.chunk_id,
            "token_count": self.token_count,
            "sequence": self.sequence,
        }


class IngestResponse(BaseModel):
    status: str
    document: str
    pages_processed: int
    chapters_detected: int
    sections_detected: int
    chunks_created: int
    duration_seconds: float


# ---------------------------------------------------------------------------
# Corpus introspection (documents / chapters / stats / config)
# ---------------------------------------------------------------------------

class ChapterInfo(BaseModel):
    chapter: str
    chapter_title: Optional[str] = None
    start_page: int
    end_page: int
    sections: List[str] = Field(default_factory=list)
    chunk_count: int = 0


class DocumentInfo(BaseModel):
    document: str
    total_pages: int
    total_chapters: int
    total_chunks: int
    ingested_at: Optional[str] = None


class StatsResponse(BaseModel):
    document: str
    total_pages: int
    total_chunks: int
    total_chapters: int
    total_sections: int
    avg_chunk_tokens: float
    embedding_model: str
    rerank_enabled: bool
    rerank_model: Optional[str] = None
    llm_model: str
    ingested_at: Optional[str] = None


class ConfigResponse(BaseModel):
    """Safe, non-secret subset of the running configuration (no hosts/paths that
    could leak local filesystem layout beyond what's already public in the repo)."""

    book_title: str
    embedding_model: str
    embedding_device: str
    rerank_enabled: bool
    rerank_model: str
    rerank_device: str
    llm_provider: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    vector_top_k: int
    bm25_top_k: int
    hybrid_candidate_k: int
    rrf_k: int
    rerank_top_k: int
    context_max_tokens: int
    context_max_chunks: int
    query_rewrite_enabled: bool
    debug_mode: bool


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryFilters(BaseModel):
    chapter: Optional[str] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None  # overrides RERANK_TOP_K / number of chunks used for the answer
    chat_history: List[ChatTurn] = Field(default_factory=list)
    filters: Optional[QueryFilters] = None
    rerank: Optional[bool] = None  # override RERANK_ENABLED for this request
    debug: Optional[bool] = None  # override DEBUG_MODE for this request


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    document: str
    chapter: Optional[str] = None
    chapter_title: Optional[str] = None
    section: Optional[str] = None
    start_page: int
    end_page: int
    sequence: int = -1
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
    source: str = "hybrid"  # "bm25" | "vector" | "hybrid"


class Source(BaseModel):
    chunk_id: str
    chapter: Optional[str] = None
    chapter_title: Optional[str] = None
    section: Optional[str] = None
    page: int  # = start_page, the field name callers expect
    start_page: int
    end_page: int
    label: str  # human readable e.g. "Chapter III — Page 27"
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    sufficient_evidence: bool
    standalone_query: Optional[str] = None
    retrieved_count: int
    debug: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    chroma_ready: bool
    bm25_ready: bool
    ollama_ready: bool
    indexed_chunks: int
