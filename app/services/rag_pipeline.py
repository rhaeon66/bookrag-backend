"""End-to-end RAG orchestration: ingestion pipeline and query pipeline.

Also owns the small on-disk ingestion manifest (data/processed/manifest.json)
that powers the /api/stats, /api/documents and /api/chapters endpoints
without having to rescan the whole BM25/Chroma index on every request.
"""
import json
import logging
import time
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.generation import llm, prompts
from app.generation.context import build_context
from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.structure import build_page_records, build_structure_index
from app.models.schemas import (
    ChapterInfo,
    ChatTurn,
    ChunkRecord,
    IngestResponse,
    QueryFilters,
    QueryRequest,
    QueryResponse,
)
from app.retrieval import bm25 as bm25_module
from app.retrieval import vector_store
from app.retrieval.embeddings import embed_documents
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank as rerank_chunks

logger = logging.getLogger("bookrag")


# ---------------------------------------------------------------------------
# Ingestion manifest (backs /api/stats, /api/documents, /api/chapters)
# ---------------------------------------------------------------------------

def _build_chapter_index(chunks: List[ChunkRecord]) -> List[ChapterInfo]:
    chapters: List[ChapterInfo] = []
    for chapter, group_iter in groupby(chunks, key=lambda c: c.chapter):
        group = list(group_iter)
        sections: List[str] = []
        for c in group:
            if c.section and c.section not in sections:
                sections.append(c.section)
        chapters.append(
            ChapterInfo(
                chapter=chapter or "(untitled)",
                chapter_title=next((c.chapter_title for c in group if c.chapter_title), None),
                start_page=min(c.start_page for c in group),
                end_page=max(c.end_page for c in group),
                sections=sections,
                chunk_count=len(group),
            )
        )
    return chapters


def _save_manifest(settings: Settings, total_pages: int, chunks: List[ChunkRecord]) -> dict:
    chapters = _build_chapter_index(chunks)
    manifest = {
        "document": settings.book_title,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "avg_chunk_tokens": round(sum(c.token_count for c in chunks) / len(chunks), 1) if chunks else 0.0,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "chapters": [c.model_dump() for c in chapters],
    }
    path = Path(settings.manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def get_manifest(settings: Optional[Settings] = None) -> Optional[dict]:
    settings = settings or get_settings()
    path = Path(settings.manifest_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_book(pdf_path: Optional[str] = None, settings: Optional[Settings] = None) -> IngestResponse:
    settings = settings or get_settings()
    pdf_path = pdf_path or settings.book_pdf_path
    started = time.time()
    logger.info("ingest: starting for %s", pdf_path)

    parsed_pages = parse_pdf(pdf_path)
    structure = build_structure_index(pdf_path)
    page_records = build_page_records(settings.book_title, parsed_pages, structure)
    logger.debug("ingest: parsed %d pages", len(page_records))

    chunks = chunk_pages(settings.book_title, page_records, settings)
    logger.debug("ingest: created %d chunks", len(chunks))

    embeddings = embed_documents([c.text for c in chunks])

    vector_store.reset_collection()
    vector_store.add_chunks(chunks, embeddings)
    bm25_module.build_and_save(chunks)

    _save_manifest(settings, len(page_records), chunks)

    chapters_detected = len({c.chapter for c in chunks if c.chapter})
    sections_detected = len({(c.chapter, c.section) for c in chunks if c.section})
    duration = round(time.time() - started, 2)
    logger.info("ingest: done in %.2fs (%d chunks, %d chapters)", duration, len(chunks), chapters_detected)

    return IngestResponse(
        status="ok",
        document=settings.book_title,
        pages_processed=len(page_records),
        chapters_detected=chapters_detected,
        sections_detected=sections_detected,
        chunks_created=len(chunks),
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------

def _normalize_query(question: str) -> str:
    return " ".join(question.strip().split())


def _resolve_standalone_query(question: str, chat_history: List[ChatTurn], settings: Settings) -> Optional[str]:
    """Rewrite a follow-up question into a standalone query using chat history.

    Only runs when there IS history and rewriting is enabled; the rewritten
    query is used purely to steer retrieval — it never overrides book
    evidence, and the original question is still what gets answered.
    """
    if not settings.query_rewrite_enabled or not chat_history:
        return None
    history_text = "\n".join(f"{turn.role}: {turn.content}" for turn in chat_history[-6:])
    prompt = prompts.build_query_rewrite_prompt(history_text, question)
    try:
        rewritten = llm.generate(prompts.QUERY_REWRITE_SYSTEM_PROMPT, prompt, settings)
        rewritten = rewritten.strip().strip('"')
        return rewritten or None
    except Exception:
        logger.warning("query rewrite failed, falling back to the original question", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Query / answer
# ---------------------------------------------------------------------------

def answer_question(request: QueryRequest, settings: Optional[Settings] = None) -> QueryResponse:
    settings = settings or get_settings()
    debug_enabled = settings.debug_mode if request.debug is None else request.debug
    trace: Optional[Dict[str, Any]] = {} if debug_enabled else None
    timings: Dict[str, float] = {}
    total_started = time.perf_counter()

    normalized_question = _normalize_query(request.query)

    t0 = time.perf_counter()
    standalone_query = _resolve_standalone_query(normalized_question, request.chat_history, settings)
    timings["query_rewrite_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    retrieval_query = standalone_query or normalized_question

    filters = request.filters or QueryFilters()
    hybrid_trace: Optional[dict] = {} if debug_enabled else None
    t0 = time.perf_counter()
    candidates = hybrid_search(retrieval_query, filters, settings, trace=hybrid_trace)
    timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    rerank_enabled = settings.rerank_enabled if request.rerank is None else request.rerank
    t0 = time.perf_counter()
    if rerank_enabled and candidates:
        candidates = rerank_chunks(retrieval_query, candidates, settings, top_k=request.top_k)
    else:
        candidates = candidates[: (request.top_k or settings.rerank_top_k)]
    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    if trace is not None:
        trace.update(
            original_query=normalized_question,
            rewritten_query=standalone_query,
            rerank_enabled=rerank_enabled,
            **(hybrid_trace or {}),
            selected_chunks=[c.chunk_id for c in candidates],
        )

    def _insufficient_response(retrieved_count: int) -> QueryResponse:
        timings["total_ms"] = round((time.perf_counter() - total_started) * 1000, 1)
        if trace is not None:
            trace["timings_ms"] = timings
        return QueryResponse(
            answer=settings.insufficient_evidence_message,
            sources=[],
            sufficient_evidence=False,
            standalone_query=standalone_query,
            retrieved_count=retrieved_count,
            debug=trace,
        )

    if not candidates:
        logger.info("query: no candidates retrieved for %r", normalized_question)
        return _insufficient_response(0)

    context_block, sources = build_context(candidates, settings, max_chunks=request.top_k)

    if not sources:
        # Every candidate was filtered out (e.g. by CONTEXT_MIN_RERANK_SCORE)
        # -- nothing usable is left, so skip the LLM call entirely rather
        # than prompting it with empty context.
        logger.info("query: all candidates filtered out by context selection for %r", normalized_question)
        return _insufficient_response(len(candidates))

    system_prompt = prompts.build_system_prompt(settings.book_title, settings.insufficient_evidence_message)
    user_prompt = prompts.build_user_prompt(normalized_question, context_block)

    t0 = time.perf_counter()
    answer = llm.generate(system_prompt, user_prompt, settings)
    timings["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    timings["total_ms"] = round((time.perf_counter() - total_started) * 1000, 1)

    sufficient = settings.insufficient_evidence_message.strip() not in answer

    if trace is not None:
        trace["timings_ms"] = timings

    logger.info(
        "query: %r -> %d candidates, sufficient=%s, total=%.0fms",
        normalized_question, len(candidates), sufficient, timings["total_ms"],
    )

    return QueryResponse(
        answer=answer,
        sources=sources if sufficient else [],
        sufficient_evidence=sufficient,
        standalone_query=standalone_query,
        retrieved_count=len(candidates),
        debug=trace,
    )
