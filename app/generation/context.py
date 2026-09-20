"""Context selection / builder (Context Selection stage).

Turns a ranked list of retrieved chunks into:
    - a token-budgeted, deduplicated context block for the LLM prompt
    - the matching [Source N] labelled citations for the API response

Responsibilities: drop exact/near-duplicate chunks, respect the context
token budget instead of dumping every candidate into the prompt, keep
higher-ranked chunks first, and optionally pull in an immediately
neighbouring chunk for continuity.
"""
import re
from typing import List, Optional, Tuple

from app.config import Settings, get_settings
from app.models.schemas import RetrievedChunk, Source
from app.retrieval import bm25 as bm25_module
from app.retrieval.embeddings import count_tokens

_WORD_RE = re.compile(r"\w+")


def _token_set(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _dedupe(chunks: List[RetrievedChunk], threshold: float) -> List[RetrievedChunk]:
    kept: List[RetrievedChunk] = []
    kept_token_sets: List[set] = []
    seen_text = set()
    for chunk in chunks:
        if chunk.text in seen_text:
            continue
        tokens = _token_set(chunk.text)
        if any(_jaccard(tokens, existing) >= threshold for existing in kept_token_sets):
            continue
        kept.append(chunk)
        kept_token_sets.append(tokens)
        seen_text.add(chunk.text)
    return kept


def _find_neighbor(chunk: RetrievedChunk, direction: int) -> Optional[RetrievedChunk]:
    """Look up the next/previous chunk in document order from the BM25 chunk store."""
    index = bm25_module.get_bm25_index()
    if index is None or chunk.sequence < 0:
        return None
    target_sequence = chunk.sequence + direction
    for chunk_id, entry in index.chunks.items():
        meta = entry["metadata"]
        if meta.get("sequence") == target_sequence and meta.get("chapter") == (chunk.chapter or ""):
            return RetrievedChunk(
                chunk_id=chunk_id,
                text=entry["text"],
                document=meta.get("document", ""),
                chapter=meta.get("chapter") or None,
                chapter_title=meta.get("chapter_title") or None,
                section=meta.get("section") or None,
                start_page=meta.get("start_page", meta.get("page", 0)),
                end_page=meta.get("end_page", meta.get("page", 0)),
                sequence=meta.get("sequence", -1),
                source="neighbor",
            )
    return None


def _citation_label(chunk: RetrievedChunk) -> str:
    page_part = f"Page {chunk.start_page}" if chunk.start_page == chunk.end_page else f"Pages {chunk.start_page}–{chunk.end_page}"
    if chunk.chapter and chunk.section:
        return f"{chunk.chapter} → Section: {chunk.section} → {page_part}"
    if chunk.chapter:
        return f"{chunk.chapter} — {page_part}"
    return page_part


def build_context(
    chunks: List[RetrievedChunk],
    settings: Optional[Settings] = None,
    max_chunks: Optional[int] = None,
) -> Tuple[str, List[Source]]:
    settings = settings or get_settings()
    chunk_limit = max_chunks or settings.context_max_chunks

    if settings.context_min_rerank_score is not None:
        # Only filters chunks that actually went through reranking; a chunk
        # with no rerank_score (reranking disabled, or a neighbor-expansion
        # add-on) is left alone rather than dropped on a score it never got.
        chunks = [
            c for c in chunks
            if c.rerank_score is None or c.rerank_score >= settings.context_min_rerank_score
        ]

    deduped = _dedupe(chunks, settings.context_dedup_threshold)

    selected: List[RetrievedChunk] = []
    token_budget = 0
    for chunk in deduped:
        if len(selected) >= chunk_limit:
            break
        tokens = count_tokens(chunk.text)
        if selected and token_budget + tokens > settings.context_max_tokens:
            continue
        selected.append(chunk)
        token_budget += tokens

    if settings.context_neighbor_expansion and selected:
        top_chunk = selected[0]
        existing_ids = {c.chunk_id for c in selected}
        for direction in (1, -1):
            if len(selected) >= chunk_limit:
                break
            neighbor = _find_neighbor(top_chunk, direction)
            if not neighbor or neighbor.chunk_id in existing_ids:
                continue
            tokens = count_tokens(neighbor.text)
            if token_budget + tokens > settings.context_max_tokens:
                continue
            selected.append(neighbor)
            existing_ids.add(neighbor.chunk_id)
            token_budget += tokens

    # Present the final context in natural reading order (by page), while the
    # earlier steps already decided *which* chunks earn a place by relevance.
    selected.sort(key=lambda c: (c.start_page, c.sequence))

    source_blocks = []
    sources: List[Source] = []
    for i, chunk in enumerate(selected, start=1):
        label = _citation_label(chunk)
        page_line = f"Page {chunk.start_page}" if chunk.start_page == chunk.end_page else f"Pages {chunk.start_page}-{chunk.end_page}"
        header_lines = [f"[Source {i}]"]
        if chunk.chapter:
            header_lines.append(chunk.chapter)
        if chunk.section:
            header_lines.append(f"Section: {chunk.section}")
        header_lines.append(page_line)
        header = "\n".join(header_lines)
        source_blocks.append(f"{header}\n{chunk.text}")

        sources.append(
            Source(
                chunk_id=chunk.chunk_id,
                chapter=chunk.chapter,
                chapter_title=chunk.chapter_title,
                section=chunk.section,
                page=chunk.start_page,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                label=label,
                snippet=chunk.text[:280] + ("..." if len(chunk.text) > 280 else ""),
            )
        )

    context_block = "\n\n".join(source_blocks)
    return context_block, sources
