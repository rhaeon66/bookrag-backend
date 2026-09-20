"""Configurable, paragraph-aware chunking (Chunking stage).

Chunks are built by greedily packing whole paragraphs up to a configurable
token budget, with a configurable token overlap carried into the next chunk.
Paragraphs are only split mid-paragraph when a single paragraph exceeds the
chunk size on its own. Chunk boundaries are never allowed to cross a
chapter/section boundary, so every chunk's metadata is unambiguous; a chunk
*may* span multiple pages of the same chapter/section, in which case
start_page/end_page differ.
"""
import re
from itertools import groupby
from typing import List, Optional, Sequence, Tuple

from app.config import Settings, get_settings
from app.models.schemas import ChunkRecord, PageRecord
from app.retrieval.embeddings import count_tokens

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CHAPTER_ROMAN_RE = re.compile(r"^Chapter\s+([IVXLCDM]+)$", re.IGNORECASE)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

Atom = Tuple[int, Optional[str], Optional[str], Optional[str], str]  # page, chapter, chapter_title, section, text


def _roman_to_int(roman: str) -> int:
    roman = roman.upper()
    total = 0
    for i, char in enumerate(roman):
        value = _ROMAN_VALUES[char]
        if i + 1 < len(roman) and _ROMAN_VALUES[roman[i + 1]] > value:
            total -= value
        else:
            total += value
    return total


def _chapter_number(chapter: Optional[str]) -> int:
    """Numeric id used in chunk_id: the actual chapter number when detected
    (e.g. "Chapter III" -> 3), otherwise 0 for front/back matter (Title,
    Table of Contents, Bibliography, Endnotes, ...)."""
    if not chapter:
        return 0
    match = _CHAPTER_ROMAN_RE.match(chapter.strip())
    if not match:
        return 0
    try:
        return _roman_to_int(match.group(1))
    except KeyError:
        return 0


def _flatten_atoms(pages: Sequence[PageRecord]) -> List[Atom]:
    atoms: List[Atom] = []
    for page in pages:
        for paragraph in page.cleaned_text.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                atoms.append((page.page, page.chapter, page.chapter_title, page.section, paragraph))
    return atoms


def _split_oversized_paragraph(text: str, max_tokens: int) -> List[str]:
    """Split a single paragraph that exceeds max_tokens into sentence-packed pieces."""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    pieces: List[str] = []
    buffer = ""
    buffer_tokens = 0
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if buffer and buffer_tokens + sentence_tokens > max_tokens:
            pieces.append(buffer.strip())
            buffer, buffer_tokens = "", 0
        buffer = f"{buffer} {sentence}".strip()
        buffer_tokens += sentence_tokens
    if buffer:
        pieces.append(buffer.strip())
    return pieces or [text]


def _take_trailing_overlap(
    paragraphs: List[Tuple[int, str]], overlap_tokens: int, chunk_size: int
) -> List[Tuple[int, str]]:
    """Return the trailing paragraphs of a buffer worth ~overlap_tokens tokens.

    Normally includes at least one trailing paragraph even if it alone is
    bigger than overlap_tokens, so a boundary always gets *some* continuity.
    But a lone paragraph/piece that is itself already a large fraction of
    chunk_size (this happens with the near-chunk_size pieces produced by
    _split_oversized_paragraph) is skipped instead of force-included —
    otherwise that single "overlap" atom plus the next real atom would
    immediately blow the budget again, and every following chunk would
    silently balloon to ~2x chunk_size.
    """
    if overlap_tokens <= 0:
        return []
    kept: List[Tuple[int, str]] = []
    total = 0
    for page, text in reversed(paragraphs):
        tokens = count_tokens(text)
        if kept and total + tokens > overlap_tokens:
            break
        if not kept and tokens > chunk_size * 0.5:
            break
        kept.append((page, text))
        total += tokens
    kept.reverse()
    return kept


def _chunk_group(
    chapter: Optional[str],
    chapter_title: Optional[str],
    section: Optional[str],
    atoms: List[Tuple[int, str]],
    settings: Settings,
) -> List[Tuple[int, int, str, int]]:
    """Chunk a single (chapter, section) run of (page, paragraph) pairs.

    Returns a list of (start_page, end_page, text, token_count).
    """
    # Expand any paragraph that alone exceeds the chunk size.
    expanded: List[Tuple[int, str]] = []
    for page, text in atoms:
        tokens = count_tokens(text)
        if tokens > settings.chunk_size:
            for piece in _split_oversized_paragraph(text, settings.chunk_size):
                expanded.append((page, piece))
        else:
            expanded.append((page, text))

    chunks: List[Tuple[int, int, str, int]] = []
    buffer: List[Tuple[int, str]] = []
    buffer_tokens = 0

    def flush():
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        start_page = buffer[0][0]
        end_page = buffer[-1][0]
        text = "\n\n".join(t for _, t in buffer)
        chunks.append((start_page, end_page, text, count_tokens(text)))

    for page, text in expanded:
        tokens = count_tokens(text)
        if buffer and buffer_tokens + tokens > settings.chunk_size:
            flush()
            overlap = _take_trailing_overlap(buffer, settings.chunk_overlap, settings.chunk_size)
            buffer = list(overlap)
            buffer_tokens = sum(count_tokens(t) for _, t in buffer)
        buffer.append((page, text))
        buffer_tokens += tokens
    flush()

    # Merge a too-small trailing fragment into the previous chunk instead of
    # leaving an orphan sliver at a section boundary.
    if len(chunks) >= 2 and chunks[-1][3] < settings.chunk_min_size:
        last = chunks.pop()
        prev = chunks.pop()
        merged_text = f"{prev[2]}\n\n{last[2]}"
        chunks.append((prev[0], last[1], merged_text, count_tokens(merged_text)))

    return chunks


def chunk_pages(document: str, pages: Sequence[PageRecord], settings: Optional[Settings] = None) -> List[ChunkRecord]:
    settings = settings or get_settings()
    atoms = _flatten_atoms(pages)

    records: List[ChunkRecord] = []
    last_chapter_key = object()
    chapter_seq = 0
    sequence = 0

    for (chapter, chapter_title, section), group_iter in groupby(atoms, key=lambda a: (a[1], a[2], a[3])):
        group_atoms = [(page, text) for page, _c, _ct, _s, text in group_iter]
        if chapter != last_chapter_key:
            chapter_counter = _chapter_number(chapter)
            chapter_seq = 0
            last_chapter_key = chapter
        for start_page, end_page, text, token_count in _chunk_group(chapter, chapter_title, section, group_atoms, settings):
            chapter_seq += 1
            sequence += 1
            chunk_id = f"ch{chapter_counter:02d}_p{start_page:03d}_{chapter_seq:03d}"
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document=document,
                    text=text,
                    chapter=chapter,
                    chapter_title=chapter_title,
                    section=section,
                    start_page=start_page,
                    end_page=end_page,
                    token_count=token_count,
                    sequence=sequence,
                )
            )

    return records
