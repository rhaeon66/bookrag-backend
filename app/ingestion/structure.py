"""Chapter/Section detection + page record assembly (Metadata Extraction stage).

Uses the PDF's own embedded outline (bookmarks/table of contents) as the
source of truth for chapter and section boundaries — this is structure the
author/publisher actually put in the document, so metadata stays accurate to
the PDF instead of being guessed from font heuristics. When a PDF has no
outline, every page simply gets chapter/section = None rather than an
invented value.
"""
import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from app.ingestion.cleaner import clean_pages
from app.ingestion.pdf_parser import ParsedPage, get_toc
from app.models.schemas import PageRecord

_CHAPTER_NUM_RE = re.compile(r"^chapter\s+(\d+)\s*[:.\-]?\s*(.*)$", re.IGNORECASE)
_LEADING_NUMBER_RE = re.compile(r"^\d+[.)]\s*")
_TRAILING_DATE_PAREN_RE = re.compile(r"\s*\([^()]*\d[^()]*\)\s*$")

_ROMAN_NUMERALS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def _int_to_roman(number: int) -> str:
    result = []
    for value, symbol in _ROMAN_NUMERALS:
        count, number = divmod(number, value)
        result.append(symbol * count)
    return "".join(result)


def _normalize_chapter_title(raw_title: str) -> Tuple[str, Optional[str]]:
    """Return (chapter_label, chapter_title) from a level-1 TOC entry.

    "Chapter 3: Life of Prophet Muhammad and the Birth of Jihad" ->
    ("Chapter III", "Life of Prophet Muhammad and the Birth of Jihad")
    Non-chapter top-level entries (e.g. "Bibliography", "Endnotes") are kept
    verbatim as the chapter label with no separate title.
    """
    match = _CHAPTER_NUM_RE.match(raw_title.strip())
    if match:
        number = int(match.group(1))
        title = match.group(2).strip() or None
        return f"Chapter {_int_to_roman(number)}", title
    return raw_title.strip(), None


def _normalize_section_title(raw_title: str) -> str:
    title = _LEADING_NUMBER_RE.sub("", raw_title.strip())
    title = _TRAILING_DATE_PAREN_RE.sub("", title)
    return title.strip()


@dataclass
class _Marker:
    page: int
    label: str
    extra: Optional[str] = None


class StructureIndex:
    """Maps an absolute page number to (chapter, chapter_title, section)."""

    def __init__(self, toc: Sequence[list]):
        chapters: List[_Marker] = []
        sections: List[_Marker] = []
        for level, title, page in toc:
            if not title or page is None or page < 1:
                continue
            if level == 1:
                label, extra = _normalize_chapter_title(title)
                chapters.append(_Marker(page=page, label=label, extra=extra))
            else:
                sections.append(_Marker(page=page, label=_normalize_section_title(title)))
        chapters.sort(key=lambda m: m.page)
        sections.sort(key=lambda m: m.page)
        self._chapter_pages = [m.page for m in chapters]
        self._chapters = chapters
        self._section_pages = [m.page for m in sections]
        self._sections = sections

    def lookup(self, page: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        chapter = chapter_title = section = None

        idx = bisect_right(self._chapter_pages, page) - 1
        chapter_start_page = None
        if idx >= 0:
            marker = self._chapters[idx]
            chapter, chapter_title, chapter_start_page = marker.label, marker.extra, marker.page

        sidx = bisect_right(self._section_pages, page) - 1
        if sidx >= 0:
            section_marker = self._sections[sidx]
            # Only attach a section if it belongs to the current chapter
            # (i.e. it started at/after the current chapter began).
            if chapter_start_page is None or section_marker.page >= chapter_start_page:
                section = section_marker.label

        return chapter, chapter_title, section


def build_structure_index(pdf_path: str) -> StructureIndex:
    return StructureIndex(get_toc(pdf_path))


def build_page_records(document_name: str, pages: Sequence[ParsedPage], structure: StructureIndex) -> List[PageRecord]:
    """Combine parsing + cleaning + chapter/section detection into PageRecords."""
    records: List[PageRecord] = []
    for page, _cleaned_paragraphs, cleaned_text in clean_pages(pages):
        chapter, chapter_title, section = structure.lookup(page.page)
        records.append(
            PageRecord(
                document=document_name,
                page=page.page,
                raw_text=page.raw_text,
                cleaned_text=cleaned_text,
                chapter=chapter,
                chapter_title=chapter_title,
                section=section,
            )
        )
    return records
