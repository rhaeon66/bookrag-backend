"""PDF parsing (Parsing + Page Detection stages).

Extracts each page of the PDF as an independent record: a plain-text
rendering of the page plus a paragraph-segmented version derived from
PyMuPDF's text blocks. Page boundaries are preserved (one record per PDF
page) — the book is never flattened into a single blob of text.
"""
from dataclasses import dataclass, field
from typing import List

import pymupdf as fitz  # PyMuPDF


@dataclass
class ParsedPage:
    page: int  # 1-indexed absolute page number, matches the PDF's own outline
    raw_text: str
    paragraphs: List[str] = field(default_factory=list)


def _extract_paragraphs(page: "fitz.Page") -> List[str]:
    """Group text spans into paragraph-level strings using PyMuPDF block layout.

    Each PDF text block is treated as one paragraph. Lines within a block are
    joined with a single space; hyphenation fixes are handled later by the
    cleaner, which operates on the same block boundaries preserved here.
    """
    page_dict = page.get_text("dict")
    paragraphs: List[str] = []
    for block in page_dict.get("blocks", []):
        if "lines" not in block:
            continue  # image or non-text block
        line_texts = []
        for line in block["lines"]:
            spans = [span.get("text", "") for span in line.get("spans", [])]
            line_text = "".join(spans).strip()
            if line_text:
                line_texts.append(line_text)
        if line_texts:
            paragraphs.append("\n".join(line_texts))
    return paragraphs


def parse_pdf(pdf_path: str) -> List[ParsedPage]:
    """Parse every page of the PDF into a ParsedPage, preserving page order."""
    doc = fitz.open(pdf_path)
    try:
        pages: List[ParsedPage] = []
        for index in range(doc.page_count):
            page = doc[index]
            raw_text = page.get_text("text")
            paragraphs = _extract_paragraphs(page)
            pages.append(ParsedPage(page=index + 1, raw_text=raw_text, paragraphs=paragraphs))
        return pages
    finally:
        doc.close()


def get_toc(pdf_path: str) -> List[list]:
    """Return the PDF's embedded outline/bookmarks as [level, title, page] entries."""
    doc = fitz.open(pdf_path)
    try:
        return doc.get_toc(simple=True)
    finally:
        doc.close()
