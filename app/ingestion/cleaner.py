"""Text cleaning (Text Cleaning stage).

Cleans paragraph-segmented page text without altering the author's wording:
    - repeated running headers/footers (detected corpus-wide)
    - isolated page-number artifacts
    - isolated endnote/footnote reference markers that PDF extraction turns
      into standalone "paragraphs" (this book uses roman-numeral endnote
      markers rendered as separate text blocks by the PDF's layout engine)
    - broken/hyphenated words caused by line-wrap extraction
    - unicode ligatures (ﬁ, ﬂ, ...) and redundant whitespace

The cleaner never rewrites, paraphrases or removes substantive sentences —
only layout/extraction artifacts.
"""
import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

from app.ingestion.pdf_parser import ParsedPage

_ROMAN_NUMERAL_RE = re.compile(
    r"^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)
_DIGIT_ONLY_RE = re.compile(r"^\d{1,4}$")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_WS_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_RE = re.compile(r"\n{2,}")


def _is_artifact_paragraph(paragraph: str) -> bool:
    """A standalone block that is purely a roman numeral or bare page number.

    Real body-text blocks are never a single bare roman numeral or a bare
    number by themselves (they always contain surrounding prose), so this is
    safe to strip: it only removes footnote-marker/page-number extraction
    noise, never sentence content.
    """
    stripped = paragraph.strip()
    if not stripped or "\n" in stripped:
        return False
    if len(stripped) > 12:
        return False
    if _DIGIT_ONLY_RE.match(stripped):
        return True
    if _ROMAN_NUMERAL_RE.match(stripped):
        return True
    return False


def _normalize_unicode(text: str) -> str:
    # Fixes ligatures (ﬁ -> fi, ﬂ -> fl, etc.) without changing wording.
    return unicodedata.normalize("NFKC", text)


def _dehyphenate_and_join(paragraph: str) -> str:
    """Join the visual lines of a paragraph block into flowing text.

    Rejoins words that were split across a line-wrap hyphen (`exam-\nple` ->
    `example`); lines without a hyphen break are simply joined with a space.
    """
    joined = _HYPHEN_BREAK_RE.sub(r"\1\2", paragraph)
    joined = joined.replace("\n", " ")
    joined = _MULTI_WS_RE.sub(" ", joined).strip()
    return joined


def detect_boilerplate_lines(pages: Sequence[ParsedPage], min_page_fraction: float = 0.15) -> set:
    """Detect running headers/footers repeated across many pages.

    Looks at the first and last paragraph of every page; any normalized text
    that recurs as a first-or-last paragraph on more than `min_page_fraction`
    of pages is treated as boilerplate (e.g. a repeated book/chapter title or
    footer) and stripped wherever it occurs.
    """
    if not pages:
        return set()
    counter: Counter = Counter()
    for page in pages:
        if not page.paragraphs:
            continue
        edge_candidates = {page.paragraphs[0].strip().lower(), page.paragraphs[-1].strip().lower()}
        for candidate in edge_candidates:
            if candidate and len(candidate) < 120:
                counter[candidate] += 1
    threshold = max(4, int(len(pages) * min_page_fraction))
    return {text for text, count in counter.items() if count >= threshold}


def clean_page_paragraphs(paragraphs: Iterable[str], boilerplate_lines: set) -> List[str]:
    cleaned: List[str] = []
    for paragraph in paragraphs:
        normalized = _normalize_unicode(paragraph)
        if _is_artifact_paragraph(normalized):
            continue
        if normalized.strip().lower() in boilerplate_lines:
            continue
        text = _dehyphenate_and_join(normalized)
        if text:
            cleaned.append(text)
    return cleaned


def clean_pages(pages: Sequence[ParsedPage]) -> List[Tuple[ParsedPage, List[str], str]]:
    """Clean every page, returning (page, cleaned_paragraphs, cleaned_text) tuples."""
    boilerplate_lines = detect_boilerplate_lines(pages)
    results = []
    for page in pages:
        cleaned_paragraphs = clean_page_paragraphs(page.paragraphs, boilerplate_lines)
        cleaned_text = _MULTI_BLANK_RE.sub("\n\n", "\n\n".join(cleaned_paragraphs)).strip()
        results.append((page, cleaned_paragraphs, cleaned_text))
    return results
