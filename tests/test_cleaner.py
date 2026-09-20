from app.ingestion.cleaner import (
    _is_artifact_paragraph,
    clean_page_paragraphs,
    detect_boilerplate_lines,
)
from app.ingestion.pdf_parser import ParsedPage


def test_artifact_paragraph_detection():
    assert _is_artifact_paragraph("xviii")
    assert _is_artifact_paragraph("IV")
    assert _is_artifact_paragraph("42")
    # real sentence content must never be treated as an artifact
    assert not _is_artifact_paragraph("Muhammad went to Mecca in the year 622.")
    assert not _is_artifact_paragraph("This is a normal paragraph of prose text.")


def test_dehyphenation_and_line_join():
    cleaned = clean_page_paragraphs(["This is an exam-\nple of a hyphen-\nated wrapped line."], set())
    assert cleaned == ["This is an example of a hyphenated wrapped line."]


def test_artifact_paragraphs_are_dropped():
    cleaned = clean_page_paragraphs(["xxiv", "42", "A real paragraph stays.", "iv"], set())
    assert cleaned == ["A real paragraph stays."]


def test_ligatures_are_normalized():
    cleaned = clean_page_paragraphs(["The ﬁrst ﬂight was difﬁcult."], set())
    assert cleaned == ["The first flight was difficult."]


def test_boilerplate_header_footer_detection_and_removal():
    pages = [
        ParsedPage(page=i, raw_text="", paragraphs=["RUNNING TITLE", f"Body text page {i}.", "RUNNING TITLE"])
        for i in range(1, 8)
    ]
    boilerplate = detect_boilerplate_lines(pages, min_page_fraction=0.15)
    assert "running title" in boilerplate

    cleaned = clean_page_paragraphs(pages[0].paragraphs, boilerplate)
    assert cleaned == ["Body text page 1."]
