from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.structure import build_page_records, build_structure_index


def test_build_page_records_assembles_full_metadata(tiny_pdf_path):
    pages = parse_pdf(tiny_pdf_path)
    structure = build_structure_index(tiny_pdf_path)
    records = build_page_records("Tiny Test Book", pages, structure)

    assert len(records) == 3
    r1, r2, r3 = records

    assert r1.document == "Tiny Test Book"
    assert r1.page == 1
    assert r1.chapter == "Chapter I"
    assert r1.section == "About Aardvarks"
    assert "aardvark" in r1.cleaned_text.lower()
    assert r1.raw_text  # raw text preserved alongside cleaned text

    assert r2.chapter == "Chapter I"
    assert r2.section == "About Beavers"

    assert r3.chapter == "Chapter II"
    assert r3.section is None  # chapter 2 has no subsection in the TOC -> left null, not invented
