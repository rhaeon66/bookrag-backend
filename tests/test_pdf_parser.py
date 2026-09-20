from app.ingestion.pdf_parser import get_toc, parse_pdf


def test_parse_pdf_preserves_page_boundaries(tiny_pdf_path):
    pages = parse_pdf(tiny_pdf_path)

    assert len(pages) == 3
    assert [p.page for p in pages] == [1, 2, 3]
    assert "aardvark" in pages[0].raw_text.lower()
    assert "beaver" in pages[1].raw_text.lower()
    assert "cactus" in pages[2].raw_text.lower()
    # each page's own text should not leak into another page
    assert "beaver" not in pages[0].raw_text.lower()


def test_parse_pdf_extracts_paragraphs(tiny_pdf_path):
    pages = parse_pdf(tiny_pdf_path)
    assert all(page.paragraphs for page in pages)


def test_get_toc_reads_embedded_outline(tiny_pdf_path):
    toc = get_toc(tiny_pdf_path)
    titles = [entry[1] for entry in toc]
    assert "Chapter 1: Animals" in titles
    assert "Chapter 2: Plants" in titles
