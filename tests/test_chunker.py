from app.ingestion.chunker import chunk_pages
from app.models.schemas import PageRecord


def _para(n_words: int, word: str) -> str:
    return " ".join([word] * n_words)


def test_chunk_pages_respects_chapter_and_section_boundaries(isolated_settings):
    pages = [
        PageRecord(
            document="Doc",
            page=1,
            raw_text="",
            cleaned_text=_para(30, "alpha"),
            chapter="Chapter I",
            chapter_title="First",
            section="Intro",
        ),
        PageRecord(
            document="Doc",
            page=2,
            raw_text="",
            cleaned_text=_para(30, "beta"),
            chapter="Chapter I",
            chapter_title="First",
            section="Middle",
        ),
        PageRecord(
            document="Doc",
            page=3,
            raw_text="",
            cleaned_text=_para(30, "gamma"),
            chapter="Chapter II",
            chapter_title="Second",
            section=None,
        ),
    ]

    chunks = chunk_pages("Doc", pages, isolated_settings)

    assert chunks, "expected at least one chunk"
    # No chunk mixes text from two different sections/chapters.
    for chunk in chunks:
        assert not ("alpha" in chunk.text and "beta" in chunk.text)
        assert not ("beta" in chunk.text and "gamma" in chunk.text)

    chapter_i_chunks = [c for c in chunks if c.chapter == "Chapter I"]
    chapter_ii_chunks = [c for c in chunks if c.chapter == "Chapter II"]
    assert chapter_i_chunks
    assert chapter_ii_chunks
    # chunk_id numbers chapters in reading order starting at ch01
    assert all(c.chunk_id.startswith("ch01_") for c in chapter_i_chunks)
    assert all(c.chunk_id.startswith("ch02_") for c in chapter_ii_chunks)


def test_chunk_pages_can_span_pages_within_same_section(isolated_settings):
    # A single section's content split across two pages should be able to
    # merge into one chunk (start_page != end_page) when it fits the budget.
    pages = [
        PageRecord(document="Doc", page=10, raw_text="", cleaned_text="short line one.", chapter="Chapter I", section="S"),
        PageRecord(document="Doc", page=11, raw_text="", cleaned_text="short line two.", chapter="Chapter I", section="S"),
    ]
    chunks = chunk_pages("Doc", pages, isolated_settings)
    assert len(chunks) == 1
    assert chunks[0].start_page == 10
    assert chunks[0].end_page == 11


def test_chunk_token_budget_is_respected(isolated_settings):
    # Realistic punctuated prose (unlike unpunctuated word-soup, real sentences
    # give the chunker split points to pack against the token budget).
    long_text = " ".join(f"This is sentence number {i} in the passage." for i in range(60))
    pages = [
        PageRecord(document="Doc", page=1, raw_text="", cleaned_text=long_text, chapter="Chapter I", section="S"),
    ]
    chunks = chunk_pages("Doc", pages, isolated_settings)
    assert len(chunks) > 1
    # Every chunk but the last must respect the token budget; the last chunk
    # may absorb a too-small trailing fragment (see chunk_min_size merging)
    # so it's allowed some slack instead of being left as a tiny orphan.
    for chunk in chunks[:-1]:
        assert chunk.token_count <= isolated_settings.chunk_size * 1.1
