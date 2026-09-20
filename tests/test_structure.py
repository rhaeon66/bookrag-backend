from app.ingestion.structure import StructureIndex, _int_to_roman, _normalize_chapter_title, _normalize_section_title


def test_int_to_roman():
    assert _int_to_roman(1) == "I"
    assert _int_to_roman(3) == "III"
    assert _int_to_roman(4) == "IV"
    assert _int_to_roman(9) == "IX"
    assert _int_to_roman(2024) == "MMXXIV"


def test_normalize_chapter_title_extracts_number_and_title():
    label, title = _normalize_chapter_title("Chapter 3: Life of Prophet Muhammad and the Birth of Jihad")
    assert label == "Chapter III"
    assert title == "Life of Prophet Muhammad and the Birth of Jihad"


def test_normalize_chapter_title_passthrough_for_non_chapter_entries():
    label, title = _normalize_chapter_title("Bibliography")
    assert label == "Bibliography"
    assert title is None


def test_normalize_section_title_strips_leading_number_and_trailing_date():
    assert _normalize_section_title("2. Prophetic Mission in Mecca (610–622)") == "Prophetic Mission in Mecca"
    # a non-date parenthetical must be preserved
    assert _normalize_section_title("11. The Battle of the Ditch (Trench)") == "The Battle of the Ditch (Trench)"


def test_structure_index_lookup_matches_toc(tiny_pdf_path):
    from app.ingestion.pdf_parser import get_toc

    index = StructureIndex(get_toc(tiny_pdf_path))

    chapter, chapter_title, section = index.lookup(1)
    assert chapter == "Chapter I"
    assert chapter_title == "Animals"
    assert section == "About Aardvarks"

    chapter, _, section = index.lookup(2)
    assert chapter == "Chapter I"
    assert section == "About Beavers"  # trailing date parenthetical stripped

    chapter, _, section = index.lookup(3)
    assert chapter == "Chapter II"
    assert section is None


def test_structure_index_with_no_toc_leaves_fields_none():
    index = StructureIndex([])
    chapter, chapter_title, section = index.lookup(5)
    assert chapter is None
    assert chapter_title is None
    assert section is None
