"""Shared pytest fixtures: an isolated Settings instance (temp data dirs) and
a tiny synthetic PDF so most tests don't depend on the real 549-page book.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.config import Settings


@pytest.fixture
def tiny_pdf_path(tmp_path) -> str:
    """A 3-page synthetic book with a real embedded outline (TOC):

    Chapter I: Animals (pages 1-2)
        1. About Aardvarks (page 1)
        2. About Beavers (page 2)
    Chapter II: Plants (page 3)
    """
    import pymupdf as fitz

    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 100), "The aardvark is a nocturnal mammal native to Africa. " * 3)
    p2 = doc.new_page()
    p2.insert_text((72, 100), "The beaver is a large semiaquatic rodent known for building dams. " * 3)
    p3 = doc.new_page()
    p3.insert_text((72, 100), "The cactus is a plant adapted to arid climates with minimal water. " * 3)

    doc.set_toc([
        [1, "Chapter 1: Animals", 1],
        [2, "1. About Aardvarks", 1],
        [2, "2. About Beavers (1800-1900)", 2],
        [1, "Chapter 2: Plants", 3],
    ])

    path = tmp_path / "tiny_book.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def patched_settings(isolated_settings, monkeypatch):
    """Point every module that reads the global settings singleton at
    `isolated_settings`, and reset the process-wide Chroma client / BM25
    index caches so tests never touch the real book's persisted data.

    Each module did `from app.config import get_settings`, which binds its
    own local name — patching app.config.get_settings alone would not affect
    those already-bound references, so every call site that needs isolation
    is patched explicitly here.
    """
    for module_path in [
        "app.config",
        "app.retrieval.vector_store",
        "app.retrieval.bm25",
        "app.retrieval.hybrid",
        "app.retrieval.reranker",
        "app.generation.context",
        "app.generation.llm",
        "app.services.rag_pipeline",
        "app.api.routes_ingest",
        "app.api.routes_meta",
    ]:
        monkeypatch.setattr(f"{module_path}.get_settings", lambda s=isolated_settings: s)

    from app.retrieval import bm25 as bm25_module
    from app.retrieval import vector_store

    vector_store.get_client.cache_clear()
    bm25_module._index_cache = None

    yield isolated_settings

    vector_store.get_client.cache_clear()
    bm25_module._index_cache = None


@pytest.fixture
def isolated_settings(tmp_path, tiny_pdf_path) -> Settings:
    """A Settings instance pointed entirely at a scratch directory so tests
    never touch the real book's persisted Chroma/BM25/embedding-cache data."""
    return Settings(
        book_pdf_path=tiny_pdf_path,
        book_title="Tiny Test Book",
        data_dir=str(tmp_path / "data"),
        processed_dir=str(tmp_path / "data" / "processed"),
        manifest_path=str(tmp_path / "data" / "processed" / "manifest.json"),
        chroma_persist_dir=str(tmp_path / "data" / "chroma"),
        chroma_collection_name="test_collection",
        bm25_index_path=str(tmp_path / "data" / "processed" / "bm25_index.pkl"),
        embedding_cache_dir=str(tmp_path / "data" / "processed" / "embedding_cache"),
        chunk_size=40,
        chunk_overlap=5,
        chunk_min_size=3,
        rerank_enabled=False,  # most tests don't need the cross-encoder; reranker has its own tests
    )
