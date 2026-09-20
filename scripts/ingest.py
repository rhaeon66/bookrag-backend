#!/usr/bin/env python3
"""CLI: ingest a book PDF into ChromaDB + the BM25 index without starting the API.

Usage:
    python scripts/ingest.py [--pdf PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.services.rag_pipeline import ingest_book  # noqa: E402


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest a book PDF into BookRAG.")
    parser.add_argument("--pdf", default=settings.book_pdf_path, help="Path to the source PDF.")
    args = parser.parse_args()

    print(f"Ingesting: {args.pdf}")
    result = ingest_book(pdf_path=args.pdf, settings=settings)
    print(f"Status:            {result.status}")
    print(f"Document:          {result.document}")
    print(f"Pages processed:   {result.pages_processed}")
    print(f"Chapters detected: {result.chapters_detected}")
    print(f"Sections detected: {result.sections_detected}")
    print(f"Chunks created:    {result.chunks_created}")
    print(f"Duration (s):      {result.duration_seconds}")


if __name__ == "__main__":
    main()
