"""Ingestion endpoint: run the PDF -> chunks -> embeddings -> Chroma/BM25 pipeline."""
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import get_settings
from app.models.schemas import IngestResponse
from app.services.rag_pipeline import ingest_book

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
def run_ingest(file: Optional[UploadFile] = None) -> IngestResponse:
    """Ingest a book PDF.

    If a file is uploaded, it is saved under data/raw/ and ingested. Otherwise
    the PDF at BOOK_PDF_PATH (see .env) is (re-)ingested. This is synchronous
    and can take a while for a large book — the response is only returned
    once indexing has finished.
    """
    settings = get_settings()
    pdf_path = settings.book_pdf_path

    if file is not None:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        raw_dir = Path(settings.data_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / file.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        pdf_path = str(dest)

    if not Path(pdf_path).exists():
        raise HTTPException(status_code=404, detail=f"PDF not found at {pdf_path}")

    try:
        return ingest_book(pdf_path=pdf_path, settings=settings)
    except Exception as exc:  # surfaced to the caller instead of a bare 500
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
