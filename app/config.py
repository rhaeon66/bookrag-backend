"""Application configuration loaded from environment variables / .env file."""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _auto_device() -> str:
    """Use a GPU automatically when one is available; otherwise fall back to CPU.
    Never assumes CUDA is present — this is only a convenience default and is
    always overridable via the environment (EMBEDDING_DEVICE / RERANK_DEVICE).
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():  # Apple Silicon
            return "mps"
    except Exception:
        pass
    return "cpu"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"
    debug_mode: bool = False

    # --- Book / PDF ---
    book_pdf_path: str = str(BACKEND_DIR / "data" / "raw" / "In_GOD's_Path.pdf")
    book_title: str = "In God's Path"

    # --- Data locations ---
    data_dir: str = str(BACKEND_DIR / "data")
    processed_dir: str = str(BACKEND_DIR / "data" / "processed")
    manifest_path: str = str(BACKEND_DIR / "data" / "processed" / "manifest.json")

    # --- Chunking (approx. tokens) ---
    chunk_size: int = 500
    chunk_overlap: int = 75
    chunk_min_size: int = 60  # drop/merge fragments smaller than this many tokens

    # --- Embeddings (local sentence-transformers model, never an external API) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = Field(default_factory=_auto_device)
    embedding_batch_size: int = 32
    embedding_cache_dir: str = str(BACKEND_DIR / "data" / "processed" / "embedding_cache")
    # bge models expect a query instruction prefix for best retrieval quality
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "

    # --- Vector store (ChromaDB) ---
    chroma_persist_dir: str = str(BACKEND_DIR / "data" / "chroma")
    chroma_collection_name: str = "bookrag"

    # --- BM25 ---
    bm25_index_path: str = str(BACKEND_DIR / "data" / "processed" / "bm25_index.pkl")
    bm25_top_k: int = 15

    # --- Retrieval ---
    vector_top_k: int = 10
    hybrid_candidate_k: int = 20  # size of fused candidate list before rerank
    rrf_k: int = 60

    # --- Reranking (optional; set RERANK_ENABLED=false to skip the cross-encoder step) ---
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_device: str = Field(default_factory=_auto_device)
    rerank_top_k: int = 6

    # --- Context builder ---
    context_max_tokens: int = 2200
    context_max_chunks: int = 8
    context_neighbor_expansion: bool = True
    context_dedup_threshold: float = 0.85  # jaccard similarity above which chunks are considered duplicates
    # Optional floor on cross-encoder rerank_score: chunks scoring below this are
    # dropped instead of padding out rerank_top_k with marginal matches. Only
    # applies to chunks that were actually reranked (raw cross-encoder logit,
    # unbounded — ms-marco-MiniLM-style models put clearly relevant matches
    # well above 0). None (default) keeps today's behavior: always keep the
    # top rerank_top_k regardless of score. Has no effect when reranking is
    # disabled (those chunks carry no rerank_score to filter on).
    context_min_rerank_score: Optional[float] = None

    # --- LLM (local, via Ollama — no external/paid API key required) ---
    llm_provider: str = "ollama"  # only "ollama" is implemented; kept configurable for future providers
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_request_timeout: float = 120.0

    # --- Query processing ---
    query_rewrite_enabled: bool = True

    # --- Answer format ---
    insufficient_evidence_message: str = (
        "I couldn't find enough information in the indexed book to answer this reliably."
    )

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
