"""Embedding pipeline: local sentence-transformers model, normalized vectors,
open-source tokenizer for token counting, and a disk cache so re-running
ingestion doesn't recompute embeddings for unchanged text.

Nothing here ever calls an external/paid API — the model runs locally on
CPU or GPU per EMBEDDING_DEVICE.
"""
import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np

from app.config import get_settings


@lru_cache
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    return SentenceTransformer(settings.embedding_model, device=settings.embedding_device)


@lru_cache
def get_tokenizer():
    """Tokenizer matching the configured embedding model (open-source, local)."""
    from transformers import AutoTokenizer

    settings = get_settings()
    return AutoTokenizer.from_pretrained(settings.embedding_model)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _cache_path(text: str, model_name: str) -> Path:
    settings = get_settings()
    cache_dir = Path(settings.embedding_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{model_name}::{text}".encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.npy"


def _embed_with_cache(texts: List[str], use_cache: bool) -> np.ndarray:
    settings = get_settings()
    model = get_embedding_model()
    model_name = settings.embedding_model

    if not use_cache:
        vectors = model.encode(texts, batch_size=settings.embedding_batch_size, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    vectors: List[np.ndarray] = [None] * len(texts)  # type: ignore
    to_compute: List[int] = []
    for i, text in enumerate(texts):
        path = _cache_path(text, model_name)
        if path.exists():
            vectors[i] = np.load(path)
        else:
            to_compute.append(i)

    if to_compute:
        batch = [texts[i] for i in to_compute]
        computed = model.encode(batch, batch_size=settings.embedding_batch_size, normalize_embeddings=True)
        for idx, vector in zip(to_compute, computed):
            vector = np.asarray(vector, dtype=np.float32)
            vectors[idx] = vector
            np.save(_cache_path(texts[idx], model_name), vector)

    return np.vstack(vectors)


def embed_documents(texts: List[str], use_cache: bool = True) -> np.ndarray:
    """Embed a batch of document/chunk texts. Returns L2-normalized vectors."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return _embed_with_cache(texts, use_cache=use_cache)


def embed_query(text: str) -> np.ndarray:
    """Embed a single user query. Applies the BGE query instruction prefix."""
    settings = get_settings()
    prefixed = f"{settings.embedding_query_instruction}{text}"
    model = get_embedding_model()
    vector = model.encode([prefixed], normalize_embeddings=True)[0]
    return np.asarray(vector, dtype=np.float32)
