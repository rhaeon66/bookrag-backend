"""Persistent ChromaDB vector store wrapper."""
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np

from app.config import get_settings
from app.models.schemas import ChunkRecord


@lru_cache
def get_client():
    import chromadb

    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection():
    settings = get_settings()
    client = get_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    settings = get_settings()
    client = get_client()
    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass
    return get_collection()


def add_chunks(chunks: Sequence[ChunkRecord], embeddings: np.ndarray, batch_size: int = 256) -> None:
    if not chunks:
        return
    collection = get_collection()
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]
    vectors = embeddings.tolist()

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=vectors[start:end],
        )


def _build_where(filters: Optional[Dict]) -> Optional[Dict]:
    if not filters:
        return None
    clauses = []
    chapter = filters.get("chapter")
    page_min = filters.get("page_min")
    page_max = filters.get("page_max")
    if chapter:
        clauses.append({"chapter": {"$eq": chapter}})
    if page_min is not None:
        clauses.append({"end_page": {"$gte": page_min}})
    if page_max is not None:
        clauses.append({"start_page": {"$lte": page_max}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def query(query_embedding: np.ndarray, top_k: int, filters: Optional[Dict] = None) -> List[dict]:
    """Return top_k semantic matches as [{chunk_id, text, metadata, score}, ...].

    Score is cosine similarity (1 - cosine distance), higher is better.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []
    where = _build_where(filters)
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    if not result["ids"] or not result["ids"][0]:
        return []
    out = []
    for chunk_id, text, metadata, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        out.append({
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
            "score": 1.0 - distance,
        })
    return out


def get_by_ids(chunk_ids: Sequence[str]) -> Dict[str, dict]:
    if not chunk_ids:
        return {}
    collection = get_collection()
    result = collection.get(ids=list(chunk_ids), include=["documents", "metadatas"])
    out = {}
    for chunk_id, text, metadata in zip(result["ids"], result["documents"], result["metadatas"]):
        out[chunk_id] = {"chunk_id": chunk_id, "text": text, "metadata": metadata}
    return out


def count() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0
