"""BM25 lexical retrieval index (rank-bm25), persisted locally.

BM25 complements the embedding-based vector search for exact names, dates,
Quran/verse references, rare or specific terminology, and exact phrases that
dense embeddings can blur together.
"""
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.config import get_settings
from app.models.schemas import ChunkRecord

# Keeps colon/dot/dash-joined tokens intact (e.g. "2:190", "4:52:220", "610-622")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[:./\-–][a-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    chunk_ids: List[str]
    corpus_tokens: List[List[str]]
    chunks: Dict[str, dict]  # chunk_id -> {"text":..., "metadata": {...}}
    _bm25: object = None

    def _ensure_model(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self.corpus_tokens)
        return self._bm25

    def matching_chunk_ids(
        self, chapter: Optional[str] = None, page_min: Optional[int] = None, page_max: Optional[int] = None
    ) -> set:
        """Chunk ids matching the given metadata filters (None = no constraint)."""
        if chapter is None and page_min is None and page_max is None:
            return set(self.chunk_ids)
        matched = set()
        for chunk_id, entry in self.chunks.items():
            meta = entry["metadata"]
            if chapter is not None and meta.get("chapter") != chapter:
                continue
            if page_min is not None and meta.get("end_page", meta.get("page", 0)) < page_min:
                continue
            if page_max is not None and meta.get("start_page", meta.get("page", 0)) > page_max:
                continue
            matched.add(chunk_id)
        return matched

    def search(self, query: str, top_k: int, allowed_chunk_ids: Optional[set] = None) -> List[dict]:
        if not self.chunk_ids:
            return []
        bm25 = self._ensure_model()
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        out: List[dict] = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            chunk_id = self.chunk_ids[idx]
            if allowed_chunk_ids is not None and chunk_id not in allowed_chunk_ids:
                continue
            entry = self.chunks[chunk_id]
            out.append({
                "chunk_id": chunk_id,
                "text": entry["text"],
                "metadata": entry["metadata"],
                "score": float(scores[idx]),
            })
            if len(out) >= top_k:
                break
        return out

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "corpus_tokens": self.corpus_tokens,
                "chunks": self.chunks,
            }, f)

    @classmethod
    def load(cls, path: str) -> Optional["BM25Index"]:
        p = Path(path)
        if not p.exists():
            return None
        with open(p, "rb") as f:
            data = pickle.load(f)
        return cls(chunk_ids=data["chunk_ids"], corpus_tokens=data["corpus_tokens"], chunks=data["chunks"])

    @classmethod
    def build(cls, chunks: Sequence[ChunkRecord]) -> "BM25Index":
        chunk_ids = [c.chunk_id for c in chunks]
        corpus_tokens = [tokenize(c.text) for c in chunks]
        chunk_map = {c.chunk_id: {"text": c.text, "metadata": c.to_metadata()} for c in chunks}
        return cls(chunk_ids=chunk_ids, corpus_tokens=corpus_tokens, chunks=chunk_map)


_index_cache: Optional[BM25Index] = None


def get_bm25_index(force_reload: bool = False) -> Optional[BM25Index]:
    global _index_cache
    if _index_cache is None or force_reload:
        settings = get_settings()
        _index_cache = BM25Index.load(settings.bm25_index_path)
    return _index_cache


def build_and_save(chunks: Sequence[ChunkRecord]) -> BM25Index:
    global _index_cache
    settings = get_settings()
    index = BM25Index.build(chunks)
    index.save(settings.bm25_index_path)
    _index_cache = index
    return index
