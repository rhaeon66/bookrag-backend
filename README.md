# bookrag-backend

The retrieval/generation backend for **BookRAG** — a Retrieval-Augmented
Generation API for asking questions about a book, with answers grounded
only in retrieved content and chapter/section/page citations, e.g.:

```
Chapter III → Section: Prophetic Mission in Mecca → Page 27
```

When retrieved passages don't answer the question, the API says so instead
of guessing — it never falls back on the LLM's outside knowledge.

Everything runs locally on open-source components: no paid API keys, no
data leaving the machine. Pairs with the companion frontend at
[bookrag-frontend](https://github.com/rhaeon66/bookrag-frontend).

## Tech stack

| Layer | Choice |
|---|---|
| API | Python, FastAPI, Pydantic |
| PDF processing | PyMuPDF |
| Text processing | Python + a HuggingFace tokenizer matching the embedding model (no tiktoken) |
| Embeddings | sentence-transformers, default `BAAI/bge-small-en-v1.5` (configurable) |
| Vector database | ChromaDB (persistent, local, HNSW ANN index — see [docs/vector-search.md](docs/vector-search.md)) |
| Lexical retrieval | rank-bm25 |
| Fusion | Reciprocal Rank Fusion (BM25 + vector) |
| Reranking (optional) | sentence-transformers `CrossEncoder`, default `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Local model via Ollama, default `qwen2.5:3b` (configurable, no external API key) |

## Architecture

```
PDF → Parsing → Page Detection → Chapter/Section Detection → Text Cleaning
    → Metadata Extraction → Chunking → Embedding → ChromaDB
                                                   → BM25 Index
User Query → Query Processing → Hybrid Retrieval (BM25 + Vector) → Fusion (RRF)
    → Candidate Selection → Optional Cross-Encoder Reranking → Context Selection
    → Prompt Construction → Local LLM → Grounded Answer → Source Citations
```

Chapter/section metadata comes from the PDF's own embedded outline/bookmarks
(not guessed from font sizes), so citations are accurate to what's actually
in the document; when a page's chapter or section can't be determined, that
field is left `null` rather than invented.

## Layout

```
app/
├── main.py             # FastAPI app entrypoint
├── config.py            # Settings (env-driven, see .env.example)
├── api/                  # HTTP routes: query, ingest, health, stats/documents/chapters/config
├── ingestion/            # PDF parsing -> cleaning -> structure -> chunking
├── retrieval/            # Embeddings, vector store, BM25, hybrid search, reranking
├── generation/           # LLM client, prompts, context assembly
├── models/                # Pydantic schemas
└── services/              # RAG pipeline orchestration (ingest + query)
scripts/
├── ingest.py             # CLI: ingest the book PDF
└── evaluate.py           # CLI: run the retrieval/generation evaluation
data/
├── raw/                  # put your source PDF here (not committed)
├── processed/             # BM25 index, embedding cache, ingestion manifest (not committed)
├── chroma/                 # persistent vector store (not committed)
└── eval/                   # hand-built evaluation question set
tests/                     # pytest suite
docs/vector-search.md      # what ANN/HNSW is and why Chroma uses it
```

## Getting started

### 1. Install Ollama and pull a model

The LLM runs locally through [Ollama](https://ollama.com) — no API key,
nothing leaves your machine.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b   # small, works on ~8GB RAM / no dedicated GPU
```

The model is fully configurable via `LLM_MODEL` in `.env`.

### 2. Install and configure

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Add the book and ingest

Place your own copy of the PDF at `data/raw/<your-book>.pdf`, then set
`BOOK_PDF_PATH`/`BOOK_TITLE` in `.env` accordingly (defaults assume
`data/raw/In_GOD's_Path.pdf`).

```bash
python scripts/ingest.py
```

### 4. Run

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Configuration

All configuration is environment-driven (`app/config.py` / `.env.example`),
including the pieces most relevant to running on modest hardware:

- `EMBEDDING_MODEL` / `EMBEDDING_DEVICE` — swap the embedding model; device
  defaults to auto-detecting a GPU (CUDA or Apple MPS) and otherwise falling
  back to CPU, so nothing assumes CUDA is present.
- `RERANK_ENABLED` — the cross-encoder reranking step is fully optional;
  when `false`, the fused hybrid ranking is used as-is and the reranker
  model is never loaded.
- `CONTEXT_MIN_RERANK_SCORE` — optional floor on cross-encoder score; chunks
  below it are dropped instead of padding out `RERANK_TOP_K` with marginal
  matches. Unset by default (keeps today's behavior).
- `LLM_PROVIDER` / `OLLAMA_BASE_URL` / `LLM_MODEL` — the local LLM is never
  hard-coded; only `ollama` is implemented today, but the provider field is
  kept configurable for future backends.
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — chunking is paragraph-aware and
  configurable (defaults: ~500 tokens / ~75 token overlap).
- `DEBUG_MODE` (or `debug: true` per request) — see **Debugging** below.

Models are loaded lazily (only on first use) and as singletons, so the
embedding model, reranker, and LLM are never all forced into memory at once
just by starting the server — the Ollama model additionally runs as a
separate local process outside the Python process entirely.

## API

All endpoints are under `/api`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | readiness of Chroma, BM25, and Ollama |
| POST | `/api/ingest` | (re-)ingest the book PDF (upload a file, or re-ingest `BOOK_PDF_PATH`) |
| POST | `/api/query` | ask a grounded question, get an answer + sources |
| GET | `/api/stats` | corpus stats (pages, chunks, chapters, models in use) |
| GET | `/api/documents` | ingested document(s) |
| GET | `/api/chapters` | chapter/section index with page ranges |
| GET | `/api/config` | safe, non-secret subset of the running configuration |

### `POST /api/query`

```json
{
  "query": "What does the book say about the migration to Medina?",
  "top_k": 8,
  "rerank": true
}
```

```json
{
  "answer": "...",
  "sources": [
    {
      "chapter": "Chapter III",
      "chapter_title": "Life of Prophet Muhammad and the Birth of Jihad",
      "section": "Was Muhammad Driven Out of Mecca?",
      "page": 29,
      "start_page": 29,
      "end_page": 29,
      "chunk_id": "ch03_p029_008",
      "label": "Chapter III → Section: Was Muhammad Driven Out of Mecca? → Page 29",
      "snippet": "..."
    }
  ],
  "sufficient_evidence": true,
  "standalone_query": null,
  "retrieved_count": 6,
  "debug": null
}
```

`chat_history` (for conversational follow-ups), `filters` (`chapter`,
`page_min`, `page_max`), and `debug` (per-request override of `DEBUG_MODE`)
are also accepted — see `app/models/schemas.py::QueryRequest`.

When the retrieved context doesn't answer the question, `answer` is exactly:

> I couldn't find enough information in the indexed book to answer this reliably.

and `sources` is empty — the system never fabricates a citation.

## Debugging / observability

Set `DEBUG_MODE=true`, or pass `"debug": true` on an individual
`/api/query` request, to get a `debug` block in the response (and DEBUG-level
server logs) containing: the original and rewritten query, the raw BM25
results, the raw vector results, the fused (RRF) results, the final selected
chunk ids, and per-stage latency (`query_rewrite_ms`, `retrieval_ms`,
`rerank_ms`, `llm_ms`, `total_ms`). Nothing sensitive is logged — just the
query text and retrieval internals.

## Evaluation

`data/eval/eval_dataset.json` has 23 hand-written questions grounded in the
default book's real table of contents (20 answerable with known expected
pages, 3 deliberately out-of-scope to test correct refusal). If you swap in
a different book, write your own dataset in the same shape.

```bash
python scripts/evaluate.py                # full eval (needs Ollama running)
python scripts/evaluate.py --no-generation # retrieval-only, no LLM required
```

Reports Recall@K / Precision@K / MRR for retrieval, and (when Ollama is
reachable) citation correctness, two heuristic word-overlap proxies for
faithfulness and answer relevance, and refusal accuracy on the unanswerable
questions. See the script's module docstring for exactly what each metric
does and doesn't measure — these are simple/explainable proxies, not an
LLM-judge pipeline.

## Testing

```bash
pytest -q
```

The suite (`tests/`) covers PDF parsing, text cleaning, chapter/section
detection, chunking, embedding + caching, BM25, vector retrieval, hybrid
fusion, reranking, context construction, all API endpoints, and a handful of
full end-to-end RAG tests — all against a tiny synthetic 3-page PDF built on
the fly, so the suite never depends on a real book or a running Ollama
instance (LLM calls are monkeypatched in tests that don't specifically test
the Ollama client itself).

## Low-hardware notes

Designed to run on ~8GB RAM with no dedicated GPU:

- Default embedding model (`bge-small-en-v1.5`) and reranker
  (`ms-marco-MiniLM-L-6-v2`) are both small, CPU-friendly models.
- Default LLM (`qwen2.5:3b`) is sized for constrained machines.
- Reranking is optional and skippable.
- Embeddings are batched (`EMBEDDING_BATCH_SIZE`) and disk-cached
  (`EMBEDDING_CACHE_DIR`) so re-ingesting doesn't recompute unchanged chunks.
- A GPU is used automatically when available (`EMBEDDING_DEVICE`/
  `RERANK_DEVICE` auto-detect CUDA/MPS); nothing assumes CUDA is present.

test edit
