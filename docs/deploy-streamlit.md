# Streamlit-only deployment

`streamlit_app.py` runs the whole RAG pipeline in one process (no FastAPI, no Vercel).

## Run locally
```
cd backend
./.venv/bin/pip install -r requirements.txt
./.venv/bin/streamlit run streamlit_app.py
```
Uses `.env` (local Ollama by default).

## Deploy on Streamlit Community Cloud (free)
1. The repo must contain the index: `data/chroma/` and `data/processed/bm25_index.pkl`
   (+ `manifest.json`). They are gitignored; on a deploy branch use `git add -f` on them.
   Community Cloud needs a GitHub repo it can read, and public repos are the free default,
   so the book-derived index would be public. Only do this if you have the rights.
2. https://share.streamlit.io -> New app -> pick the repo/branch, main file `streamlit_app.py`.
   If the repo root is not `backend/`, use the path `backend/streamlit_app.py`.
3. Advanced settings -> Python 3.11 or 3.12 (torch/chromadb wheels are safest there).
4. Secrets (TOML), see `.streamlit/secrets.toml.example`:
   ```
   OLLAMA_BASE_URL = "https://ollama.com"
   OLLAMA_API_KEY = "<new key>"
   LLM_MODEL = "gpt-oss:20b"
   ```
   A local Ollama at localhost is NOT reachable from Community Cloud, so Ollama Cloud
   (or another hosted LLM) is required there.
5. Deploy. First boot downloads the embedding + reranker models (a few minutes). If it runs
   out of memory (free tier is ~2.7 GB), set `RERANK_ENABLED = "false"` in Secrets.
