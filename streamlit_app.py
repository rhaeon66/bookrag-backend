"""Streamlit front end for BookRAG. Runs the RAG pipeline in-process (no FastAPI,
no Vercel): `streamlit run streamlit_app.py`.

Config comes from .env locally, or from Streamlit secrets when deployed
(Community Cloud -> App settings -> Secrets, TOML: OLLAMA_API_KEY = "...").
"""
import os

import streamlit as st

# Copy Streamlit secrets into the environment *before* app.config reads settings.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(str(_k).upper(), str(_v))
except Exception:
    pass  # no secrets file: fall back to .env / defaults

from app.config import get_settings  # noqa: E402
from app.generation import llm  # noqa: E402
from app.models.schemas import ChatTurn, QueryFilters, QueryRequest  # noqa: E402
from app.retrieval import vector_store  # noqa: E402
from app.services.rag_pipeline import answer_question, get_manifest  # noqa: E402

settings = get_settings()
st.set_page_config(page_title=settings.book_title, page_icon="📖", layout="wide")


@st.cache_resource(show_spinner="Loading models and index…")
def warm_up() -> int:
    """Load the vector store once per process; returns the number of indexed chunks."""
    return vector_store.count()


indexed = warm_up()
manifest = get_manifest(settings)

with st.sidebar:
    st.header(settings.book_title)
    if manifest:
        st.caption(f"{manifest['total_pages']} pages · {len(manifest['chapters'])} chapters · {indexed} chunks")
    st.caption(f"LLM: `{settings.llm_model}`")
    chapters = ["All chapters"] + [
        f"{c['chapter']}" + (f" — {c['chapter_title']}" if c.get("chapter_title") else "")
        for c in (manifest or {}).get("chapters", [])
    ]
    chapter_choice = st.selectbox("Chapter filter", chapters)
    top_k = st.slider("Passages used", 2, 12, settings.rerank_top_k)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title(f"📖 {settings.book_title}")
st.caption("Answers come only from the indexed book, with page citations.")

if indexed == 0:
    st.error("The book index is empty. Run `python scripts/ingest.py` (or commit data/chroma) first.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['label']}**" + (f" · {s['section']}" if s.get("section") else ""))
            st.caption(s["snippet"])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources"))

if prompt := st.chat_input("Ask a question about the book"):
    history = [ChatTurn(role=m["role"], content=m["content"]) for m in st.session_state.messages[-6:]]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    filters = None
    if chapter_choice != "All chapters":
        filters = QueryFilters(chapter=chapter_choice.split(" — ")[0])

    with st.chat_message("assistant"):
        with st.spinner("Searching the book…"):
            try:
                resp = answer_question(
                    QueryRequest(query=prompt, top_k=top_k, chat_history=history, filters=filters)
                )
                sources = [s.model_dump() for s in resp.sources]
                st.markdown(resp.answer)
                render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": resp.answer, "sources": sources}
                )
            except Exception as exc:
                st.error(f"Could not get an answer. Is the LLM reachable? ({exc})")
                if not llm.is_available():
                    st.info("Set OLLAMA_BASE_URL / OLLAMA_API_KEY / LLM_MODEL (see Secrets).")
