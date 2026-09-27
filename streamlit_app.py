"""Streamlit front end for BookRAG. Runs the RAG pipeline in-process (no FastAPI,
no Vercel): `streamlit run streamlit_app.py`.

Config comes from .env locally, or from Streamlit secrets when deployed
(Community Cloud -> App settings -> Secrets, TOML: OLLAMA_API_KEY = "...").
"""
import os
from pathlib import Path

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

PDF_PATH = Path(settings.book_pdf_path)
CHAT_VIEW, BOOK_VIEW = "💬 Ask the book", "📖 Read the book"


@st.cache_data(show_spinner=False)
def page_count(path: str) -> int:
    import pymupdf  # already a dependency

    with pymupdf.open(path) as doc:
        return doc.page_count


@st.cache_data(show_spinner=False, max_entries=64)
def render_page(path: str, page: int, zoom: float = 1.8) -> bytes:
    """Render one 1-indexed PDF page to PNG bytes."""
    import pymupdf

    with pymupdf.open(path) as doc:
        pix = doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return pix.tobytes("png")


def open_book(page: int) -> None:
    """Button callback: jump to the book viewer at `page`."""
    st.session_state.view = BOOK_VIEW
    st.session_state.book_page = page


st.session_state.setdefault("view", CHAT_VIEW)
st.session_state.setdefault("book_page", 1)

with st.sidebar:
    st.header(settings.book_title)
    if manifest:
        st.caption(f"{manifest['total_pages']} pages · {len(manifest['chapters'])} chapters · {indexed} chunks")
    st.caption(f"LLM: `{settings.llm_model}`")
    if PDF_PATH.exists():
        st.radio("View", [CHAT_VIEW, BOOK_VIEW], key="view")
    else:
        st.session_state.view = CHAT_VIEW
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

if st.session_state.view == BOOK_VIEW:
    total = page_count(str(PDF_PATH))
    st.session_state.book_page = min(max(st.session_state.book_page, 1), total)
    prev_col, num_col, next_col = st.columns([1, 2, 1])
    if prev_col.button("◀ Previous", width="stretch"):
        st.session_state.book_page = max(1, st.session_state.book_page - 1)
    if next_col.button("Next ▶", width="stretch"):
        st.session_state.book_page = min(total, st.session_state.book_page + 1)
    num_col.number_input("Page", 1, total, key="book_page", label_visibility="collapsed")
    st.caption(f"Page {st.session_state.book_page} of {total}")
    st.image(render_page(str(PDF_PATH), st.session_state.book_page), width="stretch")
    st.stop()

st.caption("Answers come only from the indexed book, with page citations.")

if indexed == 0:
    st.error("The book index is empty. Run `python scripts/ingest.py` (or commit data/chroma) first.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources, msg_id):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources):
            st.markdown(f"**{s['label']}**" + (f" · {s['section']}" if s.get("section") else ""))
            st.caption(s["snippet"])
            if PDF_PATH.exists():
                st.button(
                    f"Open page {s['page']} in the book",
                    key=f"open-{msg_id}-{i}",
                    on_click=open_book,
                    args=(s["page"],),
                )


for n, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources"), n)

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
                render_sources(sources, len(st.session_state.messages))
                st.session_state.messages.append(
                    {"role": "assistant", "content": resp.answer, "sources": sources}
                )
            except Exception as exc:
                st.error(f"Could not get an answer. Is the LLM reachable? ({exc})")
                if not llm.is_available():
                    st.info("Set OLLAMA_BASE_URL / OLLAMA_API_KEY / LLM_MODEL (see Secrets).")
