import pytest
from fastapi.testclient import TestClient

from app.generation import llm
from app.main import app

FAKE_ANSWER = "Answer\n\nThe aardvark forages for insects at night.\n\nSources\n\n- Chapter I, Page 1"


@pytest.fixture
def client():
    return TestClient(app)


def _fake_generate(system_prompt, user_prompt, settings=None):
    if "standalone" in system_prompt.lower() or "rewrite" in system_prompt.lower():
        return "rewritten standalone query"
    return FAKE_ANSWER


def test_health_before_ingest(client, patched_settings):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "not_ingested"
    assert body["chroma_ready"] is False


def test_query_before_ingest_returns_409(client, patched_settings):
    r = client.post("/api/query", json={"query": "anything"})
    assert r.status_code == 409


def test_query_rejects_empty_query(client, patched_settings):
    r = client.post("/api/query", json={"query": "   "})
    assert r.status_code == 400


def test_config_endpoint_exposes_safe_settings(client, patched_settings):
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["book_title"] == patched_settings.book_title
    assert body["llm_provider"] == "ollama"


def test_ingest_then_stats_documents_chapters(client, patched_settings):
    r = client.post("/api/ingest")
    assert r.status_code == 200, r.text
    ingest_body = r.json()
    assert ingest_body["pages_processed"] == 3
    assert ingest_body["chapters_detected"] == 2
    assert ingest_body["chunks_created"] >= 2

    stats = client.get("/api/stats").json()
    assert stats["total_pages"] == 3
    assert stats["total_chapters"] == 2

    docs = client.get("/api/documents").json()
    assert len(docs) == 1
    assert docs[0]["document"] == patched_settings.book_title

    chapters = client.get("/api/chapters").json()
    chapter_labels = {c["chapter"] for c in chapters}
    assert chapter_labels == {"Chapter I", "Chapter II"}


def test_full_query_flow_returns_grounded_answer_with_sources(client, patched_settings, monkeypatch):
    client.post("/api/ingest")
    monkeypatch.setattr(llm, "generate", _fake_generate)

    r = client.post("/api/query", json={"query": "Tell me about the aardvark."})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == FAKE_ANSWER
    assert body["sufficient_evidence"] is True
    assert body["sources"]
    assert body["sources"][0]["chapter"] == "Chapter I"
    assert "page" in body["sources"][0]


def test_query_with_debug_flag_returns_trace(client, patched_settings, monkeypatch):
    client.post("/api/ingest")
    monkeypatch.setattr(llm, "generate", _fake_generate)

    r = client.post("/api/query", json={"query": "Tell me about the aardvark.", "debug": True})
    body = r.json()
    assert body["debug"] is not None
    assert "timings_ms" in body["debug"]
    assert "bm25_results" in body["debug"]


def test_query_with_chapter_filter(client, patched_settings, monkeypatch):
    client.post("/api/ingest")
    monkeypatch.setattr(llm, "generate", _fake_generate)

    r = client.post("/api/query", json={"query": "content", "filters": {"chapter": "Chapter II"}})
    assert r.status_code == 200
    body = r.json()
    for source in body["sources"]:
        assert source["chapter"] == "Chapter II"


def test_query_returns_503_when_llm_unreachable(client, patched_settings, monkeypatch):
    client.post("/api/ingest")

    def unreachable(*args, **kwargs):
        raise ConnectionError("Failed to connect to Ollama")

    # Deterministic regardless of whether Ollama happens to be installed on
    # whatever machine runs this suite.
    monkeypatch.setattr(llm, "generate", unreachable)

    r = client.post("/api/query", json={"query": "Tell me about the aardvark."})
    assert r.status_code == 503
