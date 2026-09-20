"""End-to-end RAG pipeline tests: ingest a tiny synthetic book, then ask
questions through the same service layer the API uses."""
from app.generation import llm
from app.models.schemas import ChatTurn, QueryRequest
from app.services.rag_pipeline import answer_question, ingest_book

FAKE_ANSWER = "Answer\n\nThe aardvark is a nocturnal mammal.\n\nSources\n\n- Chapter I, Page 1"


def _fake_generate(system_prompt, user_prompt, settings=None):
    return FAKE_ANSWER


def test_e2e_ingest_then_answer_grounded_question(patched_settings, monkeypatch):
    result = ingest_book(settings=patched_settings)
    assert result.chunks_created > 0

    monkeypatch.setattr(llm, "generate", _fake_generate)

    response = answer_question(QueryRequest(query="Tell me about the aardvark."), patched_settings)

    assert response.sufficient_evidence is True
    assert response.answer == FAKE_ANSWER
    assert response.sources
    assert response.sources[0].chapter == "Chapter I"
    assert response.sources[0].page == 1


def test_e2e_question_with_no_matching_evidence_is_marked_insufficient(patched_settings, monkeypatch):
    ingest_book(settings=patched_settings)

    def generate_that_should_not_be_called(*args, **kwargs):
        raise AssertionError("LLM must not be called when there are zero retrieval candidates")

    monkeypatch.setattr(llm, "generate", generate_that_should_not_be_called)

    # A chapter filter that matches nothing in the corpus -> zero candidates,
    # so the pipeline must short-circuit before ever calling the LLM.
    from app.models.schemas import QueryFilters
    response = answer_question(
        QueryRequest(query="Tell me about aardvarks.", filters=QueryFilters(chapter="Chapter IX")),
        patched_settings,
    )

    assert response.sufficient_evidence is False
    assert response.sources == []
    assert response.answer == patched_settings.insufficient_evidence_message


def test_e2e_min_rerank_score_filters_everything_short_circuits_before_llm(patched_settings, monkeypatch):
    ingest_book(settings=patched_settings)

    # rerank is disabled in isolated_settings/patched_settings by default;
    # enable it here since the min-score filter only applies to reranked chunks.
    monkeypatch.setattr(patched_settings, "rerank_enabled", True, raising=False)
    # An impossibly high floor guarantees every candidate gets filtered out.
    monkeypatch.setattr(patched_settings, "context_min_rerank_score", 1000.0, raising=False)

    def generate_that_should_not_be_called(*args, **kwargs):
        raise AssertionError("LLM must not be called once every candidate is filtered out")

    monkeypatch.setattr(llm, "generate", generate_that_should_not_be_called)

    response = answer_question(QueryRequest(query="Tell me about the aardvark."), patched_settings)

    assert response.sufficient_evidence is False
    assert response.sources == []
    assert response.answer == patched_settings.insufficient_evidence_message


def test_e2e_followup_question_is_resolved_with_chat_history(patched_settings, monkeypatch):
    ingest_book(settings=patched_settings)

    calls = []

    def tracking_generate(system_prompt, user_prompt, settings=None):
        calls.append(system_prompt)
        if "standalone" in system_prompt.lower():
            return "What does the book say about beavers?"
        return FAKE_ANSWER

    monkeypatch.setattr(llm, "generate", tracking_generate)

    history = [
        ChatTurn(role="user", content="What does the book say about beavers?"),
        ChatTurn(role="assistant", content="Beavers build dams."),
    ]
    response = answer_question(QueryRequest(query="Why do they do that?", chat_history=history), patched_settings)

    assert response.standalone_query == "What does the book say about beavers?"
    # the rewrite prompt must have actually been invoked before the answer prompt
    assert any("standalone" in c.lower() for c in calls)


def test_e2e_debug_trace_includes_pipeline_stages(patched_settings, monkeypatch):
    ingest_book(settings=patched_settings)
    monkeypatch.setattr(llm, "generate", _fake_generate)

    response = answer_question(
        QueryRequest(query="Tell me about the aardvark.", debug=True), patched_settings
    )

    assert response.debug is not None
    for key in ["original_query", "bm25_results", "vector_results", "fused_results", "selected_chunks", "timings_ms"]:
        assert key in response.debug
