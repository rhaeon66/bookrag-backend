import numpy as np

from app.retrieval.embeddings import count_tokens, embed_documents, embed_query


def test_count_tokens_is_positive_and_monotonic():
    assert count_tokens("") == 0
    short = count_tokens("Mecca")
    long = count_tokens("Mecca was a major trading city on the Arabian peninsula.")
    assert short > 0
    assert long > short


def test_embed_documents_returns_normalized_vectors():
    vectors = embed_documents(["Mecca is a city.", "Medina is a different city."], use_cache=False)
    assert vectors.shape[0] == 2
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_embed_query_returns_a_single_normalized_vector():
    vector = embed_query("Was Muhammad driven out of Mecca?")
    assert vector.ndim == 1
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-3


def test_embedding_cache_avoids_recomputation(isolated_settings, monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr("app.retrieval.embeddings.get_settings", lambda: isolated_settings)

    calls = {"n": 0}
    from app.retrieval import embeddings as emb_module
    real_model = emb_module.get_embedding_model()
    original_encode = real_model.encode

    def counting_encode(*args, **kwargs):
        calls["n"] += 1
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(real_model, "encode", counting_encode)

    texts = ["A sentence about aardvarks."]
    embed_documents(texts, use_cache=True)
    first_calls = calls["n"]
    embed_documents(texts, use_cache=True)
    assert calls["n"] == first_calls  # second call was served entirely from disk cache
