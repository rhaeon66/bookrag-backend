from app.generation import llm


class _StubOllamaClient:
    def __init__(self, reply="Answer\n\nMecca is a city.\n\nSources\n\n- Chapter I, Page 1"):
        self.reply = reply
        self.last_call = None

    def chat(self, model, messages, options):
        self.last_call = {"model": model, "messages": messages, "options": options}
        return {"message": {"content": self.reply}}

    def list(self):
        return {"models": []}


class _FailingClient:
    def list(self):
        raise ConnectionError("no server")


def test_generate_calls_ollama_with_configured_model(isolated_settings, monkeypatch):
    stub = _StubOllamaClient()
    monkeypatch.setattr(llm, "get_client", lambda: stub)

    result = llm.generate("system prompt", "user prompt", isolated_settings)

    assert result.startswith("Answer")
    assert stub.last_call["model"] == isolated_settings.llm_model
    assert stub.last_call["messages"][0] == {"role": "system", "content": "system prompt"}
    assert stub.last_call["messages"][1] == {"role": "user", "content": "user prompt"}
    assert stub.last_call["options"]["temperature"] == isolated_settings.llm_temperature


def test_is_available_true_when_client_responds(monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: _StubOllamaClient())
    assert llm.is_available() is True


def test_is_available_false_when_connection_fails(monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: _FailingClient())
    assert llm.is_available() is False
