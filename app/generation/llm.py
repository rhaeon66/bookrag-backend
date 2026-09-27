"""Local LLM client via Ollama. Model name is fully configurable (OLLAMA_MODEL),
never hard-coded — the default in config.py is just a sane starting point for
small/low-GPU machines.
"""
from functools import lru_cache
from typing import Optional

from app.config import Settings, get_settings


@lru_cache
def get_client():
    import ollama

    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.ollama_api_key}"} if settings.ollama_api_key else None
    return ollama.Client(
        host=settings.ollama_base_url,
        headers=headers,
        timeout=settings.llm_request_timeout,
    )


def generate(system_prompt: str, user_prompt: str, settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    client = get_client()
    response = client.chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": settings.llm_temperature,
            "num_predict": settings.llm_max_tokens,
        },
    )
    return response["message"]["content"].strip()


def is_available(settings: Optional[Settings] = None) -> bool:
    settings = settings or get_settings()
    try:
        client = get_client()
        client.list()
        return True
    except Exception:
        return False
