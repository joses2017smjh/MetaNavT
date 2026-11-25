"""The Ollama chat model's output-token cap (M39): LLM_MAX_TOKENS, DEFAULT_LLM_MAX_TOKENS when it is not set.

The chat smoke committed in c9b4acd ran with no cap, and one model call generated until the run was stopped. These
tests check the setting and that the agent's own call, a streamed chat with tools, sends the cap to the server.
"""

import asyncio

import pytest
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.settings import Settings

from app import settings as app_settings


@pytest.fixture
def ollama(monkeypatch):
    """init_ollama() under a hash embedder; Settings is restored afterwards."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    llm, embed = Settings._llm, Settings._embed_model

    def build():
        app_settings.init_ollama()
        return Settings.llm

    yield build
    Settings._llm, Settings._embed_model = llm, embed


def test_the_chat_model_is_capped_by_default_and_by_the_setting(ollama, monkeypatch):
    assert ollama().additional_kwargs == {"num_predict": app_settings.DEFAULT_LLM_MAX_TOKENS}
    monkeypatch.setenv("LLM_MAX_TOKENS", "300")
    assert ollama().additional_kwargs == {"num_predict": 300}
    monkeypatch.setenv("LLM_MAX_TOKENS", "")  # set but empty, as a blank .env line leaves it: the default
    assert ollama().additional_kwargs == {"num_predict": app_settings.DEFAULT_LLM_MAX_TOKENS}
    for bad in ("0", "-1"):  # Ollama reads a negative num_predict as no limit
        monkeypatch.setenv("LLM_MAX_TOKENS", bad)
        with pytest.raises(ValueError, match="LLM_MAX_TOKENS"):
            ollama()


def test_the_agents_streamed_call_sends_the_cap_to_the_server(ollama):
    """FunctionAgent calls astream_chat_with_tools; the request it sends carries num_predict in its options."""
    sent = []

    class FakeAsyncClient:
        async def chat(self, **kwargs):
            sent.append(kwargs)

            async def stream():
                yield {"message": {"role": "assistant", "content": "ok"}, "done": True}

            return stream()

    llm = ollama()
    llm._async_client = FakeAsyncClient()
    llm.context_window = 4096  # otherwise the client asks the server for it (/api/show)

    async def call():
        replies = await llm.astream_chat_with_tools([], chat_history=[ChatMessage(role=MessageRole.USER, content="hi")])
        return [r async for r in replies]

    assert asyncio.run(call())[-1].message.content == "ok"
    assert sent and sent[0]["stream"] is True
    assert sent[0]["options"]["num_predict"] == app_settings.DEFAULT_LLM_MAX_TOKENS
