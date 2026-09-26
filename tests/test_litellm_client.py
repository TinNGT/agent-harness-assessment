from __future__ import annotations

import sys
import types

import pytest

from harness.llm.errors import LLMAPIError
from harness.llm.litellm_client import LiteLLMClient


def _install_fake_litellm(monkeypatch, *, content=None, raise_exc=None, capture=None):
    """Stand in for the real `litellm` package so tests never touch the network."""

    async def fake_acompletion(**kwargs):
        if capture is not None:
            capture.append(kwargs)
        if raise_exc is not None:
            raise raise_exc
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(acompletion=fake_acompletion))


@pytest.mark.asyncio
async def test_openai_style_model_uses_the_openai_key(monkeypatch):
    captured: list[dict] = []
    _install_fake_litellm(monkeypatch, content="hello", capture=captured)
    client = LiteLLMClient("gpt-4o-mini", openai_api_key="sk-openai", anthropic_api_key="sk-anthropic")

    result = await client.step([{"role": "user", "content": "hi"}], [])

    assert result == "hello"
    assert captured[0]["model"] == "gpt-4o-mini"
    assert captured[0]["api_key"] == "sk-openai"


@pytest.mark.asyncio
async def test_claude_model_uses_the_anthropic_key(monkeypatch):
    captured: list[dict] = []
    _install_fake_litellm(monkeypatch, content="hello", capture=captured)
    client = LiteLLMClient(
        "claude-3-5-sonnet-20241022", openai_api_key="sk-openai", anthropic_api_key="sk-anthropic"
    )

    await client.step([{"role": "user", "content": "hi"}], [])

    assert captured[0]["api_key"] == "sk-anthropic"


@pytest.mark.asyncio
async def test_missing_key_falls_back_to_none_instead_of_empty_string(monkeypatch):
    captured: list[dict] = []
    _install_fake_litellm(monkeypatch, content="hello", capture=captured)
    client = LiteLLMClient("gpt-4o-mini")  # no keys configured at all

    await client.step([{"role": "user", "content": "hi"}], [])

    # None (not "") lets litellm fall back to its own os.environ lookup.
    assert captured[0]["api_key"] is None


@pytest.mark.asyncio
async def test_provider_error_is_wrapped_as_llm_api_error(monkeypatch):
    _install_fake_litellm(monkeypatch, raise_exc=RuntimeError("rate limited"))
    client = LiteLLMClient("gpt-4o-mini", openai_api_key="sk-openai")

    with pytest.raises(LLMAPIError):
        await client.step([{"role": "user", "content": "hi"}], [])


@pytest.mark.asyncio
async def test_empty_response_content_raises_llm_api_error(monkeypatch):
    _install_fake_litellm(monkeypatch, content="")
    client = LiteLLMClient("gpt-4o-mini", openai_api_key="sk-openai")

    with pytest.raises(LLMAPIError):
        await client.step([{"role": "user", "content": "hi"}], [])
