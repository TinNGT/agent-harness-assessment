from __future__ import annotations

from typing import Any

from harness.llm.base import LLMClient
from harness.llm.errors import LLMAPIError
from harness.tools.registry import ToolSpec


class LiteLLMClient(LLMClient):
    """Real-LLM backend. Swap providers by changing LLM_MODEL (litellm routes by prefix,
    e.g. "gpt-4o-mini", "claude-3-5-sonnet-20241022", "gemini/gemini-1.5-flash").

    The API key is passed to litellm explicitly instead of relying on it being
    present in `os.environ`: pydantic-settings reads OPENAI_API_KEY/
    ANTHROPIC_API_KEY out of `.env` into `Settings` but does not export them
    as real process environment variables, which is what litellm reads by
    default."""

    def __init__(self, model: str, *, openai_api_key: str = "", anthropic_api_key: str = "") -> None:
        self.model = model
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key

    def _resolve_api_key(self) -> str | None:
        if self.model.startswith("claude"):
            return self.anthropic_api_key or None
        # Default: OpenAI-style models (gpt-*, o1-*, ...) and anything else
        # litellm can't infer a key for from a provider prefix (e.g. gemini/*)
        # falls back to litellm's own os.environ lookup when this is None.
        return self.openai_api_key or None

    async def step(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> str:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - dependency always declared
            raise LLMAPIError("litellm is not installed") from exc

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=0,
                api_key=self._resolve_api_key(),
            )
        except Exception as exc:  # noqa: BLE001 - normalize every provider error
            raise LLMAPIError(f"LLM API call failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMAPIError("LLM returned an empty response")
        return content
