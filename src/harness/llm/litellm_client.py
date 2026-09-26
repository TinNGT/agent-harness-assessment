from __future__ import annotations

from typing import Any

from harness.llm.base import LLMClient
from harness.llm.errors import LLMAPIError
from harness.tools.registry import ToolSpec


class LiteLLMClient(LLMClient):
    """Real-LLM backend. Swap providers by changing LLM_MODEL (litellm routes by prefix,
    e.g. "gpt-4o-mini", "claude-3-5-sonnet-20241022", "gemini/gemini-1.5-flash")."""

    def __init__(self, model: str) -> None:
        self.model = model

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
            )
        except Exception as exc:  # noqa: BLE001 - normalize every provider error
            raise LLMAPIError(f"LLM API call failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMAPIError("LLM returned an empty response")
        return content
