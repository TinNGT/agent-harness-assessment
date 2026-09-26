from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from harness.tools.registry import ToolSpec

SYSTEM_PROMPT_TEMPLATE = """You are an operations assistant. You investigate and, where \
appropriate, remediate issues by calling tools. On every turn you must reply with EXACTLY \
ONE JSON object and nothing else (no markdown fences, no prose).

Two shapes are allowed:

1. To call a tool:
   {{"type": "tool_call", "tool_call": {{"id": "call_<n>", "name": "<tool name>", "arguments": {{...}}}}}}

2. To finish and answer the user:
   {{"type": "final", "answer": "<your answer>"}}

Available tools:
{tool_list}

Rules:
- Only call tools listed above, with arguments matching their schema.
- create_incident always requires human approval before it takes effect; after \
  requesting it, wait for the observation before deciding what to do next.
- If a tool call fails, read the observation and either retry with different \
  arguments, try another tool, or give a final answer explaining the situation.
"""


def build_system_message(tools: list[ToolSpec]) -> dict[str, Any]:
    tool_list = "\n".join(
        f"- {t.name}({', '.join(t.input_model.model_fields.keys())}): {t.description}"
        + (" [requires approval]" if t.requires_approval else "")
        for t in tools
    )
    return {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(tool_list=tool_list)}


class LLMClient(ABC):
    """Common interface implemented by ScriptedLLM (tests/demo) and LiteLLMClient (real)."""

    @abstractmethod
    async def step(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> str:
        """Return the raw text content of the next decision (see SYSTEM_PROMPT_TEMPLATE)."""
        raise NotImplementedError
