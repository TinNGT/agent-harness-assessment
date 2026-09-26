from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, ValidationError, model_validator

from harness.llm.errors import LLMParseError


class ToolCallPayload(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = {}


class LLMDecision(BaseModel):
    type: Literal["tool_call", "final"]
    tool_call: Optional[ToolCallPayload] = None
    answer: Optional[str] = None

    @model_validator(mode="after")
    def _check_shape(self) -> "LLMDecision":
        if self.type == "tool_call" and self.tool_call is None:
            raise ValueError("type=='tool_call' requires a 'tool_call' object")
        if self.type == "final" and not (self.answer and self.answer.strip()):
            raise ValueError("type=='final' requires a non-empty 'answer'")
        return self


def parse_decision(raw: str) -> LLMDecision:
    """Parse raw LLM text into a validated decision.

    Tolerates the common "wrapped in a markdown code fence" mistake, but any
    other malformed JSON or missing-field response raises LLMParseError so the
    caller can feed a repair message back to the LLM.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"response is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise LLMParseError("response JSON must be an object")

    try:
        return LLMDecision.model_validate(data)
    except ValidationError as exc:
        raise LLMParseError(f"response JSON does not match the expected schema: {exc}") from exc
