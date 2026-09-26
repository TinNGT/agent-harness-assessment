from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TYPE_CHECKING, Type

from pydantic import BaseModel, ValidationError

from harness.tools.errors import ToolInputValidationError, ToolOutputValidationError

if TYPE_CHECKING:
    from harness.storage.repository import Repository


@dataclass
class ToolContext:
    """Extra, non-LLM-supplied context a handler may need.

    `attempt` lets fault-injected mocks behave deterministically (e.g. fail on
    attempts 1-2, succeed on 3) instead of using randomness. `idempotency_key`
    and `repo` are only used by the non-idempotent `create_incident` tool.
    """

    attempt: int = 1
    idempotency_key: Optional[str] = None
    repo: Optional["Repository"] = None


ToolHandler = Callable[[BaseModel, ToolContext], Awaitable[dict[str, Any]]]


class ToolSpec(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    handler: ToolHandler
    requires_approval: bool = False
    idempotent: bool = True

    def validate_input(self, raw_args: dict[str, Any]) -> BaseModel:
        try:
            return self.input_model.model_validate(raw_args)
        except ValidationError as exc:
            raise ToolInputValidationError(
                f"invalid arguments for tool '{self.name}': {exc.errors()}"
            ) from exc

    def validate_output(self, raw_output: dict[str, Any]) -> BaseModel:
        try:
            return self.output_model.model_validate(raw_output)
        except ValidationError as exc:
            raise ToolOutputValidationError(
                f"tool '{self.name}' returned output that fails its schema: {exc.errors()}"
            ) from exc

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
            "requires_approval": self.requires_approval,
            "idempotent": self.idempotent,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())
