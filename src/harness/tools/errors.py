from __future__ import annotations

from typing import Any, Optional


class ToolError(Exception):
    """Base class for all tool execution errors."""

    retryable: bool = False

    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class TransientToolError(ToolError):
    """Temporary failure (timeout, flaky dependency) — safe to retry."""

    retryable = True


class ToolTimeoutError(TransientToolError):
    """The tool did not respond within TOOL_TIMEOUT_SECONDS."""


class PermanentToolError(ToolError):
    """Failure that will not go away on retry (bad input, unknown resource)."""

    retryable = False


class ToolNotFoundError(PermanentToolError):
    """The LLM asked for a tool name that isn't registered."""


class ToolInputValidationError(PermanentToolError):
    """The LLM's tool call arguments failed schema validation."""


class ToolOutputValidationError(PermanentToolError):
    """The tool's own output failed schema validation (mock 'broken' fault)."""
