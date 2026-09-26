from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional

import structlog
from pydantic import BaseModel

from harness.observability.tracing import get_tracer
from harness.tools.errors import PermanentToolError, ToolTimeoutError, TransientToolError
from harness.tools.registry import ToolContext, ToolSpec

logger = structlog.get_logger(__name__)
tracer = get_tracer(__name__)


@dataclass
class ToolResult:
    ok: bool
    data: Optional[dict[str, Any]]
    error: Optional[dict[str, Any]]
    attempts: int
    latency_ms: int


class ToolExecutor:
    """Runs a validated tool call with a timeout and bounded retries.

    Idempotent tools (reads) retry freely on transient failures. Non-idempotent
    tools (create_incident) get a single attempt — retry safety there comes from
    the caller passing an idempotency key, not from blind re-execution here.
    """

    def __init__(self, *, timeout_seconds: float, max_retries: int) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def execute(
        self,
        tool: ToolSpec,
        args: BaseModel,
        *,
        run_id: str,
        idempotency_key: Optional[str] = None,
        repo: Any = None,
    ) -> ToolResult:
        max_attempts = (self.max_retries + 1) if tool.idempotent else 1
        last_error: Exception | None = None
        start = time.monotonic()

        for attempt in range(1, max_attempts + 1):
            ctx = ToolContext(attempt=attempt, idempotency_key=idempotency_key, repo=repo)
            try:
                with tracer.start_as_current_span(
                    f"tool_call.{tool.name}",
                    attributes={"run_id": run_id, "tool": tool.name, "attempt": attempt},
                ):
                    raw = await asyncio.wait_for(tool.handler(args, ctx), timeout=self.timeout_seconds)
                output = tool.validate_output(raw)
                latency_ms = int((time.monotonic() - start) * 1000)
                if attempt > 1:
                    logger.info(
                        "tool.retry_succeeded", run_id=run_id, tool=tool.name, attempt=attempt
                    )
                return ToolResult(
                    ok=True, data=output.model_dump(), error=None, attempts=attempt, latency_ms=latency_ms
                )
            except asyncio.TimeoutError:
                last_error = ToolTimeoutError(
                    f"tool '{tool.name}' timed out after {self.timeout_seconds}s"
                )
                logger.warning(
                    "tool.timeout", run_id=run_id, tool=tool.name, attempt=attempt,
                    timeout_seconds=self.timeout_seconds,
                )
            except TransientToolError as exc:
                last_error = exc
                logger.warning(
                    "tool.retry", run_id=run_id, tool=tool.name, attempt=attempt, error=str(exc)
                )
            except PermanentToolError as exc:
                latency_ms = int((time.monotonic() - start) * 1000)
                logger.error(
                    "tool.permanent_error", run_id=run_id, tool=tool.name, attempt=attempt,
                    error=str(exc),
                )
                return ToolResult(
                    ok=False,
                    data=None,
                    error={"type": type(exc).__name__, "message": exc.message, "retryable": False},
                    attempts=attempt,
                    latency_ms=latency_ms,
                )

        latency_ms = int((time.monotonic() - start) * 1000)
        logger.error(
            "tool.retries_exhausted", run_id=run_id, tool=tool.name, attempts=max_attempts,
            error=str(last_error),
        )
        return ToolResult(
            ok=False,
            data=None,
            error={
                "type": type(last_error).__name__ if last_error else "UnknownError",
                "message": str(last_error) if last_error else "unknown error",
                "retryable": True,
            },
            attempts=max_attempts,
            latency_ms=latency_ms,
        )
