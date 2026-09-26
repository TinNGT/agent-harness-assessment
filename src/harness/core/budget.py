from __future__ import annotations

from typing import Any

from harness.core.clock import Clock
from harness.llm.parser import parse_decision
from harness.storage.models import Run

LOOP_REPEAT_THRESHOLD = 3


class BudgetExceeded(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _extract_tool_calls(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Best-effort extraction of (tool_name, canonical_args) for every well-formed
    tool_call decision the LLM has produced so far. Malformed turns are skipped —
    they're already bounded by the repair-attempt limit, not the loop detector."""
    import json

    calls: list[tuple[str, str]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        try:
            decision = parse_decision(message["content"])
        except Exception:
            continue
        if decision.type == "tool_call" and decision.tool_call is not None:
            canonical_args = json.dumps(decision.tool_call.arguments, sort_keys=True)
            calls.append((decision.tool_call.name, canonical_args))
    return calls


class BudgetGuard:
    """Enforces step count, wall-clock deadline, and identical-call loop detection."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    def check(self, run: Run) -> None:
        if run.step_count >= run.max_steps:
            raise BudgetExceeded(f"max_steps ({run.max_steps}) exceeded")
        if self.clock.now() >= run.deadline_at:
            raise BudgetExceeded(f"max_run_seconds ({run.max_run_seconds}) exceeded")
        self._check_loop(run)

    def _check_loop(self, run: Run) -> None:
        calls = _extract_tool_calls(run.messages)
        if len(calls) < LOOP_REPEAT_THRESHOLD:
            return
        last = calls[-LOOP_REPEAT_THRESHOLD:]
        if len(set(last)) == 1:
            name, args = last[0]
            raise BudgetExceeded(
                f"tool '{name}' called {LOOP_REPEAT_THRESHOLD}x in a row with identical "
                f"arguments {args}"
            )
