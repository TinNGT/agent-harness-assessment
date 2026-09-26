from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


class TickingClock:
    """Deterministic clock that jumps forward by `step_seconds` on every call to
    now(). Lets us exceed a deadline without any real `sleep`, per the plan's
    'fake clock' testing technique."""

    def __init__(self, start: datetime, step_seconds: float) -> None:
        self._t = start
        self._step = timedelta(seconds=step_seconds)

    def now(self) -> datetime:
        current = self._t
        self._t = self._t + self._step
        return current


@pytest.mark.asyncio
async def test_max_steps_exceeded(make_service):
    service = make_service(max_steps=3)
    run = await service.create_and_run(
        objective="Search many topics", scenario="max_steps_probe"
    )

    assert run.status == "limit_exceeded"
    assert run.step_count == 3
    assert run.error["type"] == "BudgetExceeded"
    assert "max_steps" in run.error["message"]


@pytest.mark.asyncio
async def test_deadline_exceeded_using_fake_clock(make_service):
    clock = TickingClock(start=datetime.now(timezone.utc), step_seconds=40)
    service = make_service(clock=clock, max_run_seconds=60, max_steps=20)

    run = await service.create_and_run(
        objective="Search many topics", scenario="max_steps_probe"
    )

    assert run.status == "limit_exceeded"
    assert run.error["type"] == "BudgetExceeded"
    assert "max_run_seconds" in run.error["message"]


@pytest.mark.asyncio
async def test_identical_tool_call_repeated_is_detected_as_a_loop(make_service):
    service = make_service(max_steps=20)
    run = await service.create_and_run(
        objective="Keep checking payment-api", scenario="infinite_loop"
    )

    assert run.status == "limit_exceeded"
    assert run.step_count == 3  # 3 identical calls executed before the 4th is blocked
    assert run.error["type"] == "BudgetExceeded"
    assert "called 3x" in run.error["message"] or "identical" in run.error["message"]
