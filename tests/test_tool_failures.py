from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_flaky_service_retries_then_succeeds(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Check flaky-service", scenario="tool_failure_retry_success"
    )

    assert run.status == "completed"
    trace = await service.get_trace(run.id)
    result_step = next(s for s in trace if s.type == "tool_result")
    assert result_step.attempt == 3  # 2 failures + 1 success, per data/services.json
    assert result_step.payload["observation"]["status"] == "healthy"


@pytest.mark.asyncio
async def test_slow_service_times_out_after_retries_exhausted(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Check slow-service", scenario="tool_failure_timeout"
    )

    # The LLM still gets to see the failure and produce a final answer.
    assert run.status == "completed"
    trace = await service.get_trace(run.id)
    result_step = next(s for s in trace if s.type == "tool_result")
    assert result_step.attempt == 3  # tool_max_retries=2 -> 3 total attempts
    observation = result_step.payload["observation"]
    assert observation["error"]["retryable"] is True
    assert "timed out" in observation["error"]["message"]


@pytest.mark.asyncio
async def test_broken_service_output_schema_error_is_not_retried(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Check broken-service", scenario="tool_failure_schema"
    )

    assert run.status == "completed"
    trace = await service.get_trace(run.id)
    result_step = next(s for s in trace if s.type == "tool_result")
    assert result_step.attempt == 1  # permanent error: no retry
    observation = result_step.payload["observation"]
    assert observation["error"]["retryable"] is False
    assert observation["error"]["type"] == "ToolOutputValidationError"
