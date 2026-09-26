from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_llm_repairs_after_one_malformed_response(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Check search-api", scenario="malformed_then_repair"
    )

    assert run.status == "completed"
    assert run.consecutive_llm_errors == 0  # reset after the next valid tool call

    trace = await service.get_trace(run.id)
    error_steps = [s for s in trace if s.type == "error" and s.payload.get("error_type") == "LLMParseError"]
    assert len(error_steps) == 1


@pytest.mark.asyncio
async def test_llm_fails_after_repair_budget_exhausted(make_service):
    service = make_service(llm_max_repair=2)
    run = await service.create_and_run(
        objective="Check search-api", scenario="malformed_exhausted"
    )

    assert run.status == "failed"
    assert run.error["type"] == "LLMParseError"
    assert run.consecutive_llm_errors == 3  # 2 allowed repairs + 1 that tips it over


@pytest.mark.asyncio
async def test_unknown_tool_call_is_reported_and_recovered_from(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Check search-api", scenario="unknown_tool"
    )

    assert run.status == "completed"
    trace = await service.get_trace(run.id)
    error_steps = [s for s in trace if s.type == "error" and s.payload.get("error_type") == "UnknownTool"]
    assert len(error_steps) == 1
    assert "delete_everything" in error_steps[0].payload["message"]
