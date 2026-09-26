from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_happy_path_completes_without_approval(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Check payment-api and summarize", scenario="happy_path"
    )

    assert run.status == "completed"
    assert run.final_answer
    assert run.error is None

    trace = await service.get_trace(run.id)
    types = [s.type for s in trace]
    assert "tool_call" in types
    assert "tool_result" in types
    assert "final" in types
    assert "approval_requested" not in types

    tool_calls = [s.payload["tool"] for s in trace if s.type == "tool_call"]
    assert tool_calls == ["get_service_status", "search_knowledge_base"]

    status_result = next(s for s in trace if s.type == "tool_result" and s.payload["tool"] == "get_service_status")
    assert status_result.payload["observation"]["status"] == "degraded"

    kb_result = next(s for s in trace if s.type == "tool_result" and s.payload["tool"] == "search_knowledge_base")
    assert len(kb_result.payload["observation"]["results"]) > 0
