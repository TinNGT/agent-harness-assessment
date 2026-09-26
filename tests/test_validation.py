from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools.errors import ToolInputValidationError
from harness.tools.mock_tools import build_registry

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_invalid_severity_is_rejected_by_the_tool_schema():
    registry = build_registry(DATA_DIR)
    tool = registry.get("create_incident")
    with pytest.raises(ToolInputValidationError):
        tool.validate_input({"title": "x", "description": "y", "severity": "urgent"})


def test_empty_title_is_rejected_by_the_tool_schema():
    registry = build_registry(DATA_DIR)
    tool = registry.get("create_incident")
    with pytest.raises(ToolInputValidationError):
        tool.validate_input({"title": "", "description": "y", "severity": "high"})


@pytest.mark.asyncio
async def test_invalid_severity_blocks_approval_until_a_valid_call_is_made(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Open an incident for payment-api", scenario="invalid_severity"
    )

    assert run.status == "waiting_approval"
    pending = await service.get_pending_approval(run.id)
    assert pending.args["severity"] == "high"

    trace = await service.get_trace(run.id)
    error_steps = [
        s for s in trace if s.type == "error" and s.payload.get("error_type") == "ToolInputValidationError"
    ]
    assert len(error_steps) == 1
    approval_steps = [s for s in trace if s.type == "approval_requested"]
    assert len(approval_steps) == 1  # only the valid call created an approval
