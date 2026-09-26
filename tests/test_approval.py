from __future__ import annotations

import pytest
from sqlmodel import select

from harness.storage.models import Incident
from harness.storage.repository import ConflictError, NotFoundError


def _incident_count(service) -> int:
    with service._session() as session:
        return len(list(session.exec(select(Incident))))


@pytest.mark.asyncio
async def test_run_pauses_for_approval_and_incident_not_created_yet(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Handle the auth-service outage", scenario="approval_create_incident"
    )

    assert run.status == "waiting_approval"
    pending = await service.get_pending_approval(run.id)
    assert pending is not None
    assert pending.tool_name == "create_incident"
    assert pending.args["severity"] == "critical"
    assert _incident_count(service) == 0


@pytest.mark.asyncio
async def test_approve_creates_incident_with_exact_approved_arguments(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Handle the auth-service outage", scenario="approval_create_incident"
    )
    pending = await service.get_pending_approval(run.id)

    resumed = await service.decide_approval(
        run_id=run.id, approval_id=pending.id, decision="approve", approver="alice", reason=None
    )

    assert resumed.status == "completed"
    assert _incident_count(service) == 1
    with service._session() as session:
        incident = session.exec(select(Incident)).first()
    assert incident.title == pending.args["title"]
    assert incident.severity == "critical"
    assert incident.idempotency_key == pending.id


@pytest.mark.asyncio
async def test_reject_does_not_create_incident_but_run_still_completes(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Handle the auth-service outage", scenario="approval_create_incident"
    )
    pending = await service.get_pending_approval(run.id)

    resumed = await service.decide_approval(
        run_id=run.id, approval_id=pending.id, decision="reject", approver="bob",
        reason="duplicate of INC-1000",
    )

    assert resumed.status == "completed"
    assert _incident_count(service) == 0

    trace = await service.get_trace(run.id)
    decided_step = next(s for s in trace if s.type == "approval_decided")
    assert decided_step.payload["decision"] == "rejected"
    assert decided_step.payload["reason"] == "duplicate of INC-1000"


@pytest.mark.asyncio
async def test_deciding_an_already_decided_approval_conflicts(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Handle the auth-service outage", scenario="approval_create_incident"
    )
    pending = await service.get_pending_approval(run.id)

    await service.decide_approval(
        run_id=run.id, approval_id=pending.id, decision="approve", approver="alice", reason=None
    )

    with pytest.raises(ConflictError):
        await service.decide_approval(
            run_id=run.id, approval_id=pending.id, decision="approve", approver="alice", reason=None
        )


@pytest.mark.asyncio
async def test_deciding_a_nonexistent_run_or_approval_raises_not_found(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Handle the auth-service outage", scenario="approval_create_incident"
    )

    with pytest.raises(NotFoundError):
        await service.decide_approval(
            run_id=run.id, approval_id="does-not-exist", decision="approve", approver="alice", reason=None
        )

    with pytest.raises(NotFoundError):
        await service.decide_approval(
            run_id="does-not-exist", approval_id="also-missing", decision="approve",
            approver="alice", reason=None,
        )
