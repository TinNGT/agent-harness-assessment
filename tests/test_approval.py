from __future__ import annotations

import pytest
from sqlmodel import select

from harness.storage.models import Incident
from harness.storage.repository import ConflictError, NotFoundError, Repository


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


@pytest.mark.asyncio
async def test_decision_other_than_approve_or_reject_is_rejected(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Handle the auth-service outage", scenario="approval_create_incident"
    )
    pending = await service.get_pending_approval(run.id)

    with pytest.raises(ValueError):
        await service.decide_approval(
            run_id=run.id, approval_id=pending.id, decision="maybe", approver="alice", reason=None
        )

    # The bad decision must not have consumed the approval — a real one still works.
    resumed = await service.decide_approval(
        run_id=run.id, approval_id=pending.id, decision="approve", approver="alice", reason=None
    )
    assert resumed.status == "completed"


@pytest.mark.asyncio
async def test_stale_run_version_conflicts_even_if_approval_itself_is_still_pending(make_service):
    """decide_approval_atomic guards on run.version, not just approval.status — a
    caller holding a stale Run object (read before someone else's write) must not
    be able to silently overwrite a run that has moved on in the meantime."""
    service = make_service()
    run = await service.create_and_run(
        objective="Handle the auth-service outage", scenario="approval_create_incident"
    )
    pending = await service.get_pending_approval(run.id)
    stale_version = run.version

    with service._session() as session:
        repo = Repository(session)
        # Simulate a concurrent write bumping the run's version between this
        # caller's read and its decide call.
        fresh_run = repo.get_run(run.id)
        repo.save_run(fresh_run)
        assert fresh_run.version != stale_version

        with pytest.raises(ConflictError, match="version conflict"):
            repo.decide_approval_atomic(
                approval_id=pending.id,
                run_id=run.id,
                decision="approved",
                approver="alice",
                reason=None,
                now=service.clock.now(),
                expected_run_version=stale_version,
            )

        # The approval itself must still be untouched (pending), since the
        # run-version guard rejected the write before it could take effect.
        untouched = repo.get_approval(pending.id)
        assert untouched.status == "pending"
