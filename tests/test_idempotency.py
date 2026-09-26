from __future__ import annotations

import pytest
from sqlmodel import select

from harness.storage.models import Incident
from harness.storage.repository import Repository


@pytest.mark.asyncio
async def test_repeated_idempotency_key_replays_the_same_incident(make_service):
    service = make_service()

    with service._session() as session:
        repo = Repository(session)
        first, created_first = repo.create_incident_if_not_exists(
            idempotency_key="approval-123", title="X", description="Y", severity="high"
        )
        second, created_second = repo.create_incident_if_not_exists(
            idempotency_key="approval-123", title="X", description="Y", severity="high"
        )

        assert created_first is True
        assert created_second is False
        assert first.id == second.id

        all_incidents = list(session.exec(select(Incident)))
        assert len(all_incidents) == 1


@pytest.mark.asyncio
async def test_full_run_approve_then_manual_replay_does_not_duplicate(make_service):
    service = make_service()
    run = await service.create_and_run(
        objective="Handle the auth-service outage", scenario="approval_create_incident"
    )
    pending = await service.get_pending_approval(run.id)

    await service.decide_approval(
        run_id=run.id, approval_id=pending.id, decision="approve", approver="alice", reason=None
    )

    # Simulate a client blindly retrying the same already-approved request.
    with service._session() as session:
        repo = Repository(session)
        replayed, created = repo.create_incident_if_not_exists(
            idempotency_key=pending.id,
            title=pending.args["title"],
            description=pending.args["description"],
            severity=pending.args["severity"],
        )
        assert created is False

        all_incidents = list(session.exec(select(Incident)))
        assert len(all_incidents) == 1
        assert all_incidents[0].id == replayed.id
