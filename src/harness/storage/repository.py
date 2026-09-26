from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlmodel import Session, select, update

from harness.core.state import ApprovalStatus, RunStatus
from harness.storage.models import Approval, Incident, Run, Step


def args_hash(args: dict[str, Any]) -> str:
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ConflictError(Exception):
    """Raised on optimistic-lock / duplicate-decision races."""


class NotFoundError(Exception):
    """Raised when a requested entity does not exist."""


class Repository:
    """Thin persistence layer. One instance wraps one DB session/transaction."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ---------------------------------------------------------------- runs
    def create_run(
        self,
        *,
        objective: str,
        llm_provider: str,
        scenario: Optional[str],
        max_steps: int,
        max_run_seconds: int,
        now: datetime,
        system_message: dict[str, Any],
    ) -> Run:
        run = Run(
            objective=objective,
            llm_provider=llm_provider,
            scenario=scenario,
            max_steps=max_steps,
            max_run_seconds=max_run_seconds,
            started_at=now,
            deadline_at=now + timedelta(seconds=max_run_seconds),
            messages=[system_message, {"role": "user", "content": objective}],
        )
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def get_run(self, run_id: str) -> Run:
        run = self.session.get(Run, run_id)
        if run is None:
            raise NotFoundError(f"run {run_id} not found")
        return run

    def list_runs(self, status: Optional[str] = None) -> list[Run]:
        stmt = select(Run).order_by(Run.created_at.desc())
        if status:
            stmt = stmt.where(Run.status == status)
        return list(self.session.exec(stmt))

    def save_run(self, run: Run, *, bump_version: bool = True) -> Run:
        if bump_version:
            run.version += 1
        run.updated_at = datetime.now(timezone.utc)
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    # --------------------------------------------------------------- steps
    def append_step(
        self,
        run: Run,
        *,
        type: str,
        payload: dict[str, Any],
        attempt: int = 1,
        latency_ms: Optional[int] = None,
    ) -> Step:
        idx = self.session.exec(
            select(Step).where(Step.run_id == run.id).order_by(Step.idx.desc())
        ).first()
        next_idx = (idx.idx + 1) if idx else 0
        step = Step(
            run_id=run.id,
            idx=next_idx,
            type=type,
            payload=payload,
            attempt=attempt,
            latency_ms=latency_ms,
        )
        self.session.add(step)
        self.session.commit()
        self.session.refresh(step)
        return step

    def get_trace(self, run_id: str) -> list[Step]:
        stmt = select(Step).where(Step.run_id == run_id).order_by(Step.idx)
        return list(self.session.exec(stmt))

    # ----------------------------------------------------------- approvals
    def create_approval(self, run: Run, *, tool_call_id: str, tool_name: str, args: dict[str, Any]) -> Approval:
        approval = Approval(
            run_id=run.id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=args,
            args_hash=args_hash(args),
            status=ApprovalStatus.PENDING.value,
        )
        self.session.add(approval)
        self.session.commit()
        self.session.refresh(approval)
        return approval

    def get_approval(self, approval_id: str) -> Approval:
        approval = self.session.get(Approval, approval_id)
        if approval is None:
            raise NotFoundError(f"approval {approval_id} not found")
        return approval

    def get_pending_approval_for_run(self, run_id: str) -> Optional[Approval]:
        stmt = select(Approval).where(
            Approval.run_id == run_id, Approval.status == ApprovalStatus.PENDING.value
        )
        return self.session.exec(stmt).first()

    def decide_approval_atomic(
        self,
        *,
        approval_id: str,
        run_id: str,
        decision: str,
        approver: str,
        reason: Optional[str],
        now: datetime,
        expected_run_version: int,
    ) -> Approval:
        """Atomically flip a pending approval to approved/rejected.

        Two concurrent decide requests race on the `status='pending'` guard in the
        UPDATE's WHERE clause; only one can succeed. The run's `version` is bumped
        under the same guard so a stale caller gets a 409 instead of silently
        clobbering a decision made in between its read and write.
        """
        result = self.session.exec(
            update(Approval)
            .where(Approval.id == approval_id, Approval.status == ApprovalStatus.PENDING.value)
            .values(status=decision, decided_by=approver, reason=reason, decided_at=now)
        )
        if result.rowcount == 0:
            self.session.rollback()
            raise ConflictError(f"approval {approval_id} is not pending")

        run_result = self.session.exec(
            update(Run)
            .where(Run.id == run_id, Run.version == expected_run_version)
            .values(version=expected_run_version + 1, status=RunStatus.RUNNING.value)
        )
        if run_result.rowcount == 0:
            self.session.rollback()
            raise ConflictError(f"run {run_id} version conflict")

        self.session.commit()
        return self.get_approval(approval_id)

    # ------------------------------------------------------------ incidents
    def create_incident_if_not_exists(
        self, *, idempotency_key: str, title: str, description: str, severity: str
    ) -> tuple[Incident, bool]:
        """Returns (incident, created). Replays the same incident for a repeated key."""
        existing = self.session.exec(
            select(Incident).where(Incident.idempotency_key == idempotency_key)
        ).first()
        if existing:
            return existing, False

        count = len(list(self.session.exec(select(Incident))))
        incident = Incident(
            id=f"INC-{1001 + count}",
            idempotency_key=idempotency_key,
            title=title,
            description=description,
            severity=severity,
        )
        self.session.add(incident)
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            existing = self.session.exec(
                select(Incident).where(Incident.idempotency_key == idempotency_key)
            ).first()
            if existing:
                return existing, False
            raise
        self.session.refresh(incident)
        return incident, True
