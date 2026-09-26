from __future__ import annotations

from datetime import datetime
from typing import Optional

from harness.core.state import ApprovalStatus
from harness.storage.models import Approval, Run
from harness.storage.repository import Repository


class ApprovalGate:
    """Server-anchored approval: the client only ever sends approve/reject + who/why.
    The tool name and arguments always come from what the agent originally proposed,
    stored server-side — a client can never smuggle in different arguments at
    decision time."""

    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    def request(self, run: Run, *, tool_call_id: str, tool_name: str, args: dict) -> Approval:
        return self.repo.create_approval(run, tool_call_id=tool_call_id, tool_name=tool_name, args=args)

    _DECISION_TO_STATUS = {
        "approve": ApprovalStatus.APPROVED.value,
        "reject": ApprovalStatus.REJECTED.value,
    }

    def decide(
        self,
        *,
        run: Run,
        approval_id: str,
        decision: str,
        approver: str,
        reason: Optional[str],
        now: datetime,
    ) -> Approval:
        status = self._DECISION_TO_STATUS.get(decision)
        if status is None:
            raise ValueError(f"invalid decision '{decision}', expected 'approve' or 'reject'")
        return self.repo.decide_approval_atomic(
            approval_id=approval_id,
            run_id=run.id,
            decision=status,
            approver=approver,
            reason=reason,
            now=now,
            expected_run_version=run.version,
        )
