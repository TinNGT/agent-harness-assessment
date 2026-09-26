from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    LIMIT_EXCEEDED = "limit_exceeded"


class StepType(StrEnum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    ERROR = "error"
    FINAL = "final"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


TERMINAL_STATUSES = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.LIMIT_EXCEEDED}

# Allowed status transitions. WAITING_APPROVAL and RUNNING can loop back and forth;
# terminal states never transition again.
_ALLOWED_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.PENDING: {RunStatus.RUNNING},
    RunStatus.RUNNING: {
        RunStatus.RUNNING,
        RunStatus.COMPLETED,
        RunStatus.WAITING_APPROVAL,
        RunStatus.FAILED,
        RunStatus.LIMIT_EXCEEDED,
    },
    RunStatus.WAITING_APPROVAL: {RunStatus.RUNNING},
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
    RunStatus.LIMIT_EXCEEDED: set(),
}


class InvalidTransitionError(Exception):
    def __init__(self, current: RunStatus, target: RunStatus) -> None:
        super().__init__(f"Cannot transition run from {current} to {target}")
        self.current = current
        self.target = target


def can_transition(current: RunStatus, target: RunStatus) -> bool:
    return target in _ALLOWED_TRANSITIONS.get(current, set())


def transition(current: RunStatus, target: RunStatus) -> RunStatus:
    """Validate and return the new status, raising if the transition is illegal.

    This is the single choke point every status change must go through so that,
    e.g., an already-COMPLETED run can never be re-opened by a stray approval.
    """
    if not can_transition(current, target):
        raise InvalidTransitionError(current, target)
    return target
