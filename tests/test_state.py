from __future__ import annotations

import pytest

from harness.core.state import InvalidTransitionError, RunStatus, can_transition, transition


def test_valid_transition_returns_target_status():
    assert transition(RunStatus.PENDING, RunStatus.RUNNING) == RunStatus.RUNNING
    assert transition(RunStatus.RUNNING, RunStatus.WAITING_APPROVAL) == RunStatus.WAITING_APPROVAL
    assert transition(RunStatus.WAITING_APPROVAL, RunStatus.RUNNING) == RunStatus.RUNNING


def test_invalid_transition_raises():
    with pytest.raises(InvalidTransitionError):
        transition(RunStatus.COMPLETED, RunStatus.RUNNING)


def test_cannot_skip_straight_from_pending_to_completed():
    with pytest.raises(InvalidTransitionError):
        transition(RunStatus.PENDING, RunStatus.COMPLETED)


@pytest.mark.parametrize(
    "terminal", [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.LIMIT_EXCEEDED]
)
def test_terminal_statuses_have_no_outgoing_transitions(terminal):
    for target in RunStatus:
        assert can_transition(terminal, target) is False
