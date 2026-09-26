from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    """Abstraction over wall-clock time so tests can control it without real sleeps."""

    def now(self) -> datetime: ...


class SystemClock:
    """Real clock used in production."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FakeClock:
    """Deterministic clock for tests. Call `advance()` to move time forward."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime.now(timezone.utc)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        from datetime import timedelta

        self._now += timedelta(seconds=seconds)

    def set(self, when: datetime) -> None:
        self._now = when
