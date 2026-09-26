from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column
from sqlalchemy.ext.mutable import MutableList
from sqlmodel import Field, SQLModel

from harness.core.state import RunStatus


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    id: str = Field(default_factory=_uuid, primary_key=True)
    objective: str
    status: str = Field(default=RunStatus.PENDING.value, index=True)
    llm_provider: str = "scripted"
    scenario: Optional[str] = None

    step_count: int = 0
    max_steps: int
    max_run_seconds: int
    started_at: datetime = Field(default_factory=_utcnow)
    deadline_at: datetime

    # Consecutive malformed-LLM-output counter (JSON parse errors, unknown tool,
    # invalid args). Reset on any successful step. Independent from step_count.
    consecutive_llm_errors: int = 0

    # MutableList wraps the JSON column so in-place .append() calls are tracked by
    # SQLAlchemy's unit-of-work; a plain list would silently fail to persist appends.
    messages: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(MutableList.as_mutable(JSON))
    )
    final_answer: Optional[str] = None
    error: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    version: int = 0
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class Step(SQLModel, table=True):
    __tablename__ = "steps"

    id: str = Field(default_factory=_uuid, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    idx: int
    type: str
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    attempt: int = 1
    latency_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=_utcnow)


class Approval(SQLModel, table=True):
    __tablename__ = "approvals"

    id: str = Field(default_factory=_uuid, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True)
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    args_hash: str
    status: str = Field(default="pending", index=True)
    decided_by: Optional[str] = None
    reason: Optional[str] = None
    decided_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=_utcnow)


class Incident(SQLModel, table=True):
    __tablename__ = "incidents"

    id: str = Field(primary_key=True)  # e.g. INC-1001
    idempotency_key: str = Field(unique=True, index=True)
    title: str
    description: str
    severity: str
    created_at: datetime = Field(default_factory=_utcnow)
