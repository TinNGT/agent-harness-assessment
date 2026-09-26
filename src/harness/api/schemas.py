from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from harness.storage.models import Approval, Run, Step


class CreateRunRequest(BaseModel):
    objective: str = Field(min_length=1)
    llm: Literal["scripted", "real"] = "scripted"
    scenario: Optional[str] = Field(
        default=None, description="Required when llm='scripted'; picks a script from data/scenarios.json"
    )
    max_steps: Optional[int] = Field(default=None, gt=0)
    max_run_seconds: Optional[int] = Field(default=None, gt=0)


class ApprovalOut(BaseModel):
    id: str
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    args_hash: str
    status: str
    decided_by: Optional[str] = None
    reason: Optional[str] = None
    decided_at: Optional[datetime] = None
    created_at: datetime

    @classmethod
    def from_model(cls, approval: Approval) -> "ApprovalOut":
        return cls.model_validate(approval.model_dump())


class RunOut(BaseModel):
    id: str
    objective: str
    status: str
    llm_provider: str
    scenario: Optional[str] = None
    step_count: int
    max_steps: int
    max_run_seconds: int
    final_answer: Optional[str] = None
    error: Optional[dict[str, Any]] = None
    pending_approval: Optional[ApprovalOut] = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, run: Run, pending_approval: Optional[Approval] = None) -> "RunOut":
        data = run.model_dump()
        data["pending_approval"] = ApprovalOut.from_model(pending_approval) if pending_approval else None
        return cls.model_validate(data)


class ApprovalDecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    approver: str = Field(min_length=1)
    reason: Optional[str] = None


class TraceStepOut(BaseModel):
    idx: int
    type: str
    payload: dict[str, Any]
    attempt: int
    latency_ms: Optional[int] = None
    created_at: datetime

    @classmethod
    def from_model(cls, step: Step) -> "TraceStepOut":
        return cls.model_validate(step.model_dump())


class TraceOut(BaseModel):
    run_id: str
    steps: list[TraceStepOut]


class ToolSpecOut(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    requires_approval: bool
    idempotent: bool


class ErrorOut(BaseModel):
    detail: str
