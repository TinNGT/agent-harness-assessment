from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from harness.api.schemas import (
    ApprovalDecisionRequest,
    CreateRunRequest,
    RunOut,
    ToolSpecOut,
    TraceOut,
    TraceStepOut,
)
from harness.core.service import HarnessService
from harness.storage.repository import ConflictError, NotFoundError

router = APIRouter()


def get_service(request: Request) -> HarnessService:
    return request.app.state.service


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/tools", response_model=list[ToolSpecOut])
async def list_tools(request: Request) -> list[ToolSpecOut]:
    service = get_service(request)
    return [ToolSpecOut.model_validate(spec) for spec in service.list_tools()]


@router.post("/runs", response_model=RunOut, status_code=201)
async def create_run(body: CreateRunRequest, request: Request) -> RunOut:
    service = get_service(request)
    try:
        run = await service.create_and_run(
            objective=body.objective,
            llm_provider=body.llm if body.llm != "real" else "litellm",
            scenario=body.scenario,
            max_steps=body.max_steps,
            max_run_seconds=body.max_run_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pending = await service.get_pending_approval(run.id)
    return RunOut.from_model(run, pending)


@router.get("/runs", response_model=list[RunOut])
async def list_runs(request: Request, status: Optional[str] = None) -> list[RunOut]:
    service = get_service(request)
    runs = await service.list_runs(status)
    out = []
    for run in runs:
        pending = await service.get_pending_approval(run.id) if run.status == "waiting_approval" else None
        out.append(RunOut.from_model(run, pending))
    return out


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(run_id: str, request: Request) -> RunOut:
    service = get_service(request)
    try:
        run = await service.get_run(run_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    pending = await service.get_pending_approval(run_id)
    return RunOut.from_model(run, pending)


@router.get("/runs/{run_id}/trace", response_model=TraceOut)
async def get_trace(run_id: str, request: Request) -> TraceOut:
    service = get_service(request)
    try:
        steps = await service.get_trace(run_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TraceOut(run_id=run_id, steps=[TraceStepOut.from_model(s) for s in steps])


@router.post("/runs/{run_id}/approvals/{approval_id}", response_model=RunOut)
async def decide_approval(
    run_id: str, approval_id: str, body: ApprovalDecisionRequest, request: Request
) -> RunOut:
    service = get_service(request)
    try:
        run = await service.decide_approval(
            run_id=run_id,
            approval_id=approval_id,
            decision=body.decision,
            approver=body.approver,
            reason=body.reason,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    pending = await service.get_pending_approval(run_id)
    return RunOut.from_model(run, pending)
