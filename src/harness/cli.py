from __future__ import annotations

import asyncio
import json

import typer

from harness.config import get_settings
from harness.core.service import HarnessService
from harness.storage.models import Run

app = typer.Typer(add_completion=False, help="Agent Harness CLI for an Operations Assistant")

_service: HarnessService | None = None


def _get_service() -> HarnessService:
    global _service
    if _service is None:
        _service = HarnessService(get_settings())
    return _service


def _print_run(run: Run) -> None:
    typer.echo(f"run_id     : {run.id}")
    typer.echo(f"status     : {run.status}")
    typer.echo(f"steps      : {run.step_count}/{run.max_steps}")
    if run.final_answer:
        typer.echo(f"answer     : {run.final_answer}")
    if run.error:
        typer.secho(f"error      : {run.error}", fg=typer.colors.RED)


@app.command()
def run(
    objective: str = typer.Argument(..., help="The user objective for the agent"),
    llm: str = typer.Option("scripted", help="scripted | real"),
    scenario: str = typer.Option(None, help="Scripted scenario name (see `harness scenarios`)"),
    max_steps: int = typer.Option(None, help="Override MAX_STEPS for this run"),
    approver: str = typer.Option("cli-user", help="Name recorded as the approver on decisions"),
) -> None:
    """Run an objective end-to-end, prompting for approval when required."""

    async def _run() -> None:
        service = _get_service()
        current = await service.create_and_run(
            objective=objective, llm_provider=llm, scenario=scenario, max_steps=max_steps
        )
        while current.status == "waiting_approval":
            pending = await service.get_pending_approval(current.id)
            assert pending is not None
            typer.secho(
                f"\nApproval required: {pending.tool_name}({json.dumps(pending.args)})",
                fg=typer.colors.YELLOW,
            )
            approved = typer.confirm("Approve?", default=False)
            decision = "approve" if approved else "reject"
            reason = None if decision == "approve" else typer.prompt("Reason for rejection", default="")
            current = await service.decide_approval(
                run_id=current.id,
                approval_id=pending.id,
                decision=decision,
                approver=approver,
                reason=reason or None,
            )
        _print_run(current)

    asyncio.run(_run())


@app.command()
def show(run_id: str) -> None:
    """Show the current status of a run."""

    async def _show() -> None:
        service = _get_service()
        current = await service.get_run(run_id)
        _print_run(current)

    asyncio.run(_show())


@app.command()
def trace(run_id: str) -> None:
    """Print the full execution trace of a run."""

    async def _trace() -> None:
        service = _get_service()
        steps = await service.get_trace(run_id)
        for step in steps:
            typer.echo(f"[{step.idx:03d}] {step.type:20s} {json.dumps(step.payload)}")

    asyncio.run(_trace())


@app.command()
def approve(
    run_id: str,
    approval_id: str,
    approver: str = typer.Option("cli-user"),
    reject: bool = typer.Option(False, help="Reject instead of approve"),
    reason: str = typer.Option(None, help="Reason (required for reject, optional for approve)"),
) -> None:
    """Approve or reject a pending approval directly by ID."""

    async def _decide() -> None:
        service = _get_service()
        current = await service.decide_approval(
            run_id=run_id,
            approval_id=approval_id,
            decision="reject" if reject else "approve",
            approver=approver,
            reason=reason,
        )
        _print_run(current)

    asyncio.run(_decide())


@app.command()
def tools() -> None:
    """List the registered tools and their schemas."""
    service = _get_service()
    for spec in service.list_tools():
        typer.echo(json.dumps(spec, indent=2))


@app.command()
def scenarios() -> None:
    """List available scripted-LLM scenario names."""
    service = _get_service()
    for name in service.list_scenarios():
        typer.echo(name)


if __name__ == "__main__":
    app()
