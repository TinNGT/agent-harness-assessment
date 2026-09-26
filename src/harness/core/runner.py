from __future__ import annotations

import json
from typing import Optional

import structlog

from harness.config import Settings
from harness.core.budget import BudgetExceeded, BudgetGuard
from harness.core.clock import Clock
from harness.core.state import ApprovalStatus, RunStatus, StepType, transition
from harness.llm.base import LLMClient
from harness.llm.errors import LLMAPIError, LLMParseError
from harness.llm.parser import LLMDecision, parse_decision
from harness.observability.tracing import get_tracer
from harness.storage.models import Approval, Run
from harness.storage.repository import Repository
from harness.tools.errors import ToolInputValidationError
from harness.tools.executor import ToolExecutor
from harness.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)
tracer = get_tracer(__name__)


class AgentRunner:
    """Drives a single run's LLM <-> tool loop. See docs/report.md section 4."""

    def __init__(
        self,
        *,
        repo: Repository,
        registry: ToolRegistry,
        executor: ToolExecutor,
        clock: Clock,
        settings: Settings,
        llm_client: LLMClient,
    ) -> None:
        self.repo = repo
        self.registry = registry
        self.executor = executor
        self.clock = clock
        self.settings = settings
        self.llm = llm_client
        self.budget = BudgetGuard(clock)

    # ------------------------------------------------------------------ main loop
    async def drive(self, run: Run) -> Run:
        """Advance `run` until it completes, fails, hits a limit, or pauses for approval."""
        with tracer.start_as_current_span("run", attributes={"run_id": run.id}):
            return await self._drive(run)

    async def _drive(self, run: Run) -> Run:
        while True:
            try:
                self.budget.check(run)
            except BudgetExceeded as exc:
                run.status = transition(RunStatus(run.status), RunStatus.LIMIT_EXCEEDED).value
                run.error = {"type": "BudgetExceeded", "message": exc.reason}
                self.repo.append_step(run, type=StepType.ERROR.value, payload={"error": run.error})
                self.repo.save_run(run)
                logger.info("run.limit_exceeded", run_id=run.id, reason=exc.reason)
                return run

            decision = await self._llm_turn(run)
            if RunStatus(run.status) == RunStatus.FAILED:
                return run
            if decision is None:
                continue  # malformed output within repair budget; try the LLM again

            if decision.type == "final":
                run.final_answer = decision.answer
                run.status = transition(RunStatus(run.status), RunStatus.COMPLETED).value
                run.step_count += 1
                self.repo.append_step(run, type=StepType.FINAL.value, payload={"answer": decision.answer})
                self.repo.save_run(run)
                logger.info("run.completed", run_id=run.id)
                return run

            assert decision.tool_call is not None
            call = decision.tool_call
            tool = self.registry.get(call.name)

            if tool is None:
                still_alive = self._register_llm_mistake(
                    run,
                    message_for_llm=(
                        f"Unknown tool '{call.name}'. Valid tools: "
                        f"{', '.join(self.registry.names())}."
                    ),
                    error_type="UnknownTool",
                    message=f"unknown tool '{call.name}'",
                    tool_call_id=call.id,
                )
                if not still_alive:
                    return run
                continue

            try:
                args = tool.validate_input(call.arguments)
            except ToolInputValidationError as exc:
                still_alive = self._register_llm_mistake(
                    run,
                    message_for_llm=f"Tool call rejected: {exc.message}",
                    error_type="ToolInputValidationError",
                    message=exc.message,
                    tool_call_id=call.id,
                )
                if not still_alive:
                    return run
                continue

            run.consecutive_llm_errors = 0

            if tool.requires_approval:
                approval = self.repo.create_approval(
                    run, tool_call_id=call.id, tool_name=tool.name, args=args.model_dump()
                )
                self.repo.append_step(
                    run,
                    type=StepType.APPROVAL_REQUESTED.value,
                    payload={"approval_id": approval.id, "tool": tool.name, "args": args.model_dump()},
                )
                run.status = transition(RunStatus(run.status), RunStatus.WAITING_APPROVAL).value
                run.step_count += 1
                self.repo.save_run(run)
                logger.info("run.waiting_approval", run_id=run.id, approval_id=approval.id, tool=tool.name)
                return run

            self.repo.append_step(
                run,
                type=StepType.TOOL_CALL.value,
                payload={"tool_call_id": call.id, "tool": tool.name, "arguments": args.model_dump()},
            )
            result = await self.executor.execute(tool, args, run_id=run.id)
            observation = result.data if result.ok else {"error": result.error}
            self._observe(
                run, call.id, tool.name, observation, StepType.TOOL_RESULT.value,
                attempt=result.attempts, latency_ms=result.latency_ms,
            )
            run.step_count += 1
            self.repo.save_run(run)

    # ------------------------------------------------------------------ approval resume
    async def resume_after_decision(self, run: Run, approval: Approval) -> Run:
        """Apply an approve/reject decision's effect, then continue the loop."""
        if approval.status == ApprovalStatus.APPROVED.value:
            tool = self.registry.get(approval.tool_name)
            assert tool is not None, f"approved tool '{approval.tool_name}' vanished from the registry"
            args = tool.input_model.model_validate(approval.args)
            self.repo.append_step(
                run,
                type=StepType.APPROVAL_DECIDED.value,
                payload={
                    "approval_id": approval.id, "decision": "approved",
                    "approver": approval.decided_by, "reason": approval.reason,
                },
            )
            self.repo.append_step(
                run,
                type=StepType.TOOL_CALL.value,
                payload={"tool_call_id": approval.tool_call_id, "tool": tool.name, "arguments": approval.args},
            )
            result = await self.executor.execute(
                tool, args, run_id=run.id, idempotency_key=approval.id, repo=self.repo
            )
            observation = result.data if result.ok else {"error": result.error}
            self._observe(
                run, approval.tool_call_id, tool.name, observation, StepType.TOOL_RESULT.value,
                attempt=result.attempts, latency_ms=result.latency_ms,
            )
        else:
            self.repo.append_step(
                run,
                type=StepType.APPROVAL_DECIDED.value,
                payload={
                    "approval_id": approval.id, "decision": "rejected",
                    "approver": approval.decided_by, "reason": approval.reason,
                },
            )
            observation = {"rejected": True, "reason": approval.reason or "no reason given"}
            self._observe(run, approval.tool_call_id, approval.tool_name, observation, StepType.TOOL_RESULT.value)

        run.step_count += 1
        self.repo.save_run(run)
        return await self.drive(run)

    # ------------------------------------------------------------------ helpers
    async def _llm_turn(self, run: Run) -> Optional[LLMDecision]:
        raw: Optional[str] = None
        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                with tracer.start_as_current_span(
                    "llm_call", attributes={"run_id": run.id, "attempt": attempt}
                ):
                    raw = await self.llm.step(run.messages, self.registry.list())
                break
            except LLMAPIError as exc:
                self.repo.append_step(
                    run, type=StepType.ERROR.value,
                    payload={"error": "llm_api_error", "message": str(exc), "attempt": attempt},
                )
                logger.warning("llm.api_error", run_id=run.id, attempt=attempt, error=str(exc))
                if attempt >= self.settings.llm_max_retries:
                    run.status = transition(RunStatus(run.status), RunStatus.FAILED).value
                    run.error = {"type": "LLMAPIError", "message": str(exc)}
                    self.repo.save_run(run)
                    return None

        assert raw is not None
        run.messages.append({"role": "assistant", "content": raw})
        self.repo.append_step(run, type=StepType.LLM_CALL.value, payload={"raw": raw})

        try:
            return parse_decision(raw)
        except LLMParseError as exc:
            self._register_llm_mistake(
                run,
                message_for_llm=(
                    f"Your last response could not be parsed: {exc}. Reply again with a single "
                    "valid JSON object as instructed."
                ),
                error_type="LLMParseError",
                message=str(exc),
            )
            # Caller checks run.status directly to decide whether to stop or retry.
            return None

    def _register_llm_mistake(
        self,
        run: Run,
        *,
        message_for_llm: str,
        error_type: str,
        message: str,
        tool_call_id: Optional[str] = None,
    ) -> bool:
        """Record a malformed-LLM-output event and bump the repair counter.

        Returns True if the run is still alive (caller should keep looping),
        False if the repair budget just ran out and the run became FAILED.
        """
        run.consecutive_llm_errors += 1
        run.messages.append({"role": "user", "content": message_for_llm})
        self.repo.append_step(
            run, type=StepType.ERROR.value,
            payload={"error_type": error_type, "message": message, "tool_call_id": tool_call_id},
        )
        if run.consecutive_llm_errors > self.settings.llm_max_repair:
            run.status = transition(RunStatus(run.status), RunStatus.FAILED).value
            run.error = {"type": error_type, "message": message}
            logger.info("run.failed_repair_exhausted", run_id=run.id, error_type=error_type)
        self.repo.save_run(run)
        return RunStatus(run.status) != RunStatus.FAILED

    def _observe(
        self,
        run: Run,
        call_id: str,
        tool_name: str,
        observation: dict,
        step_type: str,
        *,
        attempt: int = 1,
        latency_ms: Optional[int] = None,
    ) -> None:
        run.messages.append(
            {
                "role": "user",
                "content": f"Observation for tool_call {call_id} ({tool_name}): {json.dumps(observation)}",
            }
        )
        self.repo.append_step(
            run,
            type=step_type,
            payload={"tool_call_id": call_id, "tool": tool_name, "observation": observation},
            attempt=attempt,
            latency_ms=latency_ms,
        )
