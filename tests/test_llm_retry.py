from __future__ import annotations

from typing import Any

import pytest

from harness.core.runner import AgentRunner
from harness.core.state import RunStatus, transition
from harness.llm.base import LLMClient, build_system_message
from harness.llm.errors import LLMAPIError
from harness.llm.scripted import ScriptedLLM
from harness.storage.repository import Repository
from harness.tools.executor import ToolExecutor


class FlakyAPILLM(LLMClient):
    """Raises LLMAPIError on the first `fail_times` calls, then delegates."""

    def __init__(self, fail_times: int, delegate: LLMClient) -> None:
        self.fail_times = fail_times
        self.delegate = delegate
        self.calls = 0

    async def step(self, messages: list[dict[str, Any]], tools) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise LLMAPIError(f"simulated transient API error #{self.calls}")
        return await self.delegate.step(messages, tools)


def _start_run(service, llm_client, scenario: str = "happy_path"):
    session = service._session()
    repo = Repository(session)
    executor = ToolExecutor(
        timeout_seconds=service.settings.tool_timeout_seconds,
        max_retries=service.settings.tool_max_retries,
    )
    runner = AgentRunner(
        repo=repo,
        registry=service.registry,
        executor=executor,
        clock=service.clock,
        settings=service.settings,
        llm_client=llm_client,
    )
    system_message = build_system_message(service.registry.list())
    run = repo.create_run(
        objective="Check payment-api",
        llm_provider="scripted",
        scenario=scenario,
        max_steps=service.settings.max_steps,
        max_run_seconds=service.settings.max_run_seconds,
        now=service.clock.now(),
        system_message=system_message,
    )
    run.status = transition(RunStatus(run.status), RunStatus.RUNNING).value
    repo.save_run(run)
    return repo, runner, run


@pytest.mark.asyncio
async def test_llm_api_error_retries_then_succeeds(make_service):
    service = make_service(llm_max_retries=3)
    delegate = ScriptedLLM("happy_path", service.scripts)
    flaky = FlakyAPILLM(fail_times=2, delegate=delegate)
    _repo, runner, run = _start_run(service, flaky)

    result = await runner.drive(run)

    assert result.status == "completed"
    # happy_path takes 3 LLM turns total (2 tool calls + 1 final answer); the
    # first 2 raise LLMAPIError and only tip the retry budget on turn 1, so
    # calls = 2 forced failures + 3 successful turns.
    assert flaky.calls == 5


@pytest.mark.asyncio
async def test_llm_api_error_exhausts_retries_and_fails(make_service):
    service = make_service(llm_max_retries=3)
    delegate = ScriptedLLM("happy_path", service.scripts)
    flaky = FlakyAPILLM(fail_times=99, delegate=delegate)  # always fails
    _repo, runner, run = _start_run(service, flaky)

    result = await runner.drive(run)

    assert result.status == "failed"
    assert result.error["type"] == "LLMAPIError"
    assert flaky.calls == 3  # exactly llm_max_retries attempts, then gives up
