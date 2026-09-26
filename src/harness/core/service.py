from __future__ import annotations

from typing import Any, Optional

from sqlmodel import Session

from harness.config import Settings
from harness.core.approval import ApprovalGate
from harness.core.clock import Clock, SystemClock
from harness.core.runner import AgentRunner
from harness.core.state import RunStatus, transition
from harness.llm.base import LLMClient, build_system_message
from harness.llm.litellm_client import LiteLLMClient
from harness.llm.scripted import ScriptedLLM
from harness.storage.db import get_session, init_db
from harness.storage.models import Approval, Run, Step
from harness.storage.repository import NotFoundError, Repository
from harness.tools.executor import ToolExecutor
from harness.tools.mock_tools import build_registry
from harness.tools.registry import ToolRegistry


class HarnessService:
    """The one object both the FastAPI routes and the CLI call into. Keeping the
    orchestration here (instead of in routes.py) means the API is a thin transport
    layer and the CLI never has to duplicate agent-loop logic."""

    def __init__(self, settings: Settings, clock: Optional[Clock] = None) -> None:
        self.settings = settings
        self.clock = clock or SystemClock()
        self.registry: ToolRegistry = build_registry(settings.data_path)
        self.scripts = ScriptedLLM.load_scripts(settings.data_path / "scenarios.json")
        init_db(settings.database_url)

    def _session(self) -> Session:
        return get_session(self.settings.database_url)

    def _make_llm_client(self, provider: str, scenario: Optional[str]) -> LLMClient:
        if provider == "scripted":
            if not scenario:
                raise ValueError("'scenario' is required when llm_provider='scripted'")
            return ScriptedLLM(scenario, self.scripts)
        if provider in ("litellm", "real"):
            return LiteLLMClient(
                self.settings.llm_model,
                openai_api_key=self.settings.openai_api_key,
                anthropic_api_key=self.settings.anthropic_api_key,
            )
        raise ValueError(f"unknown llm provider '{provider}'")

    def _make_runner(self, repo: Repository, llm_client: LLMClient) -> AgentRunner:
        executor = ToolExecutor(
            timeout_seconds=self.settings.tool_timeout_seconds,
            max_retries=self.settings.tool_max_retries,
        )
        return AgentRunner(
            repo=repo,
            registry=self.registry,
            executor=executor,
            clock=self.clock,
            settings=self.settings,
            llm_client=llm_client,
        )

    # -------------------------------------------------------------------- runs
    async def create_and_run(
        self,
        *,
        objective: str,
        llm_provider: str = "scripted",
        scenario: Optional[str] = None,
        max_steps: Optional[int] = None,
        max_run_seconds: Optional[int] = None,
    ) -> Run:
        llm_client = self._make_llm_client(llm_provider, scenario)  # fail fast, before writing a row
        with self._session() as session:
            repo = Repository(session)
            system_message = build_system_message(self.registry.list())
            run = repo.create_run(
                objective=objective,
                llm_provider=llm_provider,
                scenario=scenario,
                max_steps=max_steps or self.settings.max_steps,
                max_run_seconds=max_run_seconds or self.settings.max_run_seconds,
                now=self.clock.now(),
                system_message=system_message,
            )
            run.status = transition(RunStatus(run.status), RunStatus.RUNNING).value
            repo.save_run(run)
            runner = self._make_runner(repo, llm_client)
            return await runner.drive(run)

    async def get_run(self, run_id: str) -> Run:
        with self._session() as session:
            return Repository(session).get_run(run_id)

    async def list_runs(self, status: Optional[str] = None) -> list[Run]:
        with self._session() as session:
            return Repository(session).list_runs(status)

    async def get_trace(self, run_id: str) -> list[Step]:
        with self._session() as session:
            repo = Repository(session)
            repo.get_run(run_id)  # raises NotFoundError if missing
            return repo.get_trace(run_id)

    async def get_pending_approval(self, run_id: str) -> Optional[Approval]:
        with self._session() as session:
            return Repository(session).get_pending_approval_for_run(run_id)

    # --------------------------------------------------------------- approvals
    async def decide_approval(
        self,
        *,
        run_id: str,
        approval_id: str,
        decision: str,
        approver: str,
        reason: Optional[str],
    ) -> Run:
        with self._session() as session:
            repo = Repository(session)
            run = repo.get_run(run_id)
            approval_row = repo.get_approval(approval_id)
            if approval_row.run_id != run_id:
                raise NotFoundError(f"approval {approval_id} does not belong to run {run_id}")

            gate = ApprovalGate(repo)
            approval = gate.decide(
                run=run, approval_id=approval_id, decision=decision, approver=approver,
                reason=reason, now=self.clock.now(),
            )
            # status/version were updated via a raw atomic UPDATE; re-fetch the ORM object.
            run = repo.get_run(run_id)
            llm_client = self._make_llm_client(run.llm_provider, run.scenario)
            runner = self._make_runner(repo, llm_client)
            return await runner.resume_after_decision(run, approval)

    # ------------------------------------------------------------------ tools
    def list_tools(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.registry.list()]

    def list_scenarios(self) -> list[str]:
        return sorted(self.scripts.keys())
