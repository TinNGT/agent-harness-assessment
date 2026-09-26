from __future__ import annotations

import pytest
from typer.testing import CliRunner

import harness.cli as cli_module
from harness.config import get_settings

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_cli_env(tmp_path, monkeypatch):
    """Point the CLI's settings/service singletons at an isolated test DB."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'cli-test.db'}")
    monkeypatch.setenv("DATA_DIR", str(DATA_DIR))
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", "0.2")
    get_settings.cache_clear()
    cli_module._service = None
    yield
    cli_module._service = None
    get_settings.cache_clear()


def test_cli_tools_lists_all_three_tools():
    result = runner.invoke(cli_module.app, ["tools"])
    assert result.exit_code == 0
    for name in ("search_knowledge_base", "get_service_status", "create_incident"):
        assert name in result.stdout


def test_cli_scenarios_lists_happy_path():
    result = runner.invoke(cli_module.app, ["scenarios"])
    assert result.exit_code == 0
    assert "happy_path" in result.stdout


def test_cli_run_happy_path_completes():
    result = runner.invoke(cli_module.app, ["run", "Check payment-api", "--scenario", "happy_path"])
    assert result.exit_code == 0
    assert "status     : completed" in result.stdout
    assert "answer" in result.stdout


def test_cli_run_with_approval_prompts_and_approves():
    result = runner.invoke(
        cli_module.app,
        ["run", "Handle auth-service outage", "--scenario", "approval_create_incident"],
        input="y\n",
    )
    assert result.exit_code == 0
    assert "Approval required" in result.stdout
    assert "status     : completed" in result.stdout


def test_cli_run_with_approval_rejects_on_no():
    result = runner.invoke(
        cli_module.app,
        ["run", "Handle auth-service outage", "--scenario", "approval_create_incident"],
        input="n\nnot needed\n",
    )
    assert result.exit_code == 0
    assert "status     : completed" in result.stdout  # reject still lets the run finish
