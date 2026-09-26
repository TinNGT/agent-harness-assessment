from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import Settings
from harness.core.clock import FakeClock
from harness.core.service import HarnessService

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    db_path = tmp_path / "harness-test.db"
    defaults = dict(
        llm_provider="scripted",
        data_dir=str(DATA_DIR),
        database_url=f"sqlite:///{db_path}",
        max_steps=10,
        max_run_seconds=60,
        tool_timeout_seconds=0.2,
        tool_max_retries=2,
        llm_max_retries=3,
        llm_max_repair=2,
        log_level="WARNING",
        otel_exporter_otlp_endpoint="",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_service(tmp_path: Path):
    """Factory fixture: make_service(**settings_overrides) -> HarnessService."""

    created: list[HarnessService] = []

    def _make(clock=None, **overrides) -> HarnessService:
        settings = make_settings(tmp_path, **overrides)
        service = HarnessService(settings, clock=clock)
        created.append(service)
        return service

    return _make
