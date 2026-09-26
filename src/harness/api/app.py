from __future__ import annotations

from fastapi import FastAPI

from harness.api.routes import router
from harness.config import Settings, get_settings
from harness.core.service import HarnessService
from harness.observability.logging import configure_logging
from harness.observability.tracing import configure_tracing


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    configure_tracing("agent-harness", settings.otel_exporter_otlp_endpoint)

    app = FastAPI(
        title="Agent Harness — Operations Assistant",
        description=(
            "LLM-tool execution harness with approval gating, retries, timeouts, "
            "budget limits, and full execution traces."
        ),
        version="0.1.0",
    )
    app.state.service = HarnessService(settings)
    app.include_router(router)
    return app


app = create_app()
