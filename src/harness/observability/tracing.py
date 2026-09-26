from __future__ import annotations

from opentelemetry import trace

_configured = False


def configure_tracing(service_name: str, otlp_endpoint: str) -> None:
    """Optional: export spans to an OTLP collector (e.g. Jaeger via docker-compose).
    If OTLP_EXPORTER_OTLP_ENDPOINT is empty, the OpenTelemetry API defaults to a
    no-op tracer, so calling get_tracer()/start_as_current_span() elsewhere in the
    codebase stays free of `if tracing_enabled` branches."""
    global _configured
    if not otlp_endpoint or _configured:
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer(name: str):
    return trace.get_tracer(name)
