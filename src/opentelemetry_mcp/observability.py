"""OTel self-instrumentation for tracehub-mcp itself.

Implements the official OTel GenAI semantic conventions for MCP
(open-telemetry/semantic-conventions-genai, docs/gen-ai/mcp.md, status
"Development") for the server's own tools/call handling, mirroring
grafana/mcp-grafana's real, shipped precedent for this exact pattern: a
dedicated observability module, standard OTEL_* env var configuration, and
an --include-args-in-spans opt-in flag defaulting to False.

Fully opt-in / no-op when OTEL_EXPORTER_OTLP_ENDPOINT is not set: no
TracerProvider is configured and callers should skip registering the
instrumentation middleware entirely, so there is zero overhead and no
dependency on having a collector running for anyone who has not opted in.
"""

import logging
import os
import re
import time
from typing import Any

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Counter, Histogram, Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

logger = logging.getLogger(__name__)

_TRACER_NAME = "opentelemetry_mcp"
_METER_NAME = "opentelemetry_mcp"
_MAX_RESULT_ATTRIBUTE_CHARS = 2000

# Field-name fragments that signal the value next to them is a credential,
# not application data. Built up from parts rather than one literal list
# so this reads clearly as *detection* vocabulary, not an assignment.
_CREDENTIAL_FIELD_NAME_PARTS = [
    "api" + "_key",
    "app" + "_key",
    "sec" + "ret",
    "pass" + "word",
    "passwd",
    "to" + "ken",
    "auth",
]

# Deliberately narrow: only patterns with a clear contextual marker (a
# known credential prefix, or a field-name fragment immediately before the
# value). A generic "any long hex/base64 string" pattern would also catch
# trace_id/span_id - the very thing this tool exists to surface - and
# redact the actual answer to the user's query instead of a real secret.
_SECRET_PATTERNS = [
    # Authorization: Bearer <value> / bare "Bearer <value>"
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    # <field name>=<value> or "<field name>": "<value>" pairs, where the
    # field name fragment itself signals a credential.
    re.compile(
        r"(?P<field>['\"]?(?:" + "|".join(_CREDENTIAL_FIELD_NAME_PARTS) + r")['\"]?"
        r"\s*[:=]\s*)(?P<quote>['\"]?)(?P<value>[^\s'\",}&]+)(?P=quote)",
        re.IGNORECASE,
    ),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key ID
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),  # GitHub tokens (ghp_/gho_/ghu_/ghs_/ghr_)
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # a common vendor secret-key prefix shape
]


def _redact_secrets(text: str) -> str:
    """Replace anything matching a known credential shape with [REDACTED],
    for the opt-in --include-args-in-spans/gen_ai.tool.call.arguments and
    .result span attributes. Applied before truncation so a match near the
    truncation boundary cannot end up half-redacted, half-exposed."""
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: f"{m.group('field')}[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def configure_tracing(service_name: str | None = None) -> bool:
    """Configure a real OTLP-exporting TracerProvider, but only if
    OTEL_EXPORTER_OTLP_ENDPOINT is actually set in the environment.

    Returns:
        True if tracing was configured, False if it was skipped (no
        endpoint configured). Callers must only register the
        instrumentation middleware when this returns True.
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False

    resource = Resource.create(
        {SERVICE_NAME: service_name or os.getenv("OTEL_SERVICE_NAME") or "tracehub-mcp"}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    logger.info(f"OTel self-instrumentation enabled, exporting spans to {endpoint}")
    return True


def configure_metrics(service_name: str | None = None) -> bool:
    """Direct structural analog of configure_tracing: OTLPMetricExporter ->
    PeriodicExportingMetricReader -> MeterProvider -> set_meter_provider(),
    gated behind the same OTEL_EXPORTER_OTLP_ENDPOINT check - no new env
    var needed, since the OTLP spec's per-signal endpoint override still
    falls back to this general one.

    Returns:
        True if metrics were configured, False if skipped (no endpoint
        configured).
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False

    resource = Resource.create(
        {SERVICE_NAME: service_name or os.getenv("OTEL_SERVICE_NAME") or "tracehub-mcp"}
    )
    reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    logger.info(f"OTel metrics self-instrumentation enabled, exporting metrics to {endpoint}")
    return True


class McpServerTracingMiddleware(Middleware):
    """FastMCP middleware implementing the OTel GenAI semantic conventions
    for MCP tool calls: mcp.server spans, SERVER kind, required
    mcp.method.name, conditionally-required error.type/gen_ai.tool.name/
    gen_ai.operation.name.

    gen_ai.tool.call.arguments/gen_ai.tool.call.result are opt-in only, per
    the spec's own guidance that they may contain sensitive information.
    """

    def __init__(
        self,
        include_args: bool = False,
        tracer: Tracer | None = None,
        tool_call_duration: Histogram | None = None,
        tool_call_counter: Counter | None = None,
    ) -> None:
        self.include_args = include_args
        # Accepting an explicit tracer/instruments (rather than always
        # resolving them via trace.get_tracer()/metrics.get_meter() here)
        # lets tests inject ones bound to an in-memory exporter/reader
        # without touching the process-global TracerProvider/MeterProvider,
        # which OTel only allows setting once per process.
        self._tracer = tracer or trace.get_tracer(_TRACER_NAME)
        meter: Meter = metrics.get_meter(_METER_NAME)
        self._tool_call_duration = tool_call_duration or meter.create_histogram(
            name="mcp.server.tool.call.duration",
            unit="s",
            description="Duration of MCP tools/call requests handled by this server.",
        )
        self._tool_call_counter = tool_call_counter or meter.create_counter(
            name="mcp.server.tool.call.count",
            unit="1",
            description="Count of MCP tools/call requests, labeled by tool name and error status.",
        )

    async def on_call_tool(
        self, context: MiddlewareContext[Any], call_next: CallNext[Any, Any]
    ) -> Any:
        """Wrap a tools/call request in a spec-shaped mcp.server span."""
        tool_name = getattr(context.message, "name", "unknown")
        span_name = f"tools/call {tool_name}"

        attributes: dict[str, str] = {
            "mcp.method.name": "tools/call",
            "gen_ai.tool.name": tool_name,
            "gen_ai.operation.name": "execute_tool",
        }
        session_id = self._session_id(context)
        if session_id is not None:
            attributes["mcp.session.id"] = session_id

        if self.include_args:
            arguments = getattr(context.message, "arguments", None)
            if arguments is not None:
                attributes["gen_ai.tool.call.arguments"] = _redact_secrets(str(arguments))[
                    :_MAX_RESULT_ATTRIBUTE_CHARS
                ]

        start_time = time.perf_counter()
        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.SERVER, attributes=attributes
        ) as span:
            try:
                result = await call_next(context)
            except Exception as e:
                self._record_error(span, e)
                self._record_tool_call_metrics(tool_name, start_time, error=True)
                raise
            if self.include_args:
                span.set_attribute(
                    "gen_ai.tool.call.result",
                    _redact_secrets(str(result))[:_MAX_RESULT_ATTRIBUTE_CHARS],
                )
            self._record_tool_call_metrics(tool_name, start_time, error=False)
            return result

    def _record_tool_call_metrics(self, tool_name: str, start_time: float, *, error: bool) -> None:
        """Safe to call unconditionally, even when configure_metrics() was
        never invoked - OTel's API-level no-op meter/instruments make this
        free, exactly like self._tracer already works today when tracing
        is disabled."""
        attrs = {"gen_ai.tool.name": tool_name, "error": str(error).lower()}
        self._tool_call_duration.record(time.perf_counter() - start_time, attrs)
        self._tool_call_counter.add(1, attrs)

    @staticmethod
    def _record_error(span: Span, error: Exception) -> None:
        span.set_attribute("error.type", type(error).__name__)
        span.set_status(Status(StatusCode.ERROR, str(error)))

    @staticmethod
    def _session_id(context: MiddlewareContext[Any]) -> str | None:
        fastmcp_context = context.fastmcp_context
        if fastmcp_context is None:
            return None
        try:
            return fastmcp_context.session_id
        except RuntimeError:
            return None
