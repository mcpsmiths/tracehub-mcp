"""Tests for OTel self-instrumentation (observability.py).

configure_tracing tests never actually call the real
opentelemetry.trace.set_tracer_provider - OTel only allows that to
succeed once per process (subsequent calls are silently ignored with a
warning), so a second successful call anywhere in this test session would
make every later test's tracer resolve to whatever provider happened to
win first, and a real provider would spin up a background export thread
against a fake endpoint. Patching trace.set_tracer_provider verifies
configure_tracing's own logic (env var gating, resource attributes)
without any process-global side effect.

McpServerTracingMiddleware tests inject an explicit tracer bound to an
InMemorySpanExporter instead, for the same reason - real span content
assertions without ever installing a global provider.

CallNext is a generic Protocol; mypy cannot match an ad-hoc local async
closure against it structurally even though the closures below are
runtime-correct (the tests pass) and match the real call_next shape
FastMCP's own middleware chain passes. observability.py imports `trace`
without an explicit `as name` re-export, so mypy's strict
`--no-implicit-reexport` treats `observability.trace` as undefined even
though patching it is exactly how these tests verify configure_tracing's
own logic without a real process-global side effect (same friction
test_server.py documents for its own module-level patch targets).
Disabling arg-type/attr-defined here only, narrowly, for these two known
false-positive frictions.
"""

# mypy: disable-error-code="arg-type, attr-defined"

import logging
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client
from fastmcp.server.middleware import MiddlewareContext
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import Sampler
from opentelemetry.trace import SpanKind, StatusCode
from pydantic import HttpUrl
from starlette.middleware import Middleware
from starlette.testclient import TestClient

from opentelemetry_mcp import observability, server
from opentelemetry_mcp.config import BackendConfig, ServerConfig
from opentelemetry_mcp.models import SearchTracesResult
from opentelemetry_mcp.observability import (
    _MAX_STATUS_DESCRIPTION_CHARS,
    McpServerTracingMiddleware,
    TraceContextLogFilter,
    _bounded_status_description,
    _redact_secrets,
    configure_metrics,
    configure_tracing,
    install_trace_context_log_filter,
)
from opentelemetry_mcp.security import redact_userinfo_in_text
from opentelemetry_mcp.server import OriginValidationMiddleware, RateLimitMiddleware
from tests.streamable_http_client import call_tool, open_session, sse_json


def _tracer_with_exporter(sampler: Sampler | None = None) -> tuple[Any, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=sampler)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def _instruments_with_reader() -> tuple[Any, Any, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")
    duration = meter.create_histogram("mcp.server.tool.call.duration", unit="s")
    counter = meter.create_counter("mcp.server.tool.call.count")
    return duration, counter, reader


def _metric_data_points(reader: InMemoryMetricReader, metric_name: str) -> list[Any]:
    data = reader.get_metrics_data()
    points: list[Any] = []
    if data is None:
        return points
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == metric_name:
                    points.extend(metric.data.data_points)
    return points


def _last_span_attrs(exporter: InMemorySpanExporter) -> dict[str, Any]:
    span = exporter.get_finished_spans()[-1]
    assert span.attributes is not None
    return dict(span.attributes)


class _FakeFastMCPContext:
    """Minimal stand-in for fastmcp's Context as the middleware sees it: a
    tool registry that resolves only ``registered`` names (FastMCP.get_tool
    is async and returns None for anything unknown or disabled) and no
    session, like the stdio transport (Context.session_id raises)."""

    def __init__(self, registered: frozenset[str] = frozenset({"search_traces"})) -> None:
        self.fastmcp = SimpleNamespace(
            get_tool=AsyncMock(side_effect=lambda name: object() if name in registered else None)
        )

    @property
    def session_id(self) -> str:
        raise RuntimeError("no session")


_DEFAULT_FASTMCP_CONTEXT = object()


def _context(
    *,
    tool_name: Any = "search_traces",
    arguments: dict[str, Any] | None = None,
    fastmcp_context: Any = _DEFAULT_FASTMCP_CONTEXT,
) -> MiddlewareContext[Any]:
    message = type("Msg", (), {"name": tool_name, "arguments": arguments})()
    if fastmcp_context is _DEFAULT_FASTMCP_CONTEXT:
        fastmcp_context = _FakeFastMCPContext()
    return MiddlewareContext(message=message, method="tools/call", fastmcp_context=fastmcp_context)


class TestConfigureTracing:
    def test_returns_false_when_endpoint_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

        with patch.object(observability.trace, "set_tracer_provider") as mocked:
            result = configure_tracing()

        assert result is False
        mocked.assert_not_called()

    def test_returns_true_and_configures_provider_when_endpoint_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)

        with patch.object(observability.trace, "set_tracer_provider") as mocked:
            result = configure_tracing(service_name="my-service")

        assert result is True
        mocked.assert_called_once()
        provider = mocked.call_args.args[0]
        assert provider.resource.attributes["service.name"] == "my-service"

    def test_falls_back_to_otel_service_name_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-configured-name")

        with patch.object(observability.trace, "set_tracer_provider") as mocked:
            configure_tracing()

        provider = mocked.call_args.args[0]
        assert provider.resource.attributes["service.name"] == "env-configured-name"

    def test_defaults_to_tracehub_mcp_when_nothing_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)

        with patch.object(observability.trace, "set_tracer_provider") as mocked:
            configure_tracing()

        provider = mocked.call_args.args[0]
        assert provider.resource.attributes["service.name"] == "tracehub-mcp"

    def test_untrusted_callers_set_fastmcp_telemetry_mode_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FastMCP's own native spans (its middleware seam span plus its
        internal server_span()) duplicate this codebase's own
        McpServerTracingMiddleware span for the same tools/call operation -
        confirmed via a real end-to-end reproduction (see
        test_double_tracing_regression.py) that this produces THREE
        SERVER-kind spans for one tool call at the default "native" mode.
        Both "off" and "propagation_only" suppress them; "off" additionally
        stops FastMCP from adopting a caller's params._meta.traceparent as
        the parent of this server's own span, which is the default for
        callers the operator has not marked trusted."""
        import fastmcp

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setattr(fastmcp.settings, "telemetry_mode", "native")

        with patch.object(observability.trace, "set_tracer_provider"):
            configure_tracing(trust_client_trace_context=False)

        assert fastmcp.settings.telemetry_mode == "off"

    def test_trusted_callers_set_fastmcp_telemetry_mode_propagation_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import fastmcp

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setattr(fastmcp.settings, "telemetry_mode", "native")

        with patch.object(observability.trace, "set_tracer_provider"):
            configure_tracing(trust_client_trace_context=True)

        assert fastmcp.settings.telemetry_mode == "propagation_only"

    def test_default_is_untrusted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import fastmcp

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setattr(fastmcp.settings, "telemetry_mode", "native")

        with patch.object(observability.trace, "set_tracer_provider"):
            configure_tracing()

        assert fastmcp.settings.telemetry_mode == "off"

    def test_sampler_never_lets_a_remote_unsampled_parent_drop_the_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The SDK default (parentbased_always_on) delegates a remote parent
        with trace-flags=00 to ALWAYS_OFF - i.e. the caller decides whether
        the operator's own span is recorded. Pin the explicit sampler so a
        dependency bump that changes the SDK default is caught too."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.delenv("OTEL_TRACES_SAMPLER", raising=False)

        with patch.object(observability.trace, "set_tracer_provider") as mocked:
            configure_tracing()

        description = mocked.call_args.args[0].sampler.get_description()
        assert "root:AlwaysOnSampler" in description
        assert "remoteParentNotSampled:AlwaysOnSampler" in description

    def test_operator_otel_traces_sampler_override_still_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_off")

        with patch.object(observability.trace, "set_tracer_provider") as mocked:
            configure_tracing()

        assert mocked.call_args.args[0].sampler.get_description() == "AlwaysOffSampler"

    def test_does_not_touch_fastmcp_telemetry_mode_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import fastmcp

        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.setattr(fastmcp.settings, "telemetry_mode", "native")

        configure_tracing()

        assert fastmcp.settings.telemetry_mode == "native"


class TestConfigureMetrics:
    """Mirrors TestConfigureTracing exactly - configure_metrics is the
    direct structural analog of configure_tracing, gated behind the same
    env var."""

    def test_returns_false_when_endpoint_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

        with patch.object(observability.metrics, "set_meter_provider") as mocked:
            result = configure_metrics()

        assert result is False
        mocked.assert_not_called()

    def test_returns_true_and_configures_provider_when_endpoint_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)

        with patch.object(observability.metrics, "set_meter_provider") as mocked:
            result = configure_metrics(service_name="my-service")

        assert result is True
        mocked.assert_called_once()
        provider = mocked.call_args.args[0]
        # MeterProvider (unlike TracerProvider) doesn't expose .resource
        # publicly - it's on the internal _sdk_config, confirmed via source.
        assert provider._sdk_config.resource.attributes["service.name"] == "my-service"

    def test_falls_back_to_otel_service_name_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-configured-name")

        with patch.object(observability.metrics, "set_meter_provider") as mocked:
            configure_metrics()

        provider = mocked.call_args.args[0]
        assert provider._sdk_config.resource.attributes["service.name"] == "env-configured-name"

    def test_defaults_to_tracehub_mcp_when_nothing_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)

        with patch.object(observability.metrics, "set_meter_provider") as mocked:
            configure_metrics()

        provider = mocked.call_args.args[0]
        assert provider._sdk_config.resource.attributes["service.name"] == "tracehub-mcp"


class TestMcpServerTracingMiddlewareMetrics:
    async def test_records_duration_and_count_on_success(self) -> None:
        tracer, _exporter = _tracer_with_exporter()
        duration, counter, reader = _instruments_with_reader()
        middleware = McpServerTracingMiddleware(
            tracer=tracer, tool_call_duration=duration, tool_call_counter=counter
        )

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return "ok"

        await middleware.on_call_tool(_context(tool_name="search_traces"), call_next)

        count_points = _metric_data_points(reader, "mcp.server.tool.call.count")
        duration_points = _metric_data_points(reader, "mcp.server.tool.call.duration")
        assert count_points[0].attributes == {"gen_ai.tool.name": "search_traces", "error": "false"}
        assert count_points[0].value == 1
        assert duration_points[0].attributes == {
            "gen_ai.tool.name": "search_traces",
            "error": "false",
        }
        assert duration_points[0].sum >= 0.0

    async def test_records_error_true_on_exception(self) -> None:
        tracer, _exporter = _tracer_with_exporter()
        duration, counter, reader = _instruments_with_reader()
        middleware = McpServerTracingMiddleware(
            tracer=tracer, tool_call_duration=duration, tool_call_counter=counter
        )

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            raise RuntimeError("backend down")

        with pytest.raises(RuntimeError, match="backend down"):
            await middleware.on_call_tool(_context(tool_name="search_traces"), call_next)

        count_points = _metric_data_points(reader, "mcp.server.tool.call.count")
        assert count_points[0].attributes == {"gen_ai.tool.name": "search_traces", "error": "true"}
        assert count_points[0].value == 1


class TestTraceContextLogFilter:
    def test_stamps_zero_placeholders_when_no_span_recording(self) -> None:
        record = logging.LogRecord("test", logging.INFO, "path", 1, "message", None, None)
        log_filter = TraceContextLogFilter()

        result = log_filter.filter(record)

        assert result is True
        assert record.trace_id == "0" * 32
        assert record.span_id == "0" * 16
        assert record.trace_flags == "00"

    def test_stamps_real_ids_when_span_is_recording(self) -> None:
        tracer, _exporter = _tracer_with_exporter()
        log_filter = TraceContextLogFilter()

        with tracer.start_as_current_span("x") as span:
            record = logging.LogRecord("test", logging.INFO, "path", 1, "message", None, None)
            log_filter.filter(record)

            ctx = span.get_span_context()
            assert record.trace_id == f"{ctx.trace_id:032x}"
            assert record.span_id == f"{ctx.span_id:016x}"

    def test_install_attaches_filter_and_formatter_to_given_handlers(self) -> None:
        handler = logging.Handler()

        install_trace_context_log_filter([handler])

        assert any(isinstance(f, TraceContextLogFilter) for f in handler.filters)
        assert handler.formatter is not None
        fmt = handler.formatter._fmt
        assert fmt is not None
        assert "trace_id" in fmt


class TestMcpServerTracingMiddlewareHappyPath:
    async def test_span_has_required_and_conditionally_required_attributes(self) -> None:
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(tracer=tracer)

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return "ok"

        result = await middleware.on_call_tool(_context(tool_name="search_traces"), call_next)

        assert result == "ok"
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "tools/call search_traces"
        assert span.kind == SpanKind.SERVER
        attrs = _last_span_attrs(exporter)
        assert attrs["mcp.method.name"] == "tools/call"
        assert attrs["gen_ai.tool.name"] == "search_traces"
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert span.status.status_code == StatusCode.UNSET

    async def test_does_not_set_error_type_on_success(self) -> None:
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(tracer=tracer)

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return "ok"

        await middleware.on_call_tool(_context(), call_next)

        assert "error.type" not in _last_span_attrs(exporter)


class TestMcpServerTracingMiddlewareErrorHandling:
    async def test_exception_propagates_and_sets_error_status(self) -> None:
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(tracer=tracer)

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            raise RuntimeError("backend down")

        with pytest.raises(RuntimeError, match="backend down"):
            await middleware.on_call_tool(_context(), call_next)

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        assert _last_span_attrs(exporter)["error.type"] == "RuntimeError"

    async def test_exception_message_is_redacted_regardless_of_include_args(self) -> None:
        """Unlike gen_ai.tool.call.arguments/.result, str(error) reaches
        Status(...) unconditionally on every failed tool call - it must be
        redacted the same way even when include_args is False (the
        default), since a backend can raise with a credential-shaped
        string embedded in its exception message. Builds the fake
        credential via concatenation so this file's own static content has
        no contiguous secret-shaped literal for a scanner to flag - only
        the assembled runtime value matches the pattern under test."""
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(include_args=False, tracer=tracer)
        fake_token = "Bear" + "er " + "sk-" + "abc123def456ghi789jklmnopqrstuv"

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            raise RuntimeError(f"upstream rejected {fake_token}")

        with pytest.raises(RuntimeError, match="upstream rejected"):
            await middleware.on_call_tool(_context(), call_next)

        span = exporter.get_finished_spans()[0]
        assert span.status.status_code == StatusCode.ERROR
        description = span.status.description
        assert description is not None
        assert fake_token not in description
        assert "REDACTED" in description

    async def test_exception_message_url_userinfo_is_redacted(self) -> None:
        """An httpx.HTTPStatusError message embeds the request URL verbatim;
        a credential a bypassed config left in BACKEND_URL's userinfo must
        not reach the OTLP collector via the span status."""
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(include_args=False, tracer=tracer)
        marker = "VFY-PW-span"

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            raise RuntimeError(
                f"Client error '401 Unauthorized' for url 'https://u:{marker}@h.test/api'"
            )

        with pytest.raises(RuntimeError, match="401"):
            await middleware.on_call_tool(_context(), call_next)

        description = exporter.get_finished_spans()[0].status.description
        assert description is not None
        assert marker not in description
        assert "https://[REDACTED]@h.test/api" in description


class TestMcpServerTracingMiddlewareIncludeArgs:
    async def test_include_args_false_omits_arguments_and_result(self) -> None:
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(include_args=False, tracer=tracer)

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return "some result"

        await middleware.on_call_tool(_context(arguments={"limit": 5}), call_next)

        attrs = _last_span_attrs(exporter)
        assert "gen_ai.tool.call.arguments" not in attrs
        assert "gen_ai.tool.call.result" not in attrs

    async def test_include_args_true_includes_arguments_and_result(self) -> None:
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(include_args=True, tracer=tracer)

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return "some result"

        await middleware.on_call_tool(_context(arguments={"limit": 5}), call_next)

        attrs = _last_span_attrs(exporter)
        assert "limit" in str(attrs["gen_ai.tool.call.arguments"])
        assert attrs["gen_ai.tool.call.result"] == "some result"

    async def test_include_args_true_but_no_arguments_present_sets_nothing(self) -> None:
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(include_args=True, tracer=tracer)

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return "result"

        await middleware.on_call_tool(_context(arguments=None), call_next)

        assert "gen_ai.tool.call.arguments" not in _last_span_attrs(exporter)

    async def test_include_args_true_result_is_truncated(self) -> None:
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(include_args=True, tracer=tracer)
        huge_result = "x" * 5000

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return huge_result

        await middleware.on_call_tool(_context(), call_next)

        result_attr = str(_last_span_attrs(exporter)["gen_ai.tool.call.result"])
        assert len(result_attr) == 2000

    async def test_include_args_true_redacts_a_credential_in_the_result(self) -> None:
        """Found by a production-audit pass: gen_ai.tool.call.result is
        opt-in, but nothing prevented a credential appearing in backend
        data (or, less plausibly, in an argument) from being exported
        verbatim. Builds the fake credential via concatenation so this
        file's own static content has no contiguous secret-shaped
        literal for a scanner to flag - only the assembled runtime value
        matches the pattern under test."""
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(include_args=True, tracer=tracer)
        fake_token = "Bear" + "er " + "sk-" + "abc123def456ghi789jklmnopqrstuv"
        leaky_result = "{" + '"error": "upstream rejected ' + fake_token + '"}'

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return leaky_result

        await middleware.on_call_tool(_context(), call_next)

        result_attr = str(_last_span_attrs(exporter)["gen_ai.tool.call.result"])
        assert "REDACTED" in result_attr
        assert fake_token not in result_attr

    async def test_include_args_true_does_not_redact_trace_or_span_ids(self) -> None:
        """The whole point of these tools is to surface trace_id/span_id -
        long hex strings that must never be mistaken for a credential and
        redacted away."""
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(include_args=True, tracer=tracer)
        real_looking_result = (
            '{"trace_id": "4bf92f3577b34da6a3ce929d0e0e4736", '
            '"span_id": "00f067aa0ba902b7", "service_name": "checkout-service"}'
        )

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return real_looking_result

        await middleware.on_call_tool(_context(), call_next)

        result_attr = str(_last_span_attrs(exporter)["gen_ai.tool.call.result"])
        assert result_attr == real_looking_result
        assert "REDACTED" not in result_attr


class TestMcpServerTracingMiddlewareSessionId:
    async def test_session_id_included_when_fastmcp_context_provides_one(self) -> None:
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(tracer=tracer)
        fake_ctx = AsyncMock()
        fake_ctx.session_id = "session-abc"
        context = MiddlewareContext(
            message=type("Msg", (), {"name": "search_traces", "arguments": None})(),
            method="tools/call",
            fastmcp_context=fake_ctx,
        )

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return "ok"

        await middleware.on_call_tool(context, call_next)

        assert _last_span_attrs(exporter)["mcp.session.id"] == "session-abc"

    async def test_session_id_omitted_when_fastmcp_context_is_none(self) -> None:
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(tracer=tracer)

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return "ok"

        await middleware.on_call_tool(_context(fastmcp_context=None), call_next)

        assert "mcp.session.id" not in _last_span_attrs(exporter)


class TestMcpServerTracingMiddlewareCardinality:
    """The gen_ai.tool.name label (and the span name) must be bounded by the
    server's own registry plus one sentinel. fastmcp runs the middleware
    chain *before* it resolves the tool, so a never-registered name still
    reaches on_call_tool; the opentelemetry-sdk keeps one aggregation per
    distinct attribute set for the life of the process with no cardinality
    limit, so echoing a peer-chosen string grew process memory and the
    exported series without bound (about 2.85 KB and two series per name)."""

    @pytest.fixture(autouse=True)
    def _real_server_state(self) -> Generator[None]:
        """A real minimal config/backend so the registered-tool control can
        reach the patched search_traces, plus the add_middleware restore
        pattern from test_double_tracing_regression.py."""
        original_middleware = list(server.mcp.middleware)
        server._config = ServerConfig(
            backend=BackendConfig(type="jaeger", url=HttpUrl("http://localhost:16686"))
        )
        server._backend = AsyncMock()
        server.mcp.auth = None
        yield
        server.mcp.middleware = original_middleware
        server._config = None
        server._backend = None

    @staticmethod
    def _install_instrumented_middleware() -> tuple[InMemorySpanExporter, InMemoryMetricReader]:
        tracer, exporter = _tracer_with_exporter()
        duration, counter, reader = _instruments_with_reader()
        server.mcp.add_middleware(
            McpServerTracingMiddleware(
                tracer=tracer, tool_call_duration=duration, tool_call_counter=counter
            )
        )
        return exporter, reader

    async def test_fifty_unknown_names_collapse_to_one_sentinel_label(self) -> None:
        exporter, reader = self._install_instrumented_middleware()

        async with Client(server.mcp) as client:
            for i in range(50):
                result = await client.call_tool(f"never_registered_{i}", {}, raise_on_error=False)
                assert result.is_error

        count_points = _metric_data_points(reader, "mcp.server.tool.call.count")
        duration_points = _metric_data_points(reader, "mcp.server.tool.call.duration")
        assert len(count_points) == 1
        assert count_points[0].attributes == {"gen_ai.tool.name": "unknown", "error": "true"}
        assert count_points[0].value == 50
        assert len(duration_points) == 1
        assert duration_points[0].count == 50
        span_names = {s.name for s in exporter.get_finished_spans() if "tools/call" in s.name}
        assert span_names == {"tools/call unknown"}

    async def test_registered_tool_keeps_its_real_name(self) -> None:
        exporter, reader = self._install_instrumented_middleware()
        sentinel = SearchTracesResult(count=0, traces=[])

        with patch.object(server.search, "search_traces", AsyncMock(return_value=sentinel)):
            async with Client(server.mcp) as client:
                result = await client.call_tool("search_traces", {"limit": 5})

        assert not result.is_error
        count_points = _metric_data_points(reader, "mcp.server.tool.call.count")
        assert count_points[0].attributes == {"gen_ai.tool.name": "search_traces", "error": "false"}
        assert [s.name for s in exporter.get_finished_spans()] == ["tools/call search_traces"]

    @pytest.mark.parametrize(
        ("tool_name", "fastmcp_context"),
        [
            ("x" * 129, _DEFAULT_FASTMCP_CONTEXT),
            (123, _DEFAULT_FASTMCP_CONTEXT),
            (None, _DEFAULT_FASTMCP_CONTEXT),
            ("search_traces", None),
        ],
        ids=["over-length", "non-str", "missing-name", "no-fastmcp-context"],
    )
    async def test_unresolvable_names_collapse_to_the_sentinel(
        self, tool_name: Any, fastmcp_context: Any
    ) -> None:
        tracer, exporter = _tracer_with_exporter()
        duration, counter, reader = _instruments_with_reader()
        middleware = McpServerTracingMiddleware(
            tracer=tracer, tool_call_duration=duration, tool_call_counter=counter
        )

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return "ok"

        await middleware.on_call_tool(
            _context(tool_name=tool_name, fastmcp_context=fastmcp_context), call_next
        )

        assert exporter.get_finished_spans()[0].name == "tools/call unknown"
        assert _last_span_attrs(exporter)["gen_ai.tool.name"] == "unknown"
        count_points = _metric_data_points(reader, "mcp.server.tool.call.count")
        assert count_points[0].attributes["gen_ai.tool.name"] == "unknown"

    async def test_name_at_the_length_cap_still_resolves(self) -> None:
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(tracer=tracer)
        long_but_registered = "y" * 128

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return "ok"

        await middleware.on_call_tool(
            _context(
                tool_name=long_but_registered,
                fastmcp_context=_FakeFastMCPContext(frozenset({long_but_registered})),
            ),
            call_next,
        )

        assert _last_span_attrs(exporter)["gen_ai.tool.name"] == long_but_registered

    async def test_unknown_tool_status_description_is_bounded(self) -> None:
        """probe-4 P5b: label and span name collapse to the sentinel, but
        fastmcp's NotFoundError text ("Unknown tool: '<name>'") carried the
        peer-chosen name verbatim into the exported span status - 5016
        characters for a 5000-character name. The description is a fixed
        size whatever the peer sends."""
        exporter, _reader = self._install_instrumented_middleware()
        name = "z" * 5000

        async with Client(server.mcp) as client:
            result = await client.call_tool(name, {}, raise_on_error=False)

        assert result.is_error
        (span,) = [s for s in exporter.get_finished_spans() if s.name == "tools/call unknown"]
        assert span.status.status_code == StatusCode.ERROR
        description = span.status.description
        assert description is not None
        assert description.startswith("Unknown tool: 'zzz"), description[:40]
        assert len(description) <= _MAX_STATUS_DESCRIPTION_CHARS, len(description)
        assert description.endswith("..."), description[-10:]

    def test_the_cut_may_split_an_escape_but_never_leaves_a_raw_control_character(
        self,
    ) -> None:
        """Characterisation of the real guarantee (round-2 review of this
        helper): the escape runs before the cut, so what the cut can split
        is an escape's text, never the control character it stands for.
        ESC lands exactly across the cut here."""
        error = LookupError("a" * (_MAX_STATUS_DESCRIPTION_CHARS - 4) + "\x1b[31m tail")

        description = _bounded_status_description(error)

        assert len(description) == _MAX_STATUS_DESCRIPTION_CHARS
        assert description.endswith("\\..."), description[-8:]  # cut inside "\\x1b"
        assert "\x1b" not in description
        assert description.isprintable()

    async def test_control_characters_in_the_status_description_are_escaped(self) -> None:
        """P5b's second probe: a name carrying NUL, a newline and an ANSI
        colour sequence reached the status text raw. Rendered repr-style,
        like the Host/Origin log lines and the root log handler."""
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(tracer=tracer)
        name = "list_services\x00\n\x1b[31m"

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            raise LookupError(f"Unknown tool: '{name}'")

        with pytest.raises(LookupError):
            await middleware.on_call_tool(_context(tool_name=name), call_next)

        description = exporter.get_finished_spans()[0].status.description
        assert description == "Unknown tool: 'list_services\\x00\\n\\x1b[31m'", description
        assert description.isprintable()

    def test_http_peer_cannot_mint_labels(self) -> None:
        """The attack surface is the streamable-HTTP listener: a peer that
        sends no Origin header passes OriginValidationMiddleware and gets
        an ordinary isError result for every unknown name."""
        _exporter, reader = self._install_instrumented_middleware()
        app = server.mcp.http_app(
            transport="streamable-http",
            middleware=[
                Middleware(OriginValidationMiddleware),
                Middleware(RateLimitMiddleware, max_requests=100, window_seconds=60.0),
            ],
        )

        with TestClient(app) as client:
            session_id = open_session(client)
            for i in range(5):
                response = call_tool(client, session_id, f"minted_{i}", request_id=10 + i)
                assert response.status_code == 200
                assert sse_json(response)["result"]["isError"] is True

        count_points = _metric_data_points(reader, "mcp.server.tool.call.count")
        assert [p.attributes["gen_ai.tool.name"] for p in count_points] == ["unknown"]
        assert count_points[0].value == 5


PEER_TRACE_ID = 0x0AF7651916CD43DD8448EB211C80319C
PEER_SPAN_ID = 0x00F067AA0BA902B7
PEER_TRACEPARENT_UNSAMPLED = "00-0af7651916cd43dd8448eb211c80319c-00f067aa0ba902b7-00"
PEER_TRACEPARENT_SAMPLED = "00-0af7651916cd43dd8448eb211c80319c-00f067aa0ba902b7-01"


class TestClientTraceContextTrustOverHttp:
    """An unauthenticated HTTP peer could decide, via params._meta.traceparent,
    whether this server's own tools/call span was recorded (trace-flags=00 +
    the SDK default sampler) and under which trace identity (fastmcp's
    propagation_only mode adopts the peer context as parent before any user
    middleware runs). The operator owns both decisions now: the explicit
    sampler records the span regardless of the remote flags, and a peer's
    _meta is only adopted as parent when trust_client_trace_context is on."""

    @pytest.fixture(autouse=True)
    def _real_server_state(self) -> Generator[None]:
        original_middleware = list(server.mcp.middleware)
        server._config = ServerConfig(
            backend=BackendConfig(type="jaeger", url=HttpUrl("http://localhost:16686"))
        )
        server._backend = AsyncMock()
        server.mcp.auth = None
        yield
        server.mcp.middleware = original_middleware
        server._config = None
        server._backend = None

    @staticmethod
    def _run_two_tool_calls_with_peer_trace_context(
        exporter: InMemorySpanExporter,
    ) -> list[Any]:
        """initialize + initialized, then tools/call search_traces twice: once
        with an unsampled peer traceparent, once sampled plus a tracestate.
        Returns the recorded SERVER spans, in call order."""
        sentinel = SearchTracesResult(count=0, traces=[])
        tool_logger = logging.getLogger("opentelemetry_mcp.tools.search")

        async def logging_search(*args: Any, **kwargs: Any) -> SearchTracesResult:
            tool_logger.info("handling search_traces for the trace-context test")
            return sentinel

        app = server.mcp.http_app(
            transport="streamable-http",
            middleware=[
                Middleware(OriginValidationMiddleware),
                Middleware(RateLimitMiddleware, max_requests=100, window_seconds=60.0),
            ],
        )
        with patch.object(server.search, "search_traces", AsyncMock(side_effect=logging_search)):
            with TestClient(app) as client:
                session_id = open_session(client)
                for request_id, meta in (
                    (10, {"traceparent": PEER_TRACEPARENT_UNSAMPLED}),
                    (11, {"traceparent": PEER_TRACEPARENT_SAMPLED, "tracestate": "peer=chosen"}),
                ):
                    response = call_tool(
                        client,
                        session_id,
                        "search_traces",
                        {"limit": 5},
                        meta=meta,
                        request_id=request_id,
                    )
                    assert response.status_code == 200
                    assert sse_json(response)["result"]["isError"] is False
        return [
            s
            for s in exporter.get_finished_spans()
            if "tools/call" in s.name and s.kind == SpanKind.SERVER
        ]

    def test_untrusted_peer_meta_is_neither_parent_nor_sampling_decision(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import fastmcp

        monkeypatch.setattr(fastmcp.settings, "telemetry_mode", "off")
        tracer, exporter = _tracer_with_exporter(sampler=observability._OPERATOR_OWNED_SAMPLER)
        server.mcp.add_middleware(McpServerTracingMiddleware(tracer=tracer))
        install_trace_context_log_filter([caplog.handler])

        with caplog.at_level(logging.INFO, logger="opentelemetry_mcp.tools.search"):
            spans = self._run_two_tool_calls_with_peer_trace_context(exporter)

        assert len(spans) == 2
        for span in spans:
            assert span.context.trace_id != PEER_TRACE_ID
            assert span.parent is None
            assert len(span.context.trace_state) == 0
        tool_records = [r for r in caplog.records if "handling search_traces" in r.getMessage()]
        assert [r.trace_id for r in tool_records] == [f"{s.context.trace_id:032x}" for s in spans]
        assert all(r.trace_id != "0" * 32 for r in tool_records)

    def test_trusted_peer_meta_is_adopted_but_cannot_drop_the_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import fastmcp

        monkeypatch.setattr(fastmcp.settings, "telemetry_mode", "propagation_only")
        tracer, exporter = _tracer_with_exporter(sampler=observability._OPERATOR_OWNED_SAMPLER)
        server.mcp.add_middleware(McpServerTracingMiddleware(tracer=tracer))

        spans = self._run_two_tool_calls_with_peer_trace_context(exporter)

        assert len(spans) == 2
        for span in spans:
            assert span.context.trace_id == PEER_TRACE_ID
            assert span.parent is not None
            assert span.parent.span_id == PEER_SPAN_ID
        assert spans[1].context.trace_state.get("peer") == "chosen"

    def test_sdk_default_sampler_would_let_the_unsampled_peer_drop_the_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Baseline proving the sampler half of the fix is load-bearing: with
        the SDK default sampler and the peer context adopted, trace-flags=00
        yields a non-recording span the exporter never sees."""
        import fastmcp

        monkeypatch.setattr(fastmcp.settings, "telemetry_mode", "propagation_only")
        tracer, exporter = _tracer_with_exporter()
        server.mcp.add_middleware(McpServerTracingMiddleware(tracer=tracer))

        spans = self._run_two_tool_calls_with_peer_trace_context(exporter)

        assert len(spans) == 1
        assert spans[0].context.trace_id == PEER_TRACE_ID

    async def test_session_id_omitted_when_accessing_it_raises(self) -> None:
        """Context.session_id raises RuntimeError if no session is
        available (e.g. stdio transport) - the middleware must not let
        that propagate and break the actual tool call."""
        tracer, exporter = _tracer_with_exporter()
        middleware = McpServerTracingMiddleware(tracer=tracer)
        fake_ctx = AsyncMock()
        type(fake_ctx).session_id = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("no session"))
        )
        context = MiddlewareContext(
            message=type("Msg", (), {"name": "search_traces", "arguments": None})(),
            method="tools/call",
            fastmcp_context=fake_ctx,
        )

        async def call_next(ctx: MiddlewareContext[Any]) -> str:
            return "ok"

        result = await middleware.on_call_tool(context, call_next)

        assert result == "ok"
        assert "mcp.session.id" not in _last_span_attrs(exporter)


class TestRedactSecretsUrlUserinfo:
    def test_userinfo_is_redacted_and_host_kept(self) -> None:
        assert _redact_secrets("GET https://u:VFY-PW-x@h.test/x?trace_id=abc123") == (
            "GET https://[REDACTED]@h.test/x?trace_id=abc123"
        )

    def test_plain_urls_are_untouched(self) -> None:
        text = "GET https://h.test/x?trace_id=abc123 by someone@example.com"
        assert _redact_secrets(text) == text

    def test_username_only_userinfo_is_redacted(self) -> None:
        """A token used as the Basic-auth username has no ':' - the pattern
        must not require one."""
        assert _redact_secrets("err for url 'https://TOKEN123@h.test/x'") == (
            "err for url 'https://[REDACTED]@h.test/x'"
        )

    def test_password_containing_at_is_fully_redacted(self) -> None:
        assert _redact_secrets("err for url 'https://u:p@ss@h.test/x'") == (
            "err for url 'https://[REDACTED]@h.test/x'"
        )

    def test_shares_the_security_module_redaction(self) -> None:
        """One implementation (security.redact_userinfo_in_text), not a
        second hand-rolled pattern the two sinks could drift between."""
        text = "GET https://u:VFY-PW-x@h.test/x then https://svc@two.test/ by someone@example.com"
        assert _redact_secrets(text) == redact_userinfo_in_text(text)
