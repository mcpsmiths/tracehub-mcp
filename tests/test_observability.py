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
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp.server.middleware import MiddlewareContext
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from opentelemetry_mcp import observability
from opentelemetry_mcp.observability import (
    McpServerTracingMiddleware,
    TraceContextLogFilter,
    configure_metrics,
    configure_tracing,
    install_trace_context_log_filter,
)


def _tracer_with_exporter() -> tuple[Any, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
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


def _context(
    *, tool_name: str = "search_traces", arguments: dict[str, Any] | None = None
) -> MiddlewareContext[Any]:
    message = type("Msg", (), {"name": tool_name, "arguments": arguments})()
    return MiddlewareContext(message=message, method="tools/call", fastmcp_context=None)


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

    def test_sets_fastmcp_telemetry_mode_to_propagation_only_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FastMCP's own native spans (its middleware seam span plus its
        internal server_span()) duplicate this codebase's own
        McpServerTracingMiddleware span for the same tools/call operation -
        confirmed via a real end-to-end reproduction (see
        test_double_tracing_regression.py) that this produces THREE
        SERVER-kind spans for one tool call without this setting.
        propagation_only still correctly parents this codebase's own spans
        under any incoming distributed trace context - it only stops
        FastMCP from creating its own competing spans."""
        import fastmcp

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setattr(fastmcp.settings, "telemetry_mode", "native")

        with patch.object(observability.trace, "set_tracer_provider"):
            configure_tracing()

        assert fastmcp.settings.telemetry_mode == "propagation_only"

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

        await middleware.on_call_tool(_context(), call_next)

        assert "mcp.session.id" not in _last_span_attrs(exporter)

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
