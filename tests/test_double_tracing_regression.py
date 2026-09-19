"""Regression test for the double/triple-tracing bug fixed in
configure_tracing() (observability.py).

Confirmed via a real reproduction (not just source reading) against the
installed fastmcp 4.0.5: without setting `fastmcp.settings.telemetry_mode =
"propagation_only"`, a single tools/call operation produces THREE SERVER-kind
spans - FastMCP's own middleware seam span, FastMCP's own internal
server_span() (opened inside its recursive call_tool(run_middleware=False)
path), and this codebase's own McpServerTracingMiddleware span - all for the
same logical operation. A narrower fix scoped inside this codebase's own
middleware (fastmcp.telemetry.suppress_fastmcp_telemetry() wrapped only
around call_next()) was tried first and confirmed insufficient: FastMCP's
middleware seam span opens *before* any user middleware runs, so nothing
inside this codebase's own middleware can reach back far enough to suppress
it. The fix has to be process-wide, set at the point self-instrumentation is
enabled - see configure_tracing()'s own docstring.

This test deliberately calls the REAL opentelemetry.trace.set_tracer_provider
(unlike every other test in test_observability.py, which patches it away) -
that call only succeeds once per process, so this test resets the module-
private _TRACER_PROVIDER singleton first to force a genuinely fresh global
provider. This is the only way to exercise FastMCP's real recursive
call_tool() dispatch path (which reads the actual global provider via
fastmcp.telemetry.get_tracer()), not a mocked stand-in for it.
"""

# server.py imports its tools submodules (search, ...) without an explicit
# `as name` re-export, so mypy's strict --no-implicit-reexport treats
# server.search as an undefined attribute even though it exists at runtime -
# same friction test_server.py's own module docstring documents.
# mypy: disable-error-code="attr-defined"

from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import opentelemetry.trace as otel_trace_module
import pytest
from fastmcp import Client
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind
from opentelemetry.util._once import Once
from pydantic import HttpUrl

from opentelemetry_mcp import server
from opentelemetry_mcp.config import BackendConfig, ServerConfig
from opentelemetry_mcp.models import SearchTracesResult
from opentelemetry_mcp.observability import McpServerTracingMiddleware


@pytest.fixture
def real_global_tracer_provider(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Force a genuinely fresh global TracerProvider for this one test,
    bypassing OTel's real "only once per process" restriction - see module
    docstring for why this can't use the mocked-away pattern every other
    observability test in this repo uses.

    Resetting _TRACER_PROVIDER alone is not enough: set_tracer_provider()
    is gated by a SEPARATE _TRACER_PROVIDER_SET_ONCE guard (a
    opentelemetry.util._once.Once instance) that permanently no-ops every
    call after the first successful one *for the life of the process* -
    confirmed by reading opentelemetry/trace/__init__.py directly. Without
    resetting this too, only the first test in this file (or the whole
    session) to reach this fixture would ever get a real provider; every
    later one would silently keep whatever provider won first.
    """
    monkeypatch.setattr(otel_trace_module, "_TRACER_PROVIDER", None)
    monkeypatch.setattr(otel_trace_module, "_TRACER_PROVIDER_SET_ONCE", Once())
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture(autouse=True)
def _reset_fastmcp_telemetry_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastmcp

    monkeypatch.setattr(fastmcp.settings, "telemetry_mode", "native")


@pytest.fixture(autouse=True)
def _reset_server_globals() -> None:
    """A real (if minimal) config/backend, not None - _get_backend() raises
    RuntimeError before ever reaching the mocked search.search_traces below
    if server._config is unset, which would make every test in this file
    fail for an unrelated reason before even touching tracing."""
    server._config = ServerConfig(
        backend=BackendConfig(type="jaeger", url=HttpUrl("http://localhost:16686"))
    )
    server._backend = AsyncMock()


@pytest.fixture(autouse=True)
def _reset_mcp_middleware() -> Generator[None]:
    """server.mcp is a module-level singleton - add_middleware() appends to
    a plain mutable list on it (fastmcp/server/server.py:494,624), so
    without resetting this, each test in this file would accumulate every
    prior test's McpServerTracingMiddleware instance and break the
    exactly-one-span assertions below."""
    original = list(server.mcp.middleware)
    yield
    server.mcp.middleware = original


def _tools_call_server_spans(exporter: InMemorySpanExporter) -> list:  # type: ignore[type-arg]
    return [
        s
        for s in exporter.get_finished_spans()
        if "tools/call" in s.name and s.kind == SpanKind.SERVER
    ]


class TestDoubleTracingRegression:
    async def test_without_the_fix_fastmcp_produces_duplicate_native_spans(
        self, real_global_tracer_provider: InMemorySpanExporter
    ) -> None:
        """Baseline proving the bug is real, not hypothetical - fastmcp
        left at its default telemetry_mode ("native", set by the
        autouse fixture above) produces more than one SERVER-kind
        tools/call span for a single tool call."""
        mw = McpServerTracingMiddleware(tracer=trace.get_tracer("opentelemetry_mcp"))
        server.mcp.add_middleware(mw)
        sentinel = SearchTracesResult(count=0, traces=[])

        with patch.object(server.search, "search_traces", AsyncMock(return_value=sentinel)):
            async with Client(server.mcp) as client:
                await client.call_tool("search_traces", {"limit": 5}, raise_on_error=False)

        spans = _tools_call_server_spans(real_global_tracer_provider)
        assert len(spans) > 1, (
            "Expected the real double/triple-tracing bug to reproduce with "
            "telemetry_mode='native' - if this now fails, FastMCP's own "
            "internals changed and configure_tracing()'s fix may need "
            "re-verifying against the new behavior."
        )

    async def test_with_the_fix_exactly_one_span_survives(
        self, real_global_tracer_provider: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import fastmcp

        monkeypatch.setattr(fastmcp.settings, "telemetry_mode", "propagation_only")

        mw = McpServerTracingMiddleware(tracer=trace.get_tracer("opentelemetry_mcp"))
        server.mcp.add_middleware(mw)
        sentinel = SearchTracesResult(count=0, traces=[])

        with patch.object(server.search, "search_traces", AsyncMock(return_value=sentinel)):
            async with Client(server.mcp) as client:
                await client.call_tool("search_traces", {"limit": 5}, raise_on_error=False)

        spans = _tools_call_server_spans(real_global_tracer_provider)
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert attrs.get("gen_ai.tool.name") == "search_traces"
        assert attrs.get("gen_ai.operation.name") == "execute_tool"
        assert "mcp.session.id" in attrs

    async def test_with_the_fix_error_path_still_produces_exactly_one_span(
        self, real_global_tracer_provider: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exception path is where FastMCP's own (now-suppressed)
        server_span() would otherwise independently record its own
        error.type/exception event - confirm the single surviving span
        still carries this codebase's own error reporting correctly, not
        just the happy path above."""
        import fastmcp

        monkeypatch.setattr(fastmcp.settings, "telemetry_mode", "propagation_only")

        mw = McpServerTracingMiddleware(tracer=trace.get_tracer("opentelemetry_mcp"))
        server.mcp.add_middleware(mw)

        with patch.object(
            server.search, "search_traces", AsyncMock(side_effect=RuntimeError("boom"))
        ):
            async with Client(server.mcp) as client:
                await client.call_tool("search_traces", {"limit": 5}, raise_on_error=False)

        spans = _tools_call_server_spans(real_global_tracer_provider)
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert attrs.get("error.type") is not None
        assert spans[0].status.status_code == trace.StatusCode.ERROR
