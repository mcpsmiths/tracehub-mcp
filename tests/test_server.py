"""Tests for the FastMCP server: backend factory, error handler, lazy
backend caching, and the thin @mcp.tool() wrapper functions plus the click
CLI entrypoint.
"""

# server.py imports its tools submodules (search, trace, usage, ...) without
# an explicit `as name` re-export, so mypy's strict `--no-implicit-reexport`
# treats `server.search`, `server.trace`, etc. as undefined attributes even
# though they exist at runtime and this file relies on patching them to
# exercise each @mcp.tool() wrapper's own glue code. Disabling attr-defined
# here only (not project-wide) avoids scattering ~20 per-line ignores for a
# server.py import-style nuance that is out of scope for this test file.
# mypy: disable-error-code="attr-defined"

import json
from collections.abc import Generator
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from fastmcp import Client
from pydantic import HttpUrl

from opentelemetry_mcp import server
from opentelemetry_mcp.attributes import HealthCheckResponse
from opentelemetry_mcp.backends.datadog import DatadogBackend
from opentelemetry_mcp.backends.jaeger import JaegerBackend
from opentelemetry_mcp.backends.sentry import SentryBackend
from opentelemetry_mcp.backends.tempo import TempoBackend
from opentelemetry_mcp.backends.traceloop import TraceloopBackend
from opentelemetry_mcp.backends.xray import XRayBackend
from opentelemetry_mcp.config import BackendConfig, ServerConfig
from opentelemetry_mcp.models import SearchTracesResult, TraceDetail

FAKE_API_KEY = "dd-key1"
FAKE_APP_KEY = "dd-app1"
FAKE_SENTRY_ORG = "fake-org"
FAKE_SENTRY_PROJECT = "fake-project"
FAKE_TEMPO_INSTANCE_ID = "123456"


def _config(**overrides: object) -> ServerConfig:
    """Build a minimal valid ServerConfig, overriding BackendConfig fields."""
    backend_fields: dict[str, object] = {
        "type": "jaeger",
        "url": HttpUrl("http://localhost:16686"),
    }
    backend_fields.update(overrides)
    return ServerConfig(backend=BackendConfig(**backend_fields))  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def reset_server_globals() -> Generator[None]:
    """The module-level _backend/_config globals are shared mutable state -
    reset them before and after every test so tests can't leak into each
    other via caching in _get_backend. TestToolWrappers also reassigns the
    _get_backend function itself (not just the globals it caches) via raw
    attribute assignment rather than unittest.mock.patch, so it must be
    restored here too or a mocked _get_backend silently leaks into every
    test that runs afterward in the same session."""
    real_get_backend = server._get_backend
    server._backend = None
    server._config = None
    yield
    server._backend = None
    server._config = None
    server._get_backend = real_get_backend


class TestCreateBackend:
    """_create_backend is a pure factory - verify each of the 6 backend
    types produces the right class with the right constructor args threaded
    through, plus the ValueError branch for an unsupported type."""

    def test_jaeger(self) -> None:
        config = _config(type="jaeger", api_key=FAKE_API_KEY, timeout=12.0)
        backend = server._create_backend(config)

        assert isinstance(backend, JaegerBackend)
        assert backend.url == "http://localhost:16686/"
        assert backend.api_key == FAKE_API_KEY
        assert backend.timeout == 12.0

    def test_tempo(self) -> None:
        config = _config(
            type="tempo",
            api_key=FAKE_API_KEY,
            timeout=9.0,
            tempo_instance_id=FAKE_TEMPO_INSTANCE_ID,
        )
        backend = server._create_backend(config)

        assert isinstance(backend, TempoBackend)
        assert backend.api_key == FAKE_API_KEY
        assert backend.timeout == 9.0
        assert backend.tempo_instance_id == FAKE_TEMPO_INSTANCE_ID

    def test_traceloop(self) -> None:
        config = _config(
            type="traceloop", api_key=FAKE_API_KEY, timeout=7.0, environments=["prd", "staging"]
        )
        backend = server._create_backend(config)

        assert isinstance(backend, TraceloopBackend)
        assert backend.api_key == FAKE_API_KEY
        assert backend.environments == ["prd", "staging"]

    def test_datadog(self) -> None:
        config = _config(
            type="datadog",
            url=HttpUrl("https://api.datadoghq.com"),
            api_key=FAKE_API_KEY,
            app_key=FAKE_APP_KEY,
            timeout=11.0,
        )
        backend = server._create_backend(config)

        assert isinstance(backend, DatadogBackend)
        assert backend.api_key == FAKE_API_KEY
        assert backend.app_key == FAKE_APP_KEY
        assert backend.timeout == 11.0

    def test_sentry(self) -> None:
        config = _config(
            type="sentry",
            url=HttpUrl("https://sentry.io"),
            api_key=FAKE_API_KEY,
            sentry_org=FAKE_SENTRY_ORG,
            sentry_project=FAKE_SENTRY_PROJECT,
            timeout=13.0,
        )
        backend = server._create_backend(config)

        assert isinstance(backend, SentryBackend)
        assert backend.api_key == FAKE_API_KEY
        assert backend.org_slug == FAKE_SENTRY_ORG
        assert backend.project_slug == FAKE_SENTRY_PROJECT
        assert backend.timeout == 13.0

    def test_xray(self) -> None:
        config = _config(
            type="xray",
            url=HttpUrl("https://xray.us-east-1.amazonaws.com"),
            aws_region="us-east-1",
            timeout=8.0,
        )
        backend = server._create_backend(config)

        assert isinstance(backend, XRayBackend)
        assert backend.aws_region == "us-east-1"
        assert backend.timeout == 8.0

    def test_unsupported_backend_type_raises(self) -> None:
        config = _config(type="jaeger")
        # BackendConfig.type is a Literal, so bypass validation to simulate
        # an unsupported type reaching the factory.
        config.backend.type = "unknown"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unsupported backend type"):
            server._create_backend(config)


class TestHandleToolError:
    """_handle_tool_error logs then re-raises the original exception - it
    must never swallow it and return a value instead. Swallowing it would
    make the MCP SDK's lowlevel server treat the call as a *success* whose
    content merely looks like an error (isError=False), which violates
    SEP-2140's requirement that tool execution failures be reported as
    CallToolResult(isError=True)."""

    def test_reraises_the_original_exception(self) -> None:
        error = ValueError("boom")

        with pytest.raises(ValueError, match="boom") as exc_info:
            server._handle_tool_error("search_traces", error)

        assert exc_info.value is error

    def test_different_error_type_is_preserved(self) -> None:
        error = RuntimeError("trace not found")

        with pytest.raises(RuntimeError, match="trace not found") as exc_info:
            server._handle_tool_error("get_trace", error)

        assert exc_info.value is error

    def test_logs_tool_name(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("ERROR"), pytest.raises(Exception, match="oops"):
            server._handle_tool_error("list_services", Exception("oops"))

        assert "list_services" in caplog.text


class TestGetBackend:
    """_get_backend is async and uses module-level globals _backend/_config."""

    async def test_raises_when_config_not_set(self) -> None:
        server._config = None
        server._backend = None

        with pytest.raises(RuntimeError, match="Server configuration not set"):
            await server._get_backend()

    async def test_lazily_creates_and_caches_backend(self) -> None:
        server._config = _config()
        fake_backend = AsyncMock()
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(
                status="healthy", backend="jaeger", url="http://localhost:16686"
            )
        )

        with patch.object(server, "_create_backend", return_value=fake_backend) as mock_create:
            first = await server._get_backend()
            second = await server._get_backend()

        assert first is fake_backend
        assert second is fake_backend
        # Only created once even though _get_backend was called twice.
        mock_create.assert_called_once_with(server._config)
        fake_backend.health_check.assert_awaited_once()

    async def test_health_check_exception_is_swallowed(self) -> None:
        """Read the function: health_check failures are caught and logged,
        never re-raised - backend creation still succeeds."""
        server._config = _config()
        fake_backend = AsyncMock()
        fake_backend.health_check = AsyncMock(side_effect=RuntimeError("unreachable"))

        with patch.object(server, "_create_backend", return_value=fake_backend):
            result = await server._get_backend()

        assert result is fake_backend

    async def test_unhealthy_status_does_not_prevent_return(self) -> None:
        """A non-'healthy' status is only logged as a warning - the backend
        is still cached and returned."""
        server._config = _config()
        fake_backend = AsyncMock()
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(
                status="unhealthy",
                backend="jaeger",
                url="http://localhost:16686",
                error="down",
            )
        )

        with patch.object(server, "_create_backend", return_value=fake_backend):
            result = await server._get_backend()

        assert result is fake_backend
        assert server._backend is fake_backend


class TestToolWrappers:
    """Each @mcp.tool()-decorated function is a thin wrapper: get backend,
    call the matching tools.<module> function with its own parameters, and
    let any exception propagate through _handle_tool_error (which logs then
    re-raises, so the MCP SDK reports CallToolResult(isError=True) - see
    TestHandleToolError). Mock at the module-level names the wrapper
    actually calls (server._get_backend and server.<tools_module>.<function>)
    so we exercise the wrapper's own glue code rather than the mock."""

    async def _set_backend(self) -> AsyncMock:
        fake_backend = AsyncMock()
        server._get_backend = AsyncMock(return_value=fake_backend)
        return fake_backend

    async def test_search_traces_passes_arguments_through(self) -> None:
        await self._set_backend()
        sentinel = SearchTracesResult(count=1, traces=[])
        with patch.object(
            server.search, "search_traces", AsyncMock(return_value=sentinel)
        ) as mocked:
            result = await server.search_traces(
                service_name="svc",
                operation_name="op",
                min_duration_ms=5,
                has_error=True,
                limit=42,
            )

        assert result is sentinel
        _, kwargs = mocked.call_args
        assert kwargs["service_name"] == "svc"
        assert kwargs["operation_name"] == "op"
        assert kwargs["min_duration_ms"] == 5
        assert kwargs["has_error"] is True
        assert kwargs["limit"] == 42

    async def test_search_traces_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(
                server.search, "search_traces", AsyncMock(side_effect=ValueError("bad filter"))
            ),
            pytest.raises(ValueError, match="bad filter"),
        ):
            await server.search_traces()

    async def test_get_trace_passes_trace_id(self) -> None:
        await self._set_backend()
        sentinel = TraceDetail(
            trace_id="abc123",
            service_name="svc",
            root_operation="op",
            start_time=datetime(2024, 1, 1),
            duration_ms=1.0,
            status="OK",
            span_count=0,
            has_errors=False,
            spans=[],
            detail_level="full",
        )
        with patch.object(server.trace, "get_trace", AsyncMock(return_value=sentinel)) as mocked:
            result = await server.get_trace(trace_id="abc123")

        assert result is sentinel
        _, kwargs = mocked.call_args
        assert kwargs["trace_id"] == "abc123"
        assert kwargs["detail_level"] == "full"

    async def test_get_trace_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(server.trace, "get_trace", AsyncMock(side_effect=KeyError("nope"))),
            pytest.raises(KeyError, match="nope"),
        ):
            await server.get_trace(trace_id="missing")

    async def test_get_llm_usage_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(server.usage, "get_llm_usage", AsyncMock(return_value="{}")) as mocked:
            await server.get_llm_usage(service_name="svc", gen_ai_system="openai", limit=50)

        _, kwargs = mocked.call_args
        assert kwargs["service_name"] == "svc"
        assert kwargs["gen_ai_system"] == "openai"
        assert kwargs["limit"] == 50

    async def test_get_llm_usage_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(server.usage, "get_llm_usage", AsyncMock(side_effect=Exception("fail"))),
            pytest.raises(Exception, match="fail"),
        ):
            await server.get_llm_usage()

    async def test_list_services_calls_tool_with_backend_only(self) -> None:
        fake_backend = await self._set_backend()
        with patch.object(
            server.services, "list_services", AsyncMock(return_value='["a","b"]')
        ) as mocked:
            result = await server.list_services()

        assert result == '["a","b"]'
        mocked.assert_awaited_once_with(fake_backend)

    async def test_list_services_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(
                server.services, "list_services", AsyncMock(side_effect=RuntimeError("down"))
            ),
            pytest.raises(RuntimeError, match="down"),
        ):
            await server.list_services()

    async def test_find_errors_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(server.errors, "find_errors", AsyncMock(return_value="{}")) as mocked:
            await server.find_errors(service_name="svc", limit=5)

        _, kwargs = mocked.call_args
        assert kwargs["service_name"] == "svc"
        assert kwargs["limit"] == 5

    async def test_find_errors_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(server.errors, "find_errors", AsyncMock(side_effect=Exception("boom"))),
            pytest.raises(Exception, match="boom"),
        ):
            await server.find_errors()

    async def test_list_llm_models_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.list_models, "list_models", AsyncMock(return_value="{}")
        ) as mocked:
            await server.list_llm_models(gen_ai_system="anthropic", limit=10)

        _, kwargs = mocked.call_args
        assert kwargs["gen_ai_system"] == "anthropic"
        assert kwargs["limit"] == 10

    async def test_list_llm_models_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(
                server.list_models, "list_models", AsyncMock(side_effect=Exception("boom"))
            ),
            pytest.raises(Exception, match="boom"),
        ):
            await server.list_llm_models()

    async def test_get_llm_model_stats_passes_model_name(self) -> None:
        await self._set_backend()
        with patch.object(
            server.model_stats, "get_model_stats", AsyncMock(return_value="{}")
        ) as mocked:
            await server.get_llm_model_stats(model_name="gpt-4", service_name="svc")

        _, kwargs = mocked.call_args
        assert kwargs["model_name"] == "gpt-4"
        assert kwargs["service_name"] == "svc"

    async def test_get_llm_model_stats_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(
                server.model_stats, "get_model_stats", AsyncMock(side_effect=Exception("boom"))
            ),
            pytest.raises(Exception, match="boom"),
        ):
            await server.get_llm_model_stats(model_name="gpt-4")

    async def test_get_llm_expensive_traces_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.expensive_traces, "get_expensive_traces", AsyncMock(return_value="{}")
        ) as mocked:
            await server.get_llm_expensive_traces(limit=3, min_tokens=1000)

        _, kwargs = mocked.call_args
        assert kwargs["limit"] == 3
        assert kwargs["min_tokens"] == 1000

    async def test_get_llm_expensive_traces_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(
                server.expensive_traces,
                "get_expensive_traces",
                AsyncMock(side_effect=Exception("boom")),
            ),
            pytest.raises(Exception, match="boom"),
        ):
            await server.get_llm_expensive_traces()

    async def test_get_llm_slow_traces_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.slow_traces, "get_slow_traces", AsyncMock(return_value="{}")
        ) as mocked:
            await server.get_llm_slow_traces(limit=7, min_duration_ms=250)

        _, kwargs = mocked.call_args
        assert kwargs["limit"] == 7
        assert kwargs["min_duration_ms"] == 250

    async def test_get_llm_slow_traces_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(
                server.slow_traces, "get_slow_traces", AsyncMock(side_effect=Exception("boom"))
            ),
            pytest.raises(Exception, match="boom"),
        ):
            await server.get_llm_slow_traces()

    async def test_search_spans_tool_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.search_spans, "search_spans", AsyncMock(return_value="{}")
        ) as mocked:
            await server.search_spans_tool(service_name="svc", has_error=True)

        _, kwargs = mocked.call_args
        assert kwargs["service_name"] == "svc"
        assert kwargs["has_error"] is True

    async def test_search_spans_tool_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(
                server.search_spans, "search_spans", AsyncMock(side_effect=Exception("boom"))
            ),
            pytest.raises(Exception, match="boom"),
        ):
            await server.search_spans_tool()

    async def test_list_llm_tools_tool_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.list_llm_tools, "list_llm_tools", AsyncMock(return_value="{}")
        ) as mocked:
            await server.list_llm_tools_tool(gen_ai_system="openai", limit=99)

        _, kwargs = mocked.call_args
        assert kwargs["gen_ai_system"] == "openai"
        assert kwargs["limit"] == 99

    async def test_list_llm_tools_tool_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(
                server.list_llm_tools, "list_llm_tools", AsyncMock(side_effect=Exception("boom"))
            ),
            pytest.raises(Exception, match="boom"),
        ):
            await server.list_llm_tools_tool()

    async def test_investigate_cost_spike_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.investigate, "investigate_cost_spike", AsyncMock(return_value="{}")
        ) as mocked:
            await server.investigate_cost_spike(
                recent_start="2024-01-08T00:00:00Z",
                recent_end="2024-01-15T00:00:00Z",
                service_name="svc",
                top_n=10,
            )

        _, kwargs = mocked.call_args
        assert kwargs["recent_start"] == "2024-01-08T00:00:00Z"
        assert kwargs["recent_end"] == "2024-01-15T00:00:00Z"
        assert kwargs["service_name"] == "svc"
        assert kwargs["top_n"] == 10

    async def test_investigate_cost_spike_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(
                server.investigate,
                "investigate_cost_spike",
                AsyncMock(side_effect=ValueError("bad window")),
            ),
            pytest.raises(ValueError, match="bad window"),
        ):
            await server.investigate_cost_spike(
                recent_start="2024-01-08T00:00:00Z", recent_end="2024-01-15T00:00:00Z"
            )

    async def test_investigate_error_spike_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.investigate, "investigate_error_spike", AsyncMock(return_value="{}")
        ) as mocked:
            await server.investigate_error_spike(
                recent_start="2024-01-08T00:00:00Z",
                recent_end="2024-01-15T00:00:00Z",
                min_error_count_increase=5,
                rate_multiplier_threshold=3.0,
            )

        _, kwargs = mocked.call_args
        assert kwargs["recent_start"] == "2024-01-08T00:00:00Z"
        assert kwargs["min_error_count_increase"] == 5
        assert kwargs["rate_multiplier_threshold"] == 3.0

    async def test_investigate_error_spike_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(
                server.investigate,
                "investigate_error_spike",
                AsyncMock(side_effect=Exception("boom")),
            ),
            pytest.raises(Exception, match="boom"),
        ):
            await server.investigate_error_spike(
                recent_start="2024-01-08T00:00:00Z", recent_end="2024-01-15T00:00:00Z"
            )

    async def test_get_backend_failure_propagates(self) -> None:
        """If _get_backend itself raises (e.g. config not set), the wrapper
        must let it propagate through _handle_tool_error (logged, then
        re-raised) rather than swallowing it into a fake-success result."""
        server._get_backend = AsyncMock(side_effect=RuntimeError("Server configuration not set"))

        with pytest.raises(RuntimeError, match="Server configuration not set"):
            await server.list_services()


class TestToolErrorIsErrorFlag:
    """End-to-end (SEP-2140) proof: a failing tool call must produce a real
    CallToolResult with isError=True over an actual MCP client handshake,
    not just a JSON string whose content happens to contain an "error" key.
    This exercises the real mcp.server.lowlevel.Server plumbing that
    TestToolWrappers' direct-call tests never touch."""

    async def test_backend_failure_sets_is_error_true(self) -> None:
        server._config = None
        server._backend = None

        async with Client(server.mcp) as client:
            result = await client.call_tool("search_traces", {"limit": 5}, raise_on_error=False)

        assert result.is_error is True
        assert "Server configuration not set" in result.content[0].text

    async def test_successful_call_sets_is_error_false(self) -> None:
        server._config = _config()
        fake_backend = AsyncMock()
        server._backend = fake_backend
        sentinel = SearchTracesResult(count=0, traces=[])
        with patch.object(server.search, "search_traces", AsyncMock(return_value=sentinel)):
            async with Client(server.mcp) as client:
                result = await client.call_tool("search_traces", {"limit": 5}, raise_on_error=False)

        assert result.is_error is False


class TestToolAnnotationsComplete:
    """MCP directories (e.g. OpenAI's ChatGPT Apps submission pipeline) can
    reject tools whose annotations are missing rather than explicit
    booleans - every tool here is read-only, so all four hints must be
    real booleans, not just the three previously set."""

    async def test_every_tool_has_all_four_hints_as_explicit_booleans(self) -> None:
        async with Client(server.mcp) as client:
            tools = await client.list_tools()

        assert len(tools) == 17
        for t in tools:
            assert t.annotations is not None, f"{t.name} has no annotations at all"
            assert t.annotations.readOnlyHint is True, t.name
            assert t.annotations.destructiveHint is False, t.name
            assert t.annotations.idempotentHint is True, t.name
            assert t.annotations.openWorldHint is True, t.name

    async def test_every_tool_has_a_distinct_human_readable_title(self) -> None:
        """Anthropic's Software Directory Policy requires readOnlyHint,
        destructiveHint, and title on every tool - title lives on Tool
        itself (FastMCP's title= kwarg), not inside ToolAnnotations."""
        async with Client(server.mcp) as client:
            tools = await client.list_tools()

        assert len(tools) == 17
        titles = [t.title for t in tools]
        assert all(isinstance(title, str) and title for title in titles), titles
        assert len(set(titles)) == 17, "titles must be distinct per tool"


class TestClampLimit:
    """_clamp_limit enforces --max-traces-per-query/MAX_TRACES_PER_QUERY as
    a real server-wide ceiling. Found by a production-audit pass: the
    field was parsed, validated, and stored in ServerConfig, but nothing
    ever read it - every tool's own limit parameter was the only real cap."""

    def test_requested_limit_below_ceiling_is_unchanged(self) -> None:
        server._config = _config()
        server._config.max_traces_per_query = 500

        assert server._clamp_limit(100) == 100

    def test_requested_limit_above_ceiling_is_capped(self) -> None:
        server._config = _config()
        server._config.max_traces_per_query = 50

        assert server._clamp_limit(1000) == 50

    def test_requested_limit_equal_to_ceiling_is_unchanged(self) -> None:
        server._config = _config()
        server._config.max_traces_per_query = 100

        assert server._clamp_limit(100) == 100

    def test_config_not_set_yet_returns_the_requested_limit_unclamped(self) -> None:
        server._config = None

        assert server._clamp_limit(999999) == 999999

    async def test_search_traces_wrapper_clamps_before_calling_the_tool(self) -> None:
        """End-to-end through the actual @mcp.tool() wrapper, not just the
        helper in isolation - proves the wiring, not just the function."""
        server._config = _config()
        server._config.max_traces_per_query = 5
        fake_backend = AsyncMock()
        server._get_backend = AsyncMock(return_value=fake_backend)

        with patch.object(server.search, "search_traces", AsyncMock(return_value="{}")) as mocked:
            await server.search_traces(limit=1000)

        _, kwargs = mocked.call_args
        assert kwargs["limit"] == 5


class TestApplyToolGating:
    """_apply_tool_gating wraps mcp.local_provider.remove_tool() for
    --enabled-tools/--disable-tools. Patches server.mcp wholesale (same
    pattern TestMainCli uses) rather than mutating the real shared mcp
    singleton, which every other test file's Client(mcp) calls also rely
    on staying fully populated."""

    def test_disable_tools_removes_exactly_those_names(self) -> None:
        with patch.object(server, "mcp") as mock_mcp:
            server._apply_tool_gating(disable_tools="get_trace,find_errors", enabled_tools=None)

        removed = [call.args[0] for call in mock_mcp.local_provider.remove_tool.call_args_list]
        assert sorted(removed) == ["find_errors", "get_trace"]

    def test_disable_tools_unknown_name_raises_without_removing_anything(self) -> None:
        with (
            patch.object(server, "mcp") as mock_mcp,
            pytest.raises(ValueError, match="Unknown tool name.*not_a_real_tool"),
        ):
            server._apply_tool_gating(disable_tools="not_a_real_tool", enabled_tools=None)

        mock_mcp.local_provider.remove_tool.assert_not_called()

    def test_enabled_tools_removes_everything_not_in_the_allowlist(self) -> None:
        with patch.object(server, "mcp") as mock_mcp:
            server._apply_tool_gating(disable_tools=None, enabled_tools="search_traces,get_trace")

        removed = {call.args[0] for call in mock_mcp.local_provider.remove_tool.call_args_list}
        assert removed == server._ALL_TOOL_NAMES - {"search_traces", "get_trace"}

    def test_enabled_tools_unknown_name_raises_without_removing_anything(self) -> None:
        with (
            patch.object(server, "mcp") as mock_mcp,
            pytest.raises(ValueError, match="Unknown tool name.*not_a_real_tool"),
        ):
            server._apply_tool_gating(disable_tools=None, enabled_tools="not_a_real_tool")

        mock_mcp.local_provider.remove_tool.assert_not_called()

    def test_disable_tools_swallows_key_error_for_an_already_removed_tool(self) -> None:
        """A tool named in --disable-tools that was already removed some
        other way (e.g. by --enabled-tools) raises KeyError from the real
        remove_tool - that must be swallowed, not propagate."""
        with patch.object(server, "mcp") as mock_mcp:
            mock_mcp.local_provider.remove_tool.side_effect = KeyError("already removed")

            server._apply_tool_gating(disable_tools="get_trace", enabled_tools=None)

    def test_both_none_removes_nothing(self) -> None:
        with patch.object(server, "mcp") as mock_mcp:
            server._apply_tool_gating(disable_tools=None, enabled_tools=None)

        mock_mcp.local_provider.remove_tool.assert_not_called()

    def test_empty_strings_are_treated_like_none(self) -> None:
        with patch.object(server, "mcp") as mock_mcp:
            server._apply_tool_gating(disable_tools="", enabled_tools="")

        mock_mcp.local_provider.remove_tool.assert_not_called()


class TestMainCli:
    """main() is the click CLI entrypoint. mcp.run is mocked so it never
    blocks; ServerConfig.from_env/apply_cli_overrides are exercised for
    real (against env-based defaults) so we can assert CLI flags reach them."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Ensure a valid minimal config loads from env regardless of the
        # developer's local .env / shell environment.
        monkeypatch.setenv("BACKEND_TYPE", "jaeger")
        monkeypatch.setenv("BACKEND_URL", "http://localhost:16686")
        monkeypatch.delenv("BACKEND_API_KEY", raising=False)
        monkeypatch.delenv("BACKEND_APP_KEY", raising=False)
        monkeypatch.delenv("BACKEND_SENTRY_ORG", raising=False)
        monkeypatch.delenv("BACKEND_SENTRY_PROJECT", raising=False)
        monkeypatch.delenv("BACKEND_ENVIRONMENTS", raising=False)

    def test_cli_flags_reach_apply_cli_overrides(self) -> None:
        runner = CliRunner()
        with (
            patch.object(server, "mcp") as mock_mcp,
            patch.object(ServerConfig, "apply_cli_overrides") as mock_overrides,
        ):
            mock_mcp.run = MagicMock()
            result = runner.invoke(
                server.main,
                [
                    "--backend",
                    "datadog",
                    "--url",
                    "https://api.datadoghq.com",
                    "--api-key",
                    FAKE_API_KEY,
                    "--app-key",
                    FAKE_APP_KEY,
                    "--tempo-instance-id",
                    FAKE_TEMPO_INSTANCE_ID,
                    "--sentry-org",
                    FAKE_SENTRY_ORG,
                    "--sentry-project",
                    FAKE_SENTRY_PROJECT,
                    "--environments",
                    "prd,staging",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_overrides.assert_called_once_with(
            backend_type="datadog",
            backend_url="https://api.datadoghq.com",
            api_key=FAKE_API_KEY,
            app_key=FAKE_APP_KEY,
            sentry_org=FAKE_SENTRY_ORG,
            sentry_project=FAKE_SENTRY_PROJECT,
            tempo_instance_id=FAKE_TEMPO_INSTANCE_ID,
            aws_region=None,
            environments="prd,staging",
            log_level=None,
            max_traces_per_query=None,
            slow_request_threshold_ms=None,
            query_cache_ttl_seconds=None,
        )

    def test_no_cli_flags_skips_apply_cli_overrides(self) -> None:
        runner = CliRunner()
        with (
            patch.object(server, "mcp") as mock_mcp,
            patch.object(ServerConfig, "apply_cli_overrides") as mock_overrides,
        ):
            mock_mcp.run = MagicMock()
            result = runner.invoke(server.main, [])

        assert result.exit_code == 0, result.output
        mock_overrides.assert_not_called()

    def test_stdio_transport_calls_mcp_run_with_stdio(self) -> None:
        runner = CliRunner()
        with patch.object(server, "mcp") as mock_mcp:
            mock_mcp.run = MagicMock()
            result = runner.invoke(server.main, ["--transport", "stdio"])

        assert result.exit_code == 0, result.output
        mock_mcp.run.assert_called_once_with(transport="stdio")

    def test_http_transport_builds_app_and_serves_with_host_and_port(self) -> None:
        """HTTP transport no longer calls mcp.run() - see
        _install_shutdown_drain's docstring for why (a real SIGTERM never
        returns control to mcp.run()'s own outer lifespan wrap, so the
        drain/close logic has to live inside mcp.http_app()'s app instead,
        driven directly by a hand-constructed uvicorn.Server)."""
        runner = CliRunner()
        with (
            patch.object(server, "mcp") as mock_mcp,
            patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls,
        ):
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(
                server.main,
                ["--transport", "http", "--host", "127.0.0.1", "--port", "9001"],
            )

        assert result.exit_code == 0, result.output

        mock_mcp.http_app.assert_called_once()
        http_app_kwargs = mock_mcp.http_app.call_args.kwargs
        assert http_app_kwargs["transport"] == "streamable-http"
        # HTTP transport must wire in the Origin-validation middleware
        # (Tier 1 security fix: fastmcp disables DNS-rebinding protection
        # by default, see OriginValidationMiddleware's docstring), and the
        # rate limiter (enabled by default, see RateLimitMiddleware).
        middleware = http_app_kwargs["middleware"]
        assert len(middleware) == 2
        assert middleware[0].cls is server.OriginValidationMiddleware
        assert middleware[1].cls is server.RateLimitMiddleware

        mock_server_cls.assert_called_once()
        config_arg = mock_server_cls.call_args.args[0]
        assert config_arg.app is mock_mcp.http_app.return_value
        assert config_arg.host == "127.0.0.1"
        assert config_arg.port == 9001
        mock_server_cls.return_value.serve.assert_awaited_once()

    def test_rate_limit_disabled_when_max_requests_is_zero(self) -> None:
        runner = CliRunner()
        with (
            patch.object(server, "mcp") as mock_mcp,
            patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls,
        ):
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(
                server.main,
                ["--transport", "http", "--rate-limit-max-requests", "0"],
            )

        assert result.exit_code == 0, result.output
        middleware = mock_mcp.http_app.call_args.kwargs["middleware"]
        assert len(middleware) == 1
        assert middleware[0].cls is server.OriginValidationMiddleware

    def test_keyboard_interrupt_exits_zero(self) -> None:
        runner = CliRunner()
        with patch.object(server, "mcp") as mock_mcp:
            mock_mcp.run = MagicMock(side_effect=KeyboardInterrupt)
            result = runner.invoke(server.main, [])

        assert result.exit_code == 0

    def test_generic_exception_exits_one(self) -> None:
        runner = CliRunner()
        with patch.object(server, "mcp") as mock_mcp:
            mock_mcp.run = MagicMock(side_effect=Exception("startup failed"))
            result = runner.invoke(server.main, [])

        assert result.exit_code == 1

    def test_generic_exception_during_config_load_exits_one(self) -> None:
        """An exception raised before mcp.run is reached (e.g. bad env
        config) must also be caught and exit 1, not propagate."""
        runner = CliRunner()
        with patch.object(ServerConfig, "from_env", side_effect=ValueError("bad config")):
            result = runner.invoke(server.main, [])

        assert result.exit_code == 1

    def test_bare_invocation_still_works_after_group_conversion(self) -> None:
        """main() converted from a flat @click.command() to a
        @click.group(invoke_without_command=True) to add the doctor
        subcommand - every existing bare invocation (no subcommand token)
        must keep working exactly as before. This re-runs the flags test
        above verbatim as an explicit regression guard for that conversion."""
        runner = CliRunner()
        with (
            patch.object(server, "mcp") as mock_mcp,
            patch.object(ServerConfig, "apply_cli_overrides") as mock_overrides,
        ):
            mock_mcp.run = MagicMock()
            result = runner.invoke(
                server.main, ["--backend", "datadog", "--url", "https://api.datadoghq.com"]
            )

        assert result.exit_code == 0, result.output
        mock_overrides.assert_called_once()
        mock_mcp.run.assert_called_once()

    def test_help_lists_doctor_subcommand(self) -> None:
        runner = CliRunner()
        result = runner.invoke(server.main, ["--help"])

        assert result.exit_code == 0
        assert "doctor" in result.output


class TestPrintConfigFlag:
    """--print-config prints the resolved config as JSON and exits without
    starting the server."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BACKEND_TYPE", "jaeger")
        monkeypatch.setenv("BACKEND_URL", "http://localhost:16686")
        monkeypatch.delenv("BACKEND_API_KEY", raising=False)
        monkeypatch.delenv("BACKEND_APP_KEY", raising=False)
        monkeypatch.delenv("BACKEND_SENTRY_ORG", raising=False)
        monkeypatch.delenv("BACKEND_SENTRY_PROJECT", raising=False)
        monkeypatch.delenv("BACKEND_ENVIRONMENTS", raising=False)

    def test_outputs_valid_json_and_never_starts_the_server(self) -> None:
        runner = CliRunner()
        with patch.object(server, "mcp") as mock_mcp:
            mock_mcp.run = MagicMock()
            result = runner.invoke(server.main, ["--print-config"])

        assert result.exit_code == 0, result.output
        resolved = json.loads(result.output)
        assert resolved["backend"]["type"] == "jaeger"
        mock_mcp.run.assert_not_called()

    def test_redacts_api_key_and_app_key_to_boolean_presence(self) -> None:
        runner = CliRunner()
        with patch.object(server, "mcp") as mock_mcp:
            mock_mcp.run = MagicMock()
            result = runner.invoke(
                server.main,
                [
                    "--print-config",
                    "--backend",
                    "datadog",
                    "--url",
                    "https://api.datadoghq.com",
                    "--api-key",
                    FAKE_API_KEY,
                    "--app-key",
                    FAKE_APP_KEY,
                ],
            )

        assert FAKE_API_KEY not in result.output
        assert FAKE_APP_KEY not in result.output
        resolved = json.loads(result.output)
        assert resolved["backend"]["api_key_set"] is True
        assert resolved["backend"]["app_key_set"] is True

    def test_includes_non_secret_identifiers_with_real_values(self) -> None:
        runner = CliRunner()
        with patch.object(server, "mcp") as mock_mcp:
            mock_mcp.run = MagicMock()
            result = runner.invoke(
                server.main,
                [
                    "--print-config",
                    "--backend",
                    "sentry",
                    "--url",
                    "https://sentry.io",
                    "--sentry-org",
                    FAKE_SENTRY_ORG,
                    "--sentry-project",
                    FAKE_SENTRY_PROJECT,
                ],
            )

        resolved = json.loads(result.output)
        assert resolved["backend"]["sentry_org"] == FAKE_SENTRY_ORG
        assert resolved["backend"]["sentry_project"] == FAKE_SENTRY_PROJECT

    def test_includes_cli_only_fields_not_stored_on_the_config_model(self) -> None:
        """transport/host/port/tool-gating are never stored on ServerConfig/
        BackendConfig - they must come from this invocation's own CLI args."""
        runner = CliRunner()
        with patch.object(server, "mcp") as mock_mcp:
            mock_mcp.run = MagicMock()
            result = runner.invoke(
                server.main,
                [
                    "--print-config",
                    "--transport",
                    "http",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9001",
                    "--disable-tools",
                    "get_trace",
                    "--enabled-tools",
                    "search_traces,get_trace",
                ],
            )

        resolved = json.loads(result.output)
        assert resolved["transport"] == "http"
        assert resolved["host"] == "127.0.0.1"
        assert resolved["port"] == 9001
        assert resolved["disable_tools"] == "get_trace"
        assert resolved["enabled_tools"] == "search_traces,get_trace"


class TestDoctorCli:
    """doctor runs config load -> backend construction -> health_check ->
    a real read-only connectivity probe (list_services), printing [OK]/
    [FAIL] per step and exiting non-zero on any failure - unlike
    _get_backend, which deliberately swallows a failed health check."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BACKEND_TYPE", "jaeger")
        monkeypatch.setenv("BACKEND_URL", "http://localhost:16686")
        monkeypatch.delenv("BACKEND_API_KEY", raising=False)
        monkeypatch.delenv("BACKEND_APP_KEY", raising=False)
        monkeypatch.delenv("BACKEND_SENTRY_ORG", raising=False)
        monkeypatch.delenv("BACKEND_SENTRY_PROJECT", raising=False)
        monkeypatch.delenv("BACKEND_ENVIRONMENTS", raising=False)

    def _healthy_backend(self) -> AsyncMock:
        from opentelemetry_mcp.attributes import HealthCheckResponse

        fake_backend = AsyncMock()
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(status="healthy", backend="jaeger", url="http://x")
        )
        fake_backend.list_services = AsyncMock(return_value=["svc-a", "svc-b"])
        fake_backend.close = AsyncMock()
        return fake_backend

    def test_all_checks_pass_exits_zero(self) -> None:
        runner = CliRunner()
        with patch.object(server, "_create_backend", return_value=self._healthy_backend()):
            result = runner.invoke(server.main, ["doctor"])

        assert result.exit_code == 0, result.output
        assert result.output.count("[OK]") == 4
        assert "[FAIL]" not in result.output

    def test_config_load_failure_exits_one(self) -> None:
        runner = CliRunner()
        with patch.object(ServerConfig, "from_env", side_effect=ValueError("bad config")):
            result = runner.invoke(server.main, ["doctor"])

        assert result.exit_code == 1
        assert "[FAIL] Configuration" in result.output

    def test_backend_construction_failure_exits_one(self) -> None:
        runner = CliRunner()
        with patch.object(server, "_create_backend", side_effect=ValueError("unsupported backend")):
            result = runner.invoke(server.main, ["doctor"])

        assert result.exit_code == 1
        assert "[FAIL] Backend construction" in result.output

    def test_health_check_unhealthy_status_exits_one(self) -> None:
        from opentelemetry_mcp.attributes import HealthCheckResponse

        fake_backend = self._healthy_backend()
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(
                status="unhealthy", backend="jaeger", url="http://x", error="connection refused"
            )
        )
        runner = CliRunner()
        with patch.object(server, "_create_backend", return_value=fake_backend):
            result = runner.invoke(server.main, ["doctor"])

        assert result.exit_code == 1
        assert "[FAIL] Health check" in result.output
        assert "connection refused" in result.output

    def test_health_check_raises_exits_one(self) -> None:
        """health_check() can raise instead of returning status=unhealthy
        (per its own abstract docstring) - doctor must catch this too."""
        fake_backend = self._healthy_backend()
        fake_backend.health_check = AsyncMock(side_effect=RuntimeError("unreachable"))
        runner = CliRunner()
        with patch.object(server, "_create_backend", return_value=fake_backend):
            result = runner.invoke(server.main, ["doctor"])

        assert result.exit_code == 1
        assert "[FAIL] Health check raised" in result.output

    def test_connectivity_probe_failure_exits_one(self) -> None:
        fake_backend = self._healthy_backend()
        fake_backend.list_services = AsyncMock(side_effect=RuntimeError("timeout"))
        runner = CliRunner()
        with patch.object(server, "_create_backend", return_value=fake_backend):
            result = runner.invoke(server.main, ["doctor"])

        assert result.exit_code == 1
        assert "[FAIL] Connectivity probe" in result.output

    def test_accepts_backend_override_flags(self) -> None:
        runner = CliRunner()
        with patch.object(
            server, "_create_backend", return_value=self._healthy_backend()
        ) as mocked:
            result = runner.invoke(
                server.main,
                ["doctor", "--backend", "datadog", "--url", "https://api.datadoghq.com"],
            )

        assert result.exit_code == 0, result.output
        config = mocked.call_args.args[0]
        assert config.backend.type == "datadog"

    def test_closes_backend_client_on_completion(self) -> None:
        fake_backend = self._healthy_backend()
        runner = CliRunner()
        with patch.object(server, "_create_backend", return_value=fake_backend):
            runner.invoke(server.main, ["doctor"])

        fake_backend.close.assert_awaited_once()

    def test_closes_backend_client_even_when_connectivity_probe_fails(self) -> None:
        fake_backend = self._healthy_backend()
        fake_backend.list_services = AsyncMock(side_effect=RuntimeError("timeout"))
        runner = CliRunner()
        with patch.object(server, "_create_backend", return_value=fake_backend):
            runner.invoke(server.main, ["doctor"])

        fake_backend.close.assert_awaited_once()
