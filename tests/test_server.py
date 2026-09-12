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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from pydantic import HttpUrl

from opentelemetry_mcp import server
from opentelemetry_mcp.attributes import HealthCheckResponse
from opentelemetry_mcp.backends.datadog import DatadogBackend
from opentelemetry_mcp.backends.jaeger import JaegerBackend
from opentelemetry_mcp.backends.sentry import SentryBackend
from opentelemetry_mcp.backends.tempo import TempoBackend
from opentelemetry_mcp.backends.traceloop import TraceloopBackend
from opentelemetry_mcp.config import BackendConfig, ServerConfig

FAKE_API_KEY = "dd-key1"
FAKE_APP_KEY = "dd-app1"
FAKE_SENTRY_ORG = "fake-org"
FAKE_SENTRY_PROJECT = "fake-project"


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
    other via caching in _get_backend."""
    server._backend = None
    server._config = None
    yield
    server._backend = None
    server._config = None


class TestCreateBackend:
    """_create_backend is a pure factory - verify each of the 5 backend
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
        config = _config(type="tempo", api_key=FAKE_API_KEY, timeout=9.0)
        backend = server._create_backend(config)

        assert isinstance(backend, TempoBackend)
        assert backend.api_key == FAKE_API_KEY
        assert backend.timeout == 9.0

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

    def test_unsupported_backend_type_raises(self) -> None:
        config = _config(type="jaeger")
        # BackendConfig.type is a Literal, so bypass validation to simulate
        # an unsupported type reaching the factory.
        config.backend.type = "unknown"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unsupported backend type"):
            server._create_backend(config)


class TestHandleToolError:
    """_handle_tool_error is a pure function - confirm the JSON shape and
    that both the tool name (via logging) and the error message appear in
    the returned payload."""

    def test_returns_json_with_error_message(self) -> None:
        result = server._handle_tool_error("search_traces", ValueError("boom"))

        parsed = json.loads(result)
        assert parsed == {"error": "Tool execution failed: boom"}

    def test_different_error_message_is_reflected(self) -> None:
        result = server._handle_tool_error("get_trace", RuntimeError("trace not found"))

        parsed = json.loads(result)
        assert parsed["error"] == "Tool execution failed: trace not found"

    def test_logs_tool_name(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("ERROR"):
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
    turn any exception into _handle_tool_error's JSON. Mock at the
    module-level names the wrapper actually calls (server._get_backend and
    server.<tools_module>.<function>) so we exercise the wrapper's own glue
    code rather than the mock."""

    async def _set_backend(self) -> AsyncMock:
        fake_backend = AsyncMock()
        server._get_backend = AsyncMock(return_value=fake_backend)
        return fake_backend

    async def test_search_traces_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.search, "search_traces", AsyncMock(return_value='{"ok":1}')
        ) as mocked:
            result = await server.search_traces(
                service_name="svc",
                operation_name="op",
                min_duration_ms=5,
                has_error=True,
                limit=42,
            )

        assert result == '{"ok":1}'
        _, kwargs = mocked.call_args
        assert kwargs["service_name"] == "svc"
        assert kwargs["operation_name"] == "op"
        assert kwargs["min_duration_ms"] == 5
        assert kwargs["has_error"] is True
        assert kwargs["limit"] == 42

    async def test_search_traces_exception_becomes_error_json(self) -> None:
        await self._set_backend()
        with patch.object(
            server.search, "search_traces", AsyncMock(side_effect=ValueError("bad filter"))
        ):
            result = await server.search_traces()

        parsed = json.loads(result)
        assert parsed == {"error": "Tool execution failed: bad filter"}

    async def test_get_trace_passes_trace_id(self) -> None:
        await self._set_backend()
        with patch.object(
            server.trace, "get_trace", AsyncMock(return_value='{"trace":1}')
        ) as mocked:
            result = await server.get_trace(trace_id="abc123")

        assert result == '{"trace":1}'
        _, kwargs = mocked.call_args
        assert kwargs["trace_id"] == "abc123"

    async def test_get_trace_exception_becomes_error_json(self) -> None:
        await self._set_backend()
        with patch.object(server.trace, "get_trace", AsyncMock(side_effect=KeyError("nope"))):
            result = await server.get_trace(trace_id="missing")

        parsed = json.loads(result)
        assert "error" in parsed

    async def test_get_llm_usage_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(server.usage, "get_llm_usage", AsyncMock(return_value="{}")) as mocked:
            await server.get_llm_usage(service_name="svc", gen_ai_system="openai", limit=50)

        _, kwargs = mocked.call_args
        assert kwargs["service_name"] == "svc"
        assert kwargs["gen_ai_system"] == "openai"
        assert kwargs["limit"] == 50

    async def test_get_llm_usage_exception_becomes_error_json(self) -> None:
        await self._set_backend()
        with patch.object(server.usage, "get_llm_usage", AsyncMock(side_effect=Exception("fail"))):
            result = await server.get_llm_usage()

        assert json.loads(result)["error"] == "Tool execution failed: fail"

    async def test_list_services_calls_tool_with_backend_only(self) -> None:
        fake_backend = await self._set_backend()
        with patch.object(
            server.services, "list_services", AsyncMock(return_value='["a","b"]')
        ) as mocked:
            result = await server.list_services()

        assert result == '["a","b"]'
        mocked.assert_awaited_once_with(fake_backend)

    async def test_list_services_exception_becomes_error_json(self) -> None:
        await self._set_backend()
        with patch.object(
            server.services, "list_services", AsyncMock(side_effect=RuntimeError("down"))
        ):
            result = await server.list_services()

        assert json.loads(result)["error"] == "Tool execution failed: down"

    async def test_find_errors_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(server.errors, "find_errors", AsyncMock(return_value="{}")) as mocked:
            await server.find_errors(service_name="svc", limit=5)

        _, kwargs = mocked.call_args
        assert kwargs["service_name"] == "svc"
        assert kwargs["limit"] == 5

    async def test_find_errors_exception_becomes_error_json(self) -> None:
        await self._set_backend()
        with patch.object(server.errors, "find_errors", AsyncMock(side_effect=Exception("boom"))):
            result = await server.find_errors()

        assert json.loads(result)["error"] == "Tool execution failed: boom"

    async def test_list_llm_models_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.list_models, "list_models", AsyncMock(return_value="{}")
        ) as mocked:
            await server.list_llm_models(gen_ai_system="anthropic", limit=10)

        _, kwargs = mocked.call_args
        assert kwargs["gen_ai_system"] == "anthropic"
        assert kwargs["limit"] == 10

    async def test_list_llm_models_exception_becomes_error_json(self) -> None:
        await self._set_backend()
        with patch.object(
            server.list_models, "list_models", AsyncMock(side_effect=Exception("boom"))
        ):
            result = await server.list_llm_models()

        assert json.loads(result)["error"] == "Tool execution failed: boom"

    async def test_get_llm_model_stats_passes_model_name(self) -> None:
        await self._set_backend()
        with patch.object(
            server.model_stats, "get_model_stats", AsyncMock(return_value="{}")
        ) as mocked:
            await server.get_llm_model_stats(model_name="gpt-4", service_name="svc")

        _, kwargs = mocked.call_args
        assert kwargs["model_name"] == "gpt-4"
        assert kwargs["service_name"] == "svc"

    async def test_get_llm_model_stats_exception_becomes_error_json(self) -> None:
        await self._set_backend()
        with patch.object(
            server.model_stats, "get_model_stats", AsyncMock(side_effect=Exception("boom"))
        ):
            result = await server.get_llm_model_stats(model_name="gpt-4")

        assert json.loads(result)["error"] == "Tool execution failed: boom"

    async def test_get_llm_expensive_traces_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.expensive_traces, "get_expensive_traces", AsyncMock(return_value="{}")
        ) as mocked:
            await server.get_llm_expensive_traces(limit=3, min_tokens=1000)

        _, kwargs = mocked.call_args
        assert kwargs["limit"] == 3
        assert kwargs["min_tokens"] == 1000

    async def test_get_llm_expensive_traces_exception_becomes_error_json(self) -> None:
        await self._set_backend()
        with patch.object(
            server.expensive_traces,
            "get_expensive_traces",
            AsyncMock(side_effect=Exception("boom")),
        ):
            result = await server.get_llm_expensive_traces()

        assert json.loads(result)["error"] == "Tool execution failed: boom"

    async def test_get_llm_slow_traces_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.slow_traces, "get_slow_traces", AsyncMock(return_value="{}")
        ) as mocked:
            await server.get_llm_slow_traces(limit=7, min_duration_ms=250)

        _, kwargs = mocked.call_args
        assert kwargs["limit"] == 7
        assert kwargs["min_duration_ms"] == 250

    async def test_get_llm_slow_traces_exception_becomes_error_json(self) -> None:
        await self._set_backend()
        with patch.object(
            server.slow_traces, "get_slow_traces", AsyncMock(side_effect=Exception("boom"))
        ):
            result = await server.get_llm_slow_traces()

        assert json.loads(result)["error"] == "Tool execution failed: boom"

    async def test_search_spans_tool_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.search_spans, "search_spans", AsyncMock(return_value="{}")
        ) as mocked:
            await server.search_spans_tool(service_name="svc", has_error=True)

        _, kwargs = mocked.call_args
        assert kwargs["service_name"] == "svc"
        assert kwargs["has_error"] is True

    async def test_search_spans_tool_exception_becomes_error_json(self) -> None:
        await self._set_backend()
        with patch.object(
            server.search_spans, "search_spans", AsyncMock(side_effect=Exception("boom"))
        ):
            result = await server.search_spans_tool()

        assert json.loads(result)["error"] == "Tool execution failed: boom"

    async def test_list_llm_tools_tool_passes_arguments_through(self) -> None:
        await self._set_backend()
        with patch.object(
            server.list_llm_tools, "list_llm_tools", AsyncMock(return_value="{}")
        ) as mocked:
            await server.list_llm_tools_tool(gen_ai_system="openai", limit=99)

        _, kwargs = mocked.call_args
        assert kwargs["gen_ai_system"] == "openai"
        assert kwargs["limit"] == 99

    async def test_list_llm_tools_tool_exception_becomes_error_json(self) -> None:
        await self._set_backend()
        with patch.object(
            server.list_llm_tools, "list_llm_tools", AsyncMock(side_effect=Exception("boom"))
        ):
            result = await server.list_llm_tools_tool()

        assert json.loads(result)["error"] == "Tool execution failed: boom"

    async def test_get_backend_failure_becomes_error_json(self) -> None:
        """If _get_backend itself raises (e.g. config not set), the wrapper
        must still catch it via _handle_tool_error rather than propagating."""
        server._get_backend = AsyncMock(side_effect=RuntimeError("Server configuration not set"))

        result = await server.list_services()

        assert json.loads(result)["error"] == "Tool execution failed: Server configuration not set"


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
            environments="prd,staging",
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

    def test_http_transport_calls_mcp_run_with_host_and_port(self) -> None:
        runner = CliRunner()
        with patch.object(server, "mcp") as mock_mcp:
            mock_mcp.run = MagicMock()
            result = runner.invoke(
                server.main,
                ["--transport", "http", "--host", "127.0.0.1", "--port", "9001"],
            )

        assert result.exit_code == 0, result.output
        mock_mcp.run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=9001
        )

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
