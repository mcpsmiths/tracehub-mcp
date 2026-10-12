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

import io
import json
import logging
from collections.abc import Generator
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import click
import httpx
import pytest
from click.testing import CliRunner
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pydantic import HttpUrl
from starlette.testclient import TestClient

from opentelemetry_mcp import server
from opentelemetry_mcp.attributes import HealthCheckResponse
from opentelemetry_mcp.backends.base import BaseBackend, _RetryingTransport
from opentelemetry_mcp.backends.datadog import DatadogBackend
from opentelemetry_mcp.backends.honeycomb import HoneycombBackend, HoneycombEnterpriseRequiredError
from opentelemetry_mcp.backends.jaeger import JaegerBackend
from opentelemetry_mcp.backends.newrelic import NewRelicBackend
from opentelemetry_mcp.backends.sentry import SentryBackend
from opentelemetry_mcp.backends.tempo import TempoBackend
from opentelemetry_mcp.backends.traceloop import TraceloopBackend
from opentelemetry_mcp.backends.xray import XRayBackend
from opentelemetry_mcp.config import BackendConfig, ServerConfig
from opentelemetry_mcp.models import (
    CorrelationResult,
    SearchTracesResult,
    TraceDetail,
    TriageResult,
    TriageVerdict,
)
from opentelemetry_mcp.security import UserinfoRedactingFilter

FAKE_API_KEY = "dd-key1"
FAKE_APP_KEY = "dd-app1"
FAKE_SENTRY_ORG = "fake-org"
FAKE_SENTRY_PROJECT = "fake-project"
FAKE_TEMPO_INSTANCE_ID = "123456"
FAKE_NEWRELIC_ACCOUNT_ID = "7654321"
FAKE_HONEYCOMB_DATASET = "my-dataset"


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
    """The module-level _backend/_secondary_backend/_config globals are
    shared mutable state - reset them before and after every test so tests
    can't leak into each other via caching in _get_backend/
    _get_secondary_backend. TestToolWrappers also reassigns _get_backend/
    _get_secondary_backend themselves (not just the globals they cache) via
    raw attribute assignment rather than unittest.mock.patch, so both must
    be restored here too or a mocked version silently leaks into every test
    that runs afterward in the same session."""
    real_get_backend = server._get_backend
    real_get_secondary_backend = server._get_secondary_backend
    server._backend = None
    server._secondary_backend = None
    server._config = None
    server.mcp.auth = None
    yield
    server._backend = None
    server._secondary_backend = None
    server._config = None
    server._get_backend = real_get_backend
    server._get_secondary_backend = real_get_secondary_backend
    # main()'s HTTP branch installs the bearer-token verifier on the shared
    # FastMCP singleton; leave it unauthenticated for the next test.
    server.mcp.auth = None


class TestCreateBackend:
    """_create_backend is a pure factory - verify each of the 8 backend
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

    def test_newrelic(self) -> None:
        config = _config(
            type="newrelic",
            url=HttpUrl("https://api.newrelic.com/graphql"),
            api_key=FAKE_API_KEY,
            newrelic_account_id=FAKE_NEWRELIC_ACCOUNT_ID,
            timeout=14.0,
        )
        backend = server._create_backend(config)

        assert isinstance(backend, NewRelicBackend)
        assert backend.api_key == FAKE_API_KEY
        assert backend.account_id == int(FAKE_NEWRELIC_ACCOUNT_ID)
        assert backend.timeout == 14.0

    def test_honeycomb(self) -> None:
        config = _config(
            type="honeycomb",
            url=HttpUrl("https://api.honeycomb.io"),
            api_key=FAKE_API_KEY,
            honeycomb_dataset=FAKE_HONEYCOMB_DATASET,
            timeout=16.0,
        )
        backend = server._create_backend(config)

        assert isinstance(backend, HoneycombBackend)
        assert backend.api_key == FAKE_API_KEY
        assert backend.dataset == FAKE_HONEYCOMB_DATASET
        assert backend.timeout == 16.0

    def test_applies_max_response_bytes_from_config(self) -> None:
        """Like slow_request_threshold_ms, not a constructor parameter -
        applied uniformly after construction so every backend's transport
        gets the operator's BACKEND_MAX_RESPONSE_BYTES."""
        config = _config(type="jaeger", max_response_bytes=123456)
        backend = server._create_backend(config)

        assert backend.max_response_bytes == 123456
        transport = backend.client._transport
        assert isinstance(transport, _RetryingTransport)
        assert transport._max_response_bytes == 123456

    def test_unsupported_backend_type_raises(self) -> None:
        config = _config(type="jaeger")
        # BackendConfig.type is a Literal, so bypass validation to simulate
        # an unsupported type reaching the factory.
        config.backend.type = "unknown"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unsupported backend type"):
            server._create_backend(config)


class TestBackendUrlNeverLoggedWithUserinfo:
    """Defence in depth behind BackendConfig.validate_url's rejection: the
    startup INFO lines print the backend URL, so they go through
    redact_url_userinfo even though a validated config can no longer carry
    userinfo. model_construct() bypasses the validator on purpose."""

    MARKER = "VFY-PW-init"

    def _unvalidated_config(self, backend_type: str, **fields: object) -> BackendConfig:
        values: dict[str, Any] = {
            "type": backend_type,
            "url": HttpUrl(f"https://u:{self.MARKER}@h.test/"),
            "api_key": None,
            "app_key": None,
            "sentry_org": None,
            "sentry_project": None,
            "tempo_instance_id": None,
            "aws_region": None,
            "newrelic_account_id": None,
            "honeycomb_dataset": None,
            "timeout": 30.0,
            "max_response_bytes": 64 * 1024 * 1024,
            "environments": ["prd"],
        }
        values.update(fields)
        return BackendConfig.model_construct(None, **values)

    @pytest.mark.parametrize(
        ("backend_type", "fields"),
        [
            ("jaeger", {}),
            ("tempo", {}),
            ("traceloop", {"api_key": FAKE_API_KEY}),
            ("datadog", {"api_key": FAKE_API_KEY, "app_key": FAKE_APP_KEY}),
            ("sentry", {"api_key": FAKE_API_KEY, "sentry_org": FAKE_SENTRY_ORG}),
        ],
    )
    def test_initializing_line_strips_userinfo(
        self, backend_type: str, fields: dict[str, object], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            server._build_backend_from_config(self._unvalidated_config(backend_type, **fields))

        (line,) = [r.message for r in caplog.records if r.message.startswith("Initializing")]
        assert self.MARKER not in line
        assert "https://h.test/" in line

    async def test_secondary_health_log_is_redacted(self, caplog: pytest.LogCaptureFixture) -> None:
        server._config = _config()
        server._config.secondary_backend = BackendConfig(
            type="sentry", url=HttpUrl("https://sentry.io"), sentry_org=FAKE_SENTRY_ORG
        )
        fake_backend = AsyncMock(spec=BaseBackend)
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(
                status="unhealthy",
                backend="sentry",
                url=f"https://u:{self.MARKER}@sentry.io",
                error=f"Client error '401 Unauthorized' for url 'https://u:{self.MARKER}@sentry.io/'",
            )
        )

        with (
            caplog.at_level(logging.INFO),
            patch.object(server, "_build_backend_from_config", return_value=fake_backend),
        ):
            await server._get_secondary_backend()

        assert self.MARKER not in caplog.text
        assert "unhealthy" in caplog.text


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

    def test_credential_free_error_keeps_its_traceback_in_the_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
            server._handle_tool_error("list_services", RuntimeError("plain failure"))

        (record,) = caplog.records
        assert record.exc_info is not None

    def test_credential_bearing_error_becomes_a_redacted_tool_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The one case where the identity re-raise is replaced: httpx's
        HTTPStatusError text embeds the request URL, so a credential a
        bypassed config left in BACKEND_URL's userinfo would otherwise reach
        the MCP client as tool-error content and the log as both message
        and traceback. ToolError keeps isError=True and is logged by
        fastmcp without exc_info."""
        marker = "VFY-PW-tool"
        request = httpx.Request("GET", f"https://u:{marker}@h.test/api/services")
        error = httpx.HTTPStatusError(
            f"Client error '401 Unauthorized' for url 'https://u:{marker}@h.test/api/services'",
            request=request,
            response=httpx.Response(401, request=request),
        )

        with caplog.at_level("ERROR"), pytest.raises(ToolError) as exc_info:
            server._handle_tool_error("list_services", error)

        assert exc_info.value.__cause__ is error
        assert marker not in str(exc_info.value)
        assert "401 Unauthorized" in str(exc_info.value)
        assert "h.test/api/services" in str(exc_info.value)
        assert marker not in caplog.text  # message AND any formatted traceback
        assert "list_services" in caplog.text

    def test_credential_bearing_cause_also_becomes_a_redacted_tool_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The whole chain is judged, not just the top message: honeycomb
        raises HoneycombEnterpriseRequiredError `from` the HTTPStatusError
        whose text embeds the request URL. An identity re-raise would land
        in fastmcp's generic `except Exception`, whose logger.exception()
        prints the chained cause through the 'fastmcp' logger's own
        RichHandler - that logger does not propagate to the root handler,
        so UserinfoRedactingFilter never sees it. ToolError is logged by
        fastmcp with exc_info=False, so the chain is never printed."""
        marker = "VFY-PW-cause"
        request = httpx.Request("POST", f"https://u:{marker}@h.test/1/queries/ds")
        cause = httpx.HTTPStatusError(
            f"Client error '403 Forbidden' for url 'https://u:{marker}@h.test/1/queries/ds'",
            request=request,
            response=httpx.Response(403, request=request),
        )
        try:
            raise HoneycombEnterpriseRequiredError(
                "Honeycomb backend requires an Enterprise-tier account (HTTP 403)"
            ) from cause
        except HoneycombEnterpriseRequiredError as chained:
            error = chained

        with caplog.at_level("ERROR"), pytest.raises(ToolError) as exc_info:
            server._handle_tool_error("search_traces", error)

        assert exc_info.value.__cause__ is error
        assert str(exc_info.value) == str(error)
        (record,) = caplog.records
        assert record.exc_info is None
        assert marker not in caplog.text
        assert "search_traces" in caplog.text


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

    async def test_health_log_reports_status_and_redacted_error_not_the_repr(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """HealthCheckResponse carries url= and error= verbatim; logging the
        model repr would print both. The lazy-init INFO line logs the
        status and the redacted error only."""
        marker = "VFY-PW-lazy"
        server._config = _config()
        fake_backend = AsyncMock()
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(
                status="unhealthy",
                backend="jaeger",
                url=f"https://u:{marker}@h.test",
                error=f"Client error '401 Unauthorized' for url 'https://u:{marker}@h.test/api'",
            )
        )

        with (
            caplog.at_level(logging.INFO),
            patch.object(server, "_create_backend", return_value=fake_backend),
        ):
            await server._get_backend()

        assert marker not in caplog.text
        health_lines = [r.message for r in caplog.records if "health check" in r.message.lower()]
        assert any("unhealthy" in line and "h.test/api" in line for line in health_lines)

    async def test_health_check_exception_log_is_redacted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        marker = "VFY-PW-lazy2"
        server._config = _config()
        fake_backend = AsyncMock()
        fake_backend.health_check = AsyncMock(
            side_effect=RuntimeError(f"cannot reach https://u:{marker}@h.test/")
        )

        with (
            caplog.at_level(logging.INFO),
            patch.object(server, "_create_backend", return_value=fake_backend),
        ):
            await server._get_backend()

        assert marker not in caplog.text
        assert "[REDACTED]@h.test" in caplog.text

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


class TestGetSecondaryBackend:
    """_get_secondary_backend mirrors _get_backend's own lazy-init/caching
    pattern, plus a config.secondary_backend is None -> ValueError branch
    _get_backend has no equivalent of (the primary backend is always
    required, so it has nothing analogous to check)."""

    async def test_raises_when_config_not_set(self) -> None:
        server._config = None
        server._secondary_backend = None

        with pytest.raises(RuntimeError, match="Server configuration not set"):
            await server._get_secondary_backend()

    async def test_raises_when_no_secondary_backend_configured(self) -> None:
        server._config = _config()
        assert server._config.secondary_backend is None

        with pytest.raises(ValueError, match="No secondary backend configured"):
            await server._get_secondary_backend()

    async def test_lazily_creates_and_caches_secondary_backend(self) -> None:
        server._config = _config()
        server._config.secondary_backend = BackendConfig(
            type="sentry", url=HttpUrl("https://sentry.io"), sentry_org=FAKE_SENTRY_ORG
        )
        fake_backend = AsyncMock(spec=BaseBackend)
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(
                status="healthy", backend="sentry", url="https://sentry.io"
            )
        )

        with patch.object(
            server, "_build_backend_from_config", return_value=fake_backend
        ) as mock_build:
            first = await server._get_secondary_backend()
            second = await server._get_secondary_backend()

        assert first is fake_backend
        assert second is fake_backend
        mock_build.assert_called_once_with(server._config.secondary_backend)
        fake_backend.health_check.assert_awaited_once()

    async def test_applies_secondary_max_response_bytes(self) -> None:
        server._config = _config()
        server._config.secondary_backend = BackendConfig(
            type="sentry",
            url=HttpUrl("https://sentry.io"),
            sentry_org=FAKE_SENTRY_ORG,
            max_response_bytes=654321,
        )
        fake_backend = AsyncMock(spec=BaseBackend)
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(
                status="healthy", backend="sentry", url="https://sentry.io"
            )
        )

        with patch.object(server, "_build_backend_from_config", return_value=fake_backend):
            await server._get_secondary_backend()

        assert fake_backend.max_response_bytes == 654321

    async def test_health_check_exception_is_swallowed(self) -> None:
        server._config = _config()
        server._config.secondary_backend = BackendConfig(
            type="sentry", url=HttpUrl("https://sentry.io"), sentry_org=FAKE_SENTRY_ORG
        )
        fake_backend = AsyncMock(spec=BaseBackend)
        fake_backend.health_check = AsyncMock(side_effect=RuntimeError("unreachable"))

        with patch.object(server, "_build_backend_from_config", return_value=fake_backend):
            result = await server._get_secondary_backend()

        assert result is fake_backend


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

    async def test_triage_trace_passes_trace_id(self) -> None:
        await self._set_backend()
        sentinel = TriageResult(
            trace_id="abc123",
            verdict=TriageVerdict(likely_root_cause=None, confidence="low", reasoning="no spans"),
            critical_path=[],
            top_latency_contributors=[],
            error_chain=None,
        )
        with patch.object(
            server.triage, "triage_trace", AsyncMock(return_value=sentinel)
        ) as mocked:
            result = await server.triage_trace(trace_id="abc123")

        assert result is sentinel
        _, kwargs = mocked.call_args
        assert kwargs["trace_id"] == "abc123"
        assert kwargs["detail_level"] == "summary"

    async def test_triage_trace_exception_propagates(self) -> None:
        await self._set_backend()
        with (
            patch.object(server.triage, "triage_trace", AsyncMock(side_effect=KeyError("nope"))),
            pytest.raises(KeyError, match="nope"),
        ):
            await server.triage_trace(trace_id="missing")

    async def test_correlate_trace_passes_trace_id_to_both_backends(self) -> None:
        primary = await self._set_backend()
        secondary = AsyncMock()
        server._get_secondary_backend = AsyncMock(return_value=secondary)
        sentinel = CorrelationResult(primary_trace_id="abc123", matches=[], limitations=[])
        with patch.object(
            server.correlate, "correlate_trace", AsyncMock(return_value=sentinel)
        ) as mocked:
            result = await server.correlate_trace(trace_id="abc123")

        assert result is sentinel
        args, kwargs = mocked.call_args
        assert args[0] is primary
        assert args[1] is secondary
        assert kwargs["trace_id"] == "abc123"

    async def test_correlate_trace_no_secondary_backend_propagates_as_value_error(self) -> None:
        await self._set_backend()
        server._get_secondary_backend = AsyncMock(
            side_effect=ValueError("No secondary backend configured")
        )

        with pytest.raises(ValueError, match="No secondary backend configured"):
            await server.correlate_trace(trace_id="abc123")

    async def test_correlate_trace_exception_propagates(self) -> None:
        await self._set_backend()
        server._get_secondary_backend = AsyncMock(return_value=AsyncMock())
        with (
            patch.object(
                server.correlate, "correlate_trace", AsyncMock(side_effect=KeyError("nope"))
            ),
            pytest.raises(KeyError, match="nope"),
        ):
            await server.correlate_trace(trace_id="missing")

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

        assert len(tools) == 19
        for t in tools:
            assert t.annotations is not None, f"{t.name} has no annotations at all"
            assert t.annotations.read_only_hint is True, t.name
            assert t.annotations.destructive_hint is False, t.name
            assert t.annotations.idempotent_hint is True, t.name
            assert t.annotations.open_world_hint is True, t.name

    async def test_every_tool_has_a_distinct_human_readable_title(self) -> None:
        """Anthropic's Software Directory Policy requires readOnlyHint,
        destructiveHint, and title on every tool - title lives on Tool
        itself (FastMCP's title= kwarg), not inside ToolAnnotations."""
        async with Client(server.mcp) as client:
            tools = await client.list_tools()

        assert len(tools) == 19
        titles = [t.title for t in tools]
        assert all(isinstance(title, str) and title for title in titles), titles
        assert len(set(titles)) == 19, "titles must be distinct per tool"


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


class TestFixedWindowRateLimiterEviction:
    """_FixedWindowRateLimiter._buckets must be swept of expired entries,
    not retained forever - see _sweep_expired_locked's docstring. Without
    this, every distinct key (client IP) that ever hits the server adds a
    permanent entry that's never removed, even long after its window
    expired."""

    async def test_buckets_do_not_grow_unboundedly_across_many_expired_windows(
        self,
    ) -> None:
        """Regression test: simulates many distinct, never-repeated client
        IPs (as a rotating-source bot, or simply many one-off clients,
        would produce) across several fully-elapsed windows. Before the
        fix, _buckets accumulated one permanent entry per distinct IP
        (len == 20 here); with the sweep, only recently-active windows
        survive."""
        clock_time = [0.0]

        def fake_clock() -> float:
            return clock_time[0]

        limiter = server._FixedWindowRateLimiter(
            max_requests=10, window_seconds=60.0, clock=fake_clock
        )

        for window_index in range(20):
            # Each iteration lands in a window that is already fully
            # expired relative to the previous one (61s > the 60s window).
            clock_time[0] = window_index * 61.0
            await limiter.hit(f"client-{window_index}")

        assert len(limiter._buckets) < 20

    async def test_sweep_only_evicts_entries_whose_window_has_expired(self) -> None:
        """A bucket whose window has NOT yet expired must survive a sweep
        triggered by a different key's hit - only entries genuinely past
        their own window_seconds should ever be dropped."""
        clock_time = [0.0]

        def fake_clock() -> float:
            return clock_time[0]

        limiter = server._FixedWindowRateLimiter(
            max_requests=10, window_seconds=60.0, clock=fake_clock
        )

        clock_time[0] = 10.0
        await limiter.hit("stale-client")  # window_start=10; fully expired
        # (>=60s old) by the time the sweep below runs at t=70

        clock_time[0] = 59.0
        await limiter.hit("fresh-client")  # window_start=59; still well
        # within its own 60s window at t=70

        clock_time[0] = 70.0
        # >=60s since construction (t=0) triggers the amortized sweep.
        await limiter.hit("other-client")

        assert "stale-client" not in limiter._buckets
        assert "fresh-client" in limiter._buckets

    async def test_hit_behavior_unaffected_by_sweep(self) -> None:
        """The sweep must be purely a cleanup side effect - it must not
        change hit()'s own allow/reject decision for the key being hit."""
        clock_time = [0.0]

        def fake_clock() -> float:
            return clock_time[0]

        limiter = server._FixedWindowRateLimiter(
            max_requests=2, window_seconds=60.0, clock=fake_clock
        )

        assert await limiter.hit("a") is True
        assert await limiter.hit("a") is True
        # Third hit within the same window breaches max_requests=2.
        assert await limiter.hit("a") is False

        # Advance past the window and past the sweep threshold - a sweep
        # runs, but "a" gets a fresh window and is allowed again.
        clock_time[0] = 61.0
        assert await limiter.hit("a") is True


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
            newrelic_account_id=None,
            honeycomb_dataset=None,
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
        # fastmcp's own default for session_idle_timeout is None ("never
        # expire"), overriding the mcp SDK's 1800 s - main() must always
        # pass a finite bound in stateful mode (see _build_http_app).
        assert http_app_kwargs["session_idle_timeout"] == 1800.0
        assert http_app_kwargs["stateless_http"] is False
        # HTTP transport must wire in the Origin-validation middleware
        # (Tier 1 security fix: fastmcp disables DNS-rebinding protection
        # by default, see OriginValidationMiddleware's docstring), and the
        # rate limiter (enabled by default, see RateLimitMiddleware).
        middleware = http_app_kwargs["middleware"]
        assert len(middleware) == 2
        assert middleware[0].cls is server.OriginValidationMiddleware
        # The bound address drives automatic Host validation for loopback
        # binds; --allowed-hosts is None unless the operator names hosts.
        assert middleware[0].kwargs == {"bind_host": "127.0.0.1", "allowed_hosts": None}
        assert middleware[1].cls is server.RateLimitMiddleware

        mock_server_cls.assert_called_once()
        config_arg = mock_server_cls.call_args.args[0]
        assert config_arg.app is mock_mcp.http_app.return_value
        assert config_arg.host == "127.0.0.1"
        assert config_arg.port == 9001
        # uvicorn's own default trusts X-Forwarded-For from 127.0.0.1; the
        # limiter keys on request.client.host, so that trust is opt-in only.
        assert config_arg.proxy_headers is False
        mock_server_cls.return_value.serve.assert_awaited_once()

    def test_forwarded_allow_ips_flag_enables_proxy_headers(self) -> None:
        runner = CliRunner()
        with (
            patch.object(server, "mcp"),
            patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls,
        ):
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(
                server.main, ["--transport", "http", "--forwarded-allow-ips", "10.0.0.1"]
            )

        assert result.exit_code == 0, result.output
        config_arg = mock_server_cls.call_args.args[0]
        assert config_arg.proxy_headers is True
        assert config_arg.forwarded_allow_ips == "10.0.0.1"

    def test_forwarded_allow_ips_env_var_keeps_uvicorns_name(self) -> None:
        """Operators who already set uvicorn's FORWARDED_ALLOW_IPS see
        identical behaviour - the option deliberately reuses that name."""
        runner = CliRunner()
        with (
            patch.object(server, "mcp"),
            patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls,
        ):
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(
                server.main, ["--transport", "http"], env={"FORWARDED_ALLOW_IPS": "127.0.0.1"}
            )

        assert result.exit_code == 0, result.output
        config_arg = mock_server_cls.call_args.args[0]
        assert config_arg.proxy_headers is True
        assert config_arg.forwarded_allow_ips == "127.0.0.1"

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            ([], True),
            (["--no-trust-client-trace-context"], False),
            (["--transport", "http"], False),
            (["--transport", "http", "--trust-client-trace-context"], True),
        ],
        ids=["stdio-default", "stdio-opt-out", "http-default", "http-opt-in"],
    )
    def test_trust_client_trace_context_defaults_per_transport(
        self, args: list[str], expected: bool
    ) -> None:
        """stdio's only caller is the trusted parent process, so its _meta
        trace context is adopted by default; an HTTP peer's is not."""
        runner = CliRunner()
        with (
            patch.object(server, "mcp"),
            patch.object(server, "configure_tracing", return_value=False) as mock_tracing,
            patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls,
        ):
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(server.main, args)

        assert result.exit_code == 0, result.output
        mock_tracing.assert_called_once_with(trust_client_trace_context=expected)

    def test_trust_client_trace_context_env_var(self) -> None:
        runner = CliRunner()
        with (
            patch.object(server, "mcp"),
            patch.object(server, "configure_tracing", return_value=False) as mock_tracing,
            patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls,
        ):
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(
                server.main, ["--transport", "http"], env={"MCP_TRUST_CLIENT_TRACE_CONTEXT": "true"}
            )

        assert result.exit_code == 0, result.output
        mock_tracing.assert_called_once_with(trust_client_trace_context=True)

    def test_allowed_hosts_flag_is_parsed_into_the_origin_middleware(self) -> None:
        runner = CliRunner()
        with (
            patch.object(server, "mcp") as mock_mcp,
            patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls,
        ):
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(
                server.main,
                ["--transport", "http", "--allowed-hosts", "tracehub.example, 10.0.0.5"],
            )

        assert result.exit_code == 0, result.output
        middleware = mock_mcp.http_app.call_args.kwargs["middleware"]
        assert middleware[0].kwargs == {
            "bind_host": "127.0.0.1",
            "allowed_hosts": ["tracehub.example", "10.0.0.5"],
        }

    def test_session_idle_timeout_flag_is_forwarded(self) -> None:
        runner = CliRunner()
        with (
            patch.object(server, "mcp") as mock_mcp,
            patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls,
        ):
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(
                server.main,
                ["--transport", "http", "--session-idle-timeout-seconds", "0.5"],
            )

        assert result.exit_code == 0, result.output
        http_app_kwargs = mock_mcp.http_app.call_args.kwargs
        assert http_app_kwargs["session_idle_timeout"] == 0.5
        assert http_app_kwargs["stateless_http"] is False

    def test_stateless_http_flag_drops_the_idle_timeout(self) -> None:
        runner = CliRunner()
        with (
            patch.object(server, "mcp") as mock_mcp,
            patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls,
        ):
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(server.main, ["--transport", "http", "--stateless-http"])

        assert result.exit_code == 0, result.output
        http_app_kwargs = mock_mcp.http_app.call_args.kwargs
        assert http_app_kwargs["stateless_http"] is True
        assert http_app_kwargs["session_idle_timeout"] is None

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_non_positive_session_idle_timeout_is_a_usage_error(self, value: str) -> None:
        """The SDK raises ValueError for a non-positive idle timeout at
        lifespan start - far too late and far too cryptic. click.FloatRange
        (min=0, min_open=True) rejects it up front with exit code 2."""
        runner = CliRunner()
        with (
            patch.object(server, "mcp") as mock_mcp,
            patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls,
        ):
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(
                server.main,
                ["--transport", "http", "--session-idle-timeout-seconds", value],
            )

        assert result.exit_code == 2, result.output
        assert "--session-idle-timeout-seconds" in result.output
        assert "not in the range" in result.output
        mock_mcp.http_app.assert_not_called()

    @pytest.mark.parametrize("value", ["inf", "nan", "Infinity"])
    def test_non_finite_session_idle_timeout_is_a_usage_error(self, value: str) -> None:
        """click.FloatRange(min=0, min_open=True) still admits inf (inf > 0
        holds) and nan (every comparison against it is False), and the SDK's
        StreamableHTTPSessionManager rejects both only at lifespan start -
        the same late, cryptic 'Server error' / exit 1 the range check exists
        to prevent. The option callback turns them into exit-code-2 usage
        errors before the app is ever built."""
        runner = CliRunner()
        with (
            patch.object(server, "mcp") as mock_mcp,
            patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls,
        ):
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(
                server.main,
                ["--transport", "http", "--session-idle-timeout-seconds", value],
            )

        assert result.exit_code == 2, result.output
        assert "--session-idle-timeout-seconds" in result.output
        assert "finite" in result.output
        mock_mcp.http_app.assert_not_called()

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

    def test_click_usage_error_inside_the_try_is_not_swallowed(self) -> None:
        """click.UsageError is an Exception subclass: without an explicit
        `except click.ClickException: raise` ahead of the catch-all, a usage
        error raised anywhere inside main()'s try would be logged as
        'Server error' and exit 1 instead of click's own exit 2 with the
        message on stderr."""
        runner = CliRunner()
        with patch.object(
            ServerConfig, "from_env", side_effect=click.UsageError("bad combination of flags")
        ):
            result = runner.invoke(server.main, [])

        assert result.exit_code == 2
        assert "bad combination of flags" in result.output

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
        assert resolved["http_bearer_token_set"] is False
        assert resolved["allow_unauthenticated_http"] is False
        assert resolved["session_idle_timeout_seconds"] == 1800.0
        assert resolved["stateless_http"] is False
        assert resolved["allowed_hosts"] is None
        assert resolved["forwarded_allow_ips"] is None
        assert resolved["trust_client_trace_context"] is False


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


class TestUserinfoNeverLeavesTheProcess:
    """End to end, bypassing config on purpose (BackendConfig.validate_url
    now rejects userinfo; everything below is the defence in depth for a
    URL that got past it): a real JaegerBackend over the real
    _RetryingTransport with only the innermost transport mocked to answer
    401, driven through the real streamable-http app (/ready) and the
    in-memory fastmcp client (tools/call), with every log sink captured."""

    MARKER = "VFY-PW-e2e"

    @pytest.fixture(autouse=True)
    def _log_levels(self) -> Generator[None]:
        """Apply the production log-level path (which quiets httpx's INFO
        request line, the one sink that cannot be redacted at creation
        time) and restore both loggers afterwards."""
        root_level = logging.getLogger().level
        httpx_level = logging.getLogger("httpx").level
        server._apply_log_level("INFO")
        yield
        logging.getLogger().setLevel(root_level)
        logging.getLogger("httpx").setLevel(httpx_level)

    def _install_backend(self) -> JaegerBackend:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="unauthorized")

        backend = JaegerBackend(url=f"https://u:{self.MARKER}@h.test")
        # slow_request_threshold_ms=0 so the slow-request WARNING (which
        # prints the request URL) fires on every request too.
        transport = _RetryingTransport(
            wrapped=httpx.MockTransport(handler),
            slow_request_threshold_ms=0.0,
            allowed_host="h.test",
        )
        backend._client = httpx.AsyncClient(
            base_url=backend.url, transport=transport, follow_redirects=True
        )
        server._config = _config()
        server._backend = backend
        return backend

    def test_ready_body_and_logs_are_free_of_the_credential(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._install_backend()
        app = server.mcp.http_app(transport="streamable-http")

        with caplog.at_level(logging.INFO), TestClient(app) as client:
            response = client.get("/ready")

        assert response.status_code == 503
        assert self.MARKER not in response.text
        body = response.json()
        assert "401" in body["health_check"]["error"]
        assert "401" in body["list_services"]["error"]
        assert self.MARKER not in caplog.text
        # Not vacuous: the slow-request WARNING did fire and named the path.
        assert any("Slow backend request" in r.message for r in caplog.records)
        assert any("/api/services" in r.message for r in caplog.records)

    async def test_tool_error_content_and_logs_are_free_of_the_credential(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._install_backend()

        with caplog.at_level(logging.INFO):
            async with Client(server.mcp) as client:
                result = await client.call_tool("list_services", {}, raise_on_error=False)

        assert result.is_error is True
        text = " ".join(getattr(c, "text", "") for c in result.content)
        assert "401" in text
        assert self.MARKER not in text
        assert self.MARKER not in caplog.text
        assert any("Error executing list_services" in r.message for r in caplog.records)


class TestApplyLogLevel:
    @pytest.fixture(autouse=True)
    def _restore(self) -> Generator[None]:
        root_level = logging.getLogger().level
        httpx_level = logging.getLogger("httpx").level
        yield
        logging.getLogger().setLevel(root_level)
        logging.getLogger("httpx").setLevel(httpx_level)

    @pytest.mark.parametrize("level", ["INFO", "WARNING", "ERROR"])
    def test_above_debug_quiets_httpx_request_lines(self, level: str) -> None:
        """httpx logs 'HTTP Request: GET <full url>' at INFO on every
        request - the one URL sink that cannot be redacted at creation time."""
        server._apply_log_level(level)

        assert logging.getLogger().level == logging.getLevelName(level)
        assert logging.getLogger("httpx").level == logging.WARNING
        assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)

    def test_debug_keeps_httpx_request_lines(self) -> None:
        """Operators debugging connectivity rely on those lines; at DEBUG
        they stay and pass through the redacting handler filter."""
        server._apply_log_level("DEBUG")

        assert logging.getLogger().level == logging.DEBUG
        assert logging.getLogger("httpx").level == logging.NOTSET
        assert logging.getLogger("httpx").isEnabledFor(logging.INFO)


class TestRootLogHandlerRedaction:
    """server.py builds the root stderr handler itself so the userinfo
    filter sits on the HANDLER (a filter on the root logger never sees
    records propagated from httpx/fastmcp/opentelemetry_mcp.* loggers).
    Under pytest basicConfig is a no-op (the root already has pytest's
    handlers), so drive the handler object directly."""

    def test_handler_carries_the_filter(self) -> None:
        assert any(isinstance(f, UserinfoRedactingFilter) for f in server._ROOT_LOG_HANDLER.filters)

    def test_handler_emits_redacted_text_for_a_foreign_logger(self) -> None:
        marker = "VFY-PW-root"
        stream = io.StringIO()
        previous = server._ROOT_LOG_HANDLER.setStream(stream)
        try:
            record = logging.LogRecord(
                name="httpx",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="HTTP Request: %s %s",
                args=("GET", f"https://u:{marker}@h.test/api/services"),
                exc_info=None,
            )
            server._ROOT_LOG_HANDLER.handle(record)
        finally:
            server._ROOT_LOG_HANDLER.setStream(previous)  # type: ignore[arg-type]

        out = stream.getvalue()
        assert marker not in out
        assert "https://[REDACTED]@h.test/api/services" in out
        assert "httpx" in out and "INFO" in out


class TestConfigErrorsNeverEchoUserinfo:
    """pydantic's ValidationError str() echoes input_value='<the URL>', and
    exc_info=True would print it again in a traceback - so the config phase
    of main() and doctor report the redacted text only, without a
    traceback. CliRunner never sees log lines (the root handler is bound to
    the real stderr at import), so exit codes and stdout are asserted here
    and the log content via caplog."""

    MARKER = "VFY-PW-cli"

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BACKEND_TYPE", "jaeger")
        monkeypatch.setenv("BACKEND_URL", "http://localhost:16686")
        monkeypatch.delenv("BACKEND_API_KEY", raising=False)
        monkeypatch.delenv("SECONDARY_BACKEND_TYPE", raising=False)
        monkeypatch.delenv("SECONDARY_BACKEND_URL", raising=False)

    def test_print_config_with_a_userinfo_url_flag_exits_non_zero_and_redacts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = CliRunner()
        with caplog.at_level(logging.ERROR), patch.object(server, "mcp") as mock_mcp:
            mock_mcp.run = MagicMock()
            result = runner.invoke(
                server.main,
                [
                    "--print-config",
                    "--backend",
                    "jaeger",
                    "--url",
                    f"https://u:{self.MARKER}@h.test",
                ],
            )

        assert result.exit_code == 1
        assert self.MARKER not in result.output
        assert self.MARKER not in caplog.text
        (record,) = [r for r in caplog.records if "Configuration error" in r.message]
        assert record.exc_info is None
        assert "must not embed credentials" in record.message
        mock_mcp.run.assert_not_called()

    def test_userinfo_in_backend_url_env_var_is_rejected_without_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The env path raises pydantic's ValidationError (which echoes the
        raw input) rather than the CLI path's plain ValueError."""
        monkeypatch.setenv("BACKEND_URL", f"https://u:{self.MARKER}@h.test")
        runner = CliRunner()
        with caplog.at_level(logging.ERROR), patch.object(server, "mcp") as mock_mcp:
            mock_mcp.run = MagicMock()
            result = runner.invoke(server.main, ["--print-config"])

        assert result.exit_code == 1
        assert self.MARKER not in result.output
        assert self.MARKER not in caplog.text
        (record,) = [r for r in caplog.records if "Configuration error" in r.message]
        assert record.exc_info is None
        assert "[REDACTED]@h.test" in record.message

    def test_password_containing_at_in_backend_url_env_var_is_fully_redacted(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """pydantic accepts a literal '@' inside the password (WHATWG: host
        h.test, password p%40<marker>) and echoes the raw string in
        input_value; a first-'@' redaction would leave the password's tail
        in the one message guaranteed to be emitted for a rejected URL."""
        monkeypatch.setenv("BACKEND_URL", f"https://u:p@{self.MARKER}@h.test")
        runner = CliRunner()
        with caplog.at_level(logging.ERROR), patch.object(server, "mcp") as mock_mcp:
            mock_mcp.run = MagicMock()
            result = runner.invoke(server.main, ["--print-config"])

        assert result.exit_code == 1
        assert self.MARKER not in result.output
        assert self.MARKER not in caplog.text
        (record,) = [r for r in caplog.records if "Configuration error" in r.message]
        assert record.exc_info is None
        assert "[REDACTED]@h.test" in record.message
        assert "p@" not in record.message

    def test_doctor_prints_a_redacted_configuration_failure(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            server.main,
            ["doctor", "--backend", "jaeger", "--url", f"https://u:{self.MARKER}@h.test"],
        )

        assert result.exit_code == 1
        assert "[FAIL] Configuration" in result.output
        assert "must not embed credentials" in result.output
        assert self.MARKER not in result.output

    def test_doctor_env_path_prints_a_redacted_configuration_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BACKEND_URL", f"https://u:{self.MARKER}@h.test")
        runner = CliRunner()
        result = runner.invoke(server.main, ["doctor"])

        assert result.exit_code == 1
        assert "[FAIL] Configuration" in result.output
        assert "[REDACTED]@h.test" in result.output
        assert self.MARKER not in result.output

    def test_doctor_backend_lines_are_redacted(self) -> None:
        """doctor's own [OK]/[FAIL] lines print the URL and the probe
        errors; _run_backend_checks redacts the errors and doctor the URL."""
        from opentelemetry_mcp.attributes import HealthCheckResponse

        fake_backend = AsyncMock()
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(
                status="unhealthy",
                backend="jaeger",
                url=f"https://u:{self.MARKER}@h.test",
                error=f"Client error '401 Unauthorized' for url 'https://u:{self.MARKER}@h.test/api'",
            )
        )
        fake_backend.list_services = AsyncMock(
            side_effect=RuntimeError(
                f"Client error '401' for url 'https://u:{self.MARKER}@h.test/api'"
            )
        )
        fake_backend.close = AsyncMock()
        unvalidated = ServerConfig(
            backend=BackendConfig(type="jaeger", url=HttpUrl("http://localhost"))
        )
        unvalidated.backend = BackendConfig.model_construct(
            type="jaeger",
            url=HttpUrl(f"https://u:{self.MARKER}@h.test/"),
            api_key=None,
            app_key=None,
            sentry_org=None,
            sentry_project=None,
            tempo_instance_id=None,
            aws_region=None,
            newrelic_account_id=None,
            honeycomb_dataset=None,
            timeout=30.0,
            max_response_bytes=64 * 1024 * 1024,
            environments=["prd"],
        )
        runner = CliRunner()
        with (
            patch.object(ServerConfig, "from_env", return_value=unvalidated),
            patch.object(server, "_create_backend", return_value=fake_backend),
        ):
            result = runner.invoke(server.main, ["doctor"])

        assert result.exit_code == 1
        assert "[OK] Backend constructed: jaeger @ https://h.test/" in result.output
        assert "[FAIL] Health check" in result.output
        assert "[FAIL] Connectivity probe" in result.output
        assert self.MARKER not in result.output

    def test_print_config_url_is_redacted_even_for_an_unvalidated_config(self) -> None:
        unvalidated = BackendConfig.model_construct(
            type="jaeger",
            url=HttpUrl(f"https://u:{self.MARKER}@h.test/"),
            api_key=None,
            app_key=None,
            sentry_org=None,
            sentry_project=None,
            tempo_instance_id=None,
            aws_region=None,
            newrelic_account_id=None,
            honeycomb_dataset=None,
            timeout=30.0,
            max_response_bytes=64 * 1024 * 1024,
            environments=["prd"],
        )

        printed = server._print_config_backend_dict(unvalidated)

        assert printed["url"] == "https://h.test/"
        assert self.MARKER not in json.dumps(printed)

    def test_stdio_startup_line_is_redacted(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        unvalidated = ServerConfig(
            backend=BackendConfig(type="jaeger", url=HttpUrl("http://localhost"))
        )
        unvalidated.backend = BackendConfig.model_construct(
            type="jaeger",
            url=HttpUrl(f"https://u:{self.MARKER}@h.test/"),
            api_key=None,
            app_key=None,
            sentry_org=None,
            sentry_project=None,
            tempo_instance_id=None,
            aws_region=None,
            newrelic_account_id=None,
            honeycomb_dataset=None,
            timeout=30.0,
            max_response_bytes=64 * 1024 * 1024,
            environments=["prd"],
        )
        runner = CliRunner()
        with (
            caplog.at_level(logging.INFO),
            patch.object(ServerConfig, "from_env", return_value=unvalidated),
            patch.object(server, "mcp") as mock_mcp,
        ):
            mock_mcp.run = MagicMock()
            result = runner.invoke(server.main, ["--transport", "stdio"])

        assert result.exit_code == 0, result.output
        (line,) = [r.message for r in caplog.records if "connected to:" in r.message]
        assert "https://h.test/" in line
        assert self.MARKER not in line
