"""Tests for the streamable-HTTP listener's caller authentication: the
OperatorTokenVerifier that binds every /mcp request to the operator's shared
bearer token, the startup gate in main() that refuses to serve HTTP on a
non-loopback interface without it, and the fastmcp RequireAuthMiddleware
wiring that protects /mcp while leaving /health and /ready (the Docker
HEALTHCHECK probe) reachable.

Drives the real Starlette app FastMCP assembles - exactly as main() builds
it - through starlette's TestClient with raw JSON-RPC bodies (see
tests/streamable_http_client.py), because the in-memory fastmcp.Client never
touches the HTTP layer where authentication is enforced.

The fixture credential is assembled from fragments (the pattern
test_observability.py already uses) so this file's own static content has no
contiguous secret-shaped literal for a scanner to flag.
"""

# server.py imports hmac as a module and calls hmac.compare_digest through it,
# so `server.hmac` exists at runtime but mypy's strict --no-implicit-reexport
# treats it as undefined - same friction test_server.py's docstring documents.
# mypy: disable-error-code="attr-defined"

import hmac
import json
from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner
from starlette.middleware import Middleware
from starlette.testclient import TestClient

from opentelemetry_mcp import server
from opentelemetry_mcp.attributes import HealthCheckResponse
from opentelemetry_mcp.server import (
    OperatorTokenVerifier,
    OriginValidationMiddleware,
    RateLimitMiddleware,
)
from tests.streamable_http_client import open_session, post_initialize, post_request, sse_json

# 46 characters, comfortably over the 32-character floor.
TEST_TOKEN = "-".join(["unit", "test", "operator", "credential", "0123456789abcdef"])
SHORT_VALUE = "-".join(["too", "short"])
# The test subject is exactly the all-interfaces bind the gate exists to guard.
ALL_INTERFACES = "0.0.0.0"  # noqa: S104


@pytest.fixture(autouse=True)
def _reset_mcp_auth() -> Generator[None]:
    """server.mcp is a module-level singleton; its .auth attribute is what
    main() mutates to install the verifier, so every test here must leave
    it as it found it (None) or later HTTP tests would start answering 401."""
    server.mcp.auth = None
    yield
    server.mcp.auth = None


@pytest.fixture
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKEND_TYPE", "jaeger")
    monkeypatch.setenv("BACKEND_URL", "http://localhost:16686")
    for name in (
        "BACKEND_API_KEY",
        "BACKEND_APP_KEY",
        "BACKEND_SENTRY_ORG",
        "BACKEND_SENTRY_PROJECT",
        "BACKEND_ENVIRONMENTS",
        "MCP_HTTP_BEARER_TOKEN",
        "MCP_ALLOW_UNAUTHENTICATED_HTTP",
        "MCP_HOST",
        "MCP_TRANSPORT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)


def _build_app_as_main_does() -> TestClient:
    app = server.mcp.http_app(
        transport="streamable-http",
        middleware=[
            Middleware(OriginValidationMiddleware),
            Middleware(RateLimitMiddleware, max_requests=100, window_seconds=60.0),
        ],
    )
    return TestClient(app)


def _healthy_backend() -> AsyncMock:
    fake_backend = AsyncMock()
    fake_backend.health_check = AsyncMock(
        return_value=HealthCheckResponse(status="healthy", backend="jaeger", url="http://x")
    )
    fake_backend.list_services = AsyncMock(return_value=["svc-a"])
    # _run_backend_checks probes through the never-cached list_services_uncached.
    fake_backend.list_services_uncached = fake_backend.list_services
    return fake_backend


def _bearer(value: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


class TestOperatorTokenVerifier:
    async def test_exact_token_yields_operator_access_token(self) -> None:
        verifier = OperatorTokenVerifier(TEST_TOKEN)

        access = await verifier.verify_token(TEST_TOKEN)

        assert access is not None
        assert access.client_id == "operator"
        assert access.scopes == []

    async def test_token_differing_in_last_character_is_rejected(self) -> None:
        verifier = OperatorTokenVerifier(TEST_TOKEN)

        assert await verifier.verify_token(TEST_TOKEN[:-1] + "X") is None
        assert await verifier.verify_token("") is None

    async def test_comparison_is_constant_time(self) -> None:
        """The whole point of a dedicated verifier over fastmcp's
        StaticTokenVerifier (whose docstring forbids production use) is the
        constant-time compare - pin that hmac.compare_digest is the primitive."""
        verifier = OperatorTokenVerifier(TEST_TOKEN)
        with patch.object(server.hmac, "compare_digest", wraps=hmac.compare_digest) as spy:
            await verifier.verify_token(TEST_TOKEN)

        spy.assert_called_once()


class TestMcpRouteRequiresBearerToken:
    def test_initialize_without_authorization_is_401_with_challenge(self) -> None:
        server.mcp.auth = OperatorTokenVerifier(TEST_TOKEN)

        with _build_app_as_main_does() as client:
            response = post_initialize(client)

        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Bearer")
        assert "mcp-session-id" not in response.headers

    def test_wrong_token_is_401_and_exact_token_opens_a_session(self) -> None:
        server.mcp.auth = OperatorTokenVerifier(TEST_TOKEN)

        with _build_app_as_main_does() as client:
            wrong = post_initialize(client, headers=_bearer(TEST_TOKEN + "x"))
            assert wrong.status_code == 401
            assert "mcp-session-id" not in wrong.headers

            session_id = open_session(client, headers=_bearer(TEST_TOKEN))
            listed = post_request(client, session_id, "tools/list", headers=_bearer(TEST_TOKEN))

        assert listed.status_code == 200
        assert len(sse_json(listed)["result"]["tools"]) == 19

    def test_health_and_ready_stay_reachable_without_a_token(self) -> None:
        """RequireAuthMiddleware wraps only the /mcp route; custom routes are
        appended as plain routes, so docker_healthcheck.py (GET /ready, no
        Authorization header) keeps working with auth configured."""
        server.mcp.auth = OperatorTokenVerifier(TEST_TOKEN)

        with patch.object(server, "_get_backend", AsyncMock(return_value=_healthy_backend())):
            with _build_app_as_main_does() as client:
                health = client.get("/health")
                ready = client.get("/ready")

        assert health.status_code == 200
        assert ready.status_code == 200


@pytest.mark.usefixtures("_clean_env")
class TestMainHttpListenerGate:
    def test_network_bind_without_token_or_opt_out_is_a_usage_error(self) -> None:
        runner = CliRunner()
        with patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls:
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(server.main, ["--transport", "http", "--host", ALL_INTERFACES])

        assert result.exit_code == 2, result.output
        assert "MCP_HTTP_BEARER_TOKEN" in result.output
        assert "--allow-unauthenticated-http" in result.output
        mock_server_cls.return_value.serve.assert_not_awaited()

    def test_explicit_opt_out_serves_unauthenticated_on_a_network_bind(self) -> None:
        runner = CliRunner()
        with patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls:
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(
                server.main,
                ["--transport", "http", "--host", ALL_INTERFACES, "--allow-unauthenticated-http"],
            )

        assert result.exit_code == 0, result.output
        mock_server_cls.return_value.serve.assert_awaited_once()
        assert server.mcp.auth is None

    def test_plain_http_transport_defaults_to_loopback_and_needs_no_token(self) -> None:
        runner = CliRunner()
        with patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls:
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(server.main, ["--transport", "http"])

        assert result.exit_code == 0, result.output
        config_arg = mock_server_cls.call_args.args[0]
        assert config_arg.host == "127.0.0.1"
        mock_server_cls.return_value.serve.assert_awaited_once()
        assert server.mcp.auth is None

    def test_token_from_env_installs_the_verifier_for_a_network_bind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", TEST_TOKEN)
        runner = CliRunner()
        with patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls:
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(server.main, ["--transport", "http", "--host", ALL_INTERFACES])

        assert result.exit_code == 0, result.output
        mock_server_cls.return_value.serve.assert_awaited_once()
        assert isinstance(server.mcp.auth, OperatorTokenVerifier)

    def test_short_token_is_rejected_with_a_usage_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", SHORT_VALUE)
        runner = CliRunner()
        with patch("opentelemetry_mcp.server.uvicorn.Server") as mock_server_cls:
            mock_server_cls.return_value.serve = AsyncMock()
            result = runner.invoke(server.main, ["--transport", "http", "--host", ALL_INTERFACES])

        assert result.exit_code == 2, result.output
        assert "32" in result.output
        mock_server_cls.return_value.serve.assert_not_awaited()
        assert server.mcp.auth is None

    def test_token_with_stdio_transport_logs_a_warning_and_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", TEST_TOKEN)
        runner = CliRunner()
        with (
            patch.object(server.mcp, "run") as mock_run,
            caplog.at_level("WARNING", logger="opentelemetry_mcp.server"),
        ):
            result = runner.invoke(server.main, ["--transport", "stdio"])

        assert result.exit_code == 0, result.output
        mock_run.assert_called_once_with(transport="stdio")
        assert server.mcp.auth is None
        assert any("MCP_HTTP_BEARER_TOKEN" in r.getMessage() for r in caplog.records)
        assert TEST_TOKEN not in caplog.text

    def test_print_config_reports_token_presence_never_its_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCP_HTTP_BEARER_TOKEN", TEST_TOKEN)
        runner = CliRunner()
        result = runner.invoke(server.main, ["--print-config", "--transport", "http"])

        assert result.exit_code == 0, result.output
        assert TEST_TOKEN not in result.output
        resolved = json.loads(result.output)
        assert resolved["http_bearer_token_set"] is True
        assert resolved["allow_unauthenticated_http"] is False
        assert resolved["host"] == "127.0.0.1"
