"""Tests for the /health and /ready HTTP routes, registered via
FastMCP.custom_route (server.py's health_route/ready_route). These only
apply to the streamable-http transport - stdio mode has no HTTP server for
them to attach to.

Builds the real Starlette app FastMCP assembles for streamable-http (the
one gap the exploration flagged: no existing test drives a real HTTP
request against a real route), reusing test_http_origin_middleware.py's
TestClient pattern and test_server.py's TestDoctorCli AsyncMock-backend
pattern for mocking _get_backend.
"""

from unittest.mock import AsyncMock, patch

from starlette.middleware import Middleware
from starlette.testclient import TestClient

from opentelemetry_mcp import server
from opentelemetry_mcp.attributes import HealthCheckResponse
from opentelemetry_mcp.server import OriginValidationMiddleware
from tests.streamable_http_client import post_initialize


def _healthy_backend() -> AsyncMock:
    fake_backend = AsyncMock()
    fake_backend.health_check = AsyncMock(
        return_value=HealthCheckResponse(status="healthy", backend="jaeger", url="http://x")
    )
    fake_backend.list_services = AsyncMock(return_value=["svc-a", "svc-b"])
    fake_backend.close = AsyncMock()
    return fake_backend


def _build_client() -> TestClient:
    app = server.mcp.http_app(transport="streamable-http")
    return TestClient(app)


def test_health_returns_200_without_touching_backend() -> None:
    """Liveness must not do backend I/O - patch _get_backend with something
    that raises if called at all, to prove /health never touches it."""
    with patch.object(
        server, "_get_backend", AsyncMock(side_effect=AssertionError("must not be called"))
    ):
        with _build_client() as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_200_when_backend_healthy() -> None:
    fake_backend = _healthy_backend()
    with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
        with _build_client() as client:
            response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["health_check"]["status"] == "healthy"
    assert body["list_services"]["count"] == 2


def test_ready_returns_503_when_health_check_unhealthy() -> None:
    fake_backend = _healthy_backend()
    fake_backend.health_check = AsyncMock(
        return_value=HealthCheckResponse(
            status="unhealthy", backend="jaeger", url="http://x", error="connection refused"
        )
    )
    with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
        with _build_client() as client:
            response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["health_check"]["error"] == "connection refused"


def test_ready_returns_503_when_list_services_raises() -> None:
    fake_backend = _healthy_backend()
    fake_backend.list_services = AsyncMock(side_effect=RuntimeError("timeout"))
    with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
        with _build_client() as client:
            response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["list_services"]["error"] == "timeout"


def test_ready_returns_503_when_get_backend_itself_raises() -> None:
    """_get_backend can raise (e.g. server configuration not set yet) -
    /ready must report 503, not an unhandled 500."""
    with patch.object(
        server, "_get_backend", AsyncMock(side_effect=RuntimeError("Server configuration not set"))
    ):
        with _build_client() as client:
            response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["error"] == "Server configuration not set"


class TestReadyNeverEchoesUrlUserinfo:
    """/ready answers unauthenticated peers; the error strings it copies out
    of health_check/list_services embed the request URL (httpx's
    HTTPStatusError text), so any credential a bypassed config left in the
    URL must be redacted inside _run_backend_checks - doctor inherits it."""

    MARKER = "VFY-PW-ready"
    ERROR = f"Client error '401 Unauthorized' for url 'https://u:{MARKER}@h.test/api/services'"

    def test_health_check_error_is_redacted(self) -> None:
        fake_backend = _healthy_backend()
        fake_backend.health_check = AsyncMock(
            return_value=HealthCheckResponse(
                status="unhealthy",
                backend="jaeger",
                url=f"https://u:{self.MARKER}@h.test",
                error=self.ERROR,
            )
        )
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                response = client.get("/ready")

        assert response.status_code == 503
        assert self.MARKER not in response.text
        assert "h.test/api/services" in response.json()["health_check"]["error"]

    def test_health_check_exception_and_list_services_error_are_redacted(self) -> None:
        fake_backend = _healthy_backend()
        fake_backend.health_check = AsyncMock(side_effect=RuntimeError(self.ERROR))
        fake_backend.list_services = AsyncMock(side_effect=RuntimeError(self.ERROR))
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with _build_client() as client:
                response = client.get("/ready")

        assert response.status_code == 503
        assert self.MARKER not in response.text
        body = response.json()
        assert "[REDACTED]@h.test" in body["health_check"]["error"]
        assert "[REDACTED]@h.test" in body["list_services"]["error"]

    def test_get_backend_failure_is_redacted(self) -> None:
        with patch.object(server, "_get_backend", AsyncMock(side_effect=RuntimeError(self.ERROR))):
            with _build_client() as client:
                response = client.get("/ready")

        assert response.status_code == 503
        assert self.MARKER not in response.text


class TestReadyBehindHostValidation:
    """/ready drives real backend calls, so a DNS-rebound browser's
    Origin-less GET must be refused by the Host check before the route runs
    when the listener is loopback-bound (the shape main() now defaults to)."""

    def _build_client(self) -> TestClient:
        app = server.mcp.http_app(
            transport="streamable-http",
            middleware=[Middleware(OriginValidationMiddleware, bind_host="127.0.0.1")],
        )
        return TestClient(app)

    def test_foreign_host_without_origin_is_421_and_touches_no_backend(self) -> None:
        fake_backend = _healthy_backend()
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with self._build_client() as client:
                response = client.get("/ready", headers={"host": "attacker.example:8000"})

        assert response.status_code == 421
        fake_backend.health_check.assert_not_awaited()
        fake_backend.list_services.assert_not_awaited()

    def test_loopback_host_still_reaches_ready(self) -> None:
        fake_backend = _healthy_backend()
        with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
            with self._build_client() as client:
                response = client.get("/ready", headers={"host": "localhost:8000"})

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_loopback_host_still_opens_an_mcp_session(self) -> None:
        with self._build_client() as client:
            response = post_initialize(client, headers={"host": "127.0.0.1:8000"})

        assert response.status_code == 200
        assert response.headers.get("mcp-session-id")


def test_ready_never_closes_the_cached_backend() -> None:
    """The key behavioral difference from `doctor`: /ready reuses the
    server's cached backend and must never close it, or every subsequent
    real tool call sharing that instance would break."""
    fake_backend = _healthy_backend()
    with patch.object(server, "_get_backend", AsyncMock(return_value=fake_backend)):
        with _build_client() as client:
            response = client.get("/ready")

    assert response.status_code == 200
    fake_backend.close.assert_not_called()
