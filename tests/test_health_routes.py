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

from starlette.testclient import TestClient

from opentelemetry_mcp import server
from opentelemetry_mcp.attributes import HealthCheckResponse


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
