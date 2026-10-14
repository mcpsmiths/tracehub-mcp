"""Tests for the streamable-HTTP session lifetime bound applied by
server._build_http_app (the single place main() constructs the HTTP app).

fastmcp 4.0.5's http_app() falls back to its own settings default of
``http_session_idle_timeout=None`` ("sessions never expire from inactivity"),
which overrides the mcp SDK's 1800 s default on the way into
StreamableHTTPSessionManager. Combined with the SDK's 10,000-session cap,
one peer could fill the shared session table with bare initialize POSTs and
make every new client's initialize answer 503 until restart. These tests
drive the real app through TestClient (lifespan on) so the SDK's idle reaper
actually runs.
"""

import time
from typing import Any

from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.testclient import TestClient

from opentelemetry_mcp.server import (
    _DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS,
    OriginValidationMiddleware,
    _build_http_app,
)
from tests.streamable_http_client import post_initialize, post_request

SHORT_IDLE_TIMEOUT_SECONDS = 0.5


def _session_manager(app: Any) -> Any:
    """The SDK session manager behind the /mcp route. Only populated once
    the app's lifespan has started (i.e. inside a TestClient context)."""
    for route in app.routes:
        if isinstance(route, Route) and route.path == "/mcp":
            endpoint: Any = route.endpoint  # fastmcp's StreamableHTTPASGIApp
            return endpoint.session_manager
    raise AssertionError("no /mcp route on the app")


def test_default_app_has_a_finite_positive_idle_timeout() -> None:
    app = _build_http_app(
        [Middleware(OriginValidationMiddleware)],
        session_idle_timeout_seconds=_DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS,
        stateless_http=False,
    )

    with TestClient(app):
        manager = _session_manager(app)

    assert manager.stateless is False
    assert isinstance(manager.session_idle_timeout, float)
    assert manager.session_idle_timeout > 0
    assert manager.session_idle_timeout == 1800.0


def test_stateless_mode_tracks_no_sessions() -> None:
    app = _build_http_app(
        [Middleware(OriginValidationMiddleware)],
        session_idle_timeout_seconds=_DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS,
        stateless_http=True,
    )

    with TestClient(app) as client:
        manager = _session_manager(app)
        response = post_initialize(client)
        assert response.status_code == 200
        assert "mcp-session-id" not in response.headers
        assert manager._server_instances == {}

    assert manager.stateless is True


def test_idle_sessions_are_reaped_and_their_ids_answer_404() -> None:
    app = _build_http_app(
        [Middleware(OriginValidationMiddleware)],
        session_idle_timeout_seconds=SHORT_IDLE_TIMEOUT_SECONDS,
        stateless_http=False,
    )

    with TestClient(app) as client:
        manager = _session_manager(app)
        assert manager.session_idle_timeout == SHORT_IDLE_TIMEOUT_SECONDS
        assert manager.stateless is False

        session_ids = []
        for _ in range(3):
            response = post_initialize(client)
            assert response.status_code == 200
            session_ids.append(response.headers["mcp-session-id"])
        assert len(set(session_ids)) == 3
        assert set(manager._server_instances) == set(session_ids)

        # No request in flight for 3x the idle timeout: the SDK's idle scope
        # fires for each session and the serve loop forgets it.
        deadline = time.monotonic() + 3 * SHORT_IDLE_TIMEOUT_SECONDS
        while manager._server_instances and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(manager._server_instances) == 0

        fresh = post_initialize(client)
        assert fresh.status_code == 200
        assert fresh.headers["mcp-session-id"] not in session_ids

        reaped = post_request(client, session_ids[0], "ping")
        assert reaped.status_code == 404
